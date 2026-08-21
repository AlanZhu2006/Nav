#!/usr/bin/env python3
"""Attach exact, uncertainty-aware SIFT/RANSAC evidence to a causal teacher.

This is an offline diagnostic builder.  It does not select thresholds, consume
development data, or modify a navigation policy.  Each output row keeps the
task-aligned co-visibility label from the pinned teacher and adds the evidence
available to the deployed geometry verifier.  Repeated deterministic RANSAC
draws expose whether a rejection is stable or merely under-supported.

Work is committed one session at a time.  A preempted Slurm job can therefore
resume without silently mixing inputs or losing completed sessions.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from functools import lru_cache
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Iterable, Mapping, Sequence

import numpy as np


SCHEMA_VERSION = "revisit_geometry_expert_evidence_v1"
REQUIRED_COLUMNS = frozenset(
    {
        "session_id",
        "sample_id",
        "split_role",
        "scene",
        "query_path",
        "candidate_path",
        "candidate_rank",
        "candidate_frame",
        "dino_cosine",
        "teacher_covis",
        "covisibility",
        "label",
        "query_content_sha256",
        "candidate_rgb_content_sha256",
        "manifest_sha256",
        "runtime_identity_sha256",
        "no_future",
    }
)
EVIDENCE_COLUMNS = (
    "geometry_query_keypoints",
    "geometry_candidate_keypoints",
    "geometry_matches",
    "geometry_inliers",
    "geometry_inlier_ratio",
    "geometry_essential_available",
    "geometry_pose_recovered",
    "geometry_error",
    "geometry_hard_pass",
    "geometry_ransac_repeats",
    "geometry_essential_available_rate",
    "geometry_pose_recovered_rate",
    "geometry_pass_rate",
    "geometry_inliers_min",
    "geometry_inliers_median",
    "geometry_inliers_max",
    "geometry_inliers_std",
    "geometry_inliers_json",
    "geometry_state",
)


_FALLBACK_MAPS: tuple[tuple[str, str], ...] = ()
_VERIFY_HASHES = True
_RANSAC_REPEATS = 5
_MIN_MATCHES = 20
_MIN_INLIERS = 12
_MIN_RATIO = 0.50


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_fallback_maps(values: Sequence[str]) -> tuple[tuple[str, str], ...]:
    result = []
    for value in values:
        if "=" not in value:
            raise ValueError(f"fallback map must be OLD=NEW, got {value!r}")
        old, new = value.split("=", 1)
        old, new = old.rstrip("/"), new.rstrip("/")
        if not old or not new or not old.startswith("/") or not new.startswith("/"):
            raise ValueError(f"fallback map must use absolute prefixes: {value!r}")
        result.append((old, new))
    return tuple(result)


def parse_explicit_true(value: object) -> bool:
    """Accept the CSV serializers used by audited teacher generations only."""
    return str(value).strip().lower() in {"1", "true"}


def resolve_input_path(raw: str) -> Path:
    path = Path(raw)
    if path.is_file():
        return path
    for old, new in _FALLBACK_MAPS:
        if raw == old or raw.startswith(old + "/"):
            candidate = Path(new + raw[len(old):])
            if candidate.is_file():
                return candidate
    raise FileNotFoundError(f"no readable source for {raw}")


@lru_cache(maxsize=2048)
def cached_sha256(path: str) -> str:
    return sha256(Path(path))


@lru_cache(maxsize=512)
def describe_image(path: str) -> tuple[np.ndarray, np.ndarray | None]:
    import cv2

    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"failed to decode {path}")
    sift = cv2.SIFT_create(nfeatures=4000)
    keypoints, descriptors = sift.detectAndCompute(image, None)
    points = np.asarray([item.pt for item in keypoints], dtype=np.float32)
    if points.size == 0:
        points = np.empty((0, 2), dtype=np.float32)
    return points, descriptors


@lru_cache(maxsize=128)
def load_intrinsic(candidate_path: str) -> np.ndarray:
    import pandas as pd

    path = Path(candidate_path)
    if path.parent.name != "observation.images.rgb":
        raise ValueError(f"unexpected RGB path layout: {path}")
    episode_root = path.parents[3]
    parquet = episode_root / "data" / "chunk-000" / "episode_000000.parquet"
    if not parquet.is_file():
        raise FileNotFoundError(parquet)
    frame = pd.read_parquet(parquet, columns=["observation.camera_intrinsic"])
    raw = frame.iloc[0]["observation.camera_intrinsic"]
    intrinsic = np.stack([np.asarray(row, dtype=np.float64) for row in raw])
    if intrinsic.shape != (3, 3) or not np.isfinite(intrinsic).all():
        raise ValueError(f"invalid camera intrinsic in {parquet}")
    return intrinsic


def pair_seed(row: Mapping[str, str], repeat: int) -> int:
    identity = (
        f"{row['session_id']}\0{row['candidate_path']}\0{repeat}"
    ).encode("utf-8")
    value = int.from_bytes(hashlib.sha256(identity).digest()[:4], "little")
    return int(value & 0x7FFFFFFF)


def recover_inliers(
    candidate_points: np.ndarray,
    query_points: np.ndarray,
    intrinsic: np.ndarray,
    seed: int,
) -> tuple[int, bool, bool]:
    import cv2

    cv2.setRNGSeed(int(seed))
    essential, ransac_mask = cv2.findEssentialMat(
        candidate_points,
        query_points,
        intrinsic,
        cv2.RANSAC,
        0.999,
        1.5,
    )
    if essential is None:
        return 0, False, False
    essential = np.asarray(essential, dtype=np.float64)
    if essential.shape == (3, 3):
        candidates = [essential]
    elif (
        essential.ndim == 2
        and essential.shape[1] == 3
        and essential.shape[0] % 3 == 0
    ):
        candidates = [
            essential[index:index + 3]
            for index in range(0, essential.shape[0], 3)
        ]
    else:
        candidates = []
    best = 0
    recovered_any = False
    for candidate in candidates:
        mask = None if ransac_mask is None else ransac_mask.copy()
        try:
            recovered = cv2.recoverPose(
                candidate,
                candidate_points,
                query_points,
                intrinsic,
                mask=mask,
            )[0]
        except cv2.error:
            continue
        recovered_any = True
        best = max(best, int(recovered))
    return best, True, recovered_any


def classify_geometry_state(
    *,
    descriptors_available: bool,
    matches: int,
    essential_rate: float,
    pass_rate: float,
) -> str:
    if not descriptors_available:
        return "insufficient_features"
    if matches < 8:
        return "insufficient_matches"
    if essential_rate <= 0.0:
        return "model_unavailable"
    if 0.0 < pass_rate < 1.0:
        return "unstable"
    if pass_rate >= 1.0:
        return "stable_support"
    return "estimable_reject"


def geometry_evidence(row: Mapping[str, str]) -> dict[str, object]:
    import cv2

    query = resolve_input_path(row["query_path"])
    candidate = resolve_input_path(row["candidate_path"])
    if _VERIFY_HASHES:
        expected_query = row["query_content_sha256"]
        expected_candidate = row["candidate_rgb_content_sha256"]
        actual_query = cached_sha256(str(query))
        actual_candidate = cached_sha256(str(candidate))
        if actual_query != expected_query:
            raise RuntimeError(
                f"query content mismatch for {row['session_id']}: "
                f"{actual_query} != {expected_query}"
            )
        if actual_candidate != expected_candidate:
            raise RuntimeError(
                f"candidate content mismatch for {row['session_id']}: "
                f"{actual_candidate} != {expected_candidate}"
            )

    query_points_all, query_desc = describe_image(str(query))
    candidate_points_all, candidate_desc = describe_image(str(candidate))
    query_count = int(len(query_points_all))
    candidate_count = int(len(candidate_points_all))
    descriptors_available = query_desc is not None and candidate_desc is not None
    matches = 0
    error: str | None = None
    inliers = [0 for _ in range(_RANSAC_REPEATS)]
    essential_available = [False for _ in range(_RANSAC_REPEATS)]
    pose_recovered = [False for _ in range(_RANSAC_REPEATS)]

    if not descriptors_available:
        error = "insufficient image features"
    else:
        pairs = cv2.BFMatcher().knnMatch(candidate_desc, query_desc, k=2)
        good = [
            pair[0]
            for pair in pairs
            if len(pair) == 2 and pair[0].distance < 0.75 * pair[1].distance
        ]
        matches = len(good)
        if matches < 8:
            error = "too few ratio-test matches"
        else:
            candidate_points = np.float32(
                [candidate_points_all[item.queryIdx] for item in good]
            )
            query_points = np.float32(
                [query_points_all[item.trainIdx] for item in good]
            )
            intrinsic = load_intrinsic(str(candidate))
            for repeat in range(_RANSAC_REPEATS):
                value, available, recovered = recover_inliers(
                    candidate_points,
                    query_points,
                    intrinsic,
                    pair_seed(row, repeat),
                )
                inliers[repeat] = value
                essential_available[repeat] = available
                pose_recovered[repeat] = recovered
            if not any(essential_available):
                error = "essential matrix unavailable"
            elif not any(pose_recovered):
                error = "pose recovery unavailable"

    canonical = int(inliers[0])
    canonical_ratio = float(canonical / matches) if matches else 0.0
    passes = [
        matches >= _MIN_MATCHES
        and value >= _MIN_INLIERS
        and (float(value / matches) if matches else 0.0) >= _MIN_RATIO
        for value in inliers
    ]
    pass_rate = float(sum(passes) / len(passes))
    essential_rate = float(sum(essential_available) / len(essential_available))
    recovered_rate = float(sum(pose_recovered) / len(pose_recovered))
    state = classify_geometry_state(
        descriptors_available=descriptors_available,
        matches=matches,
        essential_rate=essential_rate,
        pass_rate=pass_rate,
    )
    return {
        "geometry_query_keypoints": query_count,
        "geometry_candidate_keypoints": candidate_count,
        "geometry_matches": matches,
        "geometry_inliers": canonical,
        "geometry_inlier_ratio": canonical_ratio,
        "geometry_essential_available": int(essential_available[0]),
        "geometry_pose_recovered": int(pose_recovered[0]),
        "geometry_error": error or "",
        "geometry_hard_pass": int(passes[0]),
        "geometry_ransac_repeats": _RANSAC_REPEATS,
        "geometry_essential_available_rate": essential_rate,
        "geometry_pose_recovered_rate": recovered_rate,
        "geometry_pass_rate": pass_rate,
        "geometry_inliers_min": int(min(inliers)),
        "geometry_inliers_median": float(statistics.median(inliers)),
        "geometry_inliers_max": int(max(inliers)),
        "geometry_inliers_std": float(np.std(inliers, dtype=np.float64)),
        "geometry_inliers_json": json.dumps(inliers, separators=(",", ":")),
        "geometry_state": state,
    }


def _worker_init(
    fallback_maps: tuple[tuple[str, str], ...],
    verify_hashes: bool,
    repeats: int,
    min_matches: int,
    min_inliers: int,
    min_ratio: float,
) -> None:
    global _FALLBACK_MAPS, _VERIFY_HASHES, _RANSAC_REPEATS
    global _MIN_MATCHES, _MIN_INLIERS, _MIN_RATIO
    import cv2

    _FALLBACK_MAPS = fallback_maps
    _VERIFY_HASHES = verify_hashes
    _RANSAC_REPEATS = repeats
    _MIN_MATCHES = min_matches
    _MIN_INLIERS = min_inliers
    _MIN_RATIO = min_ratio
    cv2.setNumThreads(1)


def session_filename(session_id: str) -> str:
    digest = hashlib.sha256(session_id.encode("utf-8")).hexdigest()[:20]
    return f"{digest}.json"


def write_json_atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".partial", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, sort_keys=True, separators=(",", ":"))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def valid_shard(
    path: Path,
    *,
    input_sha: str,
    session_id: str,
    expected_indices: Sequence[int],
) -> bool:
    if not path.is_file():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return (
        payload.get("schema_version") == SCHEMA_VERSION
        and payload.get("input_sha256") == input_sha
        and payload.get("session_id") == session_id
        and [row.get("_input_index") for row in payload.get("rows", [])]
        == list(expected_indices)
    )


def process_session(
    session_id: str,
    rows: Sequence[dict[str, str]],
    input_indices: Sequence[int],
    input_sha: str,
    shard_path: str,
) -> dict[str, object]:
    started = time.perf_counter()
    output_rows = []
    for row, input_index in zip(rows, input_indices):
        output_rows.append(
            {
                "_input_index": input_index,
                **geometry_evidence(row),
            }
        )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "input_sha256": input_sha,
        "session_id": session_id,
        "rows": output_rows,
    }
    write_json_atomic(Path(shard_path), payload)
    return {
        "session_id": session_id,
        "rows": len(rows),
        "seconds": time.perf_counter() - started,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher-csv", type=Path, required=True)
    parser.add_argument("--expected-teacher-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split-role", default="train")
    parser.add_argument("--fallback-map", action="append", default=[])
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--ransac-repeats", type=int, default=5)
    parser.add_argument("--min-matches", type=int, default=20)
    parser.add_argument("--min-inliers", type=int, default=12)
    parser.add_argument("--min-inlier-ratio", type=float, default=0.50)
    parser.add_argument("--max-sessions", type=int, default=0)
    parser.add_argument("--skip-content-hash-verification", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.teacher_csv.is_file():
        raise FileNotFoundError(args.teacher_csv)
    if args.workers < 1 or args.ransac_repeats < 1 or args.max_sessions < 0:
        raise ValueError("workers/repeats must be positive and maximum non-negative")
    if (
        args.min_matches < 8
        or args.min_inliers < 1
        or not 0.0 <= args.min_inlier_ratio <= 1.0
    ):
        raise ValueError("invalid geometry thresholds")
    input_sha = sha256(args.teacher_csv)
    if input_sha != args.expected_teacher_sha256:
        raise RuntimeError(
            f"teacher SHA mismatch: {input_sha} != {args.expected_teacher_sha256}"
        )
    fallback_maps = parse_fallback_maps(args.fallback_map)

    with args.teacher_csv.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("teacher CSV has no header")
        missing = REQUIRED_COLUMNS - set(reader.fieldnames)
        if missing:
            raise ValueError(f"teacher CSV missing columns: {sorted(missing)}")
        selected_rows = [
            row for row in reader if row["split_role"] == args.split_role
        ]
        input_fields = tuple(reader.fieldnames)
    if not selected_rows:
        raise ValueError(f"no rows for split role {args.split_role!r}")
    if any(not parse_explicit_true(row["no_future"]) for row in selected_rows):
        raise RuntimeError("selected teacher includes non-causal rows")
    identity = [
        (row["session_id"], row["candidate_path"]) for row in selected_rows
    ]
    if len(identity) != len(set(identity)):
        raise ValueError("duplicate session/candidate identities")

    by_session: dict[str, list[tuple[int, dict[str, str]]]] = {}
    for index, row in enumerate(selected_rows):
        by_session.setdefault(row["session_id"], []).append((index, row))
    session_ids = sorted(by_session)
    if args.max_sessions:
        session_ids = session_ids[: args.max_sessions]
    selected_index_set = {
        index for session_id in session_ids for index, _ in by_session[session_id]
    }
    rows_for_run = [
        row for index, row in enumerate(selected_rows) if index in selected_index_set
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    shard_dir = args.output_dir / "session_shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    output_csv = args.output_dir / "geometry_evidence.csv"
    report_path = args.output_dir / "report.json"
    if output_csv.exists() or report_path.exists():
        raise FileExistsError(
            "final output already exists; use its receipt or choose a new directory"
        )

    jobs = []
    resumed_sessions = 0
    for session_id in session_ids:
        indexed = by_session[session_id]
        indices = [item[0] for item in indexed]
        shard = shard_dir / session_filename(session_id)
        if valid_shard(
            shard,
            input_sha=input_sha,
            session_id=session_id,
            expected_indices=indices,
        ):
            resumed_sessions += 1
            continue
        jobs.append(
            (
                session_id,
                [item[1] for item in indexed],
                indices,
                input_sha,
                str(shard),
            )
        )

    started = time.perf_counter()
    completed = resumed_sessions
    if jobs:
        with ProcessPoolExecutor(
            max_workers=args.workers,
            initializer=_worker_init,
            initargs=(
                fallback_maps,
                not args.skip_content_hash_verification,
                args.ransac_repeats,
                args.min_matches,
                args.min_inliers,
                args.min_inlier_ratio,
            ),
        ) as executor:
            futures = {
                executor.submit(process_session, *job): job[0] for job in jobs
            }
            for future in as_completed(futures):
                result = future.result()
                completed += 1
                print(
                    f"[geometry] {completed}/{len(session_ids)} "
                    f"{result['session_id']} rows={result['rows']} "
                    f"seconds={result['seconds']:.2f}",
                    flush=True,
                )

    evidence_by_index: dict[int, dict[str, object]] = {}
    for session_id in session_ids:
        indexed = by_session[session_id]
        shard = shard_dir / session_filename(session_id)
        indices = [item[0] for item in indexed]
        if not valid_shard(
            shard,
            input_sha=input_sha,
            session_id=session_id,
            expected_indices=indices,
        ):
            raise RuntimeError(f"missing or invalid completed shard: {shard}")
        payload = json.loads(shard.read_text(encoding="utf-8"))
        for evidence in payload["rows"]:
            input_index = int(evidence.pop("_input_index"))
            if input_index in evidence_by_index:
                raise RuntimeError(f"duplicate evidence index {input_index}")
            evidence_by_index[input_index] = evidence
    if set(evidence_by_index) != selected_index_set:
        raise RuntimeError("assembled evidence does not exactly cover selected rows")

    temporary = output_csv.with_suffix(".csv.partial")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=(*input_fields, *EVIDENCE_COLUMNS))
        writer.writeheader()
        for index, row in enumerate(selected_rows):
            if index not in selected_index_set:
                continue
            writer.writerow({**row, **evidence_by_index[index]})
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, output_csv)

    states: dict[str, int] = {}
    hard_pass = 0
    for evidence in evidence_by_index.values():
        state = str(evidence["geometry_state"])
        states[state] = states.get(state, 0) + 1
        hard_pass += int(evidence["geometry_hard_pass"])
    report = {
        "schema_version": SCHEMA_VERSION,
        "status": "train_only_evidence_complete",
        "deployment_approved": False,
        "teacher_csv": str(args.teacher_csv.resolve()),
        "teacher_sha256": input_sha,
        "split_role": args.split_role,
        "manifest_sha256_values": sorted(
            {row["manifest_sha256"] for row in rows_for_run}
        ),
        "runtime_identity_sha256_values": sorted(
            {row["runtime_identity_sha256"] for row in rows_for_run}
        ),
        "rows": len(rows_for_run),
        "sessions": len(session_ids),
        "scenes": len({row["scene"] for row in rows_for_run}),
        "labels": {
            str(label): sum(row["label"] == str(label) for row in rows_for_run)
            for label in (-1, 0, 1)
        },
        "geometry_state_counts": states,
        "hard_pass_rows": hard_pass,
        "ransac_repeats": args.ransac_repeats,
        "thresholds_reproduced": {
            "min_matches": args.min_matches,
            "min_inliers": args.min_inliers,
            "min_inlier_ratio": args.min_inlier_ratio,
        },
        "content_hash_verification": not args.skip_content_hash_verification,
        "fallback_maps": [f"{old}={new}" for old, new in fallback_maps],
        "workers": args.workers,
        "resumed_sessions": resumed_sessions,
        "elapsed_seconds": time.perf_counter() - started,
        "output_csv": str(output_csv.resolve()),
        "output_csv_sha256": sha256(output_csv),
    }
    write_json_atomic(report_path, report)
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
