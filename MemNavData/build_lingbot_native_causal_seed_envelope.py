#!/usr/bin/env python3
"""Expand a legacy candidate teacher into exact causal manifest sessions.

A Goal-B image can name more than one deployment state (for example t0 and a
later midpoint).  Therefore a legacy ``scene/episode/revisit_b`` session cannot
be assigned one decision by inference.  This CPU-only bridge instead iterates
the pinned manifest samples and creates one independent candidate set per
``sample_id``.  Candidate anchors are retained only when every requested
neighbor is available before that sample's decision.

The output remains a teacher CSV: co-visibility and pose-error values are
training targets, never deployment inputs.  Its new
``causal_manifest_sample_id`` column is the sole authoritative join consumed by
``diag_lingbot_goal_loop_closure.py``; all other causal fields are revalidated
there from the pinned manifest and scale artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Optional

import pandas as pd

try:
    from MemNavData.external_causal_scale_contract import (
        CAUSAL_SAMPLE_ID_COLUMN,
        ExternalCausalScaleContract,
        ExternalCausalScalePins,
        sha256_file,
    )
except ModuleNotFoundError:  # direct script invocation
    from external_causal_scale_contract import (  # type: ignore
        CAUSAL_SAMPLE_ID_COLUMN,
        ExternalCausalScaleContract,
        ExternalCausalScalePins,
        sha256_file,
    )


REQUIRED_TEACHER_COLUMNS = {
    "session_id",
    "scene",
    "episode",
    "kind",
    "query_path",
    "candidate_path",
    "candidate_frame",
    "dino_cosine",
    "teacher_covis",
}


def atomic_write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary_path = Path(temporary)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def atomic_write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary_path = Path(temporary)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            frame.to_csv(handle, index=False, lineterminator="\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def build_seed_envelope(
    *,
    teacher: pd.DataFrame,
    contract: ExternalCausalScaleContract,
    split_role: str,
    kind: str,
    goal_roles: tuple[str, ...],
    neighbor_offsets: tuple[int, ...],
    selected_sample_ids: Optional[tuple[str, ...]] = None,
) -> tuple[pd.DataFrame, dict]:
    missing = REQUIRED_TEACHER_COLUMNS - set(teacher.columns)
    if missing:
        raise ValueError(f"teacher CSV missing columns: {sorted(missing)}")
    if CAUSAL_SAMPLE_ID_COLUMN in teacher.columns:
        raise ValueError(
            f"teacher already contains {CAUSAL_SAMPLE_ID_COLUMN}; refusing "
            "to overwrite an existing causal binding"
        )
    if split_role not in ("train", "development"):
        raise ValueError("causal seed envelope role must be train or development")
    offsets = tuple(sorted(set(map(int, neighbor_offsets))))
    if not offsets or 0 not in offsets:
        raise ValueError("neighbor offsets must be non-empty and include zero")
    if selected_sample_ids is None:
        sample_ids = contract.selected_sample_ids(
            split_role=split_role, goal_roles=goal_roles
        )
    else:
        sample_ids = tuple(dict.fromkeys(map(str, selected_sample_ids)))
        if not sample_ids:
            raise ValueError("explicit causal sample selection is empty")

    candidates = teacher.loc[teacher["kind"].astype(str).eq(kind)].copy()
    if candidates.empty:
        raise ValueError(f"teacher contains no rows for kind={kind}")
    query_sha_cache: dict[Path, str] = {}
    output_rows: list[dict] = []
    sample_reports = []
    missing_samples = []
    for sample_id in sample_ids:
        descriptor = contract.sample_descriptor(sample_id)
        if descriptor["split_role"] != split_role:
            raise ValueError(f"explicit sample crosses split role: {sample_id}")
        if descriptor["goal_role"] not in goal_roles:
            raise ValueError(
                f"explicit sample crosses selected goal roles: {sample_id}"
            )
        scene = str(descriptor["scene"])
        episode = str(descriptor["source_episode"])
        goal_sha = str(descriptor["goal_sha256"])
        decision = int(descriptor["decision_frame"])
        group = candidates.loc[
            candidates["scene"].astype(str).eq(scene)
            & candidates["episode"].astype(str).eq(episode)
        ].copy()
        query_matches = []
        for index, row in group.iterrows():
            query_path = Path(str(row["query_path"])).resolve()
            if not query_path.is_file():
                raise FileNotFoundError(query_path)
            query_sha = query_sha_cache.get(query_path)
            if query_sha is None:
                query_sha = sha256_file(query_path)
                query_sha_cache[query_path] = query_sha
            if query_sha == goal_sha:
                query_matches.append(index)
        group = group.loc[query_matches]
        anchors_before_filter = int(len(group))
        if not group.empty:
            anchors = group["candidate_frame"].astype(int)
            eligible = anchors.add(min(offsets)).ge(
                contract.num_scale_frames
            ) & anchors.add(max(offsets)).lt(decision)
            group = group.loc[eligible]
        # Multiple legacy sessions may describe the same image/candidate pair.
        # Exact duplicate targets are harmless; conflicting targets are not.
        deduplicated: dict[tuple[str, int], dict] = {}
        for _, row in group.sort_values(
            ["candidate_frame", "candidate_path", "session_id"]
        ).iterrows():
            candidate_path = Path(str(row["candidate_path"])).resolve()
            candidate_frame = int(row["candidate_frame"])
            binding = contract.bind_seed(
                manifest_sample_id=sample_id,
                scene=scene,
                episode=episode,
                query_path=Path(str(descriptor["goal_path"])),
                candidate_path=candidate_path,
                candidate_frame=candidate_frame,
                neighbor_offsets=offsets,
                expected_split_role=split_role,
            )
            record = row.to_dict()
            record["teacher_source_session_id"] = str(record["session_id"])
            record["session_id"] = sample_id
            record["query_path"] = str(descriptor["goal_path"])
            record[CAUSAL_SAMPLE_ID_COLUMN] = sample_id
            record.update(
                {
                    "causal_split_role": binding.split_role,
                    "causal_goal_role": binding.goal_role,
                    "causal_decision_frame": binding.decision_frame,
                    "causal_prefix_sha256": binding.causal_prefix_sha256,
                    "causal_goal_sha256": binding.goal_sha256,
                }
            )
            key = str(candidate_path), candidate_frame
            previous = deduplicated.get(key)
            if previous is not None:
                for column in ("teacher_covis", "dino_cosine"):
                    if float(previous[column]) != float(record[column]):
                        raise ValueError(
                            f"conflicting duplicate teacher target for "
                            f"{sample_id}/{key}: {column}"
                        )
                continue
            deduplicated[key] = record
        rows = list(deduplicated.values())
        if not rows:
            missing_samples.append(sample_id)
        output_rows.extend(rows)
        sample_reports.append(
            {
                "sample_id": sample_id,
                "scene": scene,
                "source_episode": episode,
                "goal_role": descriptor["goal_role"],
                "decision_frame": decision,
                "matching_goal_rows": anchors_before_filter,
                "causal_candidate_rows": len(rows),
            }
        )
    if missing_samples:
        raise RuntimeError(
            f"teacher cannot exact-cover selected causal samples: {missing_samples[:8]}"
        )
    output = pd.DataFrame(output_rows)
    if output.empty:
        raise RuntimeError("causal seed envelope produced no rows")
    output = output.sort_values(
        ["session_id", "dino_cosine", "candidate_frame"],
        ascending=[True, False, True],
        kind="mergesort",
    ).reset_index(drop=True)
    report = {
        "binding_approved": True,
        "deployment_approved": False,
        "reason": (
            "exact causal training-seed envelope; teacher targets remain "
            "forbidden at deployment"
        ),
        "split_role": split_role,
        "kind": kind,
        "goal_roles": list(goal_roles),
        "neighbor_offsets": list(offsets),
        "sample_count": len(sample_ids),
        "session_count": int(output["session_id"].nunique()),
        "row_count": int(len(output)),
        "samples": sample_reports,
        "external_causal_scale": contract.summary(),
    }
    return output, report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher-csv", type=Path, required=True)
    parser.add_argument("--expected-teacher-sha256", required=True)
    parser.add_argument("--causal-manifest", type=Path, required=True)
    parser.add_argument("--expected-causal-manifest-sha256", required=True)
    parser.add_argument("--external-causal-scale-artifact", type=Path, required=True)
    parser.add_argument("--expected-external-causal-scale-sha256", required=True)
    parser.add_argument("--expected-external-scale-producer-sha256", required=True)
    parser.add_argument("--expected-external-scale-configuration-sha256", required=True)
    parser.add_argument("--expected-external-scale-lingbot-commit", required=True)
    parser.add_argument("--expected-external-scale-weights-sha256", required=True)
    parser.add_argument("--expected-external-scale-stream-source-sha256", required=True)
    parser.add_argument("--split-role", choices=("train", "development"), required=True)
    parser.add_argument("--kind", required=True)
    parser.add_argument("--goal-role", action="append", choices=("B", "C"))
    parser.add_argument("--sample-id", action="append")
    parser.add_argument("--neighbor-offset", type=int, action="append")
    parser.add_argument("--out-csv", type=Path, required=True)
    parser.add_argument("--out-report", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for path in (
        args.teacher_csv,
        args.causal_manifest,
        args.external_causal_scale_artifact,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    if sha256_file(args.teacher_csv) != args.expected_teacher_sha256:
        raise RuntimeError("teacher CSV SHA changed")
    if args.out_csv.exists() or args.out_report.exists():
        raise FileExistsError("causal seed output already exists")
    contract = ExternalCausalScaleContract(
        manifest_path=args.causal_manifest,
        artifact_path=args.external_causal_scale_artifact,
        pins=ExternalCausalScalePins(
            manifest_sha256=args.expected_causal_manifest_sha256,
            artifact_sha256=args.expected_external_causal_scale_sha256,
            producer_sha256=args.expected_external_scale_producer_sha256,
            configuration_sha256=(args.expected_external_scale_configuration_sha256),
            lingbot_commit=args.expected_external_scale_lingbot_commit,
            weights_sha256=args.expected_external_scale_weights_sha256,
            stream_source_sha256=(args.expected_external_scale_stream_source_sha256),
        ),
    )
    teacher = pd.read_csv(args.teacher_csv)
    output, report = build_seed_envelope(
        teacher=teacher,
        contract=contract,
        split_role=args.split_role,
        kind=args.kind,
        goal_roles=tuple(args.goal_role or ("B", "C")),
        neighbor_offsets=tuple(args.neighbor_offset or (0,)),
        selected_sample_ids=(tuple(args.sample_id) if args.sample_id else None),
    )
    atomic_write_csv(args.out_csv, output)
    report["provenance"] = {
        "teacher_csv": str(args.teacher_csv.resolve()),
        "teacher_csv_sha256": args.expected_teacher_sha256,
        "causal_manifest": str(args.causal_manifest.resolve()),
        "causal_manifest_sha256": args.expected_causal_manifest_sha256,
        "external_scale_artifact": str(args.external_causal_scale_artifact.resolve()),
        "external_scale_artifact_sha256": (args.expected_external_causal_scale_sha256),
        "rows_csv": str(args.out_csv.resolve()),
        "rows_csv_sha256": sha256_file(args.out_csv),
        "producer_source_sha256": sha256_file(Path(__file__)),
    }
    identity = {
        "teacher_csv_sha256": args.expected_teacher_sha256,
        "causal_manifest_sha256": args.expected_causal_manifest_sha256,
        "external_scale_artifact_sha256": (args.expected_external_causal_scale_sha256),
        "rows_csv_sha256": report["provenance"]["rows_csv_sha256"],
    }
    report["artifact_identity_sha256"] = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    atomic_write_json(args.out_report, report)
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
