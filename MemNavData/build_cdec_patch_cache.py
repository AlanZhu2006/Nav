#!/usr/bin/env python3
"""Build the frozen-DINO patch cache for certificate distillation.

This stage is deliberately label-blind: the teacher CSV is used only for the
ordered image-pair universe and its immutable identity.  Patch tokens come
from the exact LingBot DINOv2-L trunk already deployed by the project.  Labels
and privileged LightGlue/Fundamental fields remain in the separate CSV.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import tempfile
import time
from typing import Iterable

import numpy as np

try:
    from MemNavData.diag_patch_temporal_router import load_exact_patch_tokens
    from MemNavData.patch_temporal_router import (
        directional_patch_feature_names,
        directional_patch_relation_features,
    )
except ModuleNotFoundError:  # direct script invocation
    from diag_patch_temporal_router import load_exact_patch_tokens  # type: ignore
    from patch_temporal_router import (  # type: ignore
        directional_patch_feature_names,
        directional_patch_relation_features,
    )


SCHEMA_VERSION = "certificate_distilled_patch_cache_v2_fixed_dino_batch"
REQUIRED_COLUMNS = {
    "session_id", "scene", "query_relative_path",
    "candidate_relative_path", "candidate_rank", "dino_cosine", "no_future",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_relative(value: str) -> Path:
    posix = PurePosixPath(str(value))
    if (not value or posix.is_absolute() or ".." in posix.parts
            or "." in posix.parts):
        raise ValueError(f"unsafe relative path: {value!r}")
    return Path(*posix.parts)


def validate_table(frame, *, expected_rows: int, expected_sessions: int,
                   expected_scenes: int, expected_candidates: int) -> None:
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"teacher table lacks columns: {sorted(missing)}")
    if expected_rows and len(frame) != expected_rows:
        raise RuntimeError(f"row count changed: {len(frame)} != {expected_rows}")
    if frame["session_id"].nunique() != expected_sessions:
        raise RuntimeError("session universe changed")
    if frame["scene"].nunique() != expected_scenes:
        raise RuntimeError("scene universe changed")
    counts = frame.groupby("session_id", sort=False).size().to_numpy()
    if not np.all(counts == expected_candidates):
        raise RuntimeError("candidate count is not constant per session")
    no_future = frame["no_future"]
    if no_future.dtype == bool:
        valid = no_future.to_numpy()
    else:
        valid = no_future.astype(str).str.lower().isin(("true", "1")).to_numpy()
    if not valid.all():
        raise RuntimeError("teacher pair universe consumes a future frame")
    cosine = frame["dino_cosine"].to_numpy(dtype=np.float64)
    if not np.isfinite(cosine).all():
        raise ValueError("DINO cosine contains non-finite values")


def pair_universe(frame, image_root: Path):
    raw_paths = sorted(set(frame["query_relative_path"].astype(str)).union(
        frame["candidate_relative_path"].astype(str)))
    paths = [image_root / safe_relative(value) for value in raw_paths]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} referenced images are unavailable; first={missing[0]}")
    index = {value: position for position, value in enumerate(raw_paths)}
    query_index = np.asarray(
        [index[value] for value in frame["query_relative_path"].astype(str)],
        dtype=np.int32)
    candidate_index = np.asarray(
        [index[value] for value in frame["candidate_relative_path"].astype(str)],
        dtype=np.int32)
    return raw_paths, paths, query_index, candidate_index


def build_relations(tokens: np.ndarray, query_index: np.ndarray,
                    candidate_index: np.ndarray,
                    dino_cosine: np.ndarray) -> np.ndarray:
    rows = []
    for row, (query, candidate, cosine) in enumerate(zip(
            query_index, candidate_index, dino_cosine), start=1):
        rows.append(directional_patch_relation_features(
            tokens[int(query)], tokens[int(candidate)], float(cosine)))
        if row == 1 or row % 500 == 0:
            print(f"[relation] {row}/{len(query_index)}", flush=True)
    result = np.asarray(rows, dtype=np.float32)
    expected = (len(query_index), len(directional_patch_feature_names()))
    if result.shape != expected or not np.isfinite(result).all():
        raise RuntimeError(f"invalid relation cache {result.shape}, expected {expected}")
    return result


def atomic_npz(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary_path = Path(temporary)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            np.savez(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
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


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows-csv", type=Path, required=True)
    parser.add_argument("--expected-rows-sha256", required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--lingbot-repo", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--expected-weights-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--grid-size", type=int, default=8)
    parser.add_argument("--expected-rows", type=int, default=3840)
    parser.add_argument("--expected-sessions", type=int, default=480)
    parser.add_argument("--expected-scenes", type=int, default=40)
    parser.add_argument("--expected-candidates", type=int, default=8)
    parser.add_argument("--expected-images", type=int, default=3203)
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    import pandas as pd

    args = parse_args(argv)
    if args.output.exists() or args.receipt.exists():
        raise FileExistsError("output or receipt already exists")
    if args.batch_size < 1 or args.grid_size < 2:
        raise ValueError("batch size and patch grid must be positive")
    rows_sha = sha256(args.rows_csv)
    if rows_sha != args.expected_rows_sha256:
        raise RuntimeError("teacher table SHA256 changed")
    weights_sha = sha256(args.weights)
    if weights_sha != args.expected_weights_sha256:
        raise RuntimeError("LingBot weight SHA256 changed")
    frame = pd.read_csv(args.rows_csv)
    validate_table(
        frame, expected_rows=args.expected_rows,
        expected_sessions=args.expected_sessions,
        expected_scenes=args.expected_scenes,
        expected_candidates=args.expected_candidates)
    raw_paths, paths, query_index, candidate_index = pair_universe(
        frame, args.image_root)
    if len(paths) != args.expected_images:
        raise RuntimeError(
            f"image universe changed: {len(paths)} != {args.expected_images}")

    started = time.perf_counter()
    tokens, extraction_seconds = load_exact_patch_tokens(
        paths, args.lingbot_repo, args.weights, args.device,
        args.batch_size, args.grid_size)
    relation_started = time.perf_counter()
    relations = build_relations(
        tokens, query_index, candidate_index,
        frame["dino_cosine"].to_numpy(dtype=np.float64))
    relation_seconds = time.perf_counter() - relation_started
    atomic_npz(
        args.output,
        schema_version=np.asarray(SCHEMA_VERSION),
        rows_csv_sha256=np.asarray(rows_sha),
        weights_sha256=np.asarray(weights_sha),
        grid_size=np.asarray(args.grid_size, dtype=np.int32),
        relative_paths=np.asarray(raw_paths),
        tokens=tokens.astype(np.float16),
        query_index=query_index,
        candidate_index=candidate_index,
        directional_relation=relations,
        directional_relation_names=np.asarray(
            directional_patch_feature_names()),
    )
    report = {
        "schema_version": SCHEMA_VERSION,
        "rows_csv": str(args.rows_csv.resolve()),
        "rows_csv_sha256": rows_sha,
        "weights": str(args.weights.resolve()),
        "weights_sha256": weights_sha,
        "output": str(args.output.resolve()),
        "output_bytes": args.output.stat().st_size,
        "output_sha256": sha256(args.output),
        "rows": len(frame),
        "sessions": int(frame["session_id"].nunique()),
        "scenes": int(frame["scene"].nunique()),
        "images": len(paths),
        "token_shape": list(tokens.shape),
        "relation_shape": list(relations.shape),
        "device": args.device,
        "batch_size": args.batch_size,
        "grid_size": args.grid_size,
        "extraction_seconds": extraction_seconds,
        "relation_seconds": relation_seconds,
        "elapsed_seconds": time.perf_counter() - started,
        "label_blind_extraction": True,
    }
    atomic_json(args.receipt, report)
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
