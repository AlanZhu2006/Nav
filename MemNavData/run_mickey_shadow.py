#!/usr/bin/env python3
"""Run label-blind MicKey inference on a frozen goal/history pair table."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import tempfile
from typing import Iterable

import numpy as np

from MemNavData.mickey_learned_relocalizer import MicKeyShadowAdapter


REQUIRED_COLUMNS = (
    "session_id", "query_relative_path", "candidate_relative_path",
    "candidate_rank", "dino_cosine",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


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


def safe_image_path(root: Path, value: str) -> Path:
    relative = PurePosixPath(value)
    if (not value or relative.is_absolute() or ".." in relative.parts
            or "." in relative.parts):
        raise ValueError(f"unsafe image path: {value!r}")
    root = root.resolve()
    path = root.joinpath(*relative.parts).resolve()
    if root not in path.parents or not path.is_file():
        raise FileNotFoundError(f"image is absent from frozen bundle: {value}")
    return path


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = set(REQUIRED_COLUMNS) - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"pair table lacks columns: {sorted(missing)}")
        rows = list(reader)
    if not rows:
        raise ValueError("pair table is empty")
    return rows


def selected_indices(total: int, expression: str, maximum: int) -> list[int]:
    if expression:
        values = [int(value) for value in expression.split(",") if value]
    else:
        values = list(range(total))
    if len(set(values)) != len(values):
        raise ValueError("row indices cannot contain duplicates")
    if any(value < 0 or value >= total for value in values):
        raise IndexError("row index is outside the frozen pair table")
    if maximum > 0:
        values = values[:maximum]
    if not values:
        raise ValueError("no rows were selected")
    return values


def parse_intrinsic(value: str) -> np.ndarray:
    matrix = np.asarray(json.loads(value), dtype=np.float64)
    if (matrix.shape != (3, 3) or not np.isfinite(matrix).all()
            or matrix[0, 0] <= 0 or matrix[1, 1] <= 0):
        raise ValueError("intrinsic JSON must contain a valid finite 3x3 matrix")
    return matrix


def run(args: argparse.Namespace) -> dict:
    rows_path = args.rows_csv.resolve()
    rows = read_rows(rows_path)
    indices = selected_indices(len(rows), args.row_indices, args.max_pairs)
    intrinsic = parse_intrinsic(args.intrinsic_json)
    output = args.output_jsonl.resolve()
    receipt = args.receipt.resolve()
    if output.exists() or receipt.exists():
        raise FileExistsError("shadow output/receipt already exists; use a new run path")
    output.parent.mkdir(parents=True, exist_ok=True)

    adapter = MicKeyShadowAdapter(
        repository=args.repository,
        python_dependencies=args.python_dependencies,
        config=args.config,
        checkpoint=args.checkpoint,
        dino_weights=args.dino_weights,
        device=args.device,
        resize=tuple(args.resize))

    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".tmp", dir=output.parent)
    temporary_path = Path(temporary)
    status_counts = {"ok": 0, "abstain": 0, "error": 0}
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            for batch_start in range(0, len(indices), args.batch_size):
                batch_indices = indices[batch_start:batch_start + args.batch_size]
                batch = [rows[index] for index in batch_indices]
                references = [safe_image_path(
                    args.image_root, row["candidate_relative_path"])
                    for row in batch]
                queries = [safe_image_path(
                    args.image_root, row["query_relative_path"])
                    for row in batch]
                predictions = adapter.infer(
                    references, queries,
                    [intrinsic] * len(batch), [intrinsic] * len(batch),
                    seed=args.seed + batch_start)
                for index, row, prediction in zip(
                        batch_indices, batch, predictions):
                    status_counts[prediction.status] += 1
                    record = {
                        "pair_index": index,
                        "session_id": row["session_id"],
                        "candidate_rank": int(row["candidate_rank"]),
                        "dino_cosine": float(row["dino_cosine"]),
                        "query_relative_path": row["query_relative_path"],
                        "candidate_relative_path": row[
                            "candidate_relative_path"],
                        "prediction": prediction.as_dict(),
                    }
                    handle.write(json.dumps(record, sort_keys=True) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
        os.replace(temporary_path, output)
    finally:
        temporary_path.unlink(missing_ok=True)

    report = {
        "schema_version": "mickey_shadow_receipt_v1",
        "scope": "train-only label-blind pairwise shadow; not navigation SR",
        "rows_csv": str(rows_path),
        "rows_csv_sha256": sha256(rows_path),
        "pair_table_rows": len(rows),
        "selected_pairs": len(indices),
        "selected_first_index": indices[0],
        "selected_last_index": indices[-1],
        "image_root": str(args.image_root.resolve()),
        "intrinsic": intrinsic.tolist(),
        "resize_width_height": list(args.resize),
        "batch_size": args.batch_size,
        "seed": args.seed,
        "model_id": adapter.model_id,
        "repository": str(args.repository.resolve()),
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": sha256(args.checkpoint.resolve()),
        "dino_weights": str(args.dino_weights.resolve()),
        "dino_weights_sha256": sha256(args.dino_weights.resolve()),
        "config": str(args.config.resolve()),
        "config_sha256": sha256(args.config.resolve()),
        "output_jsonl": str(output),
        "output_jsonl_sha256": sha256(output),
        "status_counts": status_counts,
        "labels_read_by_inference": False,
        "navigation_actions_changed": False,
    }
    atomic_json(receipt, report)
    return report


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows-csv", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--python-dependencies", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dino-weights", type=Path, required=True)
    parser.add_argument("--intrinsic-json", required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    # 476x266 is the nearest 14-divisible resolution to the native 480x270
    # stream, avoiding MicKey's silent bottom/right crop with minimal rescale.
    parser.add_argument("--resize", nargs=2, type=int, default=(476, 266),
                        metavar=("WIDTH", "HEIGHT"))
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260817)
    parser.add_argument("--row-indices", default="")
    parser.add_argument("--max-pairs", type=int, default=0)
    args = parser.parse_args(argv)
    if args.batch_size < 1 or args.max_pairs < 0:
        parser.error("batch-size must be positive and max-pairs non-negative")
    return args


def main(argv: Iterable[str] | None = None) -> int:
    report = run(parse_args(argv))
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
