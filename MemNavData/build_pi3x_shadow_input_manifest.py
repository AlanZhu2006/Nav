#!/usr/bin/env python3
"""Build the minimal PT1 file list needed by Pi3X train40 shadow inference."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import tempfile
from pathlib import Path, PurePosixPath

import numpy as np


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_relative(value: str) -> str:
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError(f"unsafe dataset-relative path: {value!r}")
    return str(path)


def required_paths(rows: list[dict[str, str]], *, bridge_frames: int,
                   anchor_offsets: tuple[int, ...]) -> list[str]:
    if bridge_frames < 2:
        raise ValueError("bridge_frames must be at least two")
    paths: set[str] = set()
    for row in rows:
        scene = _safe_relative(row["scene"])
        episode = _safe_relative(row["episode"])
        episode_root = PurePosixPath(scene) / episode
        anchor = int(row["candidate_frame"])
        decision = int(row["decision_frame"])
        if not 0 <= anchor < decision:
            raise ValueError(
                f"non-causal anchor {anchor} for decision {decision}"
            )
        bridge = {
            int(round(value))
            for value in np.linspace(anchor, decision - 1, num=bridge_frames)
        }
        bridge.update(
            min(max(anchor + offset, 0), decision - 1)
            for offset in anchor_offsets
        )
        bridge.add(anchor)
        bridge.add(decision)
        rgb_root = episode_root / "videos/chunk-000/observation.images.rgb"
        paths.update(str(rgb_root / f"{frame}.jpg") for frame in bridge)
        paths.add(str(episode_root / "data/chunk-000/episode_000000.parquet"))

        query = PurePosixPath(_safe_relative(row["query_relative_path"]))
        paths.add(str(query))
        paths.add(str(query.parent / "meta/gen_meta.json"))
    return sorted(paths)


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent, text=True
    )
    try:
        with os.fdopen(descriptor, "w") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def run(args: argparse.Namespace) -> dict:
    with args.rows_csv.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    rows_sha = _sha256(args.rows_csv)
    if args.expected_rows_sha256 and rows_sha != args.expected_rows_sha256:
        raise ValueError("rows CSV SHA mismatch")
    if args.expected_rows is not None and len(rows) != args.expected_rows:
        raise ValueError(f"found {len(rows)} rows, expected {args.expected_rows}")
    paths = required_paths(
        rows,
        bridge_frames=args.bridge_frames,
        anchor_offsets=tuple(args.anchor_offsets),
    )
    _atomic_text(args.output, "".join(f"{path}\n" for path in paths))
    receipt = {
        "schema_version": 1,
        "rows_csv": str(args.rows_csv),
        "rows_csv_sha256": rows_sha,
        "rows": len(rows),
        "bridge_frames": args.bridge_frames,
        "anchor_offsets": list(args.anchor_offsets),
        "required_paths": len(paths),
        "output": str(args.output),
        "output_sha256": _sha256(args.output),
    }
    _atomic_text(
        args.output.with_suffix(args.output.suffix + ".receipt.json"),
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
    )
    return receipt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-rows", type=int)
    parser.add_argument("--expected-rows-sha256")
    parser.add_argument("--bridge-frames", type=int, default=8)
    parser.add_argument("--anchor-offsets", type=int, nargs="+", default=[-8, 0, 8])
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), sort_keys=True))
