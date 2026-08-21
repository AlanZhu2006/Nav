#!/usr/bin/env python3
"""Export the exact causal images referenced by a certificate teacher table.

The MP3D episodes used by the teacher can live inside a read-only Singularity
overlay.  This utility runs inside that mount, validates every relative path,
and writes a deterministic uncompressed tar plus a provenance receipt.  It
never follows paths outside ``episode_root`` and never reads labels to choose
which images are exported.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import tarfile
import tempfile
from typing import Iterable


PATH_COLUMNS = ("query_relative_path", "candidate_relative_path")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def validated_relative_path(value: str) -> PurePosixPath:
    relative = PurePosixPath(str(value))
    if (not value or relative.is_absolute() or ".." in relative.parts
            or "." in relative.parts):
        raise ValueError(f"unsafe relative image path: {value!r}")
    return relative


def referenced_paths(rows_csv: Path) -> tuple[PurePosixPath, ...]:
    paths: set[PurePosixPath] = set()
    with rows_csv.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = set(PATH_COLUMNS) - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"teacher table lacks path columns: {sorted(missing)}")
        for row in reader:
            for column in PATH_COLUMNS:
                paths.add(validated_relative_path(row[column]))
    if not paths:
        raise ValueError("teacher table references no images")
    return tuple(sorted(paths, key=str))


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


def export_bundle(rows_csv: Path, episode_root: Path, output_tar: Path,
                  expected_images: int = 0) -> dict:
    rows_csv = rows_csv.resolve()
    episode_root = episode_root.resolve()
    paths = referenced_paths(rows_csv)
    if expected_images and len(paths) != expected_images:
        raise RuntimeError(
            f"image universe changed: {len(paths)} != {expected_images}")
    physical = [episode_root.joinpath(*relative.parts) for relative in paths]
    missing = [str(path) for path in physical if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} referenced images are missing; first={missing[0]}")

    output_tar.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{output_tar.name}.", suffix=".tmp", dir=output_tar.parent)
    os.close(descriptor)
    temporary_path = Path(temporary)
    try:
        with tarfile.open(temporary_path, mode="w", format=tarfile.PAX_FORMAT) as archive:
            for relative, source in zip(paths, physical):
                archive.add(source, arcname=str(relative), recursive=False)
        with temporary_path.open("rb+") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary_path, output_tar)
    finally:
        temporary_path.unlink(missing_ok=True)

    return {
        "schema_version": "certificate_image_bundle_v1",
        "rows_csv": str(rows_csv),
        "rows_csv_sha256": sha256(rows_csv),
        "episode_root": str(episode_root),
        "images": len(paths),
        "output_tar": str(output_tar.resolve()),
        "output_tar_bytes": output_tar.stat().st_size,
        "output_tar_sha256": sha256(output_tar),
        "first_relative_path": str(paths[0]),
        "last_relative_path": str(paths[-1]),
    }


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows-csv", type=Path, required=True)
    parser.add_argument("--episode-root", type=Path, required=True)
    parser.add_argument("--output-tar", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--expected-images", type=int, default=0)
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    if args.expected_images < 0:
        raise ValueError("expected image count cannot be negative")
    report = export_bundle(
        args.rows_csv, args.episode_root, args.output_tar,
        expected_images=args.expected_images)
    atomic_json(args.receipt, report)
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
