#!/usr/bin/env python3
"""Verify that a GNU SHA-256 manifest is relocatable and internally valid.

An outer source-bundle receipt cannot detect that a nested receipt embeds
paths from the machine that created it. This verifier rejects absolute paths,
parent traversal, duplicate entries, and paths outside the manifest directory
before hashing any payload.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re


_LINE = re.compile(r"^([0-9a-f]{64}) ([ *])(.+)$")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_manifest(manifest: Path) -> dict[str, object]:
    manifest = manifest.resolve(strict=True)
    base = manifest.parent
    entries: list[dict[str, object]] = []
    seen: set[str] = set()

    for line_number, raw_line in enumerate(
        manifest.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not raw_line:
            raise ValueError(f"blank checksum line at {line_number}")
        match = _LINE.fullmatch(raw_line)
        if match is None:
            raise ValueError(f"invalid checksum line {line_number}: {raw_line!r}")
        expected, _mode, raw_name = match.groups()
        posix_name = PurePosixPath(raw_name)
        if posix_name.is_absolute():
            raise ValueError(f"absolute checksum path at line {line_number}: {raw_name}")
        if ".." in posix_name.parts:
            raise ValueError(f"parent traversal at line {line_number}: {raw_name}")
        normalized = posix_name.as_posix()
        if normalized in {"", "."}:
            raise ValueError(f"empty checksum target at line {line_number}")
        if normalized in seen:
            raise ValueError(f"duplicate checksum target: {normalized}")
        seen.add(normalized)

        target = (base / Path(*posix_name.parts)).resolve(strict=True)
        try:
            target.relative_to(base)
        except ValueError as error:
            raise ValueError(
                f"checksum target escapes manifest directory: {raw_name}"
            ) from error
        if not target.is_file():
            raise ValueError(f"checksum target is not a regular file: {raw_name}")
        actual = _sha256(target)
        if actual != expected:
            raise ValueError(
                f"checksum mismatch for {raw_name}: expected {expected}, got {actual}"
            )
        entries.append(
            {"path": normalized, "sha256": actual, "bytes": target.stat().st_size}
        )

    if not entries:
        raise ValueError("checksum manifest is empty")
    return {
        "schema_version": "portable_checksum_verification_v1",
        "manifest": str(manifest),
        "entry_count": len(entries),
        "entries": entries,
        "verified": True,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    report = verify_manifest(arguments.manifest)
    if not arguments.quiet:
        print(json.dumps(report, indent=2, sort_keys=True))
