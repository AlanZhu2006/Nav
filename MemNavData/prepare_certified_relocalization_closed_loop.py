#!/usr/bin/env python3
"""Create immutable run receipts for the certified closed-loop evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil


DEPENDENCIES = ("gatecurr600", "navdp_checkpoint", "lingbot_map_long")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_new(path: Path, payload: str) -> None:
    with path.open("x", encoding="utf-8") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def prepare(
    source_manifest: Path,
    expected_manifest_sha: str,
    source_receipt: Path,
    expected_source_receipt_sha: str,
    run_root: Path,
) -> dict:
    if run_root.exists():
        raise RuntimeError(f"run root already exists: {run_root}")
    if sha256(source_manifest) != expected_manifest_sha:
        raise RuntimeError("upstream fresh-episode manifest SHA changed")
    if sha256(source_receipt) != expected_source_receipt_sha:
        raise RuntimeError("source receipt SHA changed")
    manifest = json.loads(source_manifest.read_text(encoding="utf-8"))
    if manifest.get("audit", {}).get("status") != "ok":
        raise RuntimeError("upstream manifest audit is not ok")
    if manifest.get("data_role_guards", {}).get("blind_allowed") is not False:
        raise RuntimeError("upstream manifest does not prohibit blind access")
    scenes = manifest.get("scenes")
    if not isinstance(scenes, list) or len(scenes) != 20:
        raise RuntimeError("upstream manifest must contain 20 scenes")
    if sum(len(manifest["episodes"].get(scene, [])) for scene in scenes) != 160:
        raise RuntimeError("upstream manifest must contain 160 episodes")

    dependencies = {}
    for name in DEPENDENCIES:
        record = manifest["dependencies"][name]
        path = Path(record["path"])
        if path.is_symlink() or not path.is_file():
            raise RuntimeError(f"missing physical dependency: {name}")
        size = path.stat().st_size
        if size != int(record["bytes"]):
            raise RuntimeError(f"dependency byte size changed: {name}")
        digest = sha256(path)
        if digest != record["sha256"]:
            raise RuntimeError(f"dependency SHA changed: {name}")
        dependencies[name] = {
            "path": str(path),
            "bytes": size,
            "sha256": digest,
        }

    run_root.mkdir(parents=True)
    manifest_target = run_root / "data_manifest.json"
    shutil.copy2(source_manifest, manifest_target)
    write_new(
        run_root / "data_manifest.json.sha256",
        f"{expected_manifest_sha}  data_manifest.json\n",
    )
    dependency_target = run_root / "dependency_receipt.json"
    dependency_payload = {
        "schema_version": "certified_relocalization_dependency_receipt_v1",
        "manifest_sha256": expected_manifest_sha,
        "dependencies": dependencies,
    }
    write_new(
        dependency_target,
        json.dumps(dependency_payload, indent=2, sort_keys=True) + "\n",
    )
    dependency_sha = sha256(dependency_target)
    write_new(
        run_root / "dependency_receipt.json.sha256",
        f"{dependency_sha}  dependency_receipt.json\n",
    )
    shutil.copy2(source_receipt, run_root / "source_bundle.sha256")
    for path in run_root.iterdir():
        path.chmod(path.stat().st_mode & ~0o222)
    return {
        "status": "prepared",
        "run_root": str(run_root),
        "manifest_sha256": expected_manifest_sha,
        "source_receipt_sha256": expected_source_receipt_sha,
        "dependency_receipt_sha256": dependency_sha,
        "scenes": 20,
        "episodes": 160,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--expected-manifest-sha", required=True)
    parser.add_argument("--source-receipt", type=Path, required=True)
    parser.add_argument("--expected-source-receipt-sha", required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = prepare(
        args.source_manifest,
        args.expected_manifest_sha,
        args.source_receipt,
        args.expected_source_receipt_sha,
        args.run_root,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
