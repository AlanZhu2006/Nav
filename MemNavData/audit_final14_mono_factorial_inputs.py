#!/usr/bin/env python3
"""Outcome-blind preflight for every consumed Final14 source dependency."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    require(sha256(manifest_path) == args.expected_manifest_sha256,
            "Final14 manifest changed")
    manifest = json.loads(manifest_path.read_text())
    episodes = manifest["episodes"]
    require(len(episodes) == 21, "Final14 history count changed")
    scenes: set[str] = set()
    parquet_hashes: set[str] = set()
    for index, item in enumerate(episodes):
        source = Path(item["online_a_episode"])
        receipt_path = source / "receipt.json"
        trace_path = source / "online_a_trace.json"
        require(receipt_path.is_file(), f"history {index}: receipt missing")
        require(trace_path.is_file(), f"history {index}: trace missing")
        require(sha256(receipt_path) == item["online_a_receipt_sha256"],
                f"history {index}: receipt hash changed")
        require(sha256(trace_path) == item["online_a_trace_sha256"],
                f"history {index}: trace hash changed")
        receipt = json.loads(receipt_path.read_text())
        scene_asset = Path(receipt["source_asset"])
        source_parquet = (
            Path(receipt["source_episode"])
            / "data/chunk-000/episode_000000.parquet"
        )
        require(scene_asset.is_file(), f"history {index}: scene asset missing")
        require(source_parquet.is_file(), f"history {index}: parquet missing")
        require(sha256(scene_asset) == receipt["source_asset_sha256"],
                f"history {index}: scene asset hash changed")
        require(sha256(source_parquet) == receipt["source_parquet_sha256"],
                f"history {index}: parquet hash changed")
        require(int(item["online_a_steps"]) >= 40,
                f"history {index}: mono scale prefix is too short")
        scenes.add(str(item["scene"]))
        parquet_hashes.add(str(receipt["source_parquet_sha256"]))
    require(len(scenes) == 10, "Final14 scene count changed")
    print(json.dumps({
        "status": "passed",
        "histories": len(episodes),
        "scenes": len(scenes),
        "source_parquets": len(parquet_hashes),
        "manifest_sha256": args.expected_manifest_sha256,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
