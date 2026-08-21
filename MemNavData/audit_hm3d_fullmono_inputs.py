#!/usr/bin/env python3
"""Outcome-blind audit of every HM3D full-mono source asset and episode."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from MemNavData.hm3d_fullmono_mixed_role import (
    bind_parent_manifest,
    expected_parent_source_count,
    resolve_parent_scene,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def audit(protocol_path: Path, parent_path: Path) -> dict:
    protocol = json.loads(protocol_path.read_text())
    parent, parent_sha = bind_parent_manifest(
        protocol, protocol_path, parent_path)
    source_count = 0
    scenes = []
    for index, spec in enumerate(protocol["dataset"]["scenes"]):
        require(int(spec["rank"]) == index, "scene ranks are not contiguous")
        _spec, scene = resolve_parent_scene(protocol, parent, index)
        asset_row = parent["assets"][scene]
        asset = Path(asset_row["glb_path"])
        require(asset.name == f"{scene}.basis.glb",
                f"{scene}: explicit HM3D .basis asset contract changed")
        require(asset.is_file() and asset.stat().st_size == int(asset_row["glb_bytes"]),
                f"{scene}: asset bytes changed")
        require(sha256(asset) == asset_row["glb_sha256"],
                f"{scene}: asset hash changed")
        episode_rows = parent["episodes"][scene]
        expected_per_scene = int(protocol["dataset"]["episodes_per_scene"])
        require(len(episode_rows) in {0, expected_per_scene},
                f"{scene}: episode count changed")
        if not episode_rows:
            require("parent_manifest_sha256" not in protocol["dataset"] and
                    index not in parent["evaluation_scene_indices"],
                    f"{scene}: unauthorized empty source scene")
        episodes = []
        for row in episode_rows:
            episode = str(row["episode"])
            for label in ("goal", "metadata", "parquet"):
                receipt = row["files"][label]
                path = Path(receipt["path"])
                require(path.is_file() and
                        path.stat().st_size == int(receipt["bytes"]),
                        f"{scene}/{episode}/{label}: bytes changed")
                require(sha256(path) == receipt["sha256"],
                        f"{scene}/{episode}/{label}: hash changed")
            episodes.append(episode)
            source_count += 1
        scenes.append({
            "scene": scene,
            "scene_index": index,
            "asset_path": str(asset),
            "asset_sha256": asset_row["glb_sha256"],
            "episodes": episodes,
        })
    require(source_count == expected_parent_source_count(protocol, parent),
            "frozen source count changed")
    return {
        "schema_version": "hm3d_fullmono_input_audit_v1_20260820",
        "verified": True,
        "outcome_blind": True,
        "parent_manifest_sha256": parent_sha,
        "protocol_sha256": sha256(protocol_path),
        "scene_count": len(scenes),
        "source_episode_count": source_count,
        "target_source_episode_count": int(
            protocol["dataset"].get("target_source_episode_count", source_count)
        ),
        "constructible_scene_count": sum(bool(row["episodes"]) for row in scenes),
        "explicit_basis_assets_verified": len(scenes),
        "scenes": scenes,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--parent-manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    result = audit(args.protocol, args.parent_manifest)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
