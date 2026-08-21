#!/usr/bin/env python3
"""Materialize one scene's mono history and build frozen mixed-role queries."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from MemNavData.build_final14_role_pair_scene import build
from MemNavData.hm3d_fullmono_mixed_role import (
    bind_parent_manifest,
    require,
    resolve_parent_scene,
)
from MemNavData.materialize_hm3d_fullmono_online_a import materialize_scene


SCHEMA = "hm3d_fullmono_role_pair_scene_v1_20260820"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--parent-manifest", type=Path, required=True)
    parser.add_argument("--scene-index", type=int, required=True)
    args = parser.parse_args()

    protocol = json.loads(args.protocol.read_text())
    parent, _parent_sha = bind_parent_manifest(
        protocol, args.protocol, args.parent_manifest)
    spec, scene = resolve_parent_scene(protocol, parent, args.scene_index)
    source_rows = parent["episodes"][scene]
    source_order = [str(row["episode"]) for row in source_rows]
    expected_count = int(protocol["dataset"]["episodes_per_scene"])
    require(len(source_order) in {0, expected_count},
            f"{scene}: source episode count changed")
    if not source_order:
        require("parent_index" not in spec and
                args.scene_index not in parent["evaluation_scene_indices"],
                f"{scene}: unauthorized empty source scene")
    asset_row = parent["assets"][scene]
    asset = Path(asset_row["glb_path"])
    require(asset.is_file() and sha256(asset) == asset_row["glb_sha256"],
            f"{scene}: explicit parent asset changed")
    episode_root = Path(parent["paths"]["generated_root"])
    trace_root = (
        args.run_root / "goal_a" / "scenes" /
        f"{args.scene_index:02d}_{scene}"
    )
    require((trace_root / "completion.json").is_file(),
            f"{scene}: mono Goal-A collection missing")
    scene_root = (
        args.run_root / "construction" / "scenes" /
        f"{args.scene_index:02d}_{scene}"
    )
    require(not scene_root.exists(), f"construction output exists: {scene_root}")
    scene_root.mkdir(parents=True)
    online_root = scene_root / "online_a"
    if source_order:
        materialization = materialize_scene(
            trace_root=trace_root,
            scene=scene,
            asset=asset,
            episode_root=episode_root,
            source_episode_order=source_order,
            out=online_root,
        )
        role_root = scene_root / "role_pairs"
        construction = build(
            online_root,
            role_root,
            scene_rank=args.scene_index,
            source_episode_order=source_order,
            maximum_histories=int(
                protocol["construction"]["maximum_histories_per_scene"]
            ),
            only_scene=scene,
        )
    else:
        materialization = {
            "source_traces": 0, "goal_a_successes": 0,
            "eligible": 0, "materialized": 0,
            "attrition": [{
                "scene": scene, "stage": "source_generation",
                "reason": "fixed_attempt_source_generation_incomplete",
            }],
            "manifest_sha256": None,
        }
        construction = {
            "retained_standard_natural_histories": 0,
            "attrition": materialization["attrition"],
        }
    completion = {
        "schema_version": SCHEMA,
        "status": "complete",
        "scene": scene,
        "scene_index": args.scene_index,
        "protocol_sha256": sha256(args.protocol),
        "query_policy_outcomes_read": False,
        "explicit_asset_path": str(asset),
        "explicit_asset_sha256": sha256(asset),
        "materialization": materialization,
        "retained_natural_histories": int(
            construction["retained_standard_natural_histories"]
        ),
        "construction_attrition": construction["attrition"],
    }
    encoded = (json.dumps(completion, indent=2, sort_keys=True) + "\n").encode()
    path = scene_root / "completion.json"
    path.write_bytes(encoded)
    (scene_root / "completion.json.sha256").write_text(
        hashlib.sha256(encoded).hexdigest() + "  completion.json\n"
    )
    print(json.dumps({
        "status": "complete", "scene": scene,
        "materialized": materialization["materialized"],
        "retained": completion["retained_natural_histories"],
        "output": str(scene_root),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
