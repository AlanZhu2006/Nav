#!/usr/bin/env python3
"""Merge the sealed fresh54 parent with a disjoint HM3D asset expansion."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from freeze_hm3d_table3_asset_expansion import require, sha256_file


SCHEMA = "hm3d_table3_combined_assets_v1_20260830"
EXPANSION_SCHEMA = "hm3d_table3_asset_expansion_v1_20260830"


def merge(
    *, parent: Path, expected_parent_sha256: str,
    expansion: Path, expected_expansion_sha256: str,
) -> dict:
    require(sha256_file(parent) == expected_parent_sha256,
            "parent manifest SHA-256 changed")
    require(sha256_file(expansion) == expected_expansion_sha256,
            "expansion manifest SHA-256 changed")
    first = json.loads(parent.read_text())
    second = json.loads(expansion.read_text())
    require(second.get("schema_version") == EXPANSION_SCHEMA,
            "expansion manifest schema changed")
    require(second["parent_manifest_sha256"] == expected_parent_sha256,
            "expansion is bound to another parent")
    require(second["episode_files_read"] is False
            and second["navigation_outcomes_read"] is False,
            "expansion read forbidden outcomes")
    first_scenes = [str(scene) for scene in first["scenes"]]
    second_scenes = [str(scene) for scene in second["scenes"]]
    require(len(first_scenes) == 54 and len(second_scenes) == 46,
            "source scene counts changed")
    require(not set(first_scenes).intersection(second_scenes),
            "combined scene identities overlap")
    scenes = first_scenes + second_scenes
    assets = {str(scene): first["assets"][scene] for scene in first_scenes}
    assets.update({str(scene): second["assets"][scene] for scene in second_scenes})
    require(len(scenes) == len(set(scenes)) == len(assets) == 100,
            "combined scene population is not exactly 100")
    return {
        "schema_version": SCHEMA,
        "scope": "100-scene result-blind HM3D asset pool for Table-3 capacity",
        "source_manifests": [
            {"path": str(parent.resolve()), "sha256": expected_parent_sha256,
             "scenes": 54},
            {"path": str(expansion.resolve()), "sha256": expected_expansion_sha256,
             "scenes": 46},
        ],
        "scene_count": 100,
        "scenes": scenes,
        "assets": assets,
        "cross_source_scene_overlap": 0,
        "query_policy_outcomes_read": False,
        "navigation_outcomes_read": False,
        "policy_evaluation_authorized": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--expected-parent-sha256", required=True)
    parser.add_argument("--expansion", type=Path, required=True)
    parser.add_argument("--expected-expansion-sha256", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = merge(
        parent=args.parent,
        expected_parent_sha256=args.expected_parent_sha256,
        expansion=args.expansion,
        expected_expansion_sha256=args.expected_expansion_sha256,
    )
    require(not args.out.exists(), f"refusing to overwrite {args.out}")
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    args.out.with_name(args.out.name + ".sha256").write_text(
        f"{sha256_file(args.out)}  {args.out.name}\n"
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
