#!/usr/bin/env python3
"""Fail-closed validation of one frozen MP3D paper source scene.

This validator deliberately reads no navigation result.  It checks only the
pre-registered scene/episode population and the frozen native NavDP checkpoint
before a causal Goal-A trace is collected.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from validate_expanded_navdp_router_eval import (
    require,
    sha256,
    validate_image,
    validate_selection,
)


FINAL14_SOURCE_SCHEMA = "final14_paper_source_manifest_v1_20260817"
FINAL14_SCENE_BUDGET_SHA256 = (
    "779e2d7d63faa0f9b9e735680b1d620f04428c11a57ac83158933306b62407ef"
)


def paper_scenes(
    manifest: dict,
    expected_scene_budget_sha256: str = FINAL14_SCENE_BUDGET_SHA256,
) -> list[str]:
    """Validate either the legacy blind16 or the sealed Final14 selection."""

    if manifest.get("schema_version") != FINAL14_SOURCE_SCHEMA:
        return validate_selection(manifest)
    selection = manifest["selection"]
    scenes = [str(value) for value in selection["selected_scenes"]]
    require(
        selection.get("method") == "exact_untouched_final14_ledger_order",
        "Final14 selection method changed",
    )
    require(len(scenes) == 14 and len(set(scenes)) == 14,
            "Final14 source must contain 14 unique scenes")
    require(scenes == sorted(scenes), "Final14 ledger order changed")
    require(
        [str(value) for value in selection["eligible_unseen_scenes"]]
        == scenes,
        "Final14 eligible population changed",
    )
    require(not selection.get("anchor_scenes"), "Final14 cannot use anchor scenes")
    require(
        manifest["final14_source"].get("scene_budget_sha256")
        == expected_scene_budget_sha256,
        "Final14 scene budget changed",
    )
    require(
        manifest["final14_source"].get("policy_outcomes_read") is False,
        "Final14 source manifest read policy outcomes",
    )
    require(
        not set(scenes) & set(map(str, manifest["training_scenes"])),
        "Final14 overlaps policy training scenes",
    )
    return scenes


def expected_scene_episode_count(manifest: dict, scene: str) -> tuple[int, int]:
    """Return actual frozen count and the per-scene target."""

    records = manifest["episodes"][scene]
    evaluation = manifest["evaluation"]
    if manifest.get("schema_version") == FINAL14_SOURCE_SCHEMA:
        target = int(evaluation["episode_target_per_scene"])
        declared = int(evaluation["episode_counts_by_scene"][scene])
        require(0 <= declared <= target, "Final14 source count is outside target")
        require(len(records) == declared, "Final14 source count differs")
        return declared, target
    expected = int(evaluation.get("episodes_per_scene", 2))
    require(expected >= 1, "paper source episode count must be positive")
    require(
        len(records) == expected,
        "paper source episode count differs from the frozen manifest",
    )
    return expected, expected


def validate(
    manifest_path: Path,
    expected_manifest_sha: str,
    scene_index: int,
    navdp_checkpoint: Path,
    expected_scene_count: int | None = None,
    expected_scene_budget_sha256: str = FINAL14_SCENE_BUDGET_SHA256,
) -> dict:
    actual_manifest_sha = sha256(manifest_path)
    require(
        actual_manifest_sha == expected_manifest_sha,
        "paper source manifest SHA256 mismatch",
    )
    manifest = json.loads(manifest_path.read_text())
    scenes = paper_scenes(manifest, expected_scene_budget_sha256)
    if expected_scene_count is not None:
        require(
            len(scenes) == int(expected_scene_count),
            "paper source scene count differs",
        )
    require(0 <= scene_index < len(scenes), "scene index is out of range")
    scene = scenes[scene_index]

    asset_root = Path(manifest["paths"]["asset_root"])
    episode_root = Path(manifest["paths"]["expanded_episode_root"])
    asset = asset_root / scene / f"{scene}.glb"
    asset_record = manifest["assets"][scene]
    require(asset.is_file(), f"missing scene asset: {asset}")
    require(asset.stat().st_size == asset_record["bytes"], "asset size mismatch")
    require(sha256(asset) == asset_record["sha256"], "asset SHA256 mismatch")

    records = manifest["episodes"][scene]
    expected_episode_count, episode_target = expected_scene_episode_count(
        manifest, scene
    )
    checked = []
    for record in records:
        episode = episode_root / scene / record["episode"]
        files = {
            "metadata": episode / "meta" / "gen_meta.json",
            "parquet": episode / "data/chunk-000/episode_000000.parquet",
            "goal": episode / "goal_1.jpg",
        }
        for label, path in files.items():
            expected = record["files"][label]
            require(path.is_file(), f"missing {label}: {path}")
            require(
                path.stat().st_size == expected["bytes"],
                f"{label} size mismatch",
            )
            require(sha256(path) == expected["sha256"], f"{label} hash mismatch")
        metadata = json.loads(files["metadata"].read_text())
        require(metadata["scene"] == f"{scene}.glb", "episode scene mismatch")
        require(int(metadata["n_legs"]) == 2, "source episode is not two-leg")
        require(
            int(metadata["n_frames"]) == int(record["n_frames"]),
            "source frame count changed",
        )
        rgb_root = episode / "videos/chunk-000/observation.images.rgb"
        expected_names = {
            f"{index}.jpg" for index in range(int(record["n_frames"]))
        }
        require(
            {path.name for path in rgb_root.glob("*.jpg")} == expected_names,
            "source RGB frame set is incomplete",
        )
        validate_image(rgb_root / "0.jpg")
        validate_image(rgb_root / f"{int(record['n_frames']) - 1}.jpg")
        validate_image(files["goal"])
        checked.append(str(record["episode"]))

    checkpoint_record = manifest["dependencies"]["navdp_checkpoint"]
    require(navdp_checkpoint.is_file(), "NavDP checkpoint is missing")
    require(
        navdp_checkpoint.stat().st_size == checkpoint_record["bytes"],
        "NavDP checkpoint size changed",
    )
    checkpoint_sha = sha256(navdp_checkpoint)
    require(
        checkpoint_sha == checkpoint_record["sha256"],
        "NavDP checkpoint SHA256 changed",
    )
    return {
        "schema_version": "paper_online_a_input_audit_v1_20260814",
        "status": "ok",
        "manifest_sha256": actual_manifest_sha,
        "scene_index": int(scene_index),
        "scene": scene,
        "asset": str(asset),
        "asset_sha256": asset_record["sha256"],
        "episode_root": str(episode_root / scene),
        "episodes": checked,
        "source_episode_count": expected_episode_count,
        "source_episode_target": episode_target,
        "source_episode_shortage": episode_target - expected_episode_count,
        "navdp_checkpoint": str(navdp_checkpoint),
        "navdp_checkpoint_sha256": checkpoint_sha,
        "policy_training_overlap": [],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--expected-manifest-sha", required=True)
    parser.add_argument("--scene-index", type=int, required=True)
    parser.add_argument("--navdp-checkpoint", type=Path, required=True)
    parser.add_argument("--expected-scene-count", type=int)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    result = validate(
        args.manifest,
        args.expected_manifest_sha,
        args.scene_index,
        args.navdp_checkpoint,
        args.expected_scene_count,
    )
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(encoded)
    print(encoded, end="")


if __name__ == "__main__":
    main()
