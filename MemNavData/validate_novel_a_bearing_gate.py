#!/usr/bin/env python3
"""Fail-closed inputs for one frozen Novel-A bearing-gate scene."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image

try:
    from validate_expanded_navdp_router_eval import (
        require,
        sha256,
        validate_selection,
    )
except ModuleNotFoundError:
    from MemNavData.validate_expanded_navdp_router_eval import (
        require,
        sha256,
        validate_selection,
    )


def validate_image(path: Path) -> None:
    with Image.open(path) as image:
        image.verify()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--expected-manifest-sha", required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--expected-protocol-sha", required=True)
    parser.add_argument("--input-overlay", type=Path, required=True)
    parser.add_argument("--expected-input-overlay-sha", required=True)
    parser.add_argument("--scene-index", type=int, required=True)
    parser.add_argument("--navdp-checkpoint", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path)
    parser.add_argument("--episode-root", type=Path)
    args = parser.parse_args()

    manifest_sha = sha256(args.manifest)
    protocol_sha = sha256(args.protocol)
    inputs_sha = sha256(args.input_overlay)
    require(manifest_sha == args.expected_manifest_sha,
            "parent manifest SHA256 mismatch")
    require(protocol_sha == args.expected_protocol_sha,
            "bearing protocol SHA256 mismatch")
    require(inputs_sha == args.expected_input_overlay_sha,
            "Goal-A input-overlay SHA256 mismatch")
    manifest = json.loads(args.manifest.read_text())
    protocol = json.loads(args.protocol.read_text())
    overlay = json.loads(args.input_overlay.read_text())
    require(protocol["manifest"]["sha256"] == manifest_sha,
            "protocol points to another parent manifest")
    require(protocol["input_overlay"]["sha256"] == inputs_sha,
            "protocol points to another Goal-A overlay")
    require(overlay["parent_manifest_sha256"] == manifest_sha,
            "Goal-A overlay points to another parent manifest")

    scenes = validate_selection(manifest)
    require(0 <= args.scene_index < len(scenes), "scene index out of range")
    scene = scenes[args.scene_index]
    asset_root = args.asset_root or Path(manifest["paths"]["asset_root"])
    episode_root = args.episode_root or Path(manifest["paths"][
        "legacy_anchor_episode_root"
        if scene in manifest["selection"]["anchor_scenes"]
        else "expanded_episode_root"])

    asset = asset_root / scene / f"{scene}.glb"
    asset_record = manifest["assets"][scene]
    require(asset.is_file(), f"missing scene asset: {asset}")
    require(asset.stat().st_size == asset_record["bytes"],
            "scene asset size mismatch")
    require(sha256(asset) == asset_record["sha256"],
            "scene asset SHA256 mismatch")

    checkpoint_record = manifest["dependencies"]["navdp_checkpoint"]
    require(args.navdp_checkpoint.is_file(), "missing NavDP checkpoint")
    require(args.navdp_checkpoint.stat().st_size == checkpoint_record["bytes"],
            "NavDP checkpoint size mismatch")
    require(sha256(args.navdp_checkpoint) == checkpoint_record["sha256"]
            == protocol["navdp_checkpoint_sha256"],
            "NavDP checkpoint SHA256 mismatch")

    checked = []
    records = manifest["episodes"][scene]
    require(len(records) == protocol["evaluation"]["episodes_per_scene"],
            "episode count changed")
    require(set(overlay["goal_a_images"][scene])
            == {record["episode"] for record in records},
            "Goal-A overlay episode set changed")
    for record in records:
        episode = episode_root / scene / record["episode"]
        metadata_path = episode / "meta" / "gen_meta.json"
        parquet_path = episode / "data/chunk-000/episode_000000.parquet"
        goal_b_path = episode / "goal_1.jpg"
        for label, path in {
                "metadata": metadata_path,
                "parquet": parquet_path,
                "goal": goal_b_path}.items():
            expected = record["files"][label]
            require(path.is_file(), f"missing {label}: {path}")
            require(path.stat().st_size == expected["bytes"],
                    f"{label} size mismatch")
            require(sha256(path) == expected["sha256"],
                    f"{label} SHA256 mismatch")
        metadata = json.loads(metadata_path.read_text())
        require(metadata["scene"] == f"{scene}.glb", "episode scene mismatch")
        require(int(metadata["n_legs"]) == 2, "episode is not two-leg")
        require(int(metadata["n_frames"]) == record["n_frames"],
                "episode frame count mismatch")
        goal_a_record = overlay["goal_a_images"][scene][record["episode"]]
        frame_index = int(metadata["switch_idx"]) - 1
        require(frame_index == int(goal_a_record["frame_index"]),
                "Goal-A frame index mismatch")
        goal_a_path = (episode /
                       "videos/chunk-000/observation.images.rgb" /
                       f"{frame_index}.jpg")
        require(goal_a_path.is_file(), f"missing Goal-A image: {goal_a_path}")
        require(goal_a_path.stat().st_size == goal_a_record["bytes"],
                "Goal-A image size mismatch")
        require(sha256(goal_a_path) == goal_a_record["sha256"],
                "Goal-A image SHA256 mismatch")
        validate_image(goal_a_path)
        checked.append({
            "episode": record["episode"],
            "goal_a_frame": frame_index,
            "goal_a_sha256": goal_a_record["sha256"],
        })

    print(json.dumps({
        "status": "ok",
        "scene_index": args.scene_index,
        "scene": scene,
        "manifest_sha256": manifest_sha,
        "protocol_sha256": protocol_sha,
        "input_overlay_sha256": inputs_sha,
        "asset_sha256": asset_record["sha256"],
        "navdp_checkpoint_sha256": checkpoint_record["sha256"],
        "asset_root": str(asset_root),
        "episode_root": str(episode_root),
        "episodes": checked,
        "policy_training_overlap": [],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
