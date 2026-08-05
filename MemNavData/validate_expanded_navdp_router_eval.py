#!/usr/bin/env python3
"""Fail-closed input validation for the expanded scene-disjoint benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from PIL import Image


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def validate_selection(manifest: dict) -> list[str]:
    selection = manifest["selection"]
    scenes = selection["selected_scenes"]
    training = set(manifest["training_scenes"])
    eligible = selection["eligible_unseen_scenes"]
    anchors = selection["anchor_scenes"]
    require(len(scenes) == len(set(scenes)), "selected scenes are not unique")
    require(len(eligible) == len(set(eligible)), "eligible scenes are not unique")
    require(set(scenes).issubset(eligible), "selected scene outside eligible pool")
    require(not set(scenes) & training, "policy-training scene leaked into evaluation")
    require(not set(eligible) & training, "eligible pool overlaps policy training")
    require(scenes[:len(anchors)] == anchors, "legacy anchor-scene order changed")

    salt = selection["salt"]
    remaining = sorted(
        set(eligible) - set(anchors),
        key=lambda scene: hashlib.sha256(
            f"{salt}:{scene}".encode()).hexdigest(),
    )
    expected = anchors + remaining[:selection["additional_scene_count"]]
    require(scenes == expected, "selected scenes do not match the frozen hash rule")
    require(len(scenes) == selection["selected_scene_count"], "scene count changed")
    return scenes


def validate_image(path: Path) -> None:
    with Image.open(path) as image:
        image.verify()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--expected-manifest-sha", required=True)
    parser.add_argument("--scene-index", type=int, required=True)
    args = parser.parse_args()

    actual_manifest_sha = sha256(args.manifest)
    require(
        actual_manifest_sha == args.expected_manifest_sha,
        "evaluation manifest SHA256 mismatch",
    )
    manifest = json.loads(args.manifest.read_text())
    scenes = validate_selection(manifest)
    require(0 <= args.scene_index < len(scenes), "scene index is out of range")
    scene = scenes[args.scene_index]

    asset_root = Path(manifest["paths"]["asset_root"])
    episode_root = Path(
        manifest["paths"][
            "legacy_anchor_episode_root"
            if scene in manifest["selection"]["anchor_scenes"]
            else "expanded_episode_root"
        ]
    )
    asset = asset_root / scene / f"{scene}.glb"
    asset_record = manifest["assets"][scene]
    require(asset.is_file(), f"missing scene asset: {asset}")
    require(asset.stat().st_size == asset_record["bytes"], "asset size mismatch")
    require(sha256(asset) == asset_record["sha256"], "asset SHA256 mismatch")

    checked_episodes = []
    episode_records = manifest["episodes"][scene]
    require(
        len(episode_records) == manifest["evaluation"]["episodes_per_scene"],
        "episode count in manifest changed",
    )
    for record in episode_records:
        episode = episode_root / scene / record["episode"]
        files = {
            "metadata": episode / "meta" / "gen_meta.json",
            "parquet": episode / "data/chunk-000/episode_000000.parquet",
            "goal": episode / "goal_1.jpg",
        }
        for label, path in files.items():
            require(path.is_file(), f"missing {label}: {path}")
            expected = record["files"][label]
            require(path.stat().st_size == expected["bytes"], f"{label} size mismatch")
            require(sha256(path) == expected["sha256"], f"{label} SHA256 mismatch")

        metadata = json.loads(files["metadata"].read_text())
        require(metadata["scene"] == f"{scene}.glb", "episode scene mismatch")
        require(int(metadata["n_legs"]) == 2, "episode is not two-leg")
        require(int(metadata["n_frames"]) == record["n_frames"], "frame count changed")
        require(
            int(metadata["goals"][0]["recall_gap"]) == record["recall_gap"],
            "recall gap changed",
        )
        rgb_root = episode / "videos/chunk-000/observation.images.rgb"
        expected_names = {f"{index}.jpg" for index in range(record["n_frames"])}
        actual_names = {path.name for path in rgb_root.glob("*.jpg")}
        require(actual_names == expected_names, "RGB frame set is incomplete")
        validate_image(rgb_root / "0.jpg")
        validate_image(rgb_root / f"{record['n_frames'] - 1}.jpg")
        validate_image(files["goal"])
        checked_episodes.append(record["episode"])

    checked_dependencies = {}
    for label, record in manifest["dependencies"].items():
        path = Path(record["path"])
        require(path.is_file(), f"missing dependency {label}: {path}")
        require(path.stat().st_size == record["bytes"], f"{label} size mismatch")
        actual = sha256(path)
        require(actual == record["sha256"], f"{label} SHA256 mismatch")
        checked_dependencies[label] = actual

    print(json.dumps({
        "status": "ok",
        "manifest_sha256": actual_manifest_sha,
        "scene_index": args.scene_index,
        "scene": scene,
        "asset_sha256": asset_record["sha256"],
        "episodes": checked_episodes,
        "dependencies": checked_dependencies,
        "policy_training_overlap": [],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
