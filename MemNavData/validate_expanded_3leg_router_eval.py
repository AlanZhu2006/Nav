#!/usr/bin/env python3
"""Fail-closed validation for the frozen scene-disjoint 3-leg benchmark."""

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


def validate_selection(manifest: dict, base_manifest: dict) -> list[str]:
    scenes = manifest["selection"]["selected_scenes"]
    base_scenes = base_manifest["selection"]["selected_scenes"]
    training = set(base_manifest["training_scenes"])
    require(len(scenes) == 10, "3-leg scene count changed")
    require(len(scenes) == len(set(scenes)), "3-leg scenes are not unique")
    require(scenes == base_scenes[:10], "3-leg selection differs from frozen rule")
    require(not set(scenes) & training, "policy-training scene leaked into 3-leg eval")
    require(set(manifest["episodes"]) == set(scenes), "episode scene keys changed")
    return scenes


def validate_image(path: Path) -> None:
    with Image.open(path) as image:
        image.verify()


def validate_file(path: Path, record: dict, label: str) -> str:
    require(path.is_file(), f"missing {label}: {path}")
    require(path.stat().st_size == record["bytes"], f"{label} size mismatch")
    actual = sha256(path)
    require(actual == record["sha256"], f"{label} SHA256 mismatch")
    return actual


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--expected-manifest-sha", required=True)
    parser.add_argument("--scene-index", type=int, required=True)
    args = parser.parse_args()

    actual_manifest_sha = sha256(args.manifest)
    require(
        actual_manifest_sha == args.expected_manifest_sha,
        "3-leg evaluation manifest SHA256 mismatch",
    )
    manifest = json.loads(args.manifest.read_text())
    base_record = manifest["base_manifest"]
    base_path = args.manifest.parent / base_record["file"]
    require(base_path.is_file(), f"missing base manifest: {base_path}")
    actual_base_sha = sha256(base_path)
    require(actual_base_sha == base_record["sha256"], "base manifest SHA256 mismatch")
    base_manifest = json.loads(base_path.read_text())

    require(manifest["evaluation"]["episodes_per_scene"] == 1,
            "3-leg episodes-per-scene changed")
    require(
        manifest["evaluation"]["goal_roles"]
        == {"A": "novel", "B": "novel", "C": "revisit"},
        "3-leg goal-role protocol changed",
    )
    require(
        manifest["paths"]["asset_root"] == base_manifest["paths"]["asset_root"],
        "3-leg asset root differs from base benchmark",
    )
    scenes = validate_selection(manifest, base_manifest)
    require(0 <= args.scene_index < len(scenes), "scene index is out of range")
    scene = scenes[args.scene_index]

    asset_root = Path(manifest["paths"]["asset_root"])
    episode_root = Path(manifest["paths"]["episode_root"])
    asset = asset_root / scene / f"{scene}.glb"
    asset_record = base_manifest["assets"][scene]
    asset_sha = validate_file(asset, asset_record, "scene asset")

    records = manifest["episodes"][scene]
    require(len(records) == 1, "3-leg manifest must select one episode per scene")
    checked_episodes = []
    for record in records:
        episode = episode_root / scene / record["episode"]
        files = {
            "metadata": episode / "meta" / "gen_meta.json",
            "parquet": episode / "data/chunk-000/episode_000000.parquet",
            "goal_b": episode / "goal_1.jpg",
            "goal_c": episode / "goal_2.jpg",
        }
        file_shas = {
            label: validate_file(path, record["files"][label], label)
            for label, path in files.items()
        }

        metadata = json.loads(files["metadata"].read_text())
        require(metadata["scene"] == f"{scene}.glb", "episode scene mismatch")
        require(int(metadata["n_legs"]) == 3, "episode is not three-leg")
        require(int(metadata["n_frames"]) == record["n_frames"],
                "frame count changed")
        switches = [int(value) for value in metadata["switches"]]
        require(switches == record["switches"], "goal switches changed")
        require(0 < switches[0] < switches[1] < record["n_frames"],
                "goal switches are out of bounds")
        require(len(metadata["goals"]) == 2, "three-leg goal count changed")
        goal_b, goal_c = metadata["goals"]
        require(goal_b["kind"] == "novel", "Goal B is not Novel")
        require(goal_c["kind"] == "revisit", "Goal C is not Revisit")
        require(int(goal_c["recall_gap"]) == record["c_recall_gap"],
                "Goal-C recall gap changed")

        rgb_root = episode / "videos/chunk-000/observation.images.rgb"
        expected_names = {f"{index}.jpg" for index in range(record["n_frames"])}
        actual_names = {path.name for path in rgb_root.glob("*.jpg")}
        require(actual_names == expected_names, "RGB frame set is incomplete")
        validate_image(rgb_root / "0.jpg")
        validate_image(rgb_root / f"{record['n_frames'] - 1}.jpg")
        validate_image(files["goal_b"])
        validate_image(files["goal_c"])
        checked_episodes.append({
            "episode": record["episode"],
            "switches": switches,
            "c_recall_gap": int(goal_c["recall_gap"]),
            "files": file_shas,
        })

    checked_dependencies = {}
    for label, dependency in base_manifest["dependencies"].items():
        path = Path(dependency["path"])
        checked_dependencies[label] = validate_file(path, dependency, label)

    print(json.dumps({
        "status": "ok",
        "manifest_sha256": actual_manifest_sha,
        "base_manifest_sha256": actual_base_sha,
        "scene_index": args.scene_index,
        "scene": scene,
        "asset_sha256": asset_sha,
        "episodes": checked_episodes,
        "dependencies": checked_dependencies,
        "policy_training_overlap": [],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
