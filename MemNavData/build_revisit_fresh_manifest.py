#!/usr/bin/env python3
"""Freeze and audit newly generated Revisit confirmation episodes."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    require(path.is_file() and not path.is_symlink(), f"missing regular file: {path}")
    return {"bytes": path.stat().st_size, "sha256": sha256_file(path)}


def expected_numbered_files(root: Path, suffix: str, count: int) -> None:
    require(root.is_dir() and not root.is_symlink(), f"missing frame directory: {root}")
    actual = {path.name for path in root.iterdir() if path.is_file()}
    expected = {f"{index}.{suffix}" for index in range(count)}
    require(actual == expected, f"frame identity/count mismatch: {root}")


def build_manifest(
    protocol_path: Path,
    historical_manifest_path: Path,
    generated_root: Path,
    asset_root: Path,
    generator_path: Path,
    dependency_paths: dict[str, Path] | None = None,
) -> dict[str, Any]:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    historical = json.loads(
        historical_manifest_path.read_text(encoding="utf-8"))
    require(protocol.get("schema_version") == 1, "unsupported protocol schema")
    scenes = protocol.get("scenes")
    require(isinstance(scenes, list) and len(scenes) == 20, "expected 20 scenes")
    require(len(set(scenes)) == len(scenes), "protocol scene list contains duplicates")
    require(
        scenes == historical["selection"]["selected_scenes"],
        "protocol scenes/order differ from the historical consumed selection",
    )
    training = set(historical.get("training_scenes", []))
    require(not training.intersection(scenes), "confirmation scenes overlap training scenes")
    per_scene = int(protocol["episodes_per_scene"])
    require(per_scene > 0, "episodes_per_scene must be positive")
    generation = protocol["generation"]
    expected_ids = [f"episode_{index:04d}" for index in range(per_scene)]
    episode_root = generated_root / "mp3d_2leg"
    require(episode_root.is_dir(), f"missing generated root: {episode_root}")

    records: dict[str, list[dict[str, Any]]] = {}
    assets: dict[str, dict[str, Any]] = {}
    old_hashes: dict[str, dict[str, set[str]]] = {}
    for scene in scenes:
        old_hashes[scene] = {
            kind: {
                row["files"][kind]["sha256"]
                for row in historical["episodes"][scene]
            }
            for kind in ("metadata", "parquet", "goal")
        }

    for scene_index, scene in enumerate(scenes):
        scene_root = episode_root / scene
        require(scene_root.is_dir() and not scene_root.is_symlink(), f"missing scene: {scene}")
        actual_ids = sorted(
            path.name for path in scene_root.iterdir()
            if path.is_dir() and path.name.startswith("episode_")
        )
        require(actual_ids == expected_ids, f"{scene}: episode IDs differ from protocol")
        asset = asset_root / scene / f"{scene}.glb"
        asset_record = file_record(asset)
        historical_asset = historical["assets"][scene]
        require(asset_record == historical_asset, f"{scene}: MP3D asset identity changed")
        assets[scene] = asset_record
        scene_seed = int(generation["base_seed"]) + scene_index
        scene_records = []
        for episode_index, episode_id in enumerate(expected_ids):
            episode = scene_root / episode_id
            metadata_path = episode / "meta" / "gen_meta.json"
            parquet_path = episode / "data" / "chunk-000" / "episode_000000.parquet"
            goal_path = episode / "goal_image.jpg"
            goal_alias = episode / "goal_1.jpg"
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            require(metadata.get("ep_idx") == episode_index, f"{scene}/{episode_id}: ep_idx")
            require(metadata.get("generation_seed") == scene_seed, f"{scene}/{episode_id}: seed")
            require(metadata.get("n_legs") == 2, f"{scene}/{episode_id}: not 2-leg")
            require(
                metadata.get("gen_protocol") == generation["expected_gen_protocol"],
                f"{scene}/{episode_id}: generator protocol changed",
            )
            require(Path(metadata.get("scene", "")).stem == scene, f"{scene}/{episode_id}: scene")
            goals = metadata.get("goals")
            require(isinstance(goals, list) and len(goals) == 1, f"{scene}/{episode_id}: goals")
            goal = goals[0]
            require(goal.get("name") == "B" and goal.get("kind") == "revisit",
                    f"{scene}/{episode_id}: Goal-B is not Revisit")
            covis = float(goal.get("covis"))
            require(math.isfinite(covis), f"{scene}/{episode_id}: non-finite covis")
            require(float(generation["covis_lo"]) <= covis <= float(generation["covis_hi"]),
                    f"{scene}/{episode_id}: covis outside protocol")
            head = float(goal.get("head_off_deg"))
            require(math.isfinite(head) and head <= float(generation["head_max_deg"]),
                    f"{scene}/{episode_id}: heading offset outside protocol")
            recall_gap = goal.get("recall_gap")
            require(isinstance(recall_gap, int) and recall_gap >= 0,
                    f"{scene}/{episode_id}: invalid recall gap")
            frame_count = int(metadata.get("n_frames", -1))
            require(frame_count > 0, f"{scene}/{episode_id}: invalid frame count")
            expected_numbered_files(
                episode / "videos" / "chunk-000" / "observation.images.rgb",
                "jpg", frame_count,
            )
            expected_numbered_files(
                episode / "videos" / "chunk-000" / "observation.images.depth",
                "png", frame_count,
            )
            files = {
                "metadata": file_record(metadata_path),
                "parquet": file_record(parquet_path),
                "goal": file_record(goal_path),
                "goal_alias": file_record(goal_alias),
            }
            require(files["goal"] == files["goal_alias"],
                    f"{scene}/{episode_id}: goal aliases differ")
            for kind in ("metadata", "parquet", "goal"):
                require(files[kind]["sha256"] not in old_hashes[scene][kind],
                        f"{scene}/{episode_id}: duplicates historical {kind}")
            scene_records.append({
                "episode": episode_id,
                "generation_seed": scene_seed,
                "n_frames": frame_count,
                "recall_gap": recall_gap,
                "covis": covis,
                "head_off_deg": head,
                "files": files,
            })
        records[scene] = scene_records

    dependencies = {}
    if dependency_paths is not None:
        require(
            set(dependency_paths) == set(historical["dependencies"]),
            "runtime dependency names differ from the historical baseline",
        )
        for name, path in dependency_paths.items():
            record = file_record(path)
            expected = {
                key: historical["dependencies"][name][key]
                for key in ("bytes", "sha256")
            }
            require(record == expected, f"runtime dependency changed: {name}")
            dependencies[name] = {"path": str(path), **record}

    return {
        "schema_version": 1,
        "protocol_id": protocol["protocol_id"],
        "scope": protocol["scope"],
        "audit": {
            "status": "ok",
            "scenes": len(scenes),
            "episodes": len(scenes) * per_scene,
            "training_scene_overlap": [],
            "development_read": False,
            "blind_read": False,
            "historical_episode_hash_overlap": False,
        },
        "inputs": {
            "protocol_sha256": sha256_file(protocol_path),
            "historical_manifest_sha256": sha256_file(historical_manifest_path),
            "generator_sha256": sha256_file(generator_path),
        },
        "paths": {
            "episode_root": str(episode_root),
            "asset_root": str(asset_root),
        },
        "scenes": scenes,
        "episodes_per_scene": per_scene,
        "generation": generation,
        "assets": assets,
        "dependencies": dependencies,
        "episodes": records,
        "evaluation": protocol["evaluation"],
        "analysis": protocol["analysis"],
        "data_role_guards": protocol["data_role_guards"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--historical-manifest", type=Path, required=True)
    parser.add_argument("--generated-root", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--generator", type=Path, required=True)
    parser.add_argument("--gatecurr-checkpoint", type=Path, required=True)
    parser.add_argument("--navdp-checkpoint", type=Path, required=True)
    parser.add_argument("--lingbot-weights", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error(f"refusing to overwrite {args.out}")
    return args


def main() -> None:
    args = parse_args()
    manifest = build_manifest(
        args.protocol,
        args.historical_manifest,
        args.generated_root,
        args.asset_root,
        args.generator,
        {
            "gatecurr600": args.gatecurr_checkpoint,
            "navdp_checkpoint": args.navdp_checkpoint,
            "lingbot_map_long": args.lingbot_weights,
        },
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest["audit"], sort_keys=True))


if __name__ == "__main__":
    main()
