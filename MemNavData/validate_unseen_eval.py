#!/usr/bin/env python3
"""Fail-closed validation for the scene-disjoint MemNav Habitat benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument(
        "--phase", choices=("assets", "episodes", "ready"), default="ready"
    )
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text())
    selected = manifest["selection"]["selected_scenes"]
    training = set(manifest["training_scenes"])
    require(len(selected) == len(set(selected)), "selected scene ids are not unique")
    overlap = sorted(set(selected) & training)
    require(not overlap, f"evaluation/training scene overlap: {overlap}")

    asset_hashes = manifest["assets"]
    require(set(asset_hashes) == set(selected), "asset hash keys do not match scenes")
    checked_assets = {}
    for scene in selected:
        path = args.run_root / "assets" / f"{scene}.glb"
        require(path.is_file(), f"missing scene asset: {path}")
        actual = sha256(path)
        require(actual == asset_hashes[scene], f"asset hash mismatch: {scene}")
        checked_assets[scene] = actual

    checked_episodes = {}
    if args.phase in ("episodes", "ready"):
        expected = int(manifest["episode_generation"]["episodes_per_scene"])
        metadata_hashes = set()
        for scene in selected:
            scene_root = args.run_root / "episodes" / scene
            episodes = sorted(scene_root.glob("episode_*"))
            episodes = [p for p in episodes if p.is_dir()]
            require(
                len(episodes) == expected,
                f"{scene}: expected {expected} episodes, found {len(episodes)}",
            )
            scene_rows = []
            for episode in episodes:
                meta_path = episode / "meta" / "gen_meta.json"
                parquet = episode / "data" / "chunk-000" / "episode_000000.parquet"
                goal = episode / "goal_1.jpg"
                require(meta_path.is_file(), f"missing metadata: {meta_path}")
                require(parquet.is_file(), f"missing parquet: {parquet}")
                require(goal.is_file(), f"missing goal image: {goal}")
                meta = json.loads(meta_path.read_text())
                require(meta.get("scene") == f"{scene}.glb", f"scene mismatch: {meta_path}")
                require(int(meta.get("n_legs", -1)) == 2, f"not a 2-leg episode: {meta_path}")
                meta_hash = sha256(meta_path)
                require(meta_hash not in metadata_hashes, f"duplicate episode metadata: {meta_path}")
                metadata_hashes.add(meta_hash)
                scene_rows.append(
                    {
                        "episode": episode.name,
                        "metadata_sha256": meta_hash,
                        "n_frames": int(meta["n_frames"]),
                        "recall_gap": int(meta["goals"][0]["recall_gap"]),
                    }
                )
            checked_episodes[scene] = scene_rows

    checked_checkpoints = {}
    if args.phase == "ready":
        for label, record in manifest["checkpoints"].items():
            path = args.run_root / "checkpoints" / f"{label}.memnav.ckpt"
            require(path.is_file(), f"missing checkpoint: {path}")
            actual = sha256(path)
            require(actual == record["sha256"], f"checkpoint hash mismatch: {label}")
            checked_checkpoints[label] = actual

    print(
        json.dumps(
            {
                "status": "ok",
                "phase": args.phase,
                "selected_scene_count": len(selected),
                "training_scene_count": len(training),
                "scene_overlap": overlap,
                "assets": checked_assets,
                "episodes": checked_episodes,
                "checkpoints": checked_checkpoints,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
