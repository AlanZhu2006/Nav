#!/usr/bin/env python3
"""Freeze the untouched remainder of the expanded two-leg scene pool."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path


BLIND_SALT = "memnav-strict-graph-blind-v1-20260806"


def sha_record(path: Path) -> dict:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return {"bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def ranked_scenes(scenes: list[str], salt: str = BLIND_SALT) -> list[str]:
    return sorted(
        scenes,
        key=lambda scene: hashlib.sha256(
            f"{salt}:{scene}".encode()).hexdigest(),
    )


def episode_record(episode: Path, expected_scene: str) -> dict:
    metadata_path = episode / "meta/gen_meta.json"
    parquet_path = episode / "data/chunk-000/episode_000000.parquet"
    goal_path = episode / "goal_1.jpg"
    for path in (metadata_path, parquet_path, goal_path):
        if not path.is_file():
            raise RuntimeError(f"missing blind input: {path}")
    metadata = json.loads(metadata_path.read_text())
    if metadata.get("scene") != f"{expected_scene}.glb":
        raise RuntimeError(f"episode scene mismatch: {episode}")
    if int(metadata.get("n_legs", -1)) != 2:
        raise RuntimeError(f"not a two-leg episode: {episode}")
    goals = metadata.get("goals")
    if not isinstance(goals, list) or not goals:
        raise RuntimeError(f"episode goal metadata is invalid: {episode}")
    frames = int(metadata["n_frames"])
    rgb_root = episode / "videos/chunk-000/observation.images.rgb"
    actual = {path.name for path in rgb_root.glob("*.jpg")}
    expected = {f"{index}.jpg" for index in range(frames)}
    if actual != expected:
        raise RuntimeError(f"incomplete RGB frames: {episode}")
    return {
        "episode": episode.name,
        "n_frames": frames,
        "recall_gap": int(goals[0]["recall_gap"]),
        "files": {
            "metadata": sha_record(metadata_path),
            "parquet": sha_record(parquet_path),
            "goal": sha_record(goal_path),
        },
    }


def build_manifest(
        source: dict, *, source_sha256: str, asset_root: Path,
        episode_root: Path, episodes_per_scene: int = 2) -> dict:
    old_selection = source["selection"]
    used = set(old_selection["selected_scenes"])
    eligible = list(old_selection["eligible_unseen_scenes"])
    remaining = sorted(set(eligible) - used)
    if not remaining:
        raise RuntimeError("expanded manifest has no untouched scenes")
    selected = ranked_scenes(remaining)

    output = copy.deepcopy(source)
    output["schema_version"] = 2
    output["created_at"] = "2026-08-06"
    output["purpose"] = (
        "one-shot blind shared-prefix direct-gap16 versus graph-gap16 "
        "evaluation; no result from these scenes may be inspected before "
        "configuration freeze")
    output["frozen_source"] = {
        "file": "expanded_navdp_router_eval_20260805.json",
        "sha256": source_sha256,
    }
    output["selection"] = {
        "method": (
            "all scenes in the frozen eligible pool not selected by the "
            "development benchmark, ordered by sha256(salt + ':' + scene)"),
        "salt": BLIND_SALT,
        "selected_scene_count": len(selected),
        "additional_scene_count": len(selected),
        "anchor_scenes": [],
        "selected_scenes": selected,
        "eligible_unseen_scenes": remaining,
    }
    output["paths"] = {
        "asset_root": str(asset_root),
        "expanded_episode_root": str(episode_root),
    }
    output["assets"] = {}
    output["episodes"] = {}
    for scene in selected:
        asset = asset_root / scene / f"{scene}.glb"
        if not asset.is_file():
            raise RuntimeError(f"missing blind scene asset: {asset}")
        output["assets"][scene] = sha_record(asset)
        candidates = sorted(
            path for path in (episode_root / scene).glob("episode_*")
            if (path / "meta/gen_meta.json").is_file()
        )
        records = []
        for candidate in candidates:
            try:
                record = episode_record(candidate, scene)
            except RuntimeError:
                continue
            records.append(record)
            if len(records) == episodes_per_scene:
                break
        if len(records) != episodes_per_scene:
            raise RuntimeError(
                f"scene {scene} has {len(records)} valid episodes, "
                f"expected {episodes_per_scene}")
        output["episodes"][scene] = records
    output["evaluation"]["episodes_per_scene"] = episodes_per_scene
    output["evaluation"]["shared_novel_prefix"] = True
    output["evaluation"]["deterministic_per_plan_seed"] = True
    output["evaluation"]["candidate_gap"] = 16
    output["evaluation"]["graph_spacing_m"] = 1.25
    output["evaluation"]["graph_arrival_m"] = 0.60
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path)
    parser.add_argument("--episode-root", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    source_bytes = args.source_manifest.read_bytes()
    source = json.loads(source_bytes)
    asset_root = args.asset_root or Path(source["paths"]["asset_root"])
    episode_root = args.episode_root or Path(
        source["paths"]["expanded_episode_root"])
    manifest = build_manifest(
        source,
        source_sha256=hashlib.sha256(source_bytes).hexdigest(),
        asset_root=asset_root,
        episode_root=episode_root,
        episodes_per_scene=int(source["evaluation"]["episodes_per_scene"]),
    )
    encoded = (json.dumps(
        manifest, indent=2, sort_keys=False, allow_nan=False) + "\n").encode()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_suffix(args.out.suffix + ".tmp")
    temporary.write_bytes(encoded)
    temporary.replace(args.out)
    print(json.dumps({
        "status": "ok",
        "output": str(args.out),
        "sha256": hashlib.sha256(encoded).hexdigest(),
        "scenes": len(manifest["selection"]["selected_scenes"]),
        "episodes": sum(map(len, manifest["episodes"].values())),
    }, indent=2))


if __name__ == "__main__":
    main()
