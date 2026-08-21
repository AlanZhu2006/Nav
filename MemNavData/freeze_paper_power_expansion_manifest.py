#!/usr/bin/env python3
"""Freeze an outcome-blind MP3D phase-2 source expansion.

The trigger is only the sealed attempt-7 construction population falling below
its pre-registered sample-size target.  The selected scenes are unchanged and
the additional episode identities are fixed lexically before any attempt-7
query outcome is read.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path


EXPECTED_BASE_MANIFEST_SHA256 = (
    "b90a03cd6c3456f7741c09c2d8aa4d8f15da1512b9fa6329d31f42b7a03c5fc9"
)
EXPECTED_TRIGGER_RECEIPT_SHA256 = (
    "2ecb102f137f0ec25abd615ec544f342cb4d259a9d945fa069041a8a5bb611bc"
)
EXPANSION_EPISODES = tuple(f"episode_{index:04d}" for index in range(2, 6))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(32 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {"bytes": path.stat().st_size, "sha256": sha256_file(path)}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def freeze(base_path: Path, trigger_path: Path, out: Path) -> dict:
    require(not out.exists(), f"output already exists: {out}")
    require(
        sha256_file(base_path) == EXPECTED_BASE_MANIFEST_SHA256,
        "base source manifest changed",
    )
    require(
        sha256_file(trigger_path) == EXPECTED_TRIGGER_RECEIPT_SHA256,
        "attempt-7 population receipt changed",
    )
    base = json.loads(base_path.read_text())
    trigger = json.loads(trigger_path.read_text())
    require(trigger["policy_outcomes_read"] is False, "trigger read policy outcomes")
    require(trigger["target_met"] is False, "power-expansion trigger disappeared")
    require(
        trigger["role_pair_constructible_histories"] == 9
        and trigger["role_pair_scene_count"] == 9,
        "attempt-7 sealed population changed",
    )
    scenes = list(base["selection"]["selected_scenes"])
    require(len(scenes) == 16 and len(set(scenes)) == 16, "scene set changed")
    episode_root = Path(base["paths"]["expanded_episode_root"])

    episodes = {}
    for scene in scenes:
        records = []
        for episode_name in EXPANSION_EPISODES:
            episode = episode_root / scene / episode_name
            metadata_path = episode / "meta/gen_meta.json"
            parquet_path = episode / "data/chunk-000/episode_000000.parquet"
            goal_path = episode / "goal_1.jpg"
            metadata = json.loads(metadata_path.read_text())
            require(metadata["scene"] == f"{scene}.glb", "episode scene changed")
            require(int(metadata["n_legs"]) == 2, "source is not two-leg")
            require(
                int(metadata["ep_idx"]) == int(episode_name.rsplit("_", 1)[1]),
                "episode index changed",
            )
            n_frames = int(metadata["n_frames"])
            require(n_frames > 0, "source has no frames")
            rgb_root = episode / "videos/chunk-000/observation.images.rgb"
            require(
                (rgb_root / "0.jpg").is_file()
                and (rgb_root / f"{n_frames - 1}.jpg").is_file(),
                "source RGB endpoints are missing",
            )
            records.append({
                "episode": episode_name,
                "n_frames": n_frames,
                "recall_gap": int(metadata["goals"][0]["recall_gap"]),
                "files": {
                    "metadata": file_record(metadata_path),
                    "parquet": file_record(parquet_path),
                    "goal": file_record(goal_path),
                },
            })
        episodes[scene] = records

    payload = copy.deepcopy(base)
    payload.update({
        "schema_version": 3,
        "created_at": "2026-08-14",
        "purpose": (
            "pre-result MP3D phase-2 power replication triggered solely by "
            "attempt-7 sealed construction N=9/9 scenes below the 20/12 target"
        ),
        "episodes": episodes,
        "power_expansion": {
            "phase": 2,
            "source_attempt": 7,
            "trigger_population_receipt": str(trigger_path.resolve()),
            "trigger_population_receipt_sha256": (
                EXPECTED_TRIGGER_RECEIPT_SHA256),
            "trigger_constructible_histories": 9,
            "trigger_scene_clusters": 9,
            "target_histories": 20,
            "target_scene_clusters": 12,
            "query_outcomes_read_before_freeze": False,
            "scene_replacement": False,
            "episode_selection": list(EXPANSION_EPISODES),
            "analysis": (
                "report phase 2 separately; any pooled analysis clusters by "
                "the unchanged scene identity"),
        },
    })
    payload["selection"] = copy.deepcopy(base["selection"])
    payload["selection"].update({
        "method": (
            "same 16 frozen scenes; exact additional episode indices 2--5 "
            "for every scene, selected before attempt-7 query results"),
        "selected_episode_ids": list(EXPANSION_EPISODES),
        "scene_replacement": False,
    })
    payload["evaluation"] = copy.deepcopy(base["evaluation"])
    payload["evaluation"].update({
        "episodes_per_scene": len(EXPANSION_EPISODES),
        "base_seed": 20260818,
        "max_steps_per_leg": 600,
    })
    payload["frozen_source"] = {
        "base_manifest": str(base_path.resolve()),
        "base_manifest_sha256": EXPECTED_BASE_MANIFEST_SHA256,
        "attempt7_population_receipt": str(trigger_path.resolve()),
        "attempt7_population_receipt_sha256": (
            EXPECTED_TRIGGER_RECEIPT_SHA256),
        "query_outcomes_read_before_freeze": False,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    out.write_text(encoded)
    digest = hashlib.sha256(encoded.encode()).hexdigest()
    out.with_name(out.name + ".sha256").write_text(
        f"{digest}  {out.name}\n"
    )
    return {
        "manifest": str(out),
        "sha256": digest,
        "scenes": len(scenes),
        "episodes": sum(len(rows) for rows in episodes.values()),
        "query_outcomes_read_before_freeze": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-manifest", type=Path, required=True)
    parser.add_argument("--trigger-population", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(freeze(
        args.base_manifest, args.trigger_population, args.out
    ), sort_keys=True))


if __name__ == "__main__":
    main()
