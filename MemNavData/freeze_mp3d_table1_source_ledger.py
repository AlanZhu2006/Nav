#!/usr/bin/env python3
"""Freeze MP3D actual-mono histories without reading prior Goal-B outcomes."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

try:
    from deterministic_eval_protocol import validate_leg1_trace
    from mp3d_table1_new_query_contract import (
        EPISODES_PER_SCENE,
        SCENE_COUNT,
        SOURCE_LEDGER_SCHEMA,
        require,
    )
except ImportError:
    from MemNavData.deterministic_eval_protocol import validate_leg1_trace
    from MemNavData.mp3d_table1_new_query_contract import (
        EPISODES_PER_SCENE,
        SCENE_COUNT,
        SOURCE_LEDGER_SCHEMA,
        require,
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _episode_index(manifest: dict[str, Any], scene: str) -> dict[str, dict]:
    rows = list(manifest["episodes"][scene])[:EPISODES_PER_SCENE]
    require(len(rows) == EPISODES_PER_SCENE,
            f"{scene}: frozen first-two source population is incomplete")
    result = {str(row["episode"]): row for row in rows}
    require(len(result) == len(rows), f"{scene}: duplicate source episode")
    return result


def freeze(*, source_run_root: Path, source_manifest_path: Path,
           out: Path) -> dict[str, Any]:
    require(not out.exists(), f"output already exists: {out}")
    manifest = json.loads(source_manifest_path.read_text())
    scenes = [str(value) for value in manifest["scenes"]]
    require(len(scenes) == SCENE_COUNT and len(set(scenes)) == SCENE_COUNT,
            "MP3D scene population changed")
    asset_root = Path(manifest["paths"]["asset_root"])
    episode_root = Path(manifest["paths"]["episode_root"])
    posthoc = source_run_root / "POSTHOC"
    source_receipts = {}
    for name in (
        "mdtec_cec_composition_summary.json",
        "mdtec_cec_composition_independent_verification.json",
        "output_receipt.sha256",
    ):
        path = posthoc / name
        require(path.is_file(), f"sealed source receipt missing: {path}")
        source_receipts[name] = sha256_file(path)
    receipt_path = posthoc / "output_receipt.sha256"
    receipt_members = set()
    for line in receipt_path.read_text().splitlines():
        expected, raw_path = line.split(None, 1)
        member = Path(raw_path.strip()).resolve()
        require(member.parent == posthoc.resolve(),
                "source POSTHOC receipt escaped its sealed directory")
        require(member.is_file() and sha256_file(member) == expected,
                f"source POSTHOC receipt mismatch: {member}")
        receipt_members.add(member.name)
    require(receipt_members == {
        "mdtec_cec_composition_summary.json",
        "mdtec_cec_composition_independent_verification.json",
    }, "source POSTHOC receipt membership changed")

    rows = []
    for scene_index, scene in enumerate(scenes):
        source_rows = _episode_index(manifest, scene)
        scene_root = source_run_root / "scenes" / f"{scene_index:02d}_{scene}"
        asset = asset_root / scene / f"{scene}.glb"
        require(asset.is_file(), f"source asset missing: {asset}")
        require(sha256_file(asset) == manifest["assets"][scene]["sha256"],
                f"source asset hash changed: {scene}")
        episodes = []
        for episode_rank, (episode, source_row) in enumerate(source_rows.items()):
            trace = (scene_root / f"{episode}_leg_a_trace"
                     / f"{episode}_leg1_trace.json")
            require(trace.is_file(), f"actual-mono Goal-A trace missing: {trace}")
            trace_payload = json.loads(trace.read_text())
            validate_leg1_trace(trace_payload)
            require(str(trace_payload["source_scene"]) == scene
                    and str(trace_payload["episode"]) == episode,
                    f"trace identity changed: {scene}/{episode}")

            source_episode = episode_root / scene / episode
            metadata_path = source_episode / "meta/gen_meta.json"
            old_goal_path = source_episode / "goal_image.jpg"
            parquet_path = (source_episode / "data/chunk-000"
                            / "episode_000000.parquet")
            for path in (metadata_path, old_goal_path, parquet_path):
                require(path.is_file(), f"source episode artifact missing: {path}")
            require(sha256_file(metadata_path)
                    == source_row["files"]["metadata"]["sha256"],
                    f"source metadata changed: {scene}/{episode}")
            require(sha256_file(old_goal_path)
                    == source_row["files"]["goal"]["sha256"],
                    f"source Goal-B changed: {scene}/{episode}")
            metadata = json.loads(metadata_path.read_text())
            goals = metadata.get("goals")
            require(isinstance(goals, list) and len(goals) == 1,
                    f"expected one source Goal-B: {scene}/{episode}")
            goal = goals[0]
            require(str(goal.get("name")) == "B",
                    f"source Goal-B identity changed: {scene}/{episode}")
            episodes.append({
                "episode": episode,
                "episode_rank": episode_rank,
                "trace_path": str(trace.resolve()),
                "trace_sha256": sha256_file(trace),
                "source_metadata_path": str(metadata_path.resolve()),
                "source_metadata_sha256": sha256_file(metadata_path),
                "source_parquet_path": str(parquet_path.resolve()),
                "source_parquet_sha256": sha256_file(parquet_path),
                "consumed_goal_b": {
                    "goal_rgb_path": str(old_goal_path.resolve()),
                    "goal_rgb_sha256": sha256_file(old_goal_path),
                    "floor_position": [float(value) for value in goal["pos"]],
                    "yaw_rad": float(goal["yaw_habitat"]),
                },
            })
        rows.append({
            "scene": scene,
            "scene_index": scene_index,
            "asset_path": str(asset.resolve()),
            "asset_sha256": sha256_file(asset),
            "episode_root": str(episode_root.resolve()),
            "episodes": episodes,
        })

    payload = {
        "schema_version": SOURCE_LEDGER_SCHEMA,
        "scope": (
            "all first-two actual-mono Goal-A traces on the consumed MP3D "
            "20-scene population; new query construction only"
        ),
        "source_run_root": str(source_run_root.resolve()),
        "source_manifest_path": str(source_manifest_path.resolve()),
        "source_manifest_sha256": sha256_file(source_manifest_path),
        "source_posthoc_receipt_hashes": source_receipts,
        "scene_count": len(rows),
        "episodes_per_scene": EPISODES_PER_SCENE,
        "previous_goal_b_policy_outcomes_read": False,
        "query_policy_outcomes_read": False,
        "history_selection_rule": "all_first_two_manifest_order_no_result_filter",
        "scenes": rows,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(out, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(descriptor, "w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run-root", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = freeze(
        source_run_root=args.source_run_root.resolve(),
        source_manifest_path=args.source_manifest.resolve(),
        out=args.out.resolve(),
    )
    print(json.dumps({
        "scene_count": result["scene_count"],
        "histories": sum(len(row["episodes"]) for row in result["scenes"]),
        "previous_goal_b_policy_outcomes_read": False,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
