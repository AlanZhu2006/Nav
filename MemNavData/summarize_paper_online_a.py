#!/usr/bin/env python3
"""Aggregate the frozen MP3D native Goal-A collection without query results."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def summarize(root: Path, manifest_path: Path, manifest_sha: str) -> dict:
    require(sha256_file(manifest_path) == manifest_sha, "manifest hash mismatch")
    manifest = json.loads(manifest_path.read_text())
    selected_scenes = manifest.get("selection", {}).get("selected_scenes")
    require(
        isinstance(selected_scenes, list) and bool(selected_scenes),
        "manifest selected scenes are invalid",
    )
    require(
        len(selected_scenes) == len(set(selected_scenes)),
        "manifest selected scenes are not unique",
    )
    episode_records = manifest.get("episodes")
    require(isinstance(episode_records, dict), "manifest episodes are invalid")
    require(
        set(episode_records) == set(selected_scenes),
        "manifest episode scenes differ from selected scenes",
    )
    evaluation = manifest.get("evaluation", {})
    final14_layout = (
        manifest.get("schema_version")
        == "final14_paper_source_manifest_v1_20260817"
    )
    if final14_layout:
        episode_target = int(evaluation.get("episode_target_per_scene", 0))
        declared_counts = evaluation.get("episode_counts_by_scene")
        require(episode_target > 0, "Final14 episode target must be positive")
        require(
            isinstance(declared_counts, dict)
            and set(declared_counts) == set(selected_scenes),
            "Final14 episode-count ledger differs",
        )
    else:
        episode_target = int(evaluation.get("episodes_per_scene", 0))
        require(episode_target > 0, "manifest episode count must be positive")
        declared_counts = {
            scene: episode_target for scene in selected_scenes
        }
    expected_episodes_by_scene = {}
    for scene in selected_scenes:
        records = episode_records[scene]
        expected_episode_count = int(declared_counts[scene])
        require(
            0 <= expected_episode_count <= episode_target,
            "manifest per-scene episode count is outside target",
        )
        require(
            isinstance(records, list) and len(records) == expected_episode_count,
            "manifest per-scene episode count changed",
        )
        episode_ids = [str(record["episode"]) for record in records]
        require(
            len(episode_ids) == len(set(episode_ids)),
            "manifest episode identities are not unique",
        )
        expected_episodes_by_scene[str(scene)] = episode_ids

    scene_roots = sorted(path for path in (root / "traces").iterdir() if path.is_dir())
    require(
        len(scene_roots) == len(selected_scenes),
        "expected all frozen source scenes",
    )
    rows = []
    materialized = []
    materialization_attrition = []
    for expected_index, (scene_root, expected_scene) in enumerate(
        zip(scene_roots, selected_scenes)
    ):
        require(
            scene_root.name == f"{expected_index:02d}_{expected_scene}",
            "scene array identity/order changed",
        )
        receipt_path = scene_root / "receipt.json"
        receipt = json.loads(receipt_path.read_text())
        require(
            receipt["schema_version"]
            == "paper_online_a_scene_receipt_v1_20260814",
            "scene receipt schema changed",
        )
        require(receipt["manifest_sha256"] == manifest_sha, "manifest hash mismatch")
        require(receipt["query_outcomes_read"] is False, "query outcome leak")
        require(receipt["scene"] == expected_scene, "scene receipt identity changed")
        trace_episodes = [str(trace["episode"]) for trace in receipt["traces"]]
        require(
            trace_episodes == expected_episodes_by_scene[expected_scene],
            "scene trace population differs from the frozen manifest",
        )
        online_manifest_path = scene_root / "online_a" / "manifest.json"
        online_manifest = json.loads(online_manifest_path.read_text())
        require(
            online_manifest["schema_version"] == "shared_online_a_materialized_v1",
            "materialized online-A schema changed",
        )
        require(
            int(online_manifest["source_trace_count"])
            == len(expected_episodes_by_scene[expected_scene]),
            "materialization did not account for all frozen source traces",
        )
        require(
            online_manifest["selection"]["all_eligible_traces_attempted"] is True,
            "eligible online-A trace was not attempted",
        )
        materialized.extend(online_manifest["episodes"])
        materialization_attrition.extend(online_manifest["attrition"])
        for trace in receipt["traces"]:
            path = scene_root / "native_a" / f"{trace['episode']}_leg1_trace.json"
            require(sha256_file(path) == trace["sha256"], "trace hash changed")
            payload = json.loads(path.read_text())
            require(payload["episode"] == trace["episode"], "trace identity changed")
            require(bool(payload["reached"]) == bool(trace["reached"]), "reach flag changed")
            require(int(payload["steps"]) == int(trace["steps"]), "step count changed")
            rows.append({
                "scene_index": expected_index,
                "scene": receipt["scene"],
                "episode": trace["episode"],
                "reached": bool(trace["reached"]),
                "steps": int(trace["steps"]),
                "trace": str(path.resolve()),
                "trace_sha256": trace["sha256"],
            })
    successes = [row for row in rows if row["reached"]]
    source_counts = {
        scene: len(expected_episodes_by_scene[scene])
        for scene in selected_scenes
    }
    unique_counts = set(source_counts.values())
    source_attrition = manifest.get("source_attrition", [])
    require(isinstance(source_attrition, list), "source attrition is invalid")
    return {
        "schema_version": "paper_online_a_inventory_v1_20260814",
        "scope": "native Goal-A source collection only; no memory query evaluated",
        "root": str(root.resolve()),
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": manifest_sha,
        "source_scenes": len(selected_scenes),
        "episodes_per_scene": (
            next(iter(unique_counts)) if len(unique_counts) == 1 else None
        ),
        "episode_target_per_scene": episode_target,
        "source_episode_counts_by_scene": source_counts,
        "source_episodes": sum(source_counts.values()),
        "source_episode_target": len(selected_scenes) * episode_target,
        "source_asset_attrition_count": len(source_attrition),
        "source_asset_attrition": source_attrition,
        "goal_a_successes": len(successes),
        "goal_a_failures": len(rows) - len(successes),
        "successful_scene_count": len({row["scene"] for row in successes}),
        "materialized_histories": len(materialized),
        "materialized_scene_count": len({row["scene"] for row in materialized}),
        "materialization_attrition_count": len(materialization_attrition),
        "materialization_attrition": materialization_attrition,
        "query_outcomes_read": False,
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-sha", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.root, args.manifest, args.manifest_sha)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        key: result[key] for key in (
            "source_scenes", "source_episodes", "goal_a_successes",
            "goal_a_failures", "successful_scene_count", "query_outcomes_read",
            "materialized_histories", "materialized_scene_count",
            "materialization_attrition_count",
        )
    }, sort_keys=True))


if __name__ == "__main__":
    main()
