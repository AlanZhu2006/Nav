#!/usr/bin/env python3
"""Snapshot and audit additive HM3D Full-Mono Goal-A repairs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from MemNavData.collect_hm3d_fullmono_goal_a import audit_episode
from MemNavData.hm3d_fullmono_mixed_role import (
    bind_parent_manifest,
    expected_parent_source_count,
    require,
    resolve_parent_scene,
)


SNAPSHOT_SCHEMA = "hm3d_fullmono_goal_a_pre_repair_v1_20260821"
BARRIER_SCHEMA = "hm3d_fullmono_goal_a_repair_barrier_v1_20260821"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_indices(value: str) -> list[int]:
    result = sorted({int(part) for part in value.split(",") if part.strip()})
    require(bool(result) and all(index >= 0 for index in result),
            "repair indices must be non-empty non-negative integers")
    return result


def episode_tree_hashes(scene_root: Path, episode: str) -> dict[str, str]:
    root = scene_root / episode
    if not root.exists():
        return {}
    require(root.is_dir() and not root.is_symlink(),
            f"invalid pre-existing episode output: {root}")
    files: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        require(not path.is_symlink(), f"repair inventory symlink: {path}")
        if path.is_file():
            files[path.relative_to(scene_root).as_posix()] = sha256(path)
    return files


def write_json_with_receipt(path: Path, payload: dict[str, Any]) -> None:
    require(not path.exists() and not path.with_name(path.name + ".sha256").exists(),
            f"immutable receipt already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    path.write_bytes(encoded)
    path.with_name(path.name + ".sha256").write_text(
        hashlib.sha256(encoded).hexdigest() + f"  {path.name}\n"
    )


def snapshot(args: argparse.Namespace) -> None:
    protocol = json.loads(args.protocol.read_text())
    parent, parent_sha = bind_parent_manifest(
        protocol, args.protocol, args.parent_manifest)
    indices = parse_indices(args.repair_indices)
    require(max(indices) < len(protocol["dataset"]["scenes"]),
            "repair index outside frozen scene list")
    rows = []
    for index in indices:
        _spec, scene = resolve_parent_scene(protocol, parent, index)
        source_episodes = [
            str(row["episode"]) for row in parent["episodes"][scene]
        ]
        scene_root = (
            args.run_root / "goal_a" / "scenes" / f"{index:02d}_{scene}"
        )
        require(not (scene_root / "completion.json").exists(),
                f"repair target is already complete: {scene}")
        episode_files = {
            episode: episode_tree_hashes(scene_root, episode)
            for episode in source_episodes
            if (scene_root / episode).exists()
        }
        rows.append({
            "scene": scene,
            "scene_index": index,
            "scene_root_existed": scene_root.exists(),
            "source_episodes": source_episodes,
            "preexisting_episode_files": episode_files,
        })
    write_json_with_receipt(args.out, {
        "schema_version": SNAPSHOT_SCHEMA,
        "run_root": str(args.run_root),
        "protocol_sha256": sha256(args.protocol),
        "parent_manifest_sha256": parent_sha,
        "repair_indices": indices,
        "query_outcomes_read": False,
        "scenes": rows,
    })
    print(json.dumps({"status": "snapshotted", "scenes": len(rows),
                      "out": str(args.out)}, sort_keys=True))


def checked_json(path: Path) -> dict[str, Any]:
    receipt = path.with_name(path.name + ".sha256")
    require(path.is_file() and receipt.is_file(), f"missing receipt: {path}")
    require(sha256(path) == receipt.read_text().split()[0],
            f"receipt hash changed: {path}")
    return json.loads(path.read_text())


def barrier(args: argparse.Namespace) -> None:
    protocol = json.loads(args.protocol.read_text())
    parent, parent_sha = bind_parent_manifest(
        protocol, args.protocol, args.parent_manifest)
    inventory = checked_json(args.pre_repair_inventory)
    require(inventory["schema_version"] == SNAPSHOT_SCHEMA,
            "wrong pre-repair inventory schema")
    require(inventory["run_root"] == str(args.run_root),
            "pre-repair run root changed")
    require(inventory["protocol_sha256"] == sha256(args.protocol),
            "pre-repair protocol changed")
    require(inventory["parent_manifest_sha256"] == parent_sha,
            "pre-repair parent manifest changed")
    repair_rows = {
        int(row["scene_index"]): row for row in inventory["scenes"]
    }
    require(sorted(repair_rows) == inventory["repair_indices"],
            "pre-repair scene inventory changed")
    require(not (args.run_root / "benchmarks").exists(),
            "query benchmark existed before repair barrier")

    total_sources = total_successes = metric_reads = 0
    scene_receipts = []
    base_seed = int(protocol["goal_a"]["base_seed"])
    expected_per_scene = int(protocol["dataset"]["episodes_per_scene"])
    for index, _scene_spec in enumerate(protocol["dataset"]["scenes"]):
        _spec, scene = resolve_parent_scene(protocol, parent, index)
        sources = parent["episodes"][scene]
        scene_root = (
            args.run_root / "goal_a" / "scenes" / f"{index:02d}_{scene}"
        )
        completion_path = scene_root / "completion.json"
        completion = checked_json(completion_path)
        require(completion["scene"] == scene and
                int(completion["scene_index"]) == index,
                f"{scene}: completion identity changed")
        require(completion["protocol_sha256"] == sha256(args.protocol),
                f"{scene}: completion protocol changed")
        require(completion["parent_manifest_sha256"] == parent_sha,
                f"{scene}: completion parent changed")
        require(completion["query_outcomes_read"] is False,
                f"{scene}: collection read query outcomes")
        require(int(completion["metric_depth_sensor_reads"]) == 0,
                f"{scene}: Goal-A consumed metric depth")
        require(completion["all_sources_retained"] is True,
                f"{scene}: source population changed")
        require(int(completion["target_source_episode_count"]) ==
                expected_per_scene,
                f"{scene}: target attempt count changed")
        expected_episodes = [str(row["episode"]) for row in sources]
        records = completion["records"]
        require([str(row["episode"]) for row in records] == expected_episodes,
                f"{scene}: source episode order changed")
        require(int(completion["source_episode_count"]) == len(sources),
                f"{scene}: source count changed")
        recomputed = []
        for rank, source in enumerate(sources):
            episode = str(source["episode"])
            recomputed.append(audit_episode(
                scene_root / episode,
                scene=scene,
                scene_index=index,
                episode=episode,
                episode_rank=rank,
                seed=base_seed + 100 * index + rank,
            ))
        require(records == recomputed,
                f"{scene}: completion differs from raw Goal-A receipts")

        if index in repair_rows:
            repair = repair_rows[index]
            require(completion.get("additive_resume") is True,
                    f"{scene}: repair did not use additive resume")
            require(completion.get(
                "preexisting_episode_outputs_overwritten") is False,
                f"{scene}: repair overwrite contract failed")
            preexisting = repair["preexisting_episode_files"]
            for relative_files in preexisting.values():
                for relative, expected_hash in relative_files.items():
                    path = scene_root / relative
                    require(path.is_file() and sha256(path) == expected_hash,
                            f"{scene}: pre-repair artifact changed: {relative}")
            preexisting_complete = sum(
                bool(relative_files) for relative_files in preexisting.values()
            )
            require(int(completion["existing_episode_count"]) ==
                    preexisting_complete,
                    f"{scene}: existing episode count changed")
            require(int(completion["executed_episode_count"]) ==
                    len(sources) - preexisting_complete,
                    f"{scene}: repair reran a completed episode")

        total_sources += len(records)
        total_successes += int(completion["goal_a_successes"])
        metric_reads += int(completion["metric_depth_sensor_reads"])
        scene_receipts.append({
            "scene": scene,
            "scene_index": index,
            "source_episode_count": len(records),
            "goal_a_successes": int(completion["goal_a_successes"]),
            "completion_sha256": sha256(completion_path),
            "was_repaired": index in repair_rows,
        })

    require(total_sources == expected_parent_source_count(protocol, parent),
            "total frozen Goal-A source count changed")
    require(metric_reads == 0, "metric depth read count is nonzero")
    write_json_with_receipt(args.out, {
        "schema_version": BARRIER_SCHEMA,
        "status": "complete",
        "run_root": str(args.run_root),
        "protocol_sha256": sha256(args.protocol),
        "parent_manifest_sha256": parent_sha,
        "pre_repair_inventory_sha256": sha256(args.pre_repair_inventory),
        "repair_indices": sorted(repair_rows),
        "source_episode_count": total_sources,
        "goal_a_successes": total_successes,
        "metric_depth_sensor_reads": metric_reads,
        "query_outcomes_read": False,
        "preexisting_episode_outputs_preserved": True,
        "scenes": scene_receipts,
    })
    print(json.dumps({"status": "complete", "sources": total_sources,
                      "goal_a_successes": total_successes,
                      "out": str(args.out)}, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="mode", required=True)
    for name in ("snapshot", "barrier"):
        sub = subparsers.add_parser(name)
        sub.add_argument("--run-root", type=Path, required=True)
        sub.add_argument("--protocol", type=Path, required=True)
        sub.add_argument("--parent-manifest", type=Path, required=True)
        sub.add_argument("--out", type=Path, required=True)
    snap = subparsers.choices["snapshot"]
    snap.add_argument("--repair-indices", required=True)
    check = subparsers.choices["barrier"]
    check.add_argument("--pre-repair-inventory", type=Path, required=True)
    args = parser.parse_args()
    if args.mode == "snapshot":
        snapshot(args)
    else:
        barrier(args)


if __name__ == "__main__":
    main()
