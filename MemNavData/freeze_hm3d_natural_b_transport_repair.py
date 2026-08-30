#!/usr/bin/env python3
"""Freeze an outcome-blind retry plan for missing factual-B identities.

The selector observes only the existence and byte integrity of completion
receipts.  It never deserializes a factual-B completion payload, so navigation
outcomes cannot affect retry membership.  Completion-less partial directories
are preserved byte-for-byte before the unchanged identity is retried.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path


SCHEMA = "hm3d_natural_b_transport_repair_plan_v1_20260830"
ARCHIVE_SCHEMA = "hm3d_natural_b_transport_partial_archive_v1_20260830"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_sidecar(path: Path) -> str:
    sidecar = path.with_name(path.name + ".sha256")
    require(path.is_file() and sidecar.is_file(), f"missing receipt: {path}")
    digest = sha256(path)
    require(sidecar.read_text().split() == [digest, path.name],
            f"invalid receipt: {path}")
    return digest


def label(index: int, row: dict) -> str:
    return f"{index:03d}_{row['scene']}_{row['episode']}"


def file_inventory(root: Path) -> dict[str, str]:
    inventory = {}
    for path in sorted(root.rglob("*")):
        require(not path.is_symlink(), f"partial output contains symlink: {path}")
        if path.is_file():
            inventory[path.relative_to(root).as_posix()] = sha256(path)
    return inventory


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--benchmark-manifest", type=Path, required=True)
    parser.add_argument("--expected-benchmark-sha256", required=True)
    parser.add_argument("--shard-manifest", type=Path, required=True)
    parser.add_argument("--expected-shard-sha256", required=True)
    parser.add_argument("--archive-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    require(not args.out.exists(), f"repair plan exists: {args.out}")
    require(not args.archive_root.exists(),
            f"repair archive exists: {args.archive_root}")
    benchmark_sha = verify_sidecar(args.benchmark_manifest)
    shard_sha = verify_sidecar(args.shard_manifest)
    require(benchmark_sha == args.expected_benchmark_sha256,
            "benchmark manifest changed")
    require(shard_sha == args.expected_shard_sha256,
            "factual-B shard manifest changed")
    benchmark = json.loads(args.benchmark_manifest.read_text())
    schedule = json.loads(args.shard_manifest.read_text())
    episodes = benchmark.get("episodes")
    require(isinstance(episodes, list) and len(episodes) == 84,
            "frozen expansion population must contain 84 histories")
    require(schedule.get("candidate_histories") == 84
            and schedule.get("all_candidates_partitioned_once") is True
            and schedule.get("navigation_outcomes_read") is False
            and schedule.get("query_policy_outcomes_read") is False,
            "factual-B schedule contract changed")
    scheduled = sorted(
        int(index)
        for shard in schedule["shards"]
        for index in shard["history_indices"]
    )
    require(scheduled == list(range(84)), "schedule identity coverage changed")

    factual_root = args.run_root / "factual_b"
    factual_root.mkdir(parents=True, exist_ok=True)
    completed, missing, partial = [], [], []
    for index, row in enumerate(episodes):
        root = factual_root / label(index, row)
        completion = root / "completion.json"
        if completion.is_file():
            # Hash verification is deliberately byte-level; never parse the
            # payload that contains reached_B or other navigation outcomes.
            verify_sidecar(completion)
            completed.append(index)
        else:
            missing.append(index)
            if root.exists():
                require(root.is_dir() and not root.is_symlink(),
                        f"unsafe partial output: {root}")
                partial.append(index)

    require(missing, "transport repair was requested but no identity is missing")
    args.archive_root.mkdir(parents=True)
    archived = []
    for index in partial:
        row = episodes[index]
        source = factual_root / label(index, row)
        inventory = file_inventory(source)
        destination = args.archive_root / source.name
        source.rename(destination)
        require(not source.exists() and destination.is_dir(),
                f"failed to preserve partial output {index}")
        archived.append({
            "history_index": index,
            "label": source.name,
            "destination": str(destination.resolve()),
            "files": inventory,
        })

    archive_receipt = {
        "schema_version": ARCHIVE_SCHEMA,
        "partial_history_indices": partial,
        "archives": archived,
        "completion_payloads_deserialized": False,
        "navigation_outcomes_read": False,
        "scientific_thresholds_changed": False,
    }
    archive_path = args.archive_root / "archive_receipt.json"
    archive_path.write_text(json.dumps(
        archive_receipt, indent=2, sort_keys=True, allow_nan=False
    ) + "\n")
    archive_path.with_name(archive_path.name + ".sha256").write_text(
        f"{sha256(archive_path)}  {archive_path.name}\n"
    )

    by_scene: dict[int, list[int]] = defaultdict(list)
    for index in missing:
        row = episodes[index]
        scene_rank = int(row["final14_scene_rank"])
        by_scene[scene_rank].append(index)
    groups = []
    for repair_index, scene_rank in enumerate(sorted(by_scene)):
        indices = sorted(by_scene[scene_rank])
        scene_ids = {str(episodes[index]["scene"]) for index in indices}
        require(len(scene_ids) == 1, "scene-rank identity mismatch")
        groups.append({
            "repair_index": repair_index,
            "scene_index": scene_rank,
            "scene": next(iter(scene_ids)),
            "history_indices": indices,
            "history_count": len(indices),
        })
    payload = {
        "schema_version": SCHEMA,
        "status": "repair_required",
        "benchmark_manifest_sha256": benchmark_sha,
        "shard_manifest_sha256": shard_sha,
        "candidate_histories": len(episodes),
        "completed_history_count": len(completed),
        "missing_history_indices": missing,
        "partial_history_indices": partial,
        "repair_group_count": len(groups),
        "repair_groups": groups,
        "completion_membership_signal": "existence_plus_byte_receipt_only",
        "completion_payloads_deserialized": False,
        "navigation_outcomes_read": False,
        "query_policy_outcomes_read": False,
        "model_or_controller_changed": False,
        "scientific_thresholds_changed": False,
        "step_budget_changed": False,
        "fallback_completion_allowed": False,
        "archive_receipt_sha256": sha256(archive_path),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(
        payload, indent=2, sort_keys=True, allow_nan=False
    ) + "\n")
    args.out.with_name(args.out.name + ".sha256").write_text(
        f"{sha256(args.out)}  {args.out.name}\n"
    )
    print(json.dumps({
        "completed": len(completed), "missing": len(missing),
        "partial": len(partial), "repair_groups": len(groups),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
