#!/usr/bin/env python3
"""Freeze a consumed-development Novel direction-control population."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
from typing import Any

from novel_memory_direction_control import (
    ARMS,
    SCHEMA_VERSION,
    sha256_file,
    validate_control_manifest,
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _best_rotation(rows: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    require(len(rows) >= 2, "a derangement group must contain at least two rows")
    ordered = sorted(rows, key=lambda row: (
        int(row["decision_frames"]), str(row["scene"]), str(row["episode"])
    ))
    best = None
    for shift in range(1, len(ordered)):
        donors = ordered[shift:] + ordered[:shift]
        cross_scene = sum(
            source["scene"] != donor["scene"]
            for source, donor in zip(ordered, donors)
        )
        frame_cost = sum(
            abs(int(source["decision_frames"]) - int(donor["decision_frames"]))
            for source, donor in zip(ordered, donors)
        )
        signature = tuple(
            (str(donor["scene"]), str(donor["episode"])) for donor in donors
        )
        candidate = ((cross_scene, frame_cost, signature), donors)
        if best is None or candidate[0] < best[0]:
            best = candidate
    assert best is not None
    return {
        (str(source["scene"]), str(source["episode"])): donor
        for source, donor in zip(ordered, best[1])
    }


def assign_donors(rows: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    """Prefer within-scene donors, then length-match singleton scenes."""

    require(len(rows) >= 2, "control population cannot be deranged")
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row["scene"])].append(row)
    assignments: dict[tuple[str, str], dict[str, Any]] = {}
    singletons = []
    multi_groups = []
    for scene in sorted(groups):
        group = groups[scene]
        if len(group) == 1:
            singletons.extend(group)
        else:
            multi_groups.append(group)

    # One singleton cannot derange itself.  Merge it with the smallest
    # multi-scene group, sacrificing the minimum two cross-scene assignments.
    if len(singletons) == 1:
        require(multi_groups, "single-record population cannot be deranged")
        selected = min(
            multi_groups,
            key=lambda group: (len(group), str(group[0]["scene"])),
        )
        multi_groups.remove(selected)
        singletons.extend(selected)

    for group in multi_groups:
        assignments.update(_best_rotation(group))
    if singletons:
        assignments.update(_best_rotation(singletons))
    require(len(assignments) == len(rows), "donor assignment is incomplete")
    return assignments


def freeze(
    *,
    benchmark_root: Path,
    scene_budget: Path,
    out: Path,
    global_seed: int,
) -> dict[str, Any]:
    require(not out.exists(), f"output already exists: {out}")
    manifest_path = benchmark_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    require(
        manifest.get("schema_version") == "shared_online_role_pair_v1_20260814",
        "role-pair benchmark schema changed",
    )
    budget = json.loads(scene_budget.read_text())
    require(
        budget.get("schema_version") == "mp3d_scene_budget_v1_20260816",
        "scene budget schema changed",
    )
    consumed_phase2 = set(budget["partitions"]["consumed_blind16"])
    untouched = set(budget["partitions"]["untouched_final14"])

    rows = []
    for episode_row in sorted(
        manifest["episodes"], key=lambda row: (row["scene"], row["episode"])
    ):
        scene = str(episode_row["scene"])
        episode = str(episode_row["episode"])
        require(scene in consumed_phase2, "control attempted to leave the consumed Phase-2 scene set")
        require(scene not in untouched, "control attempted to consume a final scene")
        sidecar = benchmark_root / scene / episode / "role_pairs.json"
        payload = json.loads(sidecar.read_text())
        source = Path(payload["online_a_episode"])
        trace_path = source / "online_a_trace.json"
        trace = json.loads(trace_path.read_text())
        novel_queries = [
            query
            for pair in payload["pairs"]
            for query in pair["queries"]
            if query["analysis_role"] == "novel"
        ]
        require(len(novel_queries) == 1, "expected exactly one Novel query")
        rows.append({
            "scene": scene,
            "episode": episode,
            "query_id": str(novel_queries[0]["query_id"]),
            "role_pairs_path": str(sidecar.resolve()),
            "role_pairs_sha256": sha256_file(sidecar),
            "online_a_episode": str(source.resolve()),
            "online_a_receipt_sha256": str(payload["online_a_receipt_sha256"]),
            "online_a_trace_sha256": str(payload["online_a_trace_sha256"]),
            "online_a_steps": int(payload["online_a_steps"]),
            "decision_frames": len(trace["plans"]),
        })
    require(bool(rows), "control population is empty")
    assignments = assign_donors(rows)
    by_identity = {(row["scene"], row["episode"]): row for row in rows}
    frozen_rows = []
    for index, row in enumerate(rows):
        donor = assignments[(row["scene"], row["episode"])]
        frozen = dict(row)
        frozen["index"] = index
        frozen["arm_order"] = list(ARMS[index % len(ARMS):] + ARMS[:index % len(ARMS)])
        frozen["donor"] = {
            key: donor[key]
            for key in (
                "scene", "episode", "online_a_episode",
                "online_a_receipt_sha256", "online_a_trace_sha256",
                "online_a_steps", "decision_frames",
            )
        }
        frozen["donor_same_scene"] = donor["scene"] == row["scene"]
        frozen["decision_frame_difference"] = abs(
            int(row["decision_frames"]) - int(donor["decision_frames"])
        )
        require(
            (donor["scene"], donor["episode"]) in by_identity,
            "donor identity disappeared",
        )
        frozen_rows.append(frozen)

    payload = {
        "schema_version": SCHEMA_VERSION,
        "frozen_date": "2026-08-16",
        "purpose": (
            "consumed-development causal mechanism control for Phase-2 Novel raw-bearing gains"
        ),
        "evaluation_stage": "consumed_development_mechanism_only",
        "confirmation_claim_allowed": False,
        "method_or_threshold_selection_allowed": False,
        "query_role": "novel",
        "global_seed": int(global_seed),
        "arms": list(ARMS),
        "benchmark_root": str(benchmark_root.resolve()),
        "benchmark_manifest_sha256": sha256_file(manifest_path),
        "scene_budget_path": str(scene_budget.resolve()),
        "scene_budget_sha256": sha256_file(scene_budget),
        "untouched_final_scenes_remain_unread": sorted(untouched),
        "episodes": frozen_rows,
        "population": {
            "episodes": len(frozen_rows),
            "scenes": len({row["scene"] for row in frozen_rows}),
            "same_scene_donors": sum(row["donor_same_scene"] for row in frozen_rows),
        },
    }
    validate_control_manifest(payload)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    out.with_name(out.name + ".sha256").write_text(
        f"{sha256_file(out)}  {out.name}\n"
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--scene-budget", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--global-seed", type=int, default=20260816)
    args = parser.parse_args()
    payload = freeze(
        benchmark_root=args.benchmark_root,
        scene_budget=args.scene_budget,
        out=args.out,
        global_seed=args.global_seed,
    )
    print(json.dumps(payload["population"], sort_keys=True))


if __name__ == "__main__":
    main()
