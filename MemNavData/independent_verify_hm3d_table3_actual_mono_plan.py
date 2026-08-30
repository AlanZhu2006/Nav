#!/usr/bin/env python3
"""Independently reproduce the result-blind Table-3 factual-history prefix."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
from typing import Any

from freeze_hm3d_table3_actual_mono_plan import (
    BIN_ORDER,
    CAPACITY_SUMMARY_SCHEMA,
    CAPACITY_VERIFY_SCHEMA,
    PLAN_SCHEMA,
    atomic_json,
    load_protocol,
    require,
    sha256_file,
)


VERIFY_SCHEMA = "hm3d_table3_actual_mono_candidate_plan_verification_v1_20260830"


def independently_order(
    by_scene: dict[str, list[dict[str, Any]]], target: int,
) -> list[tuple[str, int, dict[str, Any]]]:
    result: list[tuple[str, int, dict[str, Any]]] = []
    depth = max((len(rows) for rows in by_scene.values()), default=0)
    for rank in range(depth):
        for scene in sorted(by_scene):
            if rank < len(by_scene[scene]):
                result.append((scene, rank, by_scene[scene][rank]))
                if len(result) == target:
                    return result
    require(len(result) == target, "candidate prefix cannot be reproduced")
    return result


def verify(*, protocol_path: Path, plan_path: Path) -> dict[str, Any]:
    protocol = load_protocol(protocol_path)
    plan = json.loads(plan_path.read_text())
    require(plan.get("schema_version") == PLAN_SCHEMA,
            "candidate plan schema changed")
    require(plan["protocol_sha256"] == sha256_file(protocol_path),
            "candidate plan protocol binding changed")
    require(plan.get("factual_A_outcomes_read") is False
            and plan.get("query_policy_outcomes_read") is False
            and plan.get("navigation_outcomes_read_for_selection") is False
            and plan.get("query_policy_evaluation_authorized") is False,
            "candidate plan crossed an outcome boundary")

    source = protocol["capacity_source"]
    root = Path(source["run_root"])
    summary_path = root / source["summary"]
    capacity_verify_path = root / source["independent_verification"]
    require(sha256_file(summary_path) == source["summary_sha256"]
            == plan["capacity_summary_sha256"],
            "capacity summary binding changed")
    require(sha256_file(capacity_verify_path)
            == source["independent_verification_sha256"]
            == plan["capacity_verification_sha256"],
            "capacity verifier binding changed")
    summary = json.loads(summary_path.read_text())
    capacity_verify = json.loads(capacity_verify_path.read_text())
    require(summary.get("schema_version") == CAPACITY_SUMMARY_SCHEMA,
            "capacity summary schema changed")
    require(capacity_verify.get("schema_version") == CAPACITY_VERIFY_SCHEMA
            and capacity_verify.get("verified") is True
            and capacity_verify.get("all_geometry_capacity_gates_passed") is True,
            "capacity verifier no longer passes")

    parent_path = Path(plan["parent_manifest"])
    require(sha256_file(parent_path) == protocol["parent"]["manifest_sha256"]
            == plan["parent_manifest_sha256"],
            "parent manifest binding changed")
    parent = json.loads(parent_path.read_text())
    scene_indices = {scene: index for index, scene in enumerate(parent["scenes"])}

    candidates = {name: defaultdict(list) for name in BIN_ORDER}
    verified_fragments = 0
    for ledger in summary["scene_fragments"]:
        fragment_path = Path(ledger["path"])
        require(fragment_path.is_file()
                and sha256_file(fragment_path) == ledger["sha256"],
                "capacity fragment changed")
        fragment = json.loads(fragment_path.read_text())
        scene = str(fragment["scene"])
        require(scene == ledger["scene"] and scene in scene_indices,
                "capacity fragment scene changed")
        for name in BIN_ORDER:
            candidates[name][scene].extend(fragment["candidate_triads"][name])
        verified_fragments += 1

    expected_rows: list[tuple[str, str, int, dict[str, Any]]] = []
    diagnostics = {}
    requested = protocol["source_candidate_prefix"]["counts"]
    for name in BIN_ORDER:
        selected = independently_order(candidates[name], int(requested[name]))
        expected_rows.extend((name, scene, rank, row)
                             for scene, rank, row in selected)
        diagnostics[name] = {
            "selected_candidates": len(selected),
            "selected_scene_clusters": len({scene for scene, _rank, _row in selected}),
            "available_candidates": sum(len(rows) for rows in candidates[name].values()),
            "available_scene_clusters": len(candidates[name]),
        }
    require(diagnostics == plan["selection_diagnostics"],
            "candidate selection diagnostics changed")
    require(len(expected_rows) == len(plan["episodes"])
            == int(plan["candidate_count"]),
            "candidate plan length changed")
    seen_identities = set()
    for index, (expected, stored) in enumerate(zip(expected_rows, plan["episodes"])):
        bin_name, scene, rank, geometry = expected
        require(int(stored["history_index"]) == index
                and stored["bin_name"] == bin_name
                and stored["scene"] == scene
                and int(stored["scene_index"]) == scene_indices[scene]
                and int(stored["capacity_candidate_rank"]) == rank,
                "candidate ordering or identity changed")
        require(stored["capacity_geometry"] == geometry,
                "candidate geometry changed")
        require(stored["asset"] == parent["assets"][scene],
                "candidate asset binding changed")
        identity = str(stored["candidate_identity_sha256"])
        require(len(identity) == 64 and identity not in seen_identities,
                "candidate identity duplicated")
        seen_identities.add(identity)
        require(stored.get("factual_A_outcome_read") is False
                and stored.get("query_policy_outcomes_read") is False,
                "stored candidate read an outcome")

    result = {
        "schema_version": VERIFY_SCHEMA,
        "verified": True,
        "plan": str(plan_path.resolve()),
        "plan_sha256": sha256_file(plan_path),
        "protocol_sha256": sha256_file(protocol_path),
        "capacity_summary_sha256": sha256_file(summary_path),
        "capacity_verification_sha256": sha256_file(capacity_verify_path),
        "capacity_fragments_verified": verified_fragments,
        "candidate_count": len(expected_rows),
        "scene_clusters": len({row["scene"] for row in plan["episodes"]}),
        "selection_diagnostics": diagnostics,
        "factual_A_outcomes_read": False,
        "query_policy_outcomes_read": False,
        "query_policy_evaluation_authorized": False,
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = verify(
        protocol_path=args.protocol.resolve(), plan_path=args.plan.resolve(),
    )
    atomic_json(args.out.resolve(), result)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
