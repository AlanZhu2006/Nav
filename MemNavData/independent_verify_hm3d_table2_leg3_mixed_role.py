#!/usr/bin/env python3
"""Independent raw-file verifier for the Table-2 Leg-3 construction."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from deterministic_eval_protocol import validate_leg1_trace
from hm3d_table2_leg3_mixed_role import (
    POPULATION_SCHEMA,
    PREFIX_RECEIPT_SCHEMA,
    VERIFICATION_SCHEMA,
    load_protocol,
    power,
    require,
    sha256_file,
)
from shared_online_role_pair_contract import validate_manifest


def verify_checksum(root: Path) -> None:
    checksum = root / "CONSTRUCTION_FILES.sha256"
    require(checksum.is_file(), "Table-2 construction checksum is missing")
    for line in checksum.read_text().splitlines():
        expected, relative = line.split("  ", 1)
        path = root / relative
        require(path.is_file() and sha256_file(path) == expected,
                f"Table-2 construction artifact changed: {relative}")


def old_queries(source_root: Path, source_row: dict[str, Any]) -> list[dict]:
    benchmark = json.loads((
        source_root / "population" / source_row["benchmark"]
    ).read_text())
    scene, episode = benchmark["scene"], benchmark["episode"]
    sidecar = json.loads((
        source_root / "ab_population/role_pairs" / scene / episode
        / "role_pairs.json"
    ).read_text())
    return [query for pair in sidecar["pairs"] for query in pair["queries"]]


def pose_identity(query: dict[str, Any]) -> tuple[float, ...]:
    import math
    yaw = (float(query["yaw_rad"]) + math.pi) % (2.0 * math.pi) - math.pi
    return tuple(round(float(value), 4)
                 for value in query["floor_position"]) + (round(yaw, 4),)


def verify(root: Path, protocol_path: Path) -> dict[str, Any]:
    protocol = load_protocol(protocol_path)
    require((root / "SEALED").is_file(),
            "Table-2 construction is not sealed")
    verify_checksum(root)
    population_path = root / "population_receipt.json"
    population = json.loads(population_path.read_text())
    require(population.get("schema_version") == POPULATION_SCHEMA,
            "Table-2 population schema changed")
    require(population["protocol_sha256"] == sha256_file(protocol_path),
            "Table-2 population protocol changed")
    require(population.get("navigation_outcomes_generated") is False
            and population.get("query_outcomes_read_for_selection") is False
            and population.get("old_goal_C_outcomes_read_for_construction")
            is False, "Table-2 construction consumed a query outcome")

    source_contract = protocol["source_population"]
    source_root = Path(source_contract["run_root"])
    source_population_path = source_root / source_contract["population"]
    require(sha256_file(source_population_path)
            == source_contract["population_sha256"],
            "Table-2 source population changed")
    source_population = json.loads(source_population_path.read_text())
    manifest_path = root / "natural_direction/manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest["episodes"]:
        validate_manifest(manifest)
    require(len(manifest["episodes"])
            == int(population["leg3_constructible_histories"]),
            "Table-2 manifest/population count differs")

    source_rows = list(source_population["accepted"])
    seen_source_indices: set[int] = set()
    prefix_hashes: list[str] = []
    forbidden_overlap: list[dict[str, Any]] = []
    for row in manifest["episodes"]:
        source_index = int(row["table2_source_population_index"])
        require(source_index not in seen_source_indices,
                "Table-2 source A/B prefix was reused")
        seen_source_indices.add(source_index)
        require(0 <= source_index < len(source_rows),
                "Table-2 source population index escaped")
        scene, episode = str(row["scene"]), str(row["episode"])
        episode_root = root / "natural_direction" / scene / episode
        sidecar = episode_root / "role_pairs.json"
        require(sha256_file(sidecar) == row["role_pairs_sha256"],
                "Table-2 role-pair sidecar changed")
        prefix = root / "causal_prefix" / scene / episode
        require(Path(row["online_a_episode"]).resolve() == prefix.resolve(),
                "Table-2 role-pair points outside its sealed prefix")
        receipt_path = prefix / "receipt.json"
        trace_path = prefix / "online_a_trace.json"
        require(sha256_file(receipt_path) == row["online_a_receipt_sha256"]
                and sha256_file(trace_path) == row["online_a_trace_sha256"],
                "Table-2 prefix hash binding changed")
        receipt = json.loads(receipt_path.read_text())
        trace = json.loads(trace_path.read_text())
        validate_leg1_trace(trace)
        require(receipt.get("prefix_receipt_schema") == PREFIX_RECEIPT_SCHEMA
                and receipt.get("prefix_semantics")
                == "actual_mono_Novel_A_then_Novel_B",
                "Table-2 prefix semantics changed")
        require(int(receipt["prefix_A_steps"])
                + int(receipt["prefix_B_steps"]) == len(trace["poses"]),
                "Table-2 prefix segment lengths changed")
        audit = receipt.get("online_a_control_audit")
        require(isinstance(audit, dict) and audit.get("ok") is True,
                "Table-2 prefix contains a non-native intervention")
        require(all(plan.get("navdp_depth_source") == "monocular_sidecar"
                    and plan.get("metric_depth_sensor_consumed") is False
                    for plan in trace["plans"]),
                "Table-2 prefix is not fully monocular")
        prefix_hashes.append(row["online_a_trace_sha256"])

        old = old_queries(source_root, source_rows[source_index])
        old_hashes = {query["goal_rgb_sha256"] for query in old}
        old_poses = {pose_identity(query) for query in old}
        for pair in row["pairs"]:
            for query in pair["queries"]:
                rgb = episode_root / query["goal_rgb"]
                depth = episode_root / query["goal_depth"]
                require(sha256_file(rgb) == query["goal_rgb_sha256"]
                        and sha256_file(depth) == query["goal_depth_sha256"],
                        "Table-2 query asset changed")
                if (query["goal_rgb_sha256"] in old_hashes
                        or pose_identity(query) in old_poses):
                    forbidden_overlap.append({
                        "scene": scene, "episode": episode,
                        "query_id": query["query_id"],
                    })
    require(not forbidden_overlap,
            "Table-2 construction reused a consumed Goal-B/C identity")
    require(len(prefix_hashes) == len(set(prefix_hashes)),
            "Table-2 duplicated an A/B prefix trace")

    gate = protocol["population_gate"]
    observed_power = power(
        manifest["episodes"],
        target_histories=int(gate["minimum_histories"]),
        target_scenes=int(gate["minimum_scene_clusters"]),
        minimum_per_stratum=int(
            gate["minimum_histories_per_direction_stratum"]
        ),
    )
    require(observed_power == population["power_gate"],
            "Table-2 power gate does not reproduce")
    require(bool(population["formal_policy_evaluation_authorized"])
            == bool(observed_power["target_met"]),
            "Table-2 rollout authorization disagrees with power")
    return {
        "schema_version": VERIFICATION_SCHEMA,
        "verified": True,
        "construction_only": True,
        "policy_outcomes_read": False,
        "old_goal_C_outcomes_read": False,
        "population_receipt_sha256": sha256_file(population_path),
        "benchmark_manifest_sha256": sha256_file(manifest_path),
        "source_population_sha256": sha256_file(source_population_path),
        "histories": len(manifest["episodes"]),
        "scene_clusters": len({row["scene"] for row in manifest["episodes"]}),
        "queries": 2 * len(manifest["episodes"]),
        "forbidden_identity_overlap": forbidden_overlap,
        "power_gate": observed_power,
        "formal_policy_evaluation_authorized": bool(
            observed_power["target_met"]
        ),
    }


def write_exclusive(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(descriptor, "w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = verify(args.root.resolve(), args.protocol.resolve())
    write_exclusive(args.out.resolve(), result)
    print(json.dumps({
        "verified": result["verified"],
        "histories": result["histories"],
        "scene_clusters": result["scene_clusters"],
        "power_gate": result["power_gate"],
        "formal_policy_evaluation_authorized": result[
            "formal_policy_evaluation_authorized"
        ],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
