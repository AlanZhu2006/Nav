#!/usr/bin/env python3
"""Independent raw-file audit before any v4 C/B2/C2 navigation.

This verifier deliberately does not import the construction/finalization
implementation.  It recomputes the factual-B and sealed-population ledger from
raw traces, receipts, and copied assets using only the frozen protocol.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any


SCHEMA = (
    "hm3d_fullmono_lifelong_natural_v4_population_independent_"
    "verification_v1_20260828"
)
PROTOCOL_SCHEMA = "hm3d_fullmono_lifelong_direct_natural_power_v4_20260827"
EXPANSION_PROTOCOL_SCHEMA = (
    "hm3d_fullmono_lifelong_natural_b_expansion_execution_v5_20260830"
)
B_SCHEMA = "hm3d_fullmono_lifelong_b_collection_v1_20260824"
PREFIX_SCHEMA = "hm3d_fullmono_lifelong_factual_prefix_v1_20260824"
PREFIX_COMPLETION_SCHEMA = "hm3d_fullmono_lifelong_prefix_fragment_v1_20260824"
POPULATION_SCHEMA = "hm3d_fullmono_lifelong_population_v1_20260824"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    require(isinstance(payload, dict), f"{path}: JSON root is not an object")
    return payload


def verify_sidecar(path: Path) -> str:
    sidecar = path.with_name(path.name + ".sha256")
    require(sidecar.is_file(), f"{path}: SHA sidecar missing")
    fields = sidecar.read_text().split()
    digest = sha256(path)
    require(len(fields) == 2 and fields[0] == digest
            and fields[1] == path.name,
            f"{path}: SHA sidecar changed")
    return digest


def contained(path: Path, root: Path, message: str) -> Path:
    resolved, base = path.resolve(), root.resolve()
    require(resolved == base or base in resolved.parents, message)
    return resolved


def verify_file_ledger(root: Path, ledger_name: str,
                       excluded: set[str]) -> int:
    ledger = root / ledger_name
    require(ledger.is_file(), f"{ledger}: file ledger missing")
    seen: set[Path] = set()
    for line in ledger.read_text().splitlines():
        fields = line.split(maxsplit=1)
        require(len(fields) == 2, f"{ledger}: malformed ledger row")
        digest, relative = fields[0], fields[1].strip()
        path = contained(root / relative, root, "file ledger escaped root")
        require(path.is_file() and path not in seen,
                f"{ledger}: missing or duplicate ledger path")
        require(sha256(path) == digest, f"{path}: ledger digest changed")
        seen.add(path)
    actual = {
        path.resolve() for path in root.rglob("*")
        if path.is_file() and path.name not in excluded
    }
    require(seen == actual, f"{ledger}: ledger coverage changed")
    return len(seen)


def audit_mono_plans(plans: list[dict[str, Any]]) -> dict[str, int]:
    require(isinstance(plans, list) and plans,
            "factual-B trace has no NavDP plans")
    scale_hashes: set[str] = set()
    for plan in plans:
        require(plan.get("navdp_depth_source") == "monocular_sidecar",
                "factual-B plan depth source changed")
        require(plan.get("metric_depth_sensor_consumed") is not True,
                "factual-B plan consumed metric depth")
        receipt = plan.get("monocular_depth_receipt")
        require(isinstance(receipt, dict),
                "factual-B plan omitted monocular receipt")
        require(receipt.get("depth_contract")
                == "raw_lingbot_depth_first40_v1",
                "factual-B mono depth contract changed")
        require(receipt.get("metric_depth_sensor_consumed") is False,
                "factual-B mono receipt reports metric consumption")
        require(int(receipt.get("frame_index", -1)) >= 40,
                "factual-B unexpectedly used bootstrap depth")
        require(receipt.get("scale_active") is True,
                "factual-B mono scale was inactive")
        scale = receipt.get("scale_receipt")
        require(isinstance(scale, dict), "factual-B scale receipt missing")
        require(scale.get("scale_evidence_contract")
                == "causal_first_prefix_rgb_only_v1",
                "factual-B scale evidence contract changed")
        require(scale.get("whole_episode_ground_cache_consumed") is False,
                "factual-B scale consumed future evidence")
        scale_hash = receipt.get("scale_receipt_sha256")
        require(bool(scale_hash), "factual-B scale receipt hash missing")
        scale_hashes.add(str(scale_hash))
    require(len(scale_hashes) == 1,
            "factual-B used multiple scale receipts")
    return {
        "metric_sensor_plan_count": 0,
        "monocular_receipt_plan_count": len(plans),
        "monocular_scale_hash_count": 1,
    }


def expected_attrition_reasons(completion: dict[str, Any],
                               factual: dict[str, Any],
                               protocol: dict[str, Any]) -> set[str]:
    if not factual["reached_B"]:
        return {"actual_mono_B_failed"}
    rules = protocol["factual_b_collection"]
    reasons: set[str] = set()
    support = float(completion["B_goal_max_factual_B_covis"])
    if support < float(
        rules["B_goal_support_by_factual_B_minimum_inclusive"]
    ):
        reasons.add("B_goal_not_supported_by_factual_B")
    distance = completion.get("actual_B_end_to_C_geodesic_m")
    low, high = (
        float(value) for value in rules["actual_B_end_to_C_geodesic_band_m"]
    )
    if distance is None or not low <= float(distance) <= high:
        reasons.add("actual_B_end_to_C_geodesic_outside_band")
    return reasons


def verify(*, run_root: Path, protocol_path: Path) -> dict[str, Any]:
    protocol = read_json(protocol_path)
    protocol_schema = protocol.get("schema_version")
    require(protocol_schema in {PROTOCOL_SCHEMA, EXPANSION_PROTOCOL_SCHEMA},
            "frozen factual-B protocol schema changed")
    require(protocol.get("post_prefix_query_outcomes_read_before_freeze") is False,
            "protocol permits post-prefix outcome filtering")
    require(protocol["guards"]["no_post_prefix_outcome_filtering"] is True,
            "protocol post-prefix guard changed")
    protocol_sha = sha256(protocol_path)

    expected_candidates = int(protocol["construction_power_gate"][
        "expected_exact_candidate_histories"
    ])
    materialization_verify_name = (
        "independent_natural_b_expansion_materialization_verification.json"
        if protocol_schema == EXPANSION_PROTOCOL_SCHEMA
        else "independent_natural_v4_materialization_verification.json"
    )
    materialization_verify_path = run_root / materialization_verify_name
    verify_sidecar(materialization_verify_path)
    materialization = read_json(materialization_verify_path)
    require(materialization.get("verified") is True
            and materialization.get("factual_B_gate_verified") is True
            and materialization.get("factual_B_executed") is False
            and materialization.get("navigation_outcomes_read") is False,
            "materialization gate was not independently verified")

    ab_root = run_root / "ab_population"
    require((ab_root / "SEALED").is_file(), "A/B population is not sealed")
    ab_receipt_path = ab_root / "population_receipt.json"
    verify_sidecar(ab_receipt_path)
    ab_receipt = read_json(ab_receipt_path)
    manifest_path = ab_root / "role_pairs/manifest.json"
    manifest_sha = verify_sidecar(manifest_path)
    manifest = read_json(manifest_path)
    episodes = manifest.get("episodes")
    require(isinstance(episodes, list) and len(episodes) == expected_candidates,
            "A/B manifest population changed")
    require(ab_receipt["benchmark_manifest_sha256"] == manifest_sha,
            "A/B receipt no longer binds the manifest")
    require(ab_receipt["factual_B_authorized"] is True
            and ab_receipt["query_policy_outcomes_read"] is False
            and ab_receipt["navigation_outcome_selection"] is False,
            "A/B population did not authorize result-blind factual B")

    schedule_path = run_root / "factual_b_schedule/shards.json"
    schedule_sha = verify_sidecar(schedule_path)
    schedule = read_json(schedule_path)
    require(schedule["benchmark_manifest_sha256"] == manifest_sha,
            "factual-B schedule used another manifest")
    flattened = [
        int(index) for shard in schedule["shards"]
        for index in shard["history_indices"]
    ]
    require(sorted(flattened) == list(range(expected_candidates))
            and len(flattened) == len(set(flattened)),
            "factual-B schedule did not partition all candidates once")
    shard_size = int(schedule["maximum_histories_per_shard"])
    require(shard_size == 2, "factual-B shard size changed")
    per_scene = Counter(int(item["final14_scene_rank"]) for item in episodes)
    expected_shards = sum(
        math.ceil(count / shard_size) for count in per_scene.values()
    )
    require(schedule["shard_count"] == len(schedule["shards"])
            == expected_shards,
            "factual-B shard contract changed")
    require(schedule["query_policy_outcomes_read"] is False
            and schedule["navigation_outcomes_read"] is False,
            "factual-B schedule read navigation outcomes")

    factual_by_index: dict[int, dict[str, Any]] = {}
    factual_success = mono_plan_receipts = 0
    for index, item in enumerate(episodes):
        scene, episode = str(item["scene"]), str(item["episode"])
        label = f"{index:03d}_{scene}_{episode}"
        root = run_root / "factual_b" / label
        completion_path = root / "completion.json"
        verify_sidecar(completion_path)
        completion = read_json(completion_path)
        require(completion.get("schema_version") == B_SCHEMA
                and completion.get("status") == "complete",
                f"{label}: factual-B completion schema changed")
        require(completion["history_index"] == index
                and completion["scene"] == scene
                and completion["episode"] == episode,
                f"{label}: factual-B identity changed")
        require(completion["protocol_sha256"] == protocol_sha
                and completion["benchmark_manifest_sha256"] == manifest_sha
                and completion["role_pair_sidecar_sha256"]
                == item["role_pairs_sha256"],
                f"{label}: factual-B source binding changed")
        require(completion["runtime_role_visible"] is False
                and completion["controller"]
                == "frozen_navdp_native_sidecar"
                and completion["navdp_depth_source"] == "monocular_sidecar"
                and int(completion["metric_depth_sensor_reads"]) == 0,
                f"{label}: factual-B runtime contract changed")
        require(completion["online_A_rgb_hashes_verified"] is True,
                f"{label}: online-A RGB replay was not verified")
        trace_path = contained(
            Path(completion["B_trace_path"]), root,
            f"{label}: factual-B trace escaped its output",
        )
        require(sha256(trace_path) == completion["B_trace_sha256"],
                f"{label}: factual-B trace changed")
        trace = read_json(trace_path)
        require(trace.get("source_hybrid_route") == "native_sidecar"
                and str(trace.get("source_scene")) == scene
                and str(trace.get("episode")) == episode,
                f"{label}: factual-B trace controller/identity changed")
        require(bool(trace["reached"]) == bool(completion["reached_B"])
                and int(trace["steps"]) == int(completion["steps_B"]),
                f"{label}: factual-B trace summary changed")
        depth_audit = audit_mono_plans(trace["plans"])
        require(depth_audit == completion["depth_audit"],
                f"{label}: factual-B depth audit differs from raw plans")
        mono_plan_receipts += depth_audit["monocular_receipt_plan_count"]
        metric_path = root / "result/metric.csv"
        require(sha256(metric_path) == completion["result_metric_sha256"],
                f"{label}: raw factual-B metric changed")
        with metric_path.open(newline="") as handle:
            metric_rows = list(csv.DictReader(handle))
        require(len(metric_rows) == 1
                and metric_rows[0].get("analysis_role") == "novel",
                f"{label}: raw factual-B query population changed")
        plan_paths = list((root / "result").glob(f"{episode}_*_plans.json"))
        require(len(plan_paths) == 1
                and sha256(plan_paths[0]) == completion["result_plans_sha256"],
                f"{label}: raw factual-B plans changed")
        raw_plans = read_json(plan_paths[0])
        require(raw_plans.get("analysis_role_not_forwarded") is True,
                f"{label}: analysis role leaked into factual-B runtime")
        factual_success += int(completion["reached_B"])
        factual_by_index[index] = completion

    prefix_by_index: dict[int, dict[str, Any]] = {}
    eligible_indices: list[int] = []
    attrition_reasons: Counter[str] = Counter()
    strong_support = 0
    for index, item in enumerate(episodes):
        scene, episode = str(item["scene"]), str(item["episode"])
        label = f"{index:03d}_{scene}_{episode}"
        root = run_root / "prefix_fragments" / label
        completion_path = root / "completion.json"
        verify_sidecar(completion_path)
        completion = read_json(completion_path)
        factual = factual_by_index[index]
        require(completion.get("schema_version") == PREFIX_COMPLETION_SCHEMA
                and completion.get("status") == "complete",
                f"{label}: prefix completion schema changed")
        require(completion["history_index"] == index
                and completion["scene"] == scene
                and completion["episode"] == episode,
                f"{label}: prefix identity changed")
        require(completion["protocol_sha256"] == protocol_sha
                and completion["AB_manifest_sha256"] == manifest_sha
                and completion["factual_B_completion_sha256"]
                == sha256(run_root / "factual_b" / label / "completion.json")
                and completion["query_navigation_outcomes_read"] is False,
                f"{label}: prefix source/outcome binding changed")
        if completion["eligible"]:
            require(factual["reached_B"] is True,
                    f"{label}: failed factual B became eligible")
            support = float(completion["B_goal_max_factual_B_covis"])
            low, high = (
                float(value) for value in protocol["factual_b_collection"][
                    "actual_B_end_to_C_geodesic_band_m"
                ]
            )
            distance = float(completion["actual_B_end_to_C_geodesic_m"])
            require(support >= float(protocol["factual_b_collection"][
                "B_goal_support_by_factual_B_minimum_inclusive"
            ]) and low <= distance <= high,
                    f"{label}: eligible prefix violates frozen support gate")
            benchmark_path = root / "benchmark/benchmark.json"
            require(sha256(benchmark_path) == completion["benchmark_sha256"],
                    f"{label}: prefix benchmark changed")
            benchmark = read_json(benchmark_path)
            require(benchmark.get("schema_version") == PREFIX_SCHEMA
                    and benchmark["scene"] == scene
                    and benchmark["episode"] == episode
                    and benchmark["history_index"] == index,
                    f"{label}: prefix benchmark identity changed")
            require(benchmark["protocol_sha256"] == protocol_sha
                    and benchmark["AB_manifest_sha256"] == manifest_sha
                    and benchmark["runtime_role_visibility"] == "none"
                    and benchmark["query_outcomes_read"] is False,
                    f"{label}: prefix benchmark runtime contract changed")
            require(benchmark["online_B_trace_sha256"]
                    == factual["B_trace_sha256"],
                    f"{label}: prefix copied another factual-B trace")
            copied_trace = root / "benchmark" / benchmark["online_B_trace"]
            copied_completion = (
                root / "benchmark" / benchmark["factual_B_completion"]
            )
            require(sha256(copied_trace) == benchmark["online_B_trace_sha256"]
                    and sha256(copied_completion)
                    == benchmark["factual_B_completion_sha256"],
                    f"{label}: copied factual-B evidence changed")
            for goal in benchmark["goals"].values():
                for key in ("rgb", "depth"):
                    asset = contained(
                        root / "benchmark" / goal[key], root / "benchmark",
                        f"{label}: goal asset escaped benchmark",
                    )
                    require(sha256(asset) == goal[f"{key}_sha256"],
                            f"{label}: copied goal asset changed")
            threshold = float(protocol["factual_b_collection"][
                "B_goal_support_strong_threshold_inclusive"
            ])
            require(bool(completion["B_goal_strong_support"])
                    == (support >= threshold)
                    == bool(benchmark["B_goal_strong_support"]),
                    f"{label}: strong-support label changed")
            strong_support += int(completion["B_goal_strong_support"])
            eligible_indices.append(index)
        else:
            expected = expected_attrition_reasons(completion, factual, protocol)
            actual = set(str(completion["attrition_reason"]).split(","))
            require(actual == expected and expected,
                    f"{label}: attrition reason differs from frozen gate")
            attrition_reasons.update(actual)
            require(not (root / "benchmark").exists(),
                    f"{label}: ineligible prefix retained a benchmark")
        prefix_by_index[index] = completion

    population_root = run_root / "population"
    require((population_root / "SEALED").is_file(),
            "factual A/B population is not sealed")
    population_path = population_root / "population.json"
    population_sha = verify_sidecar(population_path)
    population = read_json(population_path)
    require(population.get("schema_version") == POPULATION_SCHEMA,
            "population schema changed")
    require(population["protocol_sha256"] == protocol_sha
            and population["AB_population_receipt_sha256"]
            == sha256(ab_receipt_path)
            and population["AB_manifest_sha256"] == manifest_sha,
            "population source binding changed")
    require(population["intention_to_collect_B"] == expected_candidates
            and population["supported_population"] == len(eligible_indices)
            and len(population["accepted"]) == len(eligible_indices)
            and len(population["attrition"])
            == expected_candidates - len(eligible_indices),
            "population counts changed")
    accepted_indices = [
        int(row["source_AB_history_index"]) for row in population["accepted"]
    ]
    attrition_indices = [
        int(row["history_index"]) for row in population["attrition"]
    ]
    require(accepted_indices == eligible_indices
            and sorted(accepted_indices + attrition_indices)
            == list(range(expected_candidates)),
            "population does not exactly reproduce prefix eligibility")
    for row in population["accepted"]:
        index = int(row["source_AB_history_index"])
        item, completion = episodes[index], prefix_by_index[index]
        require(row["scene"] == item["scene"]
                and row["episode"] == item["episode"]
                and row["prefix_completion_sha256"]
                == sha256(run_root / "prefix_fragments" /
                          f"{index:03d}_{item['scene']}_{item['episode']}" /
                          "completion.json"),
                "accepted population identity changed")
        benchmark_path = contained(
            population_root / row["benchmark"], population_root,
            "population benchmark escaped root",
        )
        require(sha256(benchmark_path) == row["benchmark_sha256"]
                == completion["benchmark_sha256"],
                "population benchmark differs from prefix benchmark")
    accepted_scenes = {str(row["scene"]) for row in population["accepted"]}
    target_histories = int(protocol["population"]["minimum_target_histories"])
    target_scenes = int(protocol["population"][
        "minimum_target_scene_clusters"
    ])
    target_met = (
        len(eligible_indices) >= target_histories
        and len(accepted_scenes) >= target_scenes
    )
    require(population["scene_clusters"] == len(accepted_scenes)
            and population["strong_support_histories"] == strong_support
            and population["target_histories"] == target_histories
            and population["target_scene_clusters"] == target_scenes
            and population["target_met"] is target_met
            and population["underpowered"] is (not target_met),
            "population power gate changed")
    require(population["selection_reads_C_B2_C2_navigation_outcomes"] is False
            and population["runtime_role_visibility"] == "none",
            "population selection read downstream outcomes")
    ledger_entries = verify_file_ledger(
        population_root, "POPULATION_FILES.sha256",
        {"POPULATION_FILES.sha256", "SEALED"},
    )
    return {
        "schema_version": (
            "hm3d_fullmono_lifelong_natural_b_expansion_population_"
            "independent_verification_v1_20260830"
            if protocol_schema == EXPANSION_PROTOCOL_SCHEMA else SCHEMA
        ),
        "verified": True,
        "scope": "pre-query factual-B and population raw-file audit only",
        "protocol_sha256": protocol_sha,
        "materialization_verification_sha256": sha256(
            materialization_verify_path
        ),
        "AB_manifest_sha256": manifest_sha,
        "factual_B_schedule_sha256": schedule_sha,
        "factual_B_rollouts": expected_candidates,
        "factual_B_successes": factual_success,
        "metric_depth_sensor_reads": 0,
        "monocular_plan_receipts_verified": mono_plan_receipts,
        "prefix_fragments_verified": expected_candidates,
        "supported_population": len(eligible_indices),
        "scene_clusters": len(accepted_scenes),
        "strong_support_histories": strong_support,
        "attrition_reasons": dict(sorted(attrition_reasons.items())),
        "target_met": target_met,
        "factual_C_B2_C2_executed": False,
        "query_navigation_outcomes_read": False,
        "population_sha256": population_sha,
        "population_file_ledger_entries_verified": ledger_entries,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    require(not args.out.exists(), "independent verification output exists")
    result = verify(run_root=args.run_root, protocol_path=args.protocol)
    args.out.write_text(json.dumps(
        result, indent=2, sort_keys=True, allow_nan=False
    ) + "\n")
    args.out.with_name(args.out.name + ".sha256").write_text(
        f"{sha256(args.out)}  {args.out.name}\n"
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
