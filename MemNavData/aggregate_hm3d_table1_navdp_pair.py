#!/usr/bin/env python3
"""Aggregate the fresh-query HM3D NavDP native/CEC Table-1 pair."""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np

from MemNavData.hm3d_fullmono_mixed_role import selected_arm_order


SCHEMAS = {
    "HM3D": "hm3d_table1_navdp_pair_summary_v1_20260829",
    "MP3D": "mp3d_table1_navdp_pair_summary_v1_20260829",
    "HM3D_TABLE2": "hm3d_table2_leg3_navdp_pair_summary_v1_20260829",
}
SCHEMA = SCHEMAS["HM3D"]
ARMS = ("mono_native", "mono_cec")
ROLES = ("novel", "revisit")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            value.update(block)
    return value.hexdigest()


def exact_mcnemar(gains: int, losses: int) -> float:
    discordant = int(gains) + int(losses)
    if discordant == 0:
        return 1.0
    tail = sum(
        math.comb(discordant, value)
        for value in range(min(int(gains), int(losses)) + 1)
    ) / (2 ** discordant)
    return min(1.0, 2.0 * tail)


def spl(row: dict[str, Any]) -> float:
    if int(row["reached"]) == 0:
        return 0.0
    shortest = float(row["geodesic_m"])
    executed = float(row["path_len_m"])
    return shortest / max(shortest, executed, 1e-12)


def scene_cluster_ci(
    rows: list[dict[str, Any]], *, samples: int, seed: int,
) -> list[float]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row["scene"])].append(row)
    scenes = sorted(groups)
    require(bool(scenes), "cannot bootstrap an empty population")
    numerators = []
    denominators = []
    for scene in scenes:
        units: dict[tuple[str, str], dict[str, int]] = {}
        for row in groups[scene]:
            key = str(row["episode"]), str(row["role"])
            units.setdefault(key, {})[str(row["arm"])] = int(row["reached"])
        require(all(set(value) == set(ARMS) for value in units.values()),
                "scene bootstrap pairing is incomplete")
        numerators.append(sum(
            value["mono_cec"] - value["mono_native"]
            for value in units.values()
        ))
        denominators.append(len(units))
    numerator = np.asarray(numerators, dtype=np.float64)
    denominator = np.asarray(denominators, dtype=np.float64)
    rng = np.random.default_rng(seed)
    values = np.empty(samples, dtype=np.float64)
    chunk = 10_000
    for start in range(0, samples, chunk):
        stop = min(samples, start + chunk)
        draws = rng.integers(0, len(scenes), size=(stop - start, len(scenes)))
        values[start:stop] = (
            numerator[draws].sum(axis=1)
            / denominator[draws].sum(axis=1)
        )
    return [
        100.0 * float(np.quantile(values, 0.025)),
        100.0 * float(np.quantile(values, 0.975)),
    ]


def _manifest_queries(item: dict[str, Any]) -> dict[str, dict[str, Any]]:
    queries = [query for pair in item["pairs"] for query in pair["queries"]]
    require(len(queries) == 2, "history does not contain two role queries")
    by_role = {str(query["analysis_role"]): query for query in queries}
    require(set(by_role) == set(ROLES), "history roles are not Novel/Revisit")
    return by_role


def _direction_stratum(item: dict[str, Any]) -> str:
    """Read the frozen stratum from its canonical Novel-query location."""

    novel = _manifest_queries(item)["novel"]
    value = str(novel.get("assigned_direction_stratum", ""))
    require(value in {"front", "side", "rear"},
            "Novel query has an invalid direction stratum")
    return value


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    units: dict[tuple[str, str, str], dict[str, dict[str, Any]]] = {}
    for row in rows:
        key = str(row["scene"]), str(row["episode"]), str(row["role"])
        units.setdefault(key, {})[str(row["arm"])] = row
    require(all(set(value) == set(ARMS) for value in units.values()),
            "paired NavDP units are incomplete")
    gains = sum(
        value["mono_cec"]["reached"] == 1
        and value["mono_native"]["reached"] == 0
        for value in units.values()
    )
    losses = sum(
        value["mono_cec"]["reached"] == 0
        and value["mono_native"]["reached"] == 1
        for value in units.values()
    )
    native = [value["mono_native"] for value in units.values()]
    cec = [value["mono_cec"] for value in units.values()]
    return {
        "n": len(units),
        "native_success": sum(int(row["reached"]) for row in native),
        "cec_success": sum(int(row["reached"]) for row in cec),
        "native_sr": mean(int(row["reached"]) for row in native),
        "cec_sr": mean(int(row["reached"]) for row in cec),
        "risk_difference_pp": 100.0 * mean(
            int(value["mono_cec"]["reached"])
            - int(value["mono_native"]["reached"])
            for value in units.values()
        ),
        "paired_gain": gains,
        "paired_loss": losses,
        "mcnemar_exact_p": exact_mcnemar(gains, losses),
        "native_spl": mean(spl(row) for row in native),
        "cec_spl": mean(spl(row) for row in cec),
        "native_mean_final_distance_m": mean(
            float(row["final_goal_dist_m"]) for row in native),
        "cec_mean_final_distance_m": mean(
            float(row["final_goal_dist_m"]) for row in cec),
        "native_mean_path_len_m": mean(
            float(row["path_len_m"]) for row in native),
        "cec_mean_path_len_m": mean(
            float(row["path_len_m"]) for row in cec),
        "native_mean_steps": mean(int(row["steps"]) for row in native),
        "cec_mean_steps": mean(int(row["steps"]) for row in cec),
        "cec_takeover_queries": sum(
            int(row["certificate_accept_plans"]) > 0 for row in cec),
        "cec_takeover_plans": sum(
            int(row["certificate_accept_plans"]) for row in cec),
    }


def aggregate(
    run_root: Path,
    benchmark_root: Path,
    construction_verification: Path,
    *,
    claim_scope: str,
    dataset: str = "HM3D",
    bootstrap_samples: int = 100_000,
    bootstrap_seed: int = 20260829,
) -> dict[str, Any]:
    require(dataset in SCHEMAS, "unsupported Table-1 dataset")
    require(bootstrap_samples > 0, "bootstrap sample count must be positive")
    verification = json.loads(construction_verification.read_text())
    require(verification.get("verified") is True
            and verification.get("construction_only") is True,
            "construction verification did not pass")
    require(verification.get("formal_policy_evaluation_authorized") is True,
            "construction power gate did not authorize policy evaluation")
    manifest_path = benchmark_root / "manifest.json"
    require(digest(manifest_path) == verification["benchmark_manifest_sha256"],
            "benchmark differs from the independently verified population")
    manifest = json.loads(manifest_path.read_text())
    episodes = manifest.get("episodes")
    require(isinstance(episodes, list) and episodes,
            "verified benchmark contains no histories")
    require(len(episodes) == int(verification["histories"]),
            "history denominator differs from construction verification")
    require(len({str(row["scene"]) for row in episodes})
            == int(verification["scene_clusters"]),
            "scene denominator differs from construction verification")
    table2 = dataset == "HM3D_TABLE2"
    if table2:
        require(
            verification.get("schema_version")
            == (
                "hm3d_table2_leg3_mixed_role_construction_"
                "verification_v1_20260829"
            ),
            "Table-2 construction verifier schema changed",
        )
        population_path = benchmark_root.parent / "population_receipt.json"
        require(
            digest(population_path)
            == verification.get("population_receipt_sha256"),
            "Table-2 population receipt changed",
        )
        table2_population = json.loads(population_path.read_text())
    else:
        table2_population = None

    rows: list[dict[str, Any]] = []
    cells = []
    order_counts: dict[tuple[str, ...], int] = defaultdict(int)
    fallback_counts = {role: 0 for role in ROLES}
    stratum_counts: dict[str, int] = defaultdict(int)
    for index, item in enumerate(episodes):
        scene, episode = str(item["scene"]), str(item["episode"])
        queries = _manifest_queries(item)
        stratum_counts[_direction_stratum(item)] += 1
        label = f"{index:03d}_{scene}_{episode}"
        root = run_root / "evaluation" / "natural_direction" / label
        completion_path = root / "completion.json"
        receipt_path = root / "completion.json.sha256"
        require(completion_path.is_file() and receipt_path.is_file(),
                f"missing completion for {label}")
        require(receipt_path.read_text().split()[0] == digest(completion_path),
                f"completion receipt changed for {label}")
        completion = json.loads(completion_path.read_text())
        require(completion.get("benchmark_manifest_sha256")
                == verification["benchmark_manifest_sha256"],
                f"manifest binding changed for {label}")
        require(completion.get("arms") == list(ARMS),
                f"wrong NavDP arms for {label}")
        expected_order = list(selected_arm_order(index, ARMS))
        require(completion.get("arm_order") == expected_order,
                f"arm order changed for {label}")
        require(completion.get("prefix_equality") is True
                and completion.get("runtime_role_visibility") == "none",
                f"paired replay or role hiding failed for {label}")
        require(completion.get("online_a_depth_source") == "monocular_sidecar",
                f"Goal-A history is not full-mono for {label}")
        if table2:
            require(
                completion.get("history_contract") == "actual_ab"
                and completion.get("shared_history_policy")
                == "actual_mono_navdp_novel_A_then_novel_B_rgb_replay",
                f"Table-2 A/B replay contract changed for {label}",
            )
            require(
                int(completion.get("prefix_A_steps", 0)) > 0
                and int(completion.get("prefix_B_steps", 0)) > 0
                and int(completion["prefix_A_steps"])
                + int(completion["prefix_B_steps"])
                == int(completion["online_a_steps"]),
                f"Table-2 A/B prefix lengths changed for {label}",
            )
        order_counts[tuple(expected_order)] += 1
        for role in ROLES:
            if completion["fully_rejected_exact_native"][role] is True:
                fallback_counts[role] += 1
        for arm in ARMS:
            metric_path = root / arm / "metric.csv"
            require(metric_path.is_file(), f"metric missing for {label}/{arm}")
            with metric_path.open(newline="") as handle:
                arm_rows = list(csv.DictReader(handle))
            require(len(arm_rows) == 2
                    and {row["analysis_role"] for row in arm_rows}
                    == set(ROLES),
                    f"role rows changed for {label}/{arm}")
            for row in arm_rows:
                role = str(row["analysis_role"])
                reached = int(row["reached"])
                final_distance = float(row["final_goal_dist_m"])
                require(reached == int(final_distance < 1.0),
                        f"success-distance mismatch for {label}/{arm}/{role}")
                require(row["navdp_depth_source"] == "monocular_sidecar"
                        and int(row["metric_depth_sensor_consumed_any"]) == 0,
                        f"metric depth leaked into {label}/{arm}/{role}")
                require(int(row["monocular_receipt_plans"]) > 0
                        and int(row["monocular_active_receipt_plans"])
                        == int(row["monocular_receipt_plans"])
                        and int(row["monocular_scale_receipt_hashes"]) == 1,
                        f"monocular receipt audit failed for {label}/{arm}/{role}")
                require(int(row["runtime_failure_plans"]) == 0,
                        f"certificate runtime failure at "
                        f"{label}/{arm}/{role}")
                rows.append({
                    "history_index": index,
                    "scene": scene,
                    "episode": episode,
                    "query_id": str(queries[role]["query_id"]),
                    "role": role,
                    "arm": arm,
                    "reached": reached,
                    "geodesic_m": float(row["geodesic_m"]),
                    "path_len_m": float(row["path_len_m"]),
                    "steps": int(row["steps"]),
                    "final_goal_dist_m": final_distance,
                    "certificate_accept_plans": int(
                        row["certificate_accept_plans"]),
                    "runtime_failure_plans": int(row["runtime_failure_plans"]),
                })
        cells.append({
            "history_index": index,
            "scene": scene,
            "episode": episode,
            "completion_sha256": digest(completion_path),
        })

    expected_rows = len(episodes) * len(ARMS) * len(ROLES)
    require(len(rows) == expected_rows, "paired NavDP row denominator changed")
    if len(episodes) > 1:
        require(set(order_counts) == {
            ("mono_native", "mono_cec"),
            ("mono_cec", "mono_native"),
        }, "both NavDP arm orders are not represented")
        require(max(order_counts.values()) - min(order_counts.values()) <= 1,
                "NavDP arm order is not balanced")

    results = {}
    for role in (*ROLES, "all"):
        selected = rows if role == "all" else [
            row for row in rows if row["role"] == role
        ]
        result = _summarize(selected)
        result["scene_cluster_bootstrap_95ci_pp"] = scene_cluster_ci(
            selected, samples=bootstrap_samples, seed=bootstrap_seed,
        )
        results[role] = result

    cec_rows = [row for row in rows if row["arm"] == "mono_cec"]
    result = {
        "schema_version": SCHEMAS[dataset],
        "verified": True,
        "dataset": dataset,
        "claim_scope": claim_scope,
        "run_root": str(run_root),
        "benchmark_root": str(benchmark_root),
        "benchmark_manifest_sha256": digest(manifest_path),
        "construction_verification": str(construction_verification),
        "construction_verification_sha256": digest(construction_verification),
        "controller": "navdp",
        "treatment": "certified_scale_free_bearing_residual",
        "reference": "unchanged_native_imagegoal_request",
        "arms": list(ARMS),
        "histories": len(episodes),
        "scene_clusters": len({str(row["scene"]) for row in episodes}),
        "queries": 2 * len(episodes),
        "runtime_role_visibility": "none",
        "shared_history_policy": (
            "actual_mono_navdp_novel_A_then_novel_B_rgb_replay"
            if table2 else "actual_mono_navdp_goal_a_rgb_replay"
        ),
        "fresh_query": True,
        "fresh_scene": False,
        "fresh_history": False,
        "bootstrap_seed": bootstrap_seed,
        "bootstrap_samples": bootstrap_samples,
        "direction_strata": dict(sorted(stratum_counts.items())),
        "arm_order_counts": {
            "/".join(key): value for key, value in sorted(order_counts.items())
        },
        "results": results,
        "safety": {
            "novel_takeover_queries": sum(
                row["role"] == "novel"
                and int(row["certificate_accept_plans"]) > 0
                for row in cec_rows),
            "revisit_takeover_queries": sum(
                row["role"] == "revisit"
                and int(row["certificate_accept_plans"]) > 0
                for row in cec_rows),
            "fully_rejected_exact_native_by_role": fallback_counts,
            "runtime_failure_plans": 0,
            "metric_depth_sensor_reads": 0,
        },
        "cells": cells,
    }
    if table2:
        assert table2_population is not None
        segment_counts: dict[str, int] = defaultdict(int)
        for row in episodes:
            segment = str(row.get("table2_selected_revisit_segment", ""))
            require(segment in {"A", "B"},
                    "Table-2 Revisit source segment changed")
            segment_counts[segment] += 1
        result.update({
            "estimand": "Leg3_C_given_factual_successful_A_and_B",
            "conditional_on_factual_AB_success": True,
            "factual_prefix_waterfall": {
                "source_A_successful_histories_entering_B": int(
                    table2_population["source_A_attempts"]
                ),
                "factual_AB_successful_prefixes": int(
                    table2_population["factual_AB_successful_prefixes"]
                ),
                "factual_AB_scene_clusters": int(
                    table2_population["factual_AB_scene_clusters"]
                ),
                "leg3_constructible_histories": int(
                    table2_population["leg3_constructible_histories"]
                ),
                "leg3_scene_clusters": int(
                    table2_population["leg3_scene_clusters"]
                ),
            },
            "revisit_source_segment_counts": dict(
                sorted(segment_counts.items())
            ),
            "unconditional_three_leg_joint_sr_reported": False,
        })
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--construction-verification", type=Path, required=True)
    parser.add_argument("--claim-scope", required=True)
    parser.add_argument("--dataset", choices=tuple(SCHEMAS), default="HM3D")
    parser.add_argument("--bootstrap-samples", type=int, default=100_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260829)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = aggregate(
        args.run_root.resolve(), args.benchmark_root.resolve(),
        args.construction_verification.resolve(),
        claim_scope=args.claim_scope,
        dataset=args.dataset,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
    )
    require(not args.out.exists(), "summary output already exists")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(
        result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({
        "verified": True,
        "histories": result["histories"],
        "native": result["results"]["all"]["native_success"],
        "cec": result["results"]["all"]["cec_success"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
