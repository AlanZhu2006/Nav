#!/usr/bin/env python3
"""Fail-closed paired summary for actual-online full-mono HM3D mixed roles."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

from MemNavData.hm3d_fullmono_mixed_role import (
    ARMS,
    PRIMARY_CONTRASTS,
    paired_contrast,
    require,
    rotated_arm_order,
    scene_cluster_interval,
)


SCHEMA = "hm3d_fullmono_mixed_role_summary_v1_20260820"
BOOTSTRAP_SEED = 2026082001
BOOTSTRAP_RESAMPLES = 100_000


def direction_stratum(degrees: float) -> str:
    wrapped = (float(degrees) + 180.0) % 360.0 - 180.0
    magnitude = abs(wrapped)
    if magnitude <= 45.0:
        return "front"
    if magnitude < 135.0:
        return "lateral"
    return "back"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_records(
    run_root: Path, manifest: dict[str, Any], manifest_sha: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    records = []
    completions = []
    for index, item in enumerate(manifest["episodes"]):
        scene, episode = str(item["scene"]), str(item["episode"])
        label = f"{index:03d}_{scene}_{episode}"
        root = run_root / "evaluation" / "natural_direction" / label
        path = root / "completion.json"
        require(path.is_file(), f"missing completion {label}")
        require(sha256(path) ==
                (root / "completion.json.sha256").read_text().split()[0],
                f"completion hash changed {label}")
        completion = json.loads(path.read_text())
        queries = [
            query for pair in item["pairs"] for query in pair["queries"]
        ]
        require({query["analysis_role"] for query in queries} ==
                {"novel", "revisit"} and len(queries) == 2,
                f"{label}: manifest role metadata changed")
        query_metadata = {query["analysis_role"]: query for query in queries}
        require(completion["benchmark_manifest_sha256"] == manifest_sha,
                f"{label}: benchmark manifest changed")
        require(completion["online_a_depth_source"] == "monocular_sidecar",
                f"{label}: Goal-A was not mono")
        require(completion["arm_order"] == list(rotated_arm_order(index)),
                f"{label}: arm order changed")
        require(completion["prefix_equality"] is True,
                f"{label}: paired replay changed")
        completions.append(completion)
        for arm in ARMS:
            metric = root / arm / "metric.csv"
            require(metric.is_file(), f"{label}/{arm}: metric missing")
            with metric.open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            require(len(rows) == 2 and
                    {row["analysis_role"] for row in rows} ==
                    {"novel", "revisit"},
                    f"{label}/{arm}: role pair changed")
            for row in rows:
                role = row["analysis_role"]
                reached = int(row["reached"])
                distance = float(row["final_goal_dist_m"])
                require(reached == int(distance < 1.0),
                        f"{label}/{arm}/{role}: success-distance mismatch")
                require(row["navdp_depth_source"] == "monocular_sidecar",
                        f"{label}/{arm}/{role}: depth source changed")
                require(int(row["metric_depth_sensor_consumed_any"]) == 0,
                        f"{label}/{arm}/{role}: metric depth consumed")
                require(int(row["monocular_receipt_plans"]) > 0 and
                        int(row["monocular_active_receipt_plans"]) ==
                        int(row["monocular_receipt_plans"]),
                        f"{label}/{arm}/{role}: mono receipts incomplete")
                require(int(row["monocular_scale_receipt_hashes"]) == 1,
                        f"{label}/{arm}/{role}: mono scale drift")
                records.append({
                    "history_index": index,
                    "scene": scene,
                    "episode": episode,
                    "role": role,
                    "arm": arm,
                    "reached": reached,
                    "final_distance_m": distance,
                    "certificate_accept_plans": int(
                        row["certificate_accept_plans"]
                    ),
                    "runtime_failure_plans": int(row["runtime_failure_plans"]),
                    "monocular_receipt_plans": int(row["monocular_receipt_plans"]),
                    "initial_path_direction_deg": float(
                        query_metadata[role][
                            "initial_path_direction_relative_to_a_end_deg"
                        ]
                    ),
                    "initial_path_direction_stratum": direction_stratum(
                        query_metadata[role][
                            "initial_path_direction_relative_to_a_end_deg"
                        ]
                    ),
                })
    require(len(records) == len(manifest["episodes"]) * len(ARMS) * 2,
            "full-mono result row count changed")
    return records, completions


def statistic_rows(records: list[dict[str, Any]], role: str) -> list[dict]:
    selected = records if role == "all" else [
        row for row in records if row["role"] == role
    ]
    return [{
        "scene": row["scene"],
        "episode": (f"{row['episode']}/{row['role']}"
                    if role == "all" else row["episode"]),
        "arm": row["arm"],
        "reached": row["reached"],
    } for row in selected]


def arm_summary(records: list[dict], arm: str, role: str) -> dict:
    rows = [row for row in records
            if row["arm"] == arm and (role == "all" or row["role"] == role)]
    successes = sum(row["reached"] for row in rows)
    return {
        "n": len(rows),
        "successes": successes,
        "sr": successes / len(rows) if rows else None,
        "mean_final_distance_m": (
            sum(row["final_distance_m"] for row in rows) / len(rows)
            if rows else None
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--bench-root", type=Path, required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    manifest_path = args.bench_root / "manifest.json"
    require(sha256(manifest_path) == args.expected_manifest_sha256,
            "sealed full-mono benchmark changed")
    manifest = json.loads(manifest_path.read_text())
    population = json.loads(
        (args.bench_root.parent / "population_receipt.json").read_text()
    )
    records, completions = load_records(
        args.run_root, manifest, args.expected_manifest_sha256
    )
    analysis_contract = population.get("analysis_contract", {})
    bootstrap_seed = int(analysis_contract.get(
        "bootstrap_seed", BOOTSTRAP_SEED))
    bootstrap_resamples = int(analysis_contract.get(
        "bootstrap_resamples", BOOTSTRAP_RESAMPLES))
    noninferiority_margin_pp = float(analysis_contract.get(
        "raw_revisit_noninferiority_margin_pp", -10.0))
    results = {}
    for role in ("novel", "revisit", "all"):
        rows = statistic_rows(records, role)
        contrasts = {}
        for treatment, reference in PRIMARY_CONTRASTS:
            key = f"{treatment}_minus_{reference}"
            contrasts[key] = paired_contrast(rows, treatment, reference)
            contrasts[key]["scene_cluster_bootstrap_95"] = (
                scene_cluster_interval(
                    rows, treatment, reference,
                    seed=bootstrap_seed,
                    resamples=bootstrap_resamples,
                )
            )
        results[role] = {
            "arms": {arm: arm_summary(records, arm, role) for arm in ARMS},
            "contrasts": contrasts,
        }

    direction_results = {}
    for role in ("novel", "revisit"):
        direction_results[role] = {}
        for stratum in ("front", "lateral", "back"):
            selected = [
                row for row in records
                if row["role"] == role and
                row["initial_path_direction_stratum"] == stratum
            ]
            if not selected:
                direction_results[role][stratum] = {
                    "histories": 0, "arms": {}, "contrasts": {}}
                continue
            rows = statistic_rows(selected, "all")
            contrasts = {}
            for treatment, reference in PRIMARY_CONTRASTS:
                key = f"{treatment}_minus_{reference}"
                contrasts[key] = paired_contrast(rows, treatment, reference)
                contrasts[key]["scene_cluster_bootstrap_95"] = (
                    scene_cluster_interval(
                        rows, treatment, reference,
                        seed=bootstrap_seed, resamples=bootstrap_resamples,
                    )
                )
            direction_results[role][stratum] = {
                "histories": len(selected) // len(ARMS),
                "arms": {
                    arm: arm_summary(selected, arm, "all") for arm in ARMS
                },
                "contrasts": contrasts,
            }

    cec_rows = [row for row in records if row["arm"] == "mono_cec"]
    summary = {
        "schema_version": SCHEMA,
        "status": "complete",
        "scope": population["scope"],
        "fresh_scene_generalization": bool(
            population["fresh_scene_generalization"]),
        "benchmark_manifest_sha256": args.expected_manifest_sha256,
        "source_goal_a_episodes": population["source_goal_a_episodes"],
        "goal_a_successes": population["goal_a_successes"],
        "materialized_histories": population["materialized_histories"],
        "histories": len(manifest["episodes"]),
        "scene_count": len({row["scene"] for row in records}),
        "queries_per_arm": 2 * len(manifest["episodes"]),
        "arms": list(ARMS),
        "runtime_role_visibility": "none",
        "shared_history_policy": "actual_mono_navdp_goal_a_rgb_replay",
        "bootstrap_seed": bootstrap_seed,
        "bootstrap_resamples": bootstrap_resamples,
        "results": results,
        "initial_path_direction_strata": direction_results,
        "raw_revisit_noninferiority": {
            "contrast": "mono_cec_minus_mono_raw_fixed",
            "margin_pp": noninferiority_margin_pp,
            "scene_cluster_ci_lower_pp": 100.0 * float(
                results["revisit"]["contrasts"][
                    "mono_cec_minus_mono_raw_fixed"
                ]["scene_cluster_bootstrap_95"][0]
            ),
            "passes_descriptive_margin": float(
                results["revisit"]["contrasts"][
                    "mono_cec_minus_mono_raw_fixed"
                ]["scene_cluster_bootstrap_95"][0]
            ) >= noninferiority_margin_pp / 100.0,
        },
        "cec_behavior": {
            "query_count": len(cec_rows),
            "accepted_queries": sum(
                row["certificate_accept_plans"] > 0 for row in cec_rows
            ),
            "accepted_queries_by_role": {
                role: sum(
                    row["role"] == role and row["certificate_accept_plans"] > 0
                    for row in cec_rows
                ) for role in ("novel", "revisit")
            },
            "fully_rejected_exact_native_by_role": {
                role: sum(
                    completion["fully_rejected_exact_native"][role]
                    for completion in completions
                ) for role in ("novel", "revisit")
            },
            "runtime_failure_plans": sum(
                row["runtime_failure_plans"] for row in cec_rows
            ),
        },
        "depth_audit": {
            "goal_a_metric_sensor_reads": population[
                "metric_depth_sensor_reads_goal_a"
            ],
            "query_metric_sensor_reads": 0,
            "query_plans_with_mono_receipts": sum(
                row["monocular_receipt_plans"] for row in records
            ),
        },
        "population_selection": population.get("population_selection"),
        "interpretation_boundary": (
            "The histories and all query controllers consume causal RGB-only "
            "monocular depth readouts. This experiment establishes fresh-scene "
            "confirmation but not mono-vs-metric non-inferiority."
            if population["fresh_scene_generalization"] else
            "The histories and all query controllers consume causal RGB-only "
            "monocular depth readouts. Scenes are reused HM3D scenes, so this "
            "is a prospective full-mono integration result, not fresh-scene "
            "generalization or mono-vs-metric non-inferiority."
        ),
    }
    require(not args.out.exists(), "summary output already exists")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
