#!/usr/bin/env python3
"""Aggregate the three-arm full-mono lifelong population."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import shutil
import tempfile
from pathlib import Path

from hm3d_fullmono_lifelong import ARMS, RESULT_SCHEMA, exact_mcnemar, require, sha256_file


SCHEMA = "hm3d_fullmono_lifelong_aggregate_v1_20260824"


def quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    location = (len(ordered) - 1) * probability
    lower, upper = math.floor(location), math.ceil(location)
    if lower == upper:
        return float(ordered[lower])
    weight = location - lower
    return float(ordered[lower] * (1.0 - weight) + ordered[upper] * weight)


def cluster_interval(records: list[dict], *, draws: int, seed: int) -> dict:
    grouped: dict[str, list[float]] = {}
    for row in records:
        grouped.setdefault(row["scene"], []).append(float(row["difference"]))
    scenes = sorted(grouped)
    require(bool(scenes), "cluster interval has no scenes")
    rng = random.Random(seed)
    estimates = []
    for _ in range(draws):
        values = []
        for _slot in scenes:
            values.extend(grouped[scenes[rng.randrange(len(scenes))]])
        estimates.append(sum(values) / len(values))
    return {
        "scene_clusters": len(scenes),
        "draws": draws,
        "seed": seed,
        "risk_difference": sum(row["difference"] for row in records) / len(records),
        "percentile_95_CI": [quantile(estimates, 0.025), quantile(estimates, 0.975)],
    }


def arm_endpoint_counts(rows: list[dict]) -> dict:
    """Return explicit numerators and causal denominators for every query."""

    episodes = len(rows)
    evaluated_b2 = sum(int(row["evaluated_B2"]) for row in rows)
    evaluated_c2 = sum(int(row["evaluated_C2"]) for row in rows)
    return {
        "episodes": episodes,
        "C": {
            "success": sum(int(row["reached_C"]) for row in rows),
            "evaluated": episodes,
        },
        "B2_given_C": {
            "success": sum(int(row["reached_B2"]) for row in rows),
            "evaluated": evaluated_b2,
        },
        "C2_given_C_B2": {
            "success": sum(int(row["reached_C2"]) for row in rows),
            "evaluated": evaluated_c2,
        },
        "prefix_survival": {
            str(k): sum(
                int(row["queries_completed_before_first_failure"]) >= k
                for row in rows
            ) for k in (1, 2, 3)
        },
        "query_joint": {
            "success": sum(int(row["query_joint_success"]) for row in rows),
            "evaluated": episodes,
        },
        "B2_factual_B_anchor": {
            "used": sum(int(row["B2_used_factual_B_anchor"]) for row in rows),
            "evaluated": evaluated_b2,
        },
    }


def paired_prefix_comparison(
    first: dict[tuple[str, str], dict],
    second: dict[tuple[str, str], dict],
    *,
    first_name: str,
    second_name: str,
) -> dict:
    """Compare full-sequence survival without post-treatment conditioning."""

    require(set(first) == set(second), "paired prefix populations differ")
    output = {}
    for k in (1, 2, 3):
        records = []
        gains = losses = 0
        for scene, episode in sorted(first):
            a = int(first[(scene, episode)][
                "queries_completed_before_first_failure"
            ]) >= k
            b = int(second[(scene, episode)][
                "queries_completed_before_first_failure"
            ]) >= k
            difference = a - b
            gains += difference == 1
            losses += difference == -1
            records.append({
                "scene": scene,
                "episode": episode,
                "difference": difference,
            })
        output[str(k)] = {
            "endpoint": f"survived_at_least_{k}_queries",
            first_name: sum(
                int(row["queries_completed_before_first_failure"]) >= k
                for row in first.values()
            ),
            second_name: sum(
                int(row["queries_completed_before_first_failure"]) >= k
                for row in second.values()
            ),
            "n": len(records),
            "paired_gains": gains,
            "paired_losses": losses,
            "exact_McNemar_p": exact_mcnemar(gains, losses),
            "scene_cluster_bootstrap": cluster_interval(
                records, draws=100000, seed=20260824 + k
            ),
        }
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--population", type=Path, required=True)
    parser.add_argument("--evaluation-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    require((args.population.parent / "SEALED").is_file(),
            "lifelong population is not sealed")
    receipt = (args.population.parent / "population.json.sha256").read_text().split()
    require(receipt and receipt[0] == sha256_file(args.population),
            "lifelong population receipt changed")
    population = json.loads(args.population.read_text())
    require(population["selection_reads_C_B2_C2_navigation_outcomes"] is False,
            "population read query outcomes")
    require(not args.out.exists(), f"aggregate output exists: {args.out}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=args.out.name + ".tmp.", dir=args.out.parent))
    rows_by_arm = {arm: [] for arm in ARMS}
    inputs = {arm: [] for arm in ARMS}
    try:
        for item in population["accepted"]:
            index = int(item["population_index"])
            scene, episode = str(item["scene"]), str(item["episode"])
            label = f"{index:03d}_{scene}_{episode}"
            for arm in ARMS:
                root = args.evaluation_root / label / arm
                metric = root / "result/metric.csv"
                plans = root / "result" / f"{episode}_plans.json"
                summary = root / "result/summary.json"
                compute = root / "compute_identity.json"
                require(metric.is_file() and plans.is_file()
                        and summary.is_file() and compute.is_file(),
                        f"missing completed arm {label}/{arm}")
                with metric.open(newline="") as handle:
                    rows = list(csv.DictReader(handle))
                require(len(rows) == 1, f"{label}/{arm}: expected one row")
                row = rows[0]
                require(row["result_schema"] == RESULT_SCHEMA,
                        f"{label}/{arm}: result schema changed")
                require(row["scene"] == scene and row["episode"] == episode,
                        f"{label}/{arm}: result identity changed")
                require(row["history_scope"] == arm,
                        f"{label}/{arm}: history scope changed")
                require(row["benchmark_sha256"] == item["benchmark_sha256"],
                        f"{label}/{arm}: benchmark changed")
                require(int(row["metric_depth_reads_queries"]) == 0,
                        f"{label}/{arm}: query consumed metric depth")
                rows_by_arm[arm].append(row)
                inputs[arm].append({
                    "scene": scene, "episode": episode,
                    "metric_sha256": sha256_file(metric),
                    "plans_sha256": sha256_file(plans),
                    "summary_sha256": sha256_file(summary),
                    "compute_identity_sha256": sha256_file(compute),
                    "run_root": str(root.resolve()),
                })
                destination = temporary / "arms" / arm
                destination.mkdir(parents=True, exist_ok=True)
                shutil.copy2(plans, destination / f"{scene}__{episode}_plans.json")
                shutil.copy2(
                    compute,
                    destination / f"{scene}__{episode}_compute_identity.json",
                )
        for arm in ARMS:
            destination = temporary / "arms" / arm / "metric.csv"
            with destination.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows_by_arm[arm][0]))
                writer.writeheader(); writer.writerows(rows_by_arm[arm])

        keyed = {
            arm: {(row["scene"], row["episode"]): row for row in rows}
            for arm, rows in rows_by_arm.items()
        }
        identities = sorted(keyed["all_prior"])
        require(all(set(keyed[arm]) == set(identities) for arm in ARMS),
                "arm populations differ")
        paired = []
        gains = losses = 0
        for identity in identities:
            all_row = keyed["all_prior"][identity]
            initial = keyed["initial_leg_only"][identity]
            require(int(all_row["reached_C"]) == int(initial["reached_C"]),
                    f"{identity}: C outcome differs before B2 treatment")
            if not int(all_row["reached_C"]):
                continue
            difference = int(all_row["reached_B2"]) - int(initial["reached_B2"])
            gains += difference == 1
            losses += difference == -1
            paired.append({
                "scene": identity[0], "episode": identity[1],
                "all_prior": int(all_row["reached_B2"]),
                "initial_leg_only": int(initial["reached_B2"]),
                "difference": difference,
                "both_success": int(all_row["reached_B2"])
                and int(initial["reached_B2"]),
                "steps_difference": int(all_row["steps_B2"])
                - int(initial["steps_B2"]),
                "path_difference_m": float(all_row["len_B2"])
                - float(initial["len_B2"]),
            })
        both = [row for row in paired if row["both_success"]]
        primary_bootstrap = (
            cluster_interval(paired, draws=100000, seed=20260824)
            if paired else None
        )
        analysis = {
            "schema_version": SCHEMA,
            "population_sha256": sha256_file(args.population),
            "episodes": len(identities),
            "scenes": len({identity[0] for identity in identities}),
            "arms": {
                arm: arm_endpoint_counts(rows_by_arm[arm]) for arm in ARMS
            },
            "primary_B2_after_shared_C": {
                "estimable": bool(paired),
                "n": len(paired),
                "all_prior_success": sum(row["all_prior"] for row in paired),
                "initial_leg_only_success": sum(
                    row["initial_leg_only"] for row in paired
                ),
                "paired_gains": gains,
                "paired_losses": losses,
                "exact_McNemar_p": (
                    exact_mcnemar(gains, losses) if paired else None
                ),
                "scene_cluster_bootstrap": primary_bootstrap,
                "both_success_n": len(both),
                "both_success_mean_steps_difference": (
                    sum(row["steps_difference"] for row in both) / len(both)
                    if both else None
                ),
                "both_success_mean_path_difference_m": (
                    sum(row["path_difference_m"] for row in both) / len(both)
                    if both else None
                ),
            },
            "query_outcome_filtered": False,
            "runtime_role_visible": False,
            "metric_depth_reads_queries": 0,
            "raw_inputs": inputs,
        }
        analysis["paired_prefix_survival"] = {
            "all_prior_vs_initial_leg_only": paired_prefix_comparison(
                keyed["all_prior"], keyed["initial_leg_only"],
                first_name="all_prior", second_name="initial_leg_only",
            ),
            "all_prior_vs_forced_reject_native": paired_prefix_comparison(
                keyed["all_prior"], keyed["forced_reject_native"],
                first_name="all_prior", second_name="forced_reject_native",
            ),
        }
        (temporary / "summary.json").write_text(json.dumps(
            analysis, indent=2, sort_keys=True, allow_nan=False
        ) + "\n")
        (temporary / "paired_B2.json").write_text(json.dumps(
            paired, indent=2, sort_keys=True, allow_nan=False
        ) + "\n")
        temporary.replace(args.out)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    print(json.dumps(analysis, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
