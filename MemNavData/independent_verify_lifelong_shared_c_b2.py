#!/usr/bin/env python3
"""Independent raw-output verification for shared-C B2 comparison."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

from lifelong_shared_c_contract import ARMS, POPULATION_SCHEMA, require, sha256_file


SCHEMA = "lifelong_shared_c_b2_independent_verification_v1_20260825"


def exact_mcnemar(gains: int, losses: int) -> float:
    discordant = gains + losses
    if not discordant:
        return 1.0
    tail = sum(math.comb(discordant, k)
               for k in range(min(gains, losses) + 1)) / (2 ** discordant)
    return min(1.0, 2.0 * tail)


def load_arm(records: list[dict], arm: str) -> tuple[dict, dict, dict]:
    metrics, plans, compute = {}, {}, {}
    for record in records:
        run = Path(record["run_root"])
        metric_path = run / "result/metric.csv"
        plans_path = run / "result" / f"{record['episode']}_plans.json"
        summary_path = run / "result/summary.json"
        compute_path = run / "compute_identity.json"
        for path, key in (
            (metric_path, "metric_sha256"), (plans_path, "plans_sha256"),
            (summary_path, "summary_sha256"),
            (compute_path, "compute_identity_sha256"),
        ):
            require(sha256_file(path) == record[key],
                    f"{arm}: raw {path.name} hash changed")
        with metric_path.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        require(len(rows) == 1, f"{arm}: raw metric rows changed")
        identity = (record["scene"], record["episode"])
        require(identity not in metrics, f"{arm}: duplicate identity")
        metrics[identity] = rows[0]
        plans[identity] = json.loads(plans_path.read_text())
        compute[identity] = json.loads(compute_path.read_text())
    return metrics, plans, compute


def process_identity(payload: dict) -> tuple:
    return (
        payload.get("host"), payload.get("gpu_uuid"),
        payload.get("memnav", {}).get("pid"),
        payload.get("memnav", {}).get("process_start_ticks"),
        payload.get("navdp", {}).get("pid"),
        payload.get("navdp", {}).get("process_start_ticks"),
        payload.get("cec_hub", {}).get("pid"),
        payload.get("cec_hub", {}).get("process_start_ticks"),
    )


def compare(first: dict, second: dict, first_name: str, second_name: str) -> dict:
    require(set(first) == set(second), "paired B2 populations differ")
    gains = losses = 0
    for identity in first:
        delta = int(first[identity]["reached_B2"]) - int(
            second[identity]["reached_B2"])
        gains += delta == 1
        losses += delta == -1
    return {
        first_name: sum(int(row["reached_B2"]) for row in first.values()),
        second_name: sum(int(row["reached_B2"]) for row in second.values()),
        "n": len(first),
        "paired_gains": gains,
        "paired_losses": losses,
        "exact_McNemar_p": exact_mcnemar(gains, losses),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--population", type=Path, required=True)
    parser.add_argument("--aggregate", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--fullmono-prefix-audit", action="store_true",
        help="independently re-audit raw actual-mono A/B receipts")
    args = parser.parse_args()
    population = json.loads(args.population.read_text())
    require(population.get("schema_version") == POPULATION_SCHEMA,
            "population schema changed")
    aggregate_path = args.aggregate / "aggregate_inputs.json"
    aggregate = json.loads(aggregate_path.read_text())
    require(aggregate["population_sha256"] == sha256_file(args.population),
            "aggregate consumed another population")
    arms = {}
    plans = {}
    compute = {}
    for arm in ARMS:
        arms[arm], plans[arm], compute[arm] = load_arm(
            aggregate["arms"][arm], arm)
    identities = set(arms[ARMS[0]])
    require(all(set(arms[arm]) == identities for arm in ARMS),
            "arm populations differ")
    fullmono_audits = []
    source_items = {}
    source_population_path = None
    if args.fullmono_prefix_audit:
        source_population_path = Path(population["source_population"])
        require(sha256_file(source_population_path)
                == population["source_population_sha256"],
                "full-mono source population changed")
        source_population = json.loads(source_population_path.read_text())
        source_items = {
            int(row["population_index"]): row
            for row in source_population["accepted"]
        }
    population_items = {
        (row["scene"], row["episode"]): row
        for row in population["accepted"]
    }
    for identity in sorted(identities):
        rows = [arms[arm][identity] for arm in ARMS]
        require(all(int(row["shared_C_prefix_replayed"]) == 1 for row in rows),
                f"{identity}: shared C was not replayed")
        require(len({row["shared_C_trace_sha256"] for row in rows}) == 1,
                f"{identity}: arms used different C traces")
        for field in (
            "shared_C_start_x", "shared_C_start_y", "shared_C_start_z",
            "shared_C_start_yaw", "B2_start_x", "B2_start_y", "B2_start_z",
            "B2_start_yaw",
        ):
            require(len({row[field] for row in rows}) == 1,
                    f"{identity}: causal B2 start differs at {field}")
        primary_plans = [plans[arm][identity] for arm in ARMS[:2]]
        for field in ("frozen_legA", "frozen_legB", "frozen_legC",
                      "rollout_traces", "memory_traces"):
            if field == "memory_traces":
                first = {k: primary_plans[0][field][k] for k in ("A", "B", "C")}
                second = {k: primary_plans[1][field][k] for k in ("A", "B", "C")}
            elif field == "rollout_traces":
                first = {k: primary_plans[0][field][k] for k in ("A", "B", "C")}
                second = {k: primary_plans[1][field][k] for k in ("A", "B", "C")}
            else:
                first, second = primary_plans[0][field], primary_plans[1][field]
            require(first == second, f"{identity}: primary shared prefix differs")
        require(process_identity(compute["all_prior"][identity])
                == process_identity(compute["initial_leg_only"][identity]),
                f"{identity}: primary arms did not share server processes")
        require(int(rows[0]["B2_candidate_ceiling"])
                == int(rows[0]["online_B_candidate_ceiling"]),
                f"{identity}: all-prior ceiling changed")
        require(int(rows[1]["B2_candidate_ceiling"])
                == int(rows[1]["online_A_candidate_ceiling"]),
                f"{identity}: initial-only ceiling changed")
        require(int(rows[2]["B2_candidate_ceiling"])
                == int(rows[2]["online_B_candidate_ceiling"]),
                f"{identity}: forced-native ceiling changed")
        for plan in plans["forced_reject_native"][identity]["B2"]:
            if plan.get("cec_takeover") is None:
                continue
            require(plan.get("cec_forced_reject_native") is True
                    and plan.get("cec_takeover") is False,
                    f"{identity}: forced-native granted takeover")
        if args.fullmono_prefix_audit:
            from independent_verify_hm3d_fullmono_lifelong import (
                verify_fullmono_prefix,
            )
            selected = population_items[identity]
            source_item = source_items[int(selected["source_population_index"])]
            require(source_item["scene"] == identity[0]
                    and source_item["episode"] == identity[1],
                    f"{identity}: full-mono source identity changed")
            fullmono_audits.append(verify_fullmono_prefix(
                source_population_path, source_item))
            require(all(int(row.get("metric_depth_reads_B2", 0)) == 0
                        for row in rows),
                    f"{identity}: B2 consumed metric depth")
    result = {
        "schema_version": SCHEMA,
        "verified": True,
        "controller": population["controller"],
        "episodes": len(identities),
        "scenes": len({scene for scene, _episode in identities}),
        "shared_C_prefix_exact_across_primary_arms": True,
        "primary_same_process_pairing": True,
        "selection_reads_B2_navigation_outcomes": False,
        "fullmono_prefix_audit": {
            "enabled": bool(args.fullmono_prefix_audit),
            "histories": len(fullmono_audits),
            "metric_depth_reads_A_B_B2": 0 if fullmono_audits else None,
        },
        "B2": {
            "all_prior_vs_initial_leg_only": compare(
                arms["all_prior"], arms["initial_leg_only"],
                "all_prior", "initial_leg_only"),
            "all_prior_vs_forced_reject_native": compare(
                arms["all_prior"], arms["forced_reject_native"],
                "all_prior", "forced_reject_native"),
        },
        "population_sha256": sha256_file(args.population),
        "aggregate_inputs_sha256": sha256_file(aggregate_path),
    }
    require(not args.out.exists(), f"verification output exists: {args.out}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(
        result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
