#!/usr/bin/env python3
"""Aggregate one-episode lifelong NNR jobs without changing the frozen set."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

from deterministic_eval_protocol import file_sha256


SCOPES = ("all_prior", "initial_leg_only")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def arm_summary(scope: str, rows: list[dict]) -> dict:
    def total(key: str) -> int:
        return sum(int(row[key]) for row in rows)

    return {
        "result_schema": "shared_online_lifelong_nnr_eval_v1_20260821",
        "history_scope": scope,
        "episodes": len(rows),
        "runtime_role_visible": False,
        "frozen_actual_online_prefix": "A_then_Novel_B",
        "query_sequence": ["C", "B2", "C2"],
        "C_success": total("reached_C"),
        "B2_success_given_C": sum(
            int(row["reached_B2"]) for row in rows if int(row["reached_C"])
        ),
        "C2_success_given_CB2": sum(
            int(row["reached_C2"])
            for row in rows
            if int(row["reached_C"]) and int(row["reached_B2"])
        ),
        "query_joint_success": total("query_joint_success"),
        "B2_factual_B_anchor_use_given_evaluated": sum(
            int(row["B2_used_factual_B_anchor"])
            for row in rows if int(row["evaluated_B2"])
        ),
        "claim_scope": (
            "internal lifelong accumulation mechanism on a result-blind, "
            "pre-sealed factual-B support population"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--population", type=Path, required=True)
    parser.add_argument("--evaluation-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    population_path = args.population.resolve()
    population_root = population_path.parent
    require((population_root / "SEALED").is_file(), "support population is not sealed")
    receipt = (population_root / "population.json.sha256").read_text().split()
    require(len(receipt) >= 1 and file_sha256(population_path) == receipt[0], "population receipt changed")
    population = load_json(population_path)
    require(population["selection_reads_query_navigation_outcomes"] is False, "population was outcome-selected")
    selected = list(population["accepted"])
    require(len(selected) == int(population["supported_population"]), "supported count changed")
    require(bool(selected), "supported population is empty")
    require(not args.out.exists(), "aggregate output already exists")

    arm_rows: dict[str, list[dict]] = {scope: [] for scope in SCOPES}
    arm_inputs: dict[str, list[dict]] = {scope: [] for scope in SCOPES}
    temporary = args.out.with_name(args.out.name + ".partial")
    require(not temporary.exists(), "stale aggregate staging directory exists")
    try:
        for scope in SCOPES:
            result_out = temporary / scope / "result"
            result_out.mkdir(parents=True)
            for index, item in enumerate(selected):
                scene = str(item["scene"])
                episode = str(item["episode"])
                run = (
                    args.evaluation_root
                    / f"{index:03d}_{scene}_{episode}"
                    / scope
                )
                result = run / "result"
                metric_path = result / "metric.csv"
                plans_path = result / f"{episode}_plans.json"
                summary_path = result / "summary.json"
                require(metric_path.is_file() and plans_path.is_file() and summary_path.is_file(), f"missing completed run {run}")
                with metric_path.open(newline="") as handle:
                    rows = list(csv.DictReader(handle))
                require(len(rows) == 1, f"{run}: expected one metric row")
                row = rows[0]
                require(row["scene"] == scene and row["episode"] == episode, f"{run}: identity mismatch")
                require(row["history_scope"] == scope, f"{run}: scope mismatch")
                require(row["benchmark_sha256"] == item["benchmark_sha256"], f"{run}: benchmark mismatch")
                require(row["online_B_trace_sha256"] == item["online_b_trace_sha256"], f"{run}: factual-B trace mismatch")
                single_summary = load_json(summary_path)
                require(int(single_summary["episodes"]) == 1 and single_summary["history_scope"] == scope, f"{run}: summary mismatch")
                destination_name = f"{scene}__{episode}_plans.json"
                shutil.copy2(plans_path, result_out / destination_name)
                arm_rows[scope].append(row)
                arm_inputs[scope].append({
                    "scene": scene,
                    "episode": episode,
                    "run_root": str(run.resolve()),
                    "metric_sha256": file_sha256(metric_path),
                    "plans_sha256": file_sha256(plans_path),
                    "summary_sha256": file_sha256(summary_path),
                })

            metric_out = result_out / "metric.csv"
            with metric_out.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(arm_rows[scope][0]))
                writer.writeheader()
                writer.writerows(arm_rows[scope])
            (result_out / "summary.json").write_text(
                json.dumps(arm_summary(scope, arm_rows[scope]), indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )

        aggregate = {
            "schema": "lifelong_nnr_expansion_aggregate_v1_20260821",
            "population_sha256": file_sha256(population_path),
            "episodes": len(selected),
            "scenes": len({item["scene"] for item in selected}),
            "selection_reads_query_navigation_outcomes": False,
            "arms": arm_inputs,
        }
        (temporary / "aggregate_inputs.json").write_text(
            json.dumps(aggregate, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.rename(args.out)
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    print(json.dumps({
        "episodes": len(selected),
        "scenes": len({item["scene"] for item in selected}),
        "out": str(args.out.resolve()),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
