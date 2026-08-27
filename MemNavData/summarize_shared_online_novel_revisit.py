#!/usr/bin/env python3
"""Audit and summarize the sealed actual-online NNR paired evaluation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from deterministic_eval_protocol import file_sha256
from finalize_shared_online_novel_revisit import FINAL_SCHEMA


REPORT_SCHEMA = "shared_online_novel_revisit_paired_report_v1_20260814"
ARMS = (
    "native",
    "known_direct",
    "certified",
    "certified_budget",
    "certified_graph",
)
CONTRASTS = (
    ("known_direct", "native"),
    ("certified", "native"),
    ("certified_graph", "native"),
    ("certified_graph", "certified_budget"),
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"JSON root is not an object: {path}")
    return value


def metric_row(path: Path) -> dict[str, str]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    require(len(rows) == 1, f"metric must contain one row: {path}")
    return rows[0]


def exact_mcnemar(gains: int, losses: int) -> float:
    discordant = gains + losses
    if discordant == 0:
        return 1.0
    smaller = min(gains, losses)
    tail = sum(math.comb(discordant, k) for k in range(smaller + 1)) / 2**discordant
    return min(1.0, 2.0 * tail)


def cluster_interval(
    rows: list[dict[str, Any]], treatment: str, control: str,
    *, samples: int = 100_000, seed: int = 20260814,
) -> list[float]:
    clusters: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        clusters[row["scene"]].append(
            float(row["outcomes"][treatment] - row["outcomes"][control])
        )
    names = sorted(clusters)
    require(bool(names), "no scene clusters")
    rng = np.random.default_rng(seed)
    estimates = np.empty(samples, dtype=np.float64)
    for index in range(samples):
        selected = rng.integers(0, len(names), size=len(names))
        values = [value for position in selected for value in clusters[names[position]]]
        estimates[index] = float(np.mean(values))
    return [
        100.0 * float(np.quantile(estimates, 0.025)),
        100.0 * float(np.quantile(estimates, 0.975)),
    ]


def canonical_plan(plan: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value for key, value in plan.items()
        if not key.endswith("_ms")
    }


def verify_certified_causal_branch(
    plans: dict[str, dict[str, Any]], metrics: dict[str, dict[str, str]],
) -> dict[str, Any]:
    direct = plans["certified"]
    budget = plans["certified_budget"]
    graph = plans["certified_graph"]
    budget_step_raw = metrics["certified_budget"].get(
        "certified_stagnation_intervention_step_C", ""
    )
    graph_step_raw = metrics["certified_graph"].get(
        "certified_stagnation_intervention_step_C", ""
    )
    budget_step = int(float(budget_step_raw)) if budget_step_raw else None
    graph_step = int(float(graph_step_raw)) if graph_step_raw else None
    require(budget_step == graph_step, "budget/graph treatment branches differ")

    if budget_step is None:
        require(
            metrics["certified_budget"]["certified_stagnation_intervention_C"] == "0"
            and metrics["certified_graph"]["certified_stagnation_intervention_C"] == "0",
            "no-branch episode reports an intervention",
        )
        require(
            direct["rollout_traces"]["legC"] == budget["rollout_traces"]["legC"]
            == graph["rollout_traces"]["legC"],
            "untreated certified physical rollouts differ",
        )
        return {
            "intervened": False,
            "intervention_step": None,
            "prefix_equal": True,
            "actual_graph_plans": 0,
        }

    require(
        metrics["certified_budget"]["certified_stagnation_intervention_C"] == "1"
        and metrics["certified_graph"]["certified_stagnation_intervention_C"] == "1",
        "paired treatment intervention missing",
    )
    require(
        int(float(metrics["certified"]["steps_C"])) == budget_step,
        "direct stuck boundary differs from treatment branch",
    )
    for arm, payload in (("certified_budget", budget), ("certified_graph", graph)):
        prefix = payload["rollout_traces"]["legC"][:budget_step]
        require(
            prefix == direct["rollout_traces"]["legC"],
            f"{arm}: physical prefix differs through treatment boundary",
        )
        memory_prefix = payload["memory_traces"]["legC"][:budget_step]
        require(
            memory_prefix == direct["memory_traces"]["legC"],
            f"{arm}: memory prefix differs through treatment boundary",
        )

    direct_plans = [canonical_plan(row) for row in direct["legC"]]
    for arm, payload in (("certified_budget", budget), ("certified_graph", graph)):
        before = [
            canonical_plan(row) for row in payload["legC"]
            if int(row["step"]) < budget_step
        ]
        require(before == direct_plans, f"{arm}: causal plan prefix differs")

    graph_active = sum(
        row.get("certified_graph_rescue_active") is True
        and row.get("certified_graph_reason") == "historical_subgoal"
        for row in graph["legC"]
    )
    require(graph_active > 0, "graph arm intervened without a historical subgoal")
    require(not any(
        row.get("certified_graph_rescue_active") is True
        for row in budget["legC"]
    ), "budget arm emitted a graph subgoal")
    return {
        "intervened": True,
        "intervention_step": budget_step,
        "prefix_equal": True,
        "actual_graph_plans": graph_active,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--expected-manifest-sha", required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    require(not args.out.exists(), f"report exists: {args.out}")
    manifest_path = args.benchmark_root / "manifest.json"
    require(file_sha256(manifest_path) == args.expected_manifest_sha,
            "benchmark manifest changed")
    manifest = read_object(manifest_path)
    require(manifest.get("schema_version") == FINAL_SCHEMA, "wrong benchmark schema")
    accepted = list(manifest.get("accepted") or [])
    require(len(accepted) == int(manifest["constructible_population_size"]),
            "constructible population count changed")

    records = []
    for expected_index, source in enumerate(accepted):
        require(source["selection_index"] == expected_index, "selection index changed")
        pattern = f"{expected_index:03d}_{source['scene']}_{source['episode']}"
        matches = list((args.run_root / "scenes").glob(pattern))
        require(len(matches) == 1, f"missing/duplicate episode output: {pattern}")
        root = matches[0]
        completion_path = root / "completion.json"
        require(completion_path.is_file(), f"completion receipt missing: {root}")
        completion_sha_path = root / "completion.json.sha256"
        require(completion_sha_path.is_file(), "completion hash receipt missing")
        expected_completion_sha = completion_sha_path.read_text().split()[0]
        require(file_sha256(completion_path) == expected_completion_sha,
                "completion receipt changed")
        contract = read_object(root / "episode_contract.json")
        require(contract["selection_index"] == expected_index, "episode index mismatch")
        require(set(contract["arm_order"]) == set(ARMS), "arm order is incomplete")
        metrics = {arm: metric_row(root / arm / "metric.csv") for arm in ARMS}
        plans = {
            arm: read_object(root / arm / f"{source['episode']}_plans.json")
            for arm in ARMS
        }
        trace_pairs = {
            (row["online_A_trace_sha256"], row["online_B_trace_sha256"])
            for row in metrics.values()
        }
        require(len(trace_pairs) == 1, "paired arms used different A/B traces")
        require(all(row["benchmark_sha256"] == source["benchmark_sha256"]
                    for row in metrics.values()), "paired arms used different benchmark")
        branch = verify_certified_causal_branch(plans, metrics)
        outcomes = {arm: int(metrics[arm]["reached_C"]) for arm in ARMS}
        if outcomes["certified_graph"] > outcomes["certified_budget"]:
            require(branch["actual_graph_plans"] > 0,
                    "graph gain lacks actual history-subgoal execution")
        records.append({
            "selection_index": expected_index,
            "scene": source["scene"],
            "episode": source["episode"],
            "outcomes": outcomes,
            "final_distance_m": {
                arm: float(metrics[arm]["final_dist_C"]) for arm in ARMS
            },
            "termination": {arm: metrics[arm]["termination_C"] for arm in ARMS},
            "arm_order": contract["arm_order"],
            "prefix_trace_sha256": list(next(iter(trace_pairs))),
            "certified_branch": branch,
        })

    arm_summary = {}
    for arm in ARMS:
        successes = sum(row["outcomes"][arm] for row in records)
        arm_summary[arm] = {
            "successes": successes,
            "episodes": len(records),
            "SR_C_given_frozen_online_AB": successes / len(records),
            "termination_distribution": dict(sorted(Counter(
                row["termination"][arm] for row in records
            ).items())),
        }

    contrasts = {}
    for treatment, control in CONTRASTS:
        gains = [row["selection_index"] for row in records
                 if row["outcomes"][treatment] > row["outcomes"][control]]
        losses = [row["selection_index"] for row in records
                  if row["outcomes"][treatment] < row["outcomes"][control]]
        contrasts[f"{treatment}_minus_{control}"] = {
            "gains": len(gains), "losses": len(losses),
            "gain_indices": gains, "loss_indices": losses,
            "risk_difference_pp": 100.0 * (len(gains) - len(losses)) / len(records),
            "exact_mcnemar_two_sided_p": exact_mcnemar(len(gains), len(losses)),
            "scene_cluster_bootstrap_95ci_pp": cluster_interval(
                records, treatment, control
            ),
        }

    report = {
        "schema_version": REPORT_SCHEMA,
        "benchmark_manifest_sha256": args.expected_manifest_sha,
        "source_population_size": int(manifest["source_population_size"]),
        "constructible_population_size": len(records),
        "construction_rejections": manifest["rejected"],
        "scene_clusters": len({row["scene"] for row in records}),
        "arms": arm_summary,
        "contrasts": contrasts,
        "intervention_episodes": sum(
            row["certified_branch"]["intervened"] for row in records
        ),
        "actual_graph_plan_count": sum(
            row["certified_branch"]["actual_graph_plans"] for row in records
        ),
        "all_shared_prefixes_equal": True,
        "all_treatment_prefixes_equal": True,
        "records": records,
        "scope": (
            "internal retest on an already consumed strict-v4 source pool; "
            "not scene-disjoint paper confirmation"
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(report, indent=2, sort_keys=True, allow_nan=False)
               + "\n").encode()
    args.out.write_bytes(encoded)
    args.out.with_suffix(args.out.suffix + ".sha256").write_text(
        hashlib.sha256(encoded).hexdigest() + f"  {args.out.name}\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "report": str(args.out),
        "report_sha256": hashlib.sha256(encoded).hexdigest(),
        "N": len(records),
        "arms": arm_summary,
        "contrasts": contrasts,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
