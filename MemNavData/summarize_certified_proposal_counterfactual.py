#!/usr/bin/env python3
"""Summarize read-only semantic-proposal versus geometry-proposal audits."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any


INPUT_SCHEMA = "certified_proposal_counterfactual_episode_v1_20260815"
OUTPUT_SCHEMA = "certified_proposal_counterfactual_summary_v1_20260815"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def exact_mcnemar(gains: int, losses: int) -> float:
    discordant = gains + losses
    if discordant == 0:
        return 1.0
    tail = sum(
        math.comb(discordant, value)
        for value in range(0, min(gains, losses) + 1)
    ) / (2 ** discordant)
    return min(1.0, 2.0 * tail)


def summarize(rows: list[dict[str, Any]], expected_count: int) -> dict[str, Any]:
    require(len(rows) == expected_count, "counterfactual population incomplete")
    identities = [(row["benchmark_root"], row["population_index"])
                  for row in rows]
    require(len(identities) == len(set(identities)), "duplicate population row")
    for row in rows:
        require(row.get("schema_version") == INPUT_SCHEMA, "input schema changed")
        require(row.get("method_action_unchanged") is True, "audit changed action")
        require(row.get("is_closed_loop_evaluation") is False,
                "input mislabeled as closed-loop")
        require(row.get("query_role_selected_for_analysis") == "revisit",
                "unexpected query role")
        require(row["counterfactual_dino_top1"]["action_authority"] is False,
                "top-1 audit gained authority")
        require(
            row["counterfactual_dino_first_certified"]["action_authority"]
            is False,
            "ordered audit gained authority",
        )

    geometry_accept = sum(bool(row["geometry_accepted"]) for row in rows)
    top1_accept = sum(bool(row["counterfactual_dino_top1"]["accepted"])
                      for row in rows)
    ordered_accept = sum(
        bool(row["counterfactual_dino_first_certified"]["accepted"])
        for row in rows
    )
    geometry_vs_top1 = Counter()
    geometry_vs_ordered = Counter()
    for row in rows:
        geometry = bool(row["geometry_accepted"])
        top1 = bool(row["counterfactual_dino_top1"]["accepted"])
        ordered = bool(row["counterfactual_dino_first_certified"]["accepted"])
        geometry_vs_top1[(geometry, top1)] += 1
        geometry_vs_ordered[(geometry, ordered)] += 1

    top1_gains = geometry_vs_top1[(False, True)]
    top1_losses = geometry_vs_top1[(True, False)]
    ordered_gains = geometry_vs_ordered[(False, True)]
    ordered_losses = geometry_vs_ordered[(True, False)]
    ranks = Counter(
        row["counterfactual_dino_first_certified"].get("selected_dino_rank")
        for row in rows
        if row["counterfactual_dino_first_certified"]["accepted"]
    )
    attempt_counts = [
        int(row["counterfactual_dino_first_certified"]["attempt_count"])
        for row in rows
    ]
    anchor_changed = [
        row for row in rows
        if row["dino_top1_anchor"] != row["geometry_selected_anchor"]
    ]

    return {
        "schema_version": OUTPUT_SCHEMA,
        "scope": "consumed_posthoc_method_development_diagnostic",
        "is_confirmation_evidence": False,
        "closed_loop_required_before_method_claim": True,
        "population": {
            "revisit_histories": len(rows),
            "scenes": len({row["scene"] for row in rows}),
            "benchmark_roots": sorted({row["benchmark_root"] for row in rows}),
        },
        "proposal_acceptance": {
            "deployed_geometry": geometry_accept,
            "dino_top1_same_certificate": top1_accept,
            "dino_order_first_certificate": ordered_accept,
            "denominator": len(rows),
        },
        "paired_acceptance": {
            "dino_top1_minus_geometry": {
                "gain_loss": [top1_gains, top1_losses],
                "exact_mcnemar_p": exact_mcnemar(top1_gains, top1_losses),
            },
            "dino_order_minus_geometry": {
                "gain_loss": [ordered_gains, ordered_losses],
                "exact_mcnemar_p": exact_mcnemar(
                    ordered_gains, ordered_losses),
            },
        },
        "selection": {
            "geometry_changed_dino_top1": len(anchor_changed),
            "same_anchor": len(rows) - len(anchor_changed),
            "accepted_dino_rank_histogram": {
                str(key): value for key, value in sorted(
                    ranks.items(), key=lambda item: int(item[0]))
            },
            "ordered_attempts_mean": (
                sum(attempt_counts) / len(attempt_counts) if attempt_counts else 0.0
            ),
            "ordered_attempts_max": max(attempt_counts, default=0),
        },
        "records": [
            {
                "scene": row["scene"],
                "episode": row["episode"],
                "benchmark_root": row["benchmark_root"],
                "population_index": row["population_index"],
                "geometry_anchor": row["geometry_selected_anchor"],
                "geometry_accepted": row["geometry_accepted"],
                "dino_top1_anchor": row["dino_top1_anchor"],
                "dino_top1_accepted": row[
                    "counterfactual_dino_top1"]["accepted"],
                "dino_first_certified_anchor": row[
                    "counterfactual_dino_first_certified"]["selected_anchor"],
                "dino_first_certified_rank": row[
                    "counterfactual_dino_first_certified"]["selected_dino_rank"],
                "dino_first_certified_accepted": row[
                    "counterfactual_dino_first_certified"]["accepted"],
            }
            for row in rows
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--expected-count", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    require(args.expected_count > 0, "expected count must be positive")
    require(not args.out.exists(), "summary output already exists")
    paths = sorted(args.input_root.glob("*.json"))
    rows = [json.loads(path.read_text()) for path in paths]
    result = summarize(rows, args.expected_count)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    print(json.dumps(result["proposal_acceptance"], sort_keys=True))


if __name__ == "__main__":
    main()
