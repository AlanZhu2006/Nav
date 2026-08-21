#!/usr/bin/env python3
"""Summarize the frozen consumed closed-loop proposal-order comparison."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
from typing import Any


INPUT_SCHEMA = "semantic_proposal_gate_b_completion_v2_20260815"
OUTPUT_SCHEMA = "semantic_proposal_gate_b_summary_v2_20260815"
SCOPE = "consumed_closed_loop_development_never_confirmation"
ARMS = ("geometry_first", "semantic_first")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def exact_mcnemar(gains: int, losses: int) -> float:
    discordant = gains + losses
    if discordant == 0:
        return 1.0
    tail = sum(
        math.comb(discordant, value)
        for value in range(min(gains, losses) + 1)
    ) / (2 ** discordant)
    return min(1.0, 2.0 * tail)


def summarize(rows: list[dict[str, Any]], expected_count: int) -> dict[str, Any]:
    require(len(rows) == expected_count, "Gate-B population incomplete")
    rows = sorted(rows, key=lambda row: int(row["population_index"]))
    require(
        [int(row["population_index"]) for row in rows]
        == list(range(expected_count)),
        "population indices changed",
    )
    identities = [(row["cohort"], row["scene"], row["episode"])
                  for row in rows]
    require(len(identities) == len(set(identities)), "duplicate history")
    first_arm_counts = Counter()
    for row in rows:
        require(row.get("schema_version") == INPUT_SCHEMA,
                "input schema changed")
        require(row.get("scope") == SCOPE, "scope changed")
        require(row.get("query_role") == "revisit", "non-Revisit entered")
        require(row.get("runtime_role_visibility") == "none", "role leaked")
        require(row.get("prefix_equality") is True, "prefixes differ")
        require(set(row.get("outcomes", {})) == set(ARMS), "arms changed")
        require(set(row.get("raw_outcomes", {})) == set(ARMS),
                "raw arms changed")
        require(all(row["outcomes"][arm] in (0, 1) for arm in ARMS),
                "outcome is not binary")
        require(all(row["raw_outcomes"][arm] in (0, 1) for arm in ARMS),
                "raw outcome is not binary")
        require(set(row.get("runtime_failure_plans", {})) == set(ARMS),
                "runtime failure receipt changed")
        for arm in ARMS:
            expected = int(
                bool(row["raw_outcomes"][arm])
                and int(row["runtime_failure_plans"][arm]) == 0
            )
            require(row["outcomes"][arm] == expected,
                    f"runtime failure accounting changed: {arm}")
        order = row.get("arm_order")
        require(isinstance(order, list) and set(order) == set(ARMS),
                "arm order changed")
        first_arm_counts[order[0]] += 1
        expected_orders = {
            "geometry_first": "geometry_first",
            "semantic_first": "dino_first_certified",
        }
        for arm in ARMS:
            observed = row.get("proposal_orders", {}).get(arm)
            require(isinstance(observed, list) and observed,
                    f"missing proposal-order receipt: {arm}")
            require(set(observed) == {expected_orders[arm]},
                    f"wrong proposal order: {arm}")
    require(set(first_arm_counts) == set(ARMS), "one arm never ran first")
    require(max(first_arm_counts.values()) - min(first_arm_counts.values()) <= 1,
            "arm order is not balanced")

    geometry = [int(row["outcomes"]["geometry_first"]) for row in rows]
    semantic = [int(row["outcomes"]["semantic_first"]) for row in rows]
    raw_geometry = [int(row["raw_outcomes"]["geometry_first"])
                    for row in rows]
    raw_semantic = [int(row["raw_outcomes"]["semantic_first"])
                    for row in rows]
    gains = sum(right and not left for left, right in zip(geometry, semantic))
    losses = sum(left and not right for left, right in zip(geometry, semantic))
    ties_success = sum(left and right for left, right in zip(geometry, semantic))
    ties_failure = sum(not left and not right
                       for left, right in zip(geometry, semantic))
    runtime_failures = {
        arm: sum(int(row["runtime_failure_plans"][arm]) for row in rows)
        for arm in ARMS
    }
    first_anchor_changed = sum(
        bool(row["selected_anchors"]["geometry_first"])
        and bool(row["selected_anchors"]["semantic_first"])
        and row["selected_anchors"]["geometry_first"][0]
        != row["selected_anchors"]["semantic_first"][0]
        for row in rows
    )
    gate_passed = gains > losses
    return {
        "schema_version": OUTPUT_SCHEMA,
        "scope": SCOPE,
        "is_confirmation_evidence": False,
        "population": {
            "histories": expected_count,
            "scenes": len({row["scene"] for row in rows}),
            "cohorts": dict(sorted(Counter(
                row["cohort"] for row in rows).items())),
        },
        "successes": {
            "geometry_first": sum(geometry),
            "semantic_first": sum(semantic),
            "denominator": expected_count,
        },
        "raw_physical_successes_before_runtime_failure_penalty": {
            "geometry_first": sum(raw_geometry),
            "semantic_first": sum(raw_semantic),
            "denominator": expected_count,
        },
        "paired_semantic_minus_geometry": {
            "gains": gains,
            "losses": losses,
            "ties_success": ties_success,
            "ties_failure": ties_failure,
            "exact_mcnemar_p": exact_mcnemar(gains, losses),
        },
        "execution_audit": {
            "first_arm_counts": dict(sorted(first_arm_counts.items())),
            "runtime_failure_plans": runtime_failures,
            "first_selected_anchor_changed": first_anchor_changed,
            "all_prefixes_equal": True,
            "runtime_role_visibility": "none",
        },
        "frozen_gate_b_rule": "promote_only_if_paired_gains_strictly_exceed_losses",
        "gate_b_passed": gate_passed,
        "next_action": (
            "freeze_semantic_first_for_fresh_scene_disjoint_confirmation"
            if gate_passed else
            "retain_geometry_first_cec"
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--expected-count", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    require(args.expected_count > 0, "expected count must be positive")
    require(not args.out.exists(), "summary output already exists")
    paths = sorted(args.input_root.glob("*/completion.json"))
    rows = [json.loads(path.read_text()) for path in paths]
    result = summarize(rows, args.expected_count)
    result["records"] = [
        {"path": str(path.resolve()), "sha256": sha256_file(path)}
        for path in paths
    ]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    print(json.dumps({
        **result["successes"],
        **result["paired_semantic_minus_geometry"],
        "gate_b_passed": result["gate_b_passed"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
