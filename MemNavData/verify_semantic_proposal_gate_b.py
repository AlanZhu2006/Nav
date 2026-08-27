#!/usr/bin/env python3
"""Independent raw-completion verifier for semantic-proposal Gate B."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


INPUT_SCHEMA = "semantic_proposal_gate_b_completion_v2_20260815"
SUMMARY_SCHEMA = "semantic_proposal_gate_b_summary_v2_20260815"
SCOPE = "consumed_closed_loop_development_never_confirmation"
ARMS = {"geometry_first", "semantic_first"}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def mcnemar(gains: int, losses: int) -> float:
    total = gains + losses
    if total == 0:
        return 1.0
    lower = min(gains, losses)
    return min(1.0, 2.0 * sum(
        math.comb(total, value) for value in range(lower + 1)
    ) / (2 ** total))


def verify(rows: list[dict], summary: dict, expected_count: int) -> dict:
    require(len(rows) == expected_count, "raw population incomplete")
    require(summary.get("schema_version") == SUMMARY_SCHEMA,
            "summary schema changed")
    indices = sorted(int(row["population_index"]) for row in rows)
    require(indices == list(range(expected_count)), "population changed")
    identities = {
        (row.get("cohort"), row.get("scene"), row.get("episode"))
        for row in rows
    }
    require(len(identities) == expected_count, "duplicate history")
    require(all(row.get("schema_version") == INPUT_SCHEMA for row in rows),
            "raw schema changed")
    require(all(row.get("scope") == SCOPE for row in rows),
            "raw scope changed")
    require(all(row.get("query_role") == "revisit" for row in rows),
            "non-Revisit query entered")
    require(all(set(row.get("outcomes", {})) == ARMS for row in rows),
            "paired arms changed")
    require(all(set(row.get("raw_outcomes", {})) == ARMS for row in rows),
            "raw paired arms changed")
    require(all(set(row.get("runtime_failure_plans", {})) == ARMS
                for row in rows), "runtime failure receipt changed")
    require(all(
        int(row["outcomes"][arm])
        == int(bool(row["raw_outcomes"][arm])
               and int(row["runtime_failure_plans"][arm]) == 0)
        for row in rows for arm in ARMS
    ), "runtime failure accounting changed")
    require(all(set(row.get("arm_order", [])) == ARMS for row in rows),
            "arm order receipt changed")
    require(all(
        set(row.get("proposal_orders", {}).get("geometry_first", []))
        == {"geometry_first"}
        and set(row.get("proposal_orders", {}).get("semantic_first", []))
        == {"dino_first_certified"}
        for row in rows
    ), "proposal-order receipt changed")
    require(all(row.get("prefix_equality") is True for row in rows),
            "prefix equality failed")
    require(all(row.get("runtime_role_visibility") == "none" for row in rows),
            "role leaked")
    geometry = sum(int(row["outcomes"]["geometry_first"]) for row in rows)
    semantic = sum(int(row["outcomes"]["semantic_first"]) for row in rows)
    raw_geometry = sum(
        int(row["raw_outcomes"]["geometry_first"]) for row in rows)
    raw_semantic = sum(
        int(row["raw_outcomes"]["semantic_first"]) for row in rows)
    gains = sum(
        not row["outcomes"]["geometry_first"]
        and row["outcomes"]["semantic_first"] for row in rows)
    losses = sum(
        row["outcomes"]["geometry_first"]
        and not row["outcomes"]["semantic_first"] for row in rows)
    paired = summary["paired_semantic_minus_geometry"]
    require(summary["successes"]["geometry_first"] == geometry,
            "geometry success differs")
    require(summary["successes"]["semantic_first"] == semantic,
            "semantic success differs")
    raw_summary = summary[
        "raw_physical_successes_before_runtime_failure_penalty"]
    require(raw_summary["geometry_first"] == raw_geometry,
            "raw geometry success differs")
    require(raw_summary["semantic_first"] == raw_semantic,
            "raw semantic success differs")
    require(paired["gains"] == gains and paired["losses"] == losses,
            "paired count differs")
    require(math.isclose(float(paired["exact_mcnemar_p"]),
                         mcnemar(gains, losses), abs_tol=1e-15),
            "McNemar differs")
    passed = gains > losses
    require(summary["gate_b_passed"] is passed, "Gate-B decision differs")
    return {
        "schema_version": "semantic_proposal_gate_b_independent_verification_v1_20260815",
        "verified": True,
        "expected_count": expected_count,
        "successes": {"geometry_first": geometry,
                      "semantic_first": semantic},
        "paired": {"gains": gains, "losses": losses,
                   "exact_mcnemar_p": mcnemar(gains, losses)},
        "gate_b_passed": passed,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--expected-count", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    require(not args.output.exists(), "verification output already exists")
    paths = sorted(args.input_root.glob("*/completion.json"))
    rows = [json.loads(path.read_text()) for path in paths]
    summary = json.loads(args.summary.read_text())
    result = verify(rows, summary, args.expected_count)
    result.update({
        "summary_sha256": sha256_file(args.summary),
        "record_sha256": {path.name + "@" + path.parent.name: sha256_file(path)
                          for path in paths},
    })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
