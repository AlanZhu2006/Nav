#!/usr/bin/env python3
"""Independent raw-record verification for the proposal audit.

This module deliberately does not import the formal summarizer.  It recomputes
the acceptance pairs and selection diagnostics from immutable episode records,
then checks the published summary before sealing a verification receipt.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
from typing import Any


INPUT_SCHEMA = "certified_proposal_counterfactual_episode_v1_20260815"
SUMMARY_SCHEMA = "certified_proposal_counterfactual_summary_v1_20260815"
OUTPUT_SCHEMA = (
    "independent_certified_proposal_counterfactual_verification_v1_20260815"
)


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


def verify_records(
    rows: list[dict[str, Any]],
    summary: dict[str, Any],
    expected_count: int,
) -> dict[str, Any]:
    require(len(rows) == expected_count, "raw population is incomplete")
    require(summary.get("schema_version") == SUMMARY_SCHEMA,
            "unexpected summary schema")
    identities = [
        (row.get("benchmark_root"), row.get("population_index"))
        for row in rows
    ]
    require(len(identities) == len(set(identities)), "duplicate raw identity")

    pairs_top1: Counter[tuple[bool, bool]] = Counter()
    pairs_ordered: Counter[tuple[bool, bool]] = Counter()
    ranks: Counter[int] = Counter()
    attempts: list[int] = []
    geometry_accept = top1_accept = ordered_accept = anchor_changed = 0
    for row in rows:
        require(row.get("schema_version") == INPUT_SCHEMA,
                "unexpected raw schema")
        require(row.get("is_closed_loop_evaluation") is False,
                "raw record was mislabeled as closed-loop")
        require(row.get("method_action_unchanged") is True,
                "counterfactual changed the deployed action")
        require(row.get("query_role_selected_for_analysis") == "revisit",
                "non-Revisit record entered the audit")
        top1 = row.get("counterfactual_dino_top1")
        ordered = row.get("counterfactual_dino_first_certified")
        require(isinstance(top1, dict) and isinstance(ordered, dict),
                "counterfactual payload is missing")
        require(top1.get("action_authority") is False,
                "DINO top-1 gained action authority")
        require(ordered.get("action_authority") is False,
                "DINO order gained action authority")

        geometry_value = bool(row.get("geometry_accepted"))
        top1_value = bool(top1.get("accepted"))
        ordered_value = bool(ordered.get("accepted"))
        geometry_accept += int(geometry_value)
        top1_accept += int(top1_value)
        ordered_accept += int(ordered_value)
        pairs_top1[(geometry_value, top1_value)] += 1
        pairs_ordered[(geometry_value, ordered_value)] += 1
        attempt_count = int(ordered.get("attempt_count"))
        require(attempt_count >= 1, "ordered audit attempted no hypothesis")
        attempts.append(attempt_count)
        if ordered_value:
            rank = int(ordered.get("selected_dino_rank"))
            require(rank >= 1, "accepted DINO rank is invalid")
            ranks[rank] += 1
        if row.get("dino_top1_anchor") != row.get("geometry_selected_anchor"):
            anchor_changed += 1

    top1_gain = pairs_top1[(False, True)]
    top1_loss = pairs_top1[(True, False)]
    ordered_gain = pairs_ordered[(False, True)]
    ordered_loss = pairs_ordered[(True, False)]
    recomputed = {
        "proposal_acceptance": {
            "deployed_geometry": geometry_accept,
            "dino_top1_same_certificate": top1_accept,
            "dino_order_first_certificate": ordered_accept,
            "denominator": expected_count,
        },
        "paired_acceptance": {
            "dino_top1_minus_geometry": {
                "gain_loss": [top1_gain, top1_loss],
                "exact_mcnemar_p": exact_mcnemar(top1_gain, top1_loss),
            },
            "dino_order_minus_geometry": {
                "gain_loss": [ordered_gain, ordered_loss],
                "exact_mcnemar_p": exact_mcnemar(
                    ordered_gain, ordered_loss),
            },
        },
        "selection": {
            "geometry_changed_dino_top1": anchor_changed,
            "same_anchor": expected_count - anchor_changed,
            "accepted_dino_rank_histogram": {
                str(key): ranks[key] for key in sorted(ranks)
            },
            "ordered_attempts_mean": sum(attempts) / len(attempts),
            "ordered_attempts_max": max(attempts),
        },
    }

    require(summary.get("proposal_acceptance")
            == recomputed["proposal_acceptance"],
            "proposal acceptance differs from raw records")
    require(summary.get("selection") == recomputed["selection"],
            "selection diagnostics differ from raw records")
    observed_pairs = summary.get("paired_acceptance")
    require(isinstance(observed_pairs, dict), "paired summary is missing")
    for name, expected in recomputed["paired_acceptance"].items():
        observed = observed_pairs.get(name)
        require(isinstance(observed, dict), f"paired contrast missing: {name}")
        require(observed.get("gain_loss") == expected["gain_loss"],
                f"gain/loss differs: {name}")
        require(math.isclose(
            float(observed.get("exact_mcnemar_p")),
            float(expected["exact_mcnemar_p"]),
            rel_tol=0.0,
            abs_tol=1e-15,
        ), f"McNemar p differs: {name}")
    return recomputed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--expected-count", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    require(args.expected_count > 0, "expected count must be positive")
    require(not args.output.exists(), "verification output already exists")
    record_paths = sorted(args.input_root.glob("*.json"))
    rows = [json.loads(path.read_text()) for path in record_paths]
    summary = json.loads(args.summary.read_text())
    recomputed = verify_records(rows, summary, args.expected_count)
    output = {
        "schema_version": OUTPUT_SCHEMA,
        "verified": True,
        "expected_count": args.expected_count,
        "summary_path": str(args.summary.resolve()),
        "summary_sha256": sha256_file(args.summary),
        "record_sha256": {
            path.name: sha256_file(path) for path in record_paths
        },
        "recomputed": recomputed,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    print(json.dumps({"verified": True, **recomputed["proposal_acceptance"]},
                     sort_keys=True))


if __name__ == "__main__":
    main()
