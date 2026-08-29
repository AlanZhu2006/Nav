#!/usr/bin/env python3
"""Audit the sealed HM3D Table-2 Leg-3 construction receipts.

This is a construction-only audit.  It never opens a navigation outcome and
does not change the frozen Novel/Revisit thresholds.  Its main purpose is to
separate failure to *sample* a candidate from failure of the requested binary
role pair to exist after the complete factual A+B history.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
from statistics import median
import tempfile
from typing import Any, Iterable, Mapping


SCHEMA = "hm3d_table2_leg3_constructibility_audit_v1_20260830"
FRAGMENT_SCHEMA = "hm3d_table2_leg3_mixed_role_fragment_v1_20260829"
POPULATION_SCHEMA = "hm3d_table2_leg3_mixed_role_population_v1_20260829"
VERIFICATION_SCHEMA = (
    "hm3d_table2_leg3_mixed_role_construction_verification_v1_20260829"
)
STRATA = ("front", "side", "rear")
NOVEL_ATTRITION = "no_new_unsupported_novel_after_combined_AB"
REVISIT_ATTRITION = "no_new_standard_revisit_after_combined_AB"

# These are mutually exclusive terminal counters in sample_natural_novel.
# The aggregate floor_or_clearance_rejects and geodesic_rejects fields are
# intentionally omitted because they duplicate counters below.
TERMINAL_STAGES = (
    ("duplicate_position", "duplicate_position_rejects"),
    ("non_navigable", "non_navigable_rejects"),
    ("floor_mismatch", "floor_mismatch_rejects"),
    ("insufficient_clearance", "clearance_rejects"),
    ("consumed_identity_separation", "candidate_separation_rejects"),
    ("unreachable", "unreachable_rejects"),
    ("outside_2_9m_band", "a_to_b_outside_band_rejects"),
    ("wrong_direction_stratum", "direction_stratum_rejects"),
    ("paired_unreachable", "paired_unreachable_rejects"),
    ("paired_too_close", "paired_below_minimum_rejects"),
    ("paired_too_far", "paired_above_maximum_rejects"),
    ("historically_supported", "support_rejects"),
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    require(not path.exists(), f"refusing to overwrite audit: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def read_object(path: Path) -> dict[str, Any]:
    require(path.is_file() and not path.is_symlink(), f"missing input: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"expected JSON object: {path}")
    return value


def completion_paths(root: Path) -> list[Path]:
    require(root.is_dir(), f"completion root is not a directory: {root}")
    paths = sorted(root.rglob("completion.json"))
    if not paths:
        paths = sorted(root.glob("completion_*.json"))
    require(bool(paths), f"no completion receipts under {root}")
    return paths


def distribution(values: list[int]) -> dict[str, int | float]:
    require(bool(values), "empty distribution")
    return {
        "n": len(values),
        "minimum": min(values),
        "median": median(values),
        "maximum": max(values),
    }


def _sum_diagnostics(
    diagnostics: Iterable[Mapping[str, Any]],
) -> Counter[str]:
    result: Counter[str] = Counter()
    for row in diagnostics:
        result.update({str(key): int(value) for key, value in row.items()
                       if isinstance(value, int) and not isinstance(value, bool)})
    return result


def _funnel(total: Counter[str]) -> tuple[list[dict[str, Any]], int]:
    incoming = int(total["attempts"])
    rows = []
    for name, counter in TERMINAL_STAGES:
        rejected = int(total[counter])
        require(rejected <= incoming,
                f"{name}: rejection count exceeds incoming candidates")
        rows.append({
            "stage": name,
            "incoming": incoming,
            "rejected": rejected,
            "surviving": incoming - rejected,
            "fraction_of_all_attempts": (
                rejected / total["attempts"] if total["attempts"] else 0.0
            ),
            "fraction_of_stage_input": (
                rejected / incoming if incoming else 0.0
            ),
        })
        incoming -= rejected
    return rows, incoming


def audit(
    completions: list[tuple[Path, dict[str, Any]]],
    population: dict[str, Any],
    verification: dict[str, Any],
    *,
    source_uri: str,
) -> dict[str, Any]:
    require(population.get("schema_version") == POPULATION_SCHEMA,
            "population schema changed")
    require(verification.get("schema_version") == VERIFICATION_SCHEMA,
            "construction verifier schema changed")
    require(verification.get("verified") is True,
            "construction verifier did not pass")
    require(population.get("navigation_outcomes_generated") is False
            and population.get("query_outcomes_read_for_selection") is False
            and population.get("old_goal_C_outcomes_read_for_construction")
            is False, "population construction consumed a query outcome")

    expected_inputs = {
        int(row["population_index"]): row
        for row in population["construction_inputs"]
    }
    require(len(expected_inputs) == 22, "population is not the sealed 22-prefix set")
    indexed: dict[int, tuple[Path, dict[str, Any]]] = {}
    for path, row in completions:
        require(row.get("schema_version") == FRAGMENT_SCHEMA
                and row.get("status") == "complete",
                f"incomplete fragment: {path}")
        index = int(row["population_index"])
        require(index not in indexed, f"duplicate completion index {index}")
        expected = expected_inputs.get(index)
        require(expected is not None, f"unexpected completion index {index}")
        require(sha256_file(path) == expected["completion_sha256"],
                f"completion {index} hash differs from sealed population")
        require(bool(row["eligible"]) == bool(expected["eligible"]),
                f"completion {index} eligibility changed")
        require(row.get("leg3_query_policy_outcomes_read") is False
                and row.get("old_goal_C_navigation_outcomes_read") is False,
                f"completion {index} consumed a query outcome")
        indexed[index] = (path, row)
    require(set(indexed) == set(range(22)), "completion universe is incomplete")
    rows = [indexed[index][1] for index in range(22)]

    attrition = Counter(
        "accepted" if row["eligible"] else str(row["attrition_reason"])
        for row in rows
    )
    require(attrition == Counter({
        "accepted": 8,
        NOVEL_ATTRITION: 13,
        REVISIT_ATTRITION: 1,
    }), "sealed attrition counts changed")
    require(int(population["leg3_constructible_histories"]) == 8
            and int(population["leg3_scene_clusters"]) == 6,
            "population constructibility counts changed")
    require(population["formal_policy_evaluation_authorized"] is False
            and population["power_gate"]["target_met"] is False,
            "failed power gate unexpectedly authorized policy evaluation")

    accepted = [row for row in rows if row["eligible"]]
    novel_failures = [row for row in rows
                      if row.get("attrition_reason") == NOVEL_ATTRITION]
    revisit_failures = [row for row in rows
                        if row.get("attrition_reason") == REVISIT_ATTRITION]
    require(all(set(row["natural_diagnostics"]) == set(STRATA)
                for row in novel_failures),
            "a Novel attrition did not exhaust all three strata")

    failed_diagnostics = [
        row["natural_diagnostics"][stratum]
        for row in novel_failures for stratum in STRATA
    ]
    totals = _sum_diagnostics(failed_diagnostics)
    funnel, survivors = _funnel(totals)
    require(int(totals["attempts"]) == 13 * len(STRATA) * 5000,
            "Novel attrition did not exhaust 5,000 attempts per stratum")
    require(survivors == 0,
            "terminal rejection counters do not account for every attempt")
    require(int(totals["support_rejects"]) > 0,
            "no candidate reached the support gate")

    by_stratum = {}
    for stratum in STRATA:
        selected = [row["natural_diagnostics"][stratum]
                    for row in novel_failures]
        subtotal = _sum_diagnostics(selected)
        subfunnel, sub_survivors = _funnel(subtotal)
        require(sub_survivors == 0,
                f"{stratum}: terminal counters do not close")
        reached_support = int(subtotal["support_rejects"])
        by_stratum[stratum] = {
            "failed_history_searches": len(selected),
            "attempts": int(subtotal["attempts"]),
            "candidates_reaching_final_support_check": reached_support,
            "rejected_as_historically_supported": reached_support,
            "unsupported_candidates_found": 0,
            "stage_funnel": subfunnel,
        }

    accepted_modes = Counter()
    for row in accepted:
        selected = row["natural_diagnostics"][row["selected_stratum"]]
        if int(selected["uniform_random_attempts"]) == 0:
            accepted_modes["deterministic_local_polar_grid"] += 1
        else:
            accepted_modes["deterministic_seeded_uniform_fallback"] += 1

    revisit = revisit_failures[0]["revisit_diagnostics"]
    require(int(revisit["fully_scored"]) == int(revisit["support_rejects"]),
            "Revisit attrition contains an unaccounted fully scored candidate")

    source_digest = hashlib.sha256("".join(
        f"{sha256_file(path)}  {int(row['population_index']):03d}\n"
        for path, row in sorted(completions,
                                key=lambda item: int(item[1]["population_index"]))
    ).encode()).hexdigest()
    return {
        "schema_version": SCHEMA,
        "status": "complete_construction_only_audit",
        "source_uri": source_uri,
        "source_completion_count": len(rows),
        "source_completion_digest_sha256": source_digest,
        "source_protocol_sha256": rows[0]["protocol_sha256"],
        "query_policy_outcomes_read": False,
        "threshold_or_method_changed": False,
        "sealed_population": {
            "factual_AB_prefixes": int(population["factual_AB_successful_prefixes"]),
            "factual_AB_scene_clusters": int(population["factual_AB_scene_clusters"]),
            "constructible_histories": len(accepted),
            "constructible_scene_clusters": len({row["scene"] for row in accepted}),
            "direction_strata": dict(population["power_gate"]["direction_strata"]),
            "formal_policy_evaluation_authorized": False,
            "attrition": dict(sorted(attrition.items())),
            "combined_prefix_steps": {
                "accepted": distribution([
                    int(row["combined_prefix_steps"]) for row in accepted]),
                "novel_attrition": distribution([
                    int(row["combined_prefix_steps"]) for row in novel_failures]),
            },
            "accepted_novel_proposal_modes": dict(sorted(accepted_modes.items())),
            "selected_revisit_source_segments": dict(sorted(Counter(
                str(row["selected_revisit_segment"]) for row in accepted
            ).items())),
        },
        "novel_attrition_audit": {
            "histories": len(novel_failures),
            "stratum_searches": len(failed_diagnostics),
            "attempts": int(totals["attempts"]),
            "deterministic_local_attempts": int(
                totals["deterministic_local_attempts"]),
            "deterministic_seeded_uniform_attempts": int(
                totals["uniform_random_attempts"]),
            "candidates_reaching_final_support_check": int(
                totals["support_rejects"]),
            "rejected_as_historically_supported": int(
                totals["support_rejects"]),
            "unsupported_candidates_found": 0,
            "stage_funnel": funnel,
            "by_direction_stratum": by_stratum,
        },
        "revisit_attrition_audit": {
            "histories": 1,
            "grid_attempts": int(revisit["grid_attempts"]),
            "fully_scored": int(revisit["fully_scored"]),
            "support_band_rejects": int(revisit["support_rejects"]),
            "interpretation_boundary": (
                "The receipt does not separate candidates below 0.55 from "
                "candidates above 0.90, so no finer Revisit cause is inferred."
            ),
        },
        "audit_conclusion": {
            "simple_attempt_budget_failure": False,
            "side_direction_sampler_failure": False,
            "binary_role_pair_constructibility_failure": True,
            "basis": (
                "Every Novel attrition exhausted all three 5,000-attempt "
                "strata. Thousands of candidates passed every preceding "
                "geometry, direction, distance, and pairing gate; all were "
                "rejected only because they retained >=0.10 covisibility "
                "with the complete factual A+B history."
            ),
            "scientific_scope": (
                "This explains why the prospective population gate failed; "
                "it is not a navigation-policy outcome and does not estimate SR."
            ),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--completions", type=Path, required=True)
    parser.add_argument("--population-receipt", type=Path, required=True)
    parser.add_argument("--construction-verification", type=Path, required=True)
    parser.add_argument("--source-uri", required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = completion_paths(args.completions)
    result = audit(
        [(path, read_object(path)) for path in paths],
        read_object(args.population_receipt),
        read_object(args.construction_verification),
        source_uri=args.source_uri,
    )
    result["source_files"] = {
        "population_receipt_sha256": sha256_file(args.population_receipt),
        "construction_verification_sha256": sha256_file(
            args.construction_verification),
    }
    atomic_json(args.out, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
