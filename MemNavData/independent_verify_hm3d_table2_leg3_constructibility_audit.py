#!/usr/bin/env python3
"""Independent recount of the HM3D Table-2 constructibility audit.

This verifier deliberately does not import the production audit module.  It
reopens the sealed fragment receipts, reconstructs the terminal rejection
funnel, and checks the published audit field by field.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping


SCHEMA = "independent_hm3d_table2_leg3_constructibility_audit_v1_20260830"
AUDIT_SCHEMA = "hm3d_table2_leg3_constructibility_audit_v1_20260830"
NOVEL_ATTRITION = "no_new_unsupported_novel_after_combined_AB"
REVISIT_ATTRITION = "no_new_standard_revisit_after_combined_AB"
STRATA = ("front", "side", "rear")
TERMINAL_COUNTERS = (
    "duplicate_position_rejects",
    "non_navigable_rejects",
    "floor_mismatch_rejects",
    "clearance_rejects",
    "candidate_separation_rejects",
    "unreachable_rejects",
    "a_to_b_outside_band_rejects",
    "direction_stratum_rejects",
    "paired_unreachable_rejects",
    "paired_below_minimum_rejects",
    "paired_above_maximum_rejects",
    "support_rejects",
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


def read_object(path: Path) -> dict[str, Any]:
    require(path.is_file() and not path.is_symlink(), f"missing input: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"expected object: {path}")
    return value


def completion_paths(root: Path) -> list[Path]:
    paths = sorted(root.rglob("completion.json"))
    if not paths:
        paths = sorted(root.glob("completion_*.json"))
    require(bool(paths), "completion receipts are missing")
    return paths


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    require(not path.exists(), f"refusing to overwrite verifier: {path}")
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


def verify(
    paths: list[Path],
    population: dict[str, Any],
    construction: dict[str, Any],
    audit: dict[str, Any],
) -> dict[str, Any]:
    require(audit.get("schema_version") == AUDIT_SCHEMA,
            "audit schema changed")
    require(audit.get("query_policy_outcomes_read") is False,
            "audit claims to have read a query outcome")
    require(construction.get("verified") is True,
            "upstream construction verification failed")
    expected = {
        int(row["population_index"]): row
        for row in population["construction_inputs"]
    }
    require(len(expected) == 22, "sealed population is not 22 prefixes")

    rows: dict[int, dict[str, Any]] = {}
    completion_hashes: dict[int, str] = {}
    for path in paths:
        row = read_object(path)
        index = int(row["population_index"])
        require(index not in rows, f"duplicate completion index {index}")
        require(index in expected, f"unexpected completion index {index}")
        digest = sha256_file(path)
        require(digest == expected[index]["completion_sha256"],
                f"completion {index} hash mismatch")
        require(row.get("leg3_query_policy_outcomes_read") is False
                and row.get("old_goal_C_navigation_outcomes_read") is False,
                f"completion {index} read a query outcome")
        rows[index] = row
        completion_hashes[index] = digest
    require(set(rows) == set(range(22)), "completion universe is incomplete")
    ordered = [rows[index] for index in range(22)]

    attrition = Counter(
        "accepted" if row["eligible"] else row["attrition_reason"]
        for row in ordered
    )
    require(attrition == Counter({
        "accepted": 8,
        NOVEL_ATTRITION: 13,
        REVISIT_ATTRITION: 1,
    }), "raw attrition changed")

    novel_failures = [row for row in ordered
                      if row.get("attrition_reason") == NOVEL_ATTRITION]
    totals = Counter()
    per_stratum: dict[str, Counter[str]] = {
        name: Counter() for name in STRATA
    }
    for row in novel_failures:
        require(set(row["natural_diagnostics"]) == set(STRATA),
                "Novel failure did not exhaust every stratum")
        for stratum in STRATA:
            diag = row["natural_diagnostics"][stratum]
            for key, value in diag.items():
                if isinstance(value, int) and not isinstance(value, bool):
                    totals[key] += value
                    per_stratum[stratum][key] += value
    require(totals["attempts"] == 195_000,
            "raw attempt count is not 13 x 3 x 5,000")
    require(sum(totals[key] for key in TERMINAL_COUNTERS)
            == totals["attempts"], "terminal counters do not close")
    require(totals["support_rejects"] == 6_660,
            "final support-gate count changed")
    expected_support = {"front": 881, "side": 2_847, "rear": 2_932}
    for stratum in STRATA:
        require(per_stratum[stratum]["attempts"] == 65_000,
                f"{stratum}: attempt count changed")
        require(sum(per_stratum[stratum][key] for key in TERMINAL_COUNTERS)
                == per_stratum[stratum]["attempts"],
                f"{stratum}: terminal counters do not close")
        require(per_stratum[stratum]["support_rejects"]
                == expected_support[stratum],
                f"{stratum}: support-gate count changed")

    accepted = [row for row in ordered if row["eligible"]]
    revisit_failure = next(row for row in ordered
                           if row.get("attrition_reason") == REVISIT_ATTRITION)
    revisit = revisit_failure["revisit_diagnostics"]
    observed = {
        "constructible_histories": len(accepted),
        "constructible_scene_clusters": len({row["scene"] for row in accepted}),
        "attrition": dict(sorted(attrition.items())),
        "novel_failure_histories": len(novel_failures),
        "novel_stratum_searches": 3 * len(novel_failures),
        "novel_attempts": int(totals["attempts"]),
        "deterministic_local_attempts": int(
            totals["deterministic_local_attempts"]),
        "uniform_attempts": int(totals["uniform_random_attempts"]),
        "final_support_checks": int(totals["support_rejects"]),
        "support_checks_by_stratum": expected_support,
        "revisit_grid_attempts": int(revisit["grid_attempts"]),
        "revisit_fully_scored": int(revisit["fully_scored"]),
        "revisit_support_rejects": int(revisit["support_rejects"]),
    }
    reported_population = audit["sealed_population"]
    reported_novel = audit["novel_attrition_audit"]
    reported_revisit = audit["revisit_attrition_audit"]
    checks = {
        "constructible_histories": reported_population["constructible_histories"],
        "constructible_scene_clusters": reported_population[
            "constructible_scene_clusters"],
        "attrition": reported_population["attrition"],
        "novel_failure_histories": reported_novel["histories"],
        "novel_stratum_searches": reported_novel["stratum_searches"],
        "novel_attempts": reported_novel["attempts"],
        "deterministic_local_attempts": reported_novel[
            "deterministic_local_attempts"],
        "uniform_attempts": reported_novel[
            "deterministic_seeded_uniform_attempts"],
        "final_support_checks": reported_novel[
            "candidates_reaching_final_support_check"],
        "support_checks_by_stratum": {
            name: reported_novel["by_direction_stratum"][name][
                "candidates_reaching_final_support_check"]
            for name in STRATA
        },
        "revisit_grid_attempts": reported_revisit["grid_attempts"],
        "revisit_fully_scored": reported_revisit["fully_scored"],
        "revisit_support_rejects": reported_revisit["support_band_rejects"],
    }
    require(observed == checks, "published audit differs from raw recount")

    digest = hashlib.sha256("".join(
        f"{completion_hashes[index]}  {index:03d}\n"
        for index in range(22)
    ).encode()).hexdigest()
    require(digest == audit["source_completion_digest_sha256"],
            "audit source digest differs")
    conclusion = audit["audit_conclusion"]
    require(conclusion.get("simple_attempt_budget_failure") is False,
            "audit incorrectly attributes failure to attempt budget")
    require(conclusion.get("side_direction_sampler_failure") is False,
            "audit incorrectly attributes failure to side sampling")
    require(conclusion.get("binary_role_pair_constructibility_failure") is True,
            "audit did not identify the role-pair constructibility failure")
    return {
        "schema_version": SCHEMA,
        "verified": True,
        "status": "independent_raw_construction_recount_passed",
        "query_policy_outcomes_read": False,
        "navigation_success_rate_computed": False,
        "completion_count": len(ordered),
        "population_receipt_sha256": None,
        "construction_verification_sha256": None,
        "audit_sha256": None,
        "source_completion_digest_sha256": digest,
        "recomputed": observed,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--completions", type=Path, required=True)
    parser.add_argument("--population-receipt", type=Path, required=True)
    parser.add_argument("--construction-verification", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = verify(
        completion_paths(args.completions),
        read_object(args.population_receipt),
        read_object(args.construction_verification),
        read_object(args.audit),
    )
    result["population_receipt_sha256"] = sha256_file(args.population_receipt)
    result["construction_verification_sha256"] = sha256_file(
        args.construction_verification)
    result["audit_sha256"] = sha256_file(args.audit)
    atomic_json(args.out, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
