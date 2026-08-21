#!/usr/bin/env python3
"""Audit the authorized full-fresh20 certified graph-rescue expansion."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from MemNavData.audit_shared_online_double_revisit_fresh import (
    exact_mcnemar,
    scene_cluster_bootstrap,
)
from MemNavData.summarize_certified_graph_rescue_pilot import (
    CONTROL_INDICES,
    KNOWN_FAILURE_INDICES,
    PILOT_INDICES,
    audit_records,
    require,
    sha256_file,
)


FULL_INDICES = tuple(range(20))
UNSELECTED_INDICES = tuple(
    index for index in FULL_INDICES if index not in PILOT_INDICES)


def arm_contrast(
        records: list[dict[str, Any]], first_arm: str, second_arm: str,
        outcome: str, *, seed: int = 260813,
        resamples: int = 100_000) -> dict[str, Any]:
    first = [bool(row[first_arm][outcome]) for row in records]
    second = [bool(row[second_arm][outcome]) for row in records]
    result = exact_mcnemar(first, second)
    result["first_arm"] = first_arm
    result["second_arm"] = second_arm
    result["outcome"] = outcome
    result["scene_cluster_bootstrap"] = scene_cluster_bootstrap(
        [
            (str(row["scene"]), left, right)
            for row, left, right in zip(records, first, second)
        ],
        seed=seed,
        resamples=resamples,
    )
    return result


def expansion_gate(records: list[dict[str, Any]]) -> dict[str, Any]:
    require(tuple(row["selection_index"] for row in records) == FULL_INDICES,
            "expansion gate requires all 20 ordered records")
    direct_failures = [
        row["selection_index"] for row in records
        if not row["direct"]["joint"]
    ]
    rescue_losses = [
        row["selection_index"] for row in records
        if row["direct"]["joint"] and not row["rescue"]["joint"]
    ]
    rescue_gains = [
        row["selection_index"] for row in records
        if not row["direct"]["joint"] and row["rescue"]["joint"]
    ]
    gains_without_graph = [
        row["selection_index"] for row in records
        if (not row["direct"]["joint"] and row["rescue"]["joint"]
            and row["rescue"]["graph_active_plans_B"] <= 0
            and row["rescue"]["graph_active_plans_C"] <= 0)
    ]
    unselected_interventions = sum(
        bool(row["prefix_audits"][arm][role]
             and row["prefix_audits"][arm][role]["attempted"])
        for row in records if row["selection_index"] in UNSELECTED_INDICES
        for arm in ("budget_control", "rescue") for role in ("B", "C")
    )
    baseline_reproduced = tuple(direct_failures) == KNOWN_FAILURE_INDICES
    decision = (
        "freeze_internal_fresh20_result"
        if baseline_reproduced and not rescue_losses
        and not gains_without_graph and unselected_interventions == 0
        else "stop_and_audit_expansion"
    )
    return {
        "direct_failure_indices": direct_failures,
        "expected_direct_failure_indices": list(KNOWN_FAILURE_INDICES),
        "baseline_classification_reproduced": baseline_reproduced,
        "rescue_gain_indices": rescue_gains,
        "rescue_loss_indices": rescue_losses,
        "rescue_gains_without_active_graph": gains_without_graph,
        "unselected_treatment_interventions": unselected_interventions,
        "allowed_unselected_treatment_interventions": 0,
        "decision": decision,
    }


def audit(
        run_root: Path, expected_manifest_sha: str,
        pilot_report: Path, expected_pilot_report_sha: str,
        *, bootstrap_seed: int = 260813,
        bootstrap_resamples: int = 100_000) -> dict[str, Any]:
    require(sha256_file(pilot_report) == expected_pilot_report_sha,
            "pilot report hash changed")
    pilot = json.loads(pilot_report.read_text())
    require(pilot["gate"]["decision"] == "expand_to_unselected_fresh20",
            "pilot did not authorize expansion")
    records = audit_records(run_root, expected_manifest_sha, FULL_INDICES)
    current_pilot = [
        row for row in records if row["selection_index"] in PILOT_INDICES
    ]
    require(current_pilot == pilot["records"],
            "reused pilot records differ from the authorizing report")
    for row in records:
        index = row["selection_index"]
        if index in UNSELECTED_INDICES:
            require(row["cohort"] == "unselected_expansion",
                    f"{index}: wrong expansion cohort")
        elif index in KNOWN_FAILURE_INDICES:
            require(row["cohort"] == "known_failure",
                    f"{index}: wrong known-failure cohort")
        elif index in CONTROL_INDICES:
            require(row["cohort"] == "control",
                    f"{index}: wrong pilot-control cohort")

    arm_summary = {}
    for arm in ("direct", "budget_control", "rescue"):
        arm_summary[arm] = {
            "B_success": sum(row[arm]["B"] for row in records),
            "C_success": sum(row[arm]["C"] for row in records),
            "joint_success": sum(row[arm]["joint"] for row in records),
            "episodes": len(records),
            "graph_active_plans_B": sum(
                row[arm]["graph_active_plans_B"] for row in records),
            "graph_active_plans_C": sum(
                row[arm]["graph_active_plans_C"] for row in records),
        }

    return {
        "schema_version": (
            "certified_stagnation_graph_fresh20_expansion_report_v1"),
        "scope": (
            "internal full-fresh20 total-effect estimate after a post-hoc "
            "three-failure mechanism gate; not scene-disjoint paper confirmation"),
        "benchmark_manifest_sha256": expected_manifest_sha,
        "pilot_report": str(pilot_report),
        "pilot_report_sha256": expected_pilot_report_sha,
        "episodes": len(records),
        "scene_clusters": len({row["scene"] for row in records}),
        "pilot_indices": list(PILOT_INDICES),
        "unselected_indices": list(UNSELECTED_INDICES),
        "arm_summary": arm_summary,
        "primary_contrast": arm_contrast(
            records, "rescue", "budget_control", "joint",
            seed=bootstrap_seed, resamples=bootstrap_resamples),
        "secondary_contrasts": {
            "rescue_minus_direct_joint": arm_contrast(
                records, "rescue", "direct", "joint",
                seed=bootstrap_seed + 1,
                resamples=bootstrap_resamples),
            "budget_control_minus_direct_joint": arm_contrast(
                records, "budget_control", "direct", "joint",
                seed=bootstrap_seed + 2,
                resamples=bootstrap_resamples),
            "rescue_minus_budget_control_B": arm_contrast(
                records, "rescue", "budget_control", "B",
                seed=bootstrap_seed + 3,
                resamples=bootstrap_resamples),
        },
        "gate": expansion_gate(records),
        "records": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--expected-manifest-sha", required=True)
    parser.add_argument("--pilot-report", type=Path, required=True)
    parser.add_argument("--expected-pilot-report-sha", required=True)
    parser.add_argument("--bootstrap-seed", type=int, default=260813)
    parser.add_argument("--bootstrap-resamples", type=int, default=100_000)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = audit(
        args.run_root, args.expected_manifest_sha, args.pilot_report,
        args.expected_pilot_report_sha,
        bootstrap_seed=args.bootstrap_seed,
        bootstrap_resamples=args.bootstrap_resamples,
    )
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "arm_summary": report["arm_summary"],
        "primary_contrast": report["primary_contrast"],
        "gate": report["gate"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
