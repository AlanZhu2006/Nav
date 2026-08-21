#!/usr/bin/env python3
"""Fail-closed audit for the post-hoc certified graph-rescue mechanism pilot."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any


PILOT_INDICES = (0, 1, 2, 3, 7, 14)
CONTROL_INDICES = (0, 1, 3)
KNOWN_FAILURE_INDICES = (2, 7, 14)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def one_metric(path: Path) -> dict[str, str]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    require(len(rows) == 1, f"expected one metric row: {path}")
    return rows[0]


def as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if str(value) in ("1", "1.0", "True", "true"):
        return True
    if str(value) in ("0", "0.0", "False", "false", ""):
        return False
    raise ValueError(f"invalid bool {value!r}")


def first_graph_plan(plans: list[dict[str, Any]]) -> int | None:
    return next((index for index, plan in enumerate(plans)
                 if plan.get("certified_graph_rescue_requested") is True), None)


def optional_int(value: Any) -> int | None:
    if value in (None, "", "None"):
        return None
    return int(value)


def first_plan_at_or_after(
        plans: list[dict[str, Any]], step: int) -> int | None:
    return next((index for index, plan in enumerate(plans)
                 if int(plan["step"]) >= int(step)), None)


def causal_payload(value: Any) -> Any:
    """Remove wall-clock fields while preserving every decision variable."""
    if isinstance(value, dict):
        return {
            key: causal_payload(item)
            for key, item in value.items()
            if not key.endswith("_ms")
        }
    if isinstance(value, list):
        return [causal_payload(item) for item in value]
    return value


def verify_leg_prefix(
        direct: dict[str, Any], treatment: dict[str, Any], role: str,
        *, attempted: bool, intervention_step: int | None,
        expects_graph: bool) -> dict[str, Any]:
    plans_d = direct[f"leg{role}"]
    plans_r = treatment[f"leg{role}"]
    rollout_d = direct["rollout_traces"][f"leg{role}"]
    rollout_r = treatment["rollout_traces"][f"leg{role}"]
    memory_d = direct["memory_traces"][f"leg{role}"]
    memory_r = treatment["memory_traces"][f"leg{role}"]
    graph_branch = first_graph_plan(plans_r)
    if attempted:
        require(intervention_step is not None,
                f"{role}: intervention has no frozen step")
        branch = first_plan_at_or_after(plans_r, intervention_step)
        require(branch is not None, f"{role}: no post-intervention plan")
        require(causal_payload(plans_d) == causal_payload(plans_r[:branch]),
                f"{role}: causal plan prefix changed")
        require(rollout_d == rollout_r[:len(rollout_d)],
                f"{role}: physical prefix changed")
        require(memory_d == memory_r[:len(memory_d)],
                f"{role}: memory prefix changed")
        if expects_graph:
            require(graph_branch == branch,
                    f"{role}: graph did not begin at the intervention branch")
        else:
            require(graph_branch is None,
                    f"{role}: budget control requested a graph")
    else:
        branch = None
        require(intervention_step is None,
                f"{role}: no-op treatment has an intervention step")
        require(graph_branch is None, f"{role}: graph request without trigger")
        require(causal_payload(plans_d) == causal_payload(plans_r),
                f"{role}: no-op causal plans changed")
        require(rollout_d == rollout_r, f"{role}: no-op rollout changed")
        require(memory_d == memory_r, f"{role}: no-op memory changed")
    return {
        "attempted": attempted,
        "intervention_step": intervention_step,
        "first_treatment_plan_index": branch,
        "first_graph_plan_index": graph_branch,
        "plan_prefix_exact": True,
        "rollout_prefix_exact": True,
        "memory_prefix_exact": True,
    }


def pilot_gate(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Apply the frozen mechanism gate to already audited episode records."""
    failures = [
        row for row in records
        if row["selection_index"] in KNOWN_FAILURE_INDICES
    ]
    controls = [
        row for row in records if row["selection_index"] in CONTROL_INDICES
    ]
    require(len(failures) == 3 and len(controls) == 3,
            "gate requires the frozen three failure / three control split")
    baseline_reproduced = bool(
        all(not row["direct"]["B"] for row in failures)
        and all(row["direct"]["joint"] for row in controls)
    )
    rescued_b = sum(row["rescue"]["B"] for row in failures)
    budget_b = sum(row["budget_control"]["B"] for row in failures)
    rescue_vs_budget_gains = sum(
        not row["budget_control"]["B"] and row["rescue"]["B"]
        for row in failures)
    rescue_vs_budget_losses = sum(
        row["budget_control"]["B"] and not row["rescue"]["B"]
        for row in failures)
    graph_active_on_every_rescue = all(
        (not row["rescue"]["B"])
        or row["rescue"]["graph_active_plans_B"] > 0
        for row in failures)
    control_losses = sum(
        row["direct"]["joint"] and not row[arm]["joint"]
        for row in controls for arm in ("budget_control", "rescue"))
    control_interventions = sum(
        bool(row["prefix_audits"][arm][role]
             and row["prefix_audits"][arm][role]["attempted"])
        for row in controls for arm in ("budget_control", "rescue")
        for role in ("B", "C"))
    decision = (
        "expand_to_unselected_fresh20"
        if baseline_reproduced and rescued_b >= 2 and budget_b <= 1
        and rescue_vs_budget_gains >= 1 and rescue_vs_budget_losses == 0
        and graph_active_on_every_rescue
        and control_losses == 0 and control_interventions == 0
        else "stop_or_repair_before_expansion"
    )
    return {
        "baseline_classification_reproduced": baseline_reproduced,
        "known_failure_B_rescues": rescued_b,
        "known_failure_budget_control_B_successes": budget_b,
        "rescue_vs_budget_B_gains": rescue_vs_budget_gains,
        "rescue_vs_budget_B_losses": rescue_vs_budget_losses,
        "graph_active_on_every_rescued_B": graph_active_on_every_rescue,
        "known_failure_count": len(failures),
        "control_joint_losses": control_losses,
        "control_interventions": control_interventions,
        "gate": {
            "required_B_rescues": 2,
            "maximum_budget_control_B_successes": 1,
            "required_rescue_vs_budget_B_gains": 1,
            "allowed_rescue_vs_budget_B_losses": 0,
            "require_graph_active_on_every_rescued_B": True,
            "allowed_control_joint_losses": 0,
            "allowed_control_interventions": 0,
            "decision": decision,
        },
    }


def audit_records(
        run_root: Path, expected_manifest_sha: str,
        expected_indices: tuple[int, ...]) -> list[dict[str, Any]]:
    """Audit paired episode outputs and return causal records.

    Keeping the per-episode audit independent of the post-hoc pilot gate lets
    the authorized expansion apply exactly the same prefix checks to all 20
    episodes without changing the frozen six-episode decision rule.
    """
    require(sha256_file(run_root / "benchmark_manifest.json") ==
            expected_manifest_sha, "copied benchmark manifest hash changed")
    episode_dirs = sorted((run_root / "scenes").iterdir())
    require(len(episode_dirs) == len(expected_indices),
            "audited episode count changed")
    records = []
    for episode_dir in episode_dirs:
        contract = json.loads((episode_dir / "episode_contract.json").read_text())
        index = int(contract["selection_index"])
        require(index in expected_indices, "unexpected audited index")
        require(contract["benchmark_manifest_sha256"] == expected_manifest_sha,
                "episode benchmark hash changed")
        arms = {}
        for arm, mode in (
                ("direct", "off"),
                ("budget_control", "budget_control"),
                ("rescue", "rescue")):
            root = episode_dir / arm
            summary = json.loads((root / "summary.json").read_text())
            run_contract = json.loads((root / "run_contract.json").read_text())
            metric = one_metric(root / "metric.csv")
            plans = json.loads((root / f"{contract['episode']}_plans.json").read_text())
            require(summary["benchmark_manifest_sha256"] == expected_manifest_sha,
                    f"{index}/{arm}: manifest changed")
            require(summary["certified_stagnation_graph"] == mode,
                    f"{index}/{arm}: graph mode changed")
            require(run_contract["certified_stagnation_graph"] == mode,
                    f"{index}/{arm}: run contract mode changed")
            require(float(run_contract["graph_subgoal_spacing_m"]) == 1.25,
                    f"{index}/{arm}: spacing changed")
            require(float(run_contract["graph_subgoal_arrival_m"]) == 0.60,
                    f"{index}/{arm}: arrival threshold changed")
            arms[arm] = {"summary": summary, "metric": metric, "plans": plans}
        md = arms["direct"]["metric"]
        for arm in ("budget_control", "rescue"):
            require(arms["direct"]["plans"]["replay"] ==
                    arms[arm]["plans"]["replay"],
                    f"{index}/{arm}: online-A replay differs")
            mt = arms[arm]["metric"]
            for key in ("scene", "episode", "seed", "variant",
                        "A_candidate_ceiling"):
                require(md[key] == mt[key],
                        f"{index}/{arm}: paired identity differs at {key}")
        prefix_audits = {}
        for arm, expects_graph in (("budget_control", False), ("rescue", True)):
            metric = arms[arm]["metric"]
            attempted_b = as_bool(
                metric["certified_stagnation_intervention_attempted_B"])
            prefix_audits[arm] = {
                "B": verify_leg_prefix(
                    arms["direct"]["plans"], arms[arm]["plans"], "B",
                    attempted=attempted_b,
                    intervention_step=optional_int(metric[
                        "certified_stagnation_intervention_step_B"]),
                    expects_graph=expects_graph,
                ),
                "C": None,
            }
            # C is pairable only when B produced an identical causal rollout.
            if (as_bool(md["reached_B"]) and as_bool(metric["reached_B"])
                    and arms["direct"]["plans"]["rollout_traces"]["legB"]
                    == arms[arm]["plans"]["rollout_traces"]["legB"]):
                attempted_c = as_bool(
                    metric["certified_stagnation_intervention_attempted_C"])
                prefix_audits[arm]["C"] = verify_leg_prefix(
                    arms["direct"]["plans"], arms[arm]["plans"], "C",
                    attempted=attempted_c,
                    intervention_step=optional_int(metric[
                        "certified_stagnation_intervention_step_C"]),
                    expects_graph=expects_graph,
                )
        outcomes = {}
        for arm in ("direct", "budget_control", "rescue"):
            metric = arms[arm]["metric"]
            outcomes[arm] = {
                "B": as_bool(metric["reached_B"]),
                "C": as_bool(metric["reached_C"]),
                "joint": as_bool(metric["joint_success"]),
                "final_dist_B": float(metric["final_dist_B"]),
                "graph_active_plans_B": int(
                    metric["certified_graph_active_plans_B"]),
                "graph_active_plans_C": int(
                    metric["certified_graph_active_plans_C"]),
            }
        require(outcomes["direct"]["graph_active_plans_B"] == 0
                and outcomes["direct"]["graph_active_plans_C"] == 0,
                f"{index}: direct arm activated graph")
        require(outcomes["budget_control"]["graph_active_plans_B"] == 0
                and outcomes["budget_control"]["graph_active_plans_C"] == 0,
                f"{index}: budget control activated graph")
        record = {
            "selection_index": index,
            "cohort": contract["cohort"],
            "scene": md["scene"], "episode": md["episode"],
            **outcomes,
            "prefix_audits": prefix_audits,
        }
        records.append(record)
    require(tuple(sorted(row["selection_index"] for row in records)) ==
            expected_indices, "audited indices incomplete")
    records.sort(key=lambda row: row["selection_index"])
    return records


def audit(run_root: Path, expected_manifest_sha: str) -> dict[str, Any]:
    records = audit_records(run_root, expected_manifest_sha, PILOT_INDICES)
    gate = pilot_gate(records)
    return {
        "schema_version": "certified_stagnation_graph_pilot_report_v2_budget_control",
        "scope": "post-hoc mechanism pilot; not a population SR estimate",
        "benchmark_manifest_sha256": expected_manifest_sha,
        "pilot_indices": list(PILOT_INDICES),
        "known_failure_indices": list(KNOWN_FAILURE_INDICES),
        "control_indices": list(CONTROL_INDICES),
        **gate,
        "records": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--expected-manifest-sha", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.run_root, args.expected_manifest_sha)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report["gate"], sort_keys=True))


if __name__ == "__main__":
    main()
