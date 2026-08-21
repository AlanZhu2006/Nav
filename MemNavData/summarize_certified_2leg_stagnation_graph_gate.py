#!/usr/bin/env python3
"""Audit and summarize the consumed fresh160 two-leg graph-rescue gate."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if str(value) in {"1", "1.0", "True", "true"}:
        return True
    if str(value) in {"0", "0.0", "False", "false", ""}:
        return False
    raise ValueError(f"invalid bool {value!r}")


def exact_mcnemar(gains: int, losses: int) -> float:
    discordant = int(gains) + int(losses)
    if discordant == 0:
        return 1.0
    tail = sum(
        math.comb(discordant, index)
        for index in range(min(int(gains), int(losses)) + 1)
    )
    return min(1.0, 2.0 * tail / (2 ** discordant))


def one_metric(path: Path) -> dict[str, str]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    require(len(rows) == 1, f"expected one metric row: {path}")
    return rows[0]


def _world_to_local_forward_left(
    current: dict[str, Any], target: dict[str, Any]
) -> tuple[float, float]:
    yaw = float(current["yaw"])
    dx = float(target["x"]) - float(current["x"])
    dz = float(target["z"]) - float(current["z"])
    return (
        -math.sin(yaw) * dx - math.cos(yaw) * dz,
        -math.cos(yaw) * dx + math.sin(yaw) * dz,
    )


def _angular_error_deg(first: Any, second: Any) -> float | None:
    left = tuple(float(value) for value in first)
    right = tuple(float(value) for value in second)
    if len(left) != 2 or len(right) != 2:
        return None
    left_norm = math.hypot(*left)
    right_norm = math.hypot(*right)
    if min(left_norm, right_norm) <= 1e-12:
        return None
    cosine = sum(a * b for a, b in zip(left, right)) / (
        left_norm * right_norm
    )
    return math.degrees(math.acos(max(-1.0, min(1.0, cosine))))


def _distance_xz(first: dict[str, Any], second: dict[str, Any]) -> float:
    return math.hypot(
        float(first["x"]) - float(second["x"]),
        float(first["z"]) - float(second["z"]),
    )


def graph_execution(
    plans: list[dict[str, Any]],
    *,
    memory_trace: list[dict[str, Any]] | None = None,
    rollout_trace: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    requested = [
        plan for plan in plans
        if plan.get("certified_graph_rescue_requested") is True
    ]
    reported_active = [
        plan for plan in requested
        if plan.get("certified_graph_rescue_active") is True
    ]
    historical_subgoals = [
        plan for plan in requested
        if plan.get("certified_graph_reason") == "historical_subgoal"
        and plan.get("certified_graph_node") is not None
    ]
    empty_route_direct = [
        plan for plan in requested
        if plan.get("certified_graph_reason") == "route_complete_direct_bearing"
    ]
    first = requested[0] if requested else {}
    post_outcome: dict[str, Any] | None = None
    if requested and memory_trace is not None and rollout_trace is not None:
        memory = {int(row["frame_idx"]): row for row in memory_trace}
        rollout = {int(row["step"]): row for row in rollout_trace}
        first_step = int(first["step"])
        target_anchor = first.get("certified_graph_target_anchor")
        route_start = first.get("certified_graph_route_start_node")
        current = rollout.get(first_step)
        target = memory.get(int(target_anchor)) if target_anchor is not None else None
        start = memory.get(int(route_start)) if route_start is not None else None
        bearing_errors = []
        pre_normalization_norms = []
        for plan in historical_subgoals:
            node = plan.get("certified_graph_node")
            step = int(plan["step"])
            predicted = plan.get("memory_bearing_unit")
            if predicted is not None and node is not None:
                plan_current = rollout.get(step)
                plan_target = memory.get(int(node))
                if plan_current is not None and plan_target is not None:
                    error = _angular_error_deg(
                        predicted,
                        _world_to_local_forward_left(plan_current, plan_target),
                    )
                    if error is not None:
                        bearing_errors.append(error)
            norm = plan.get("memory_unbounded_pointgoal_norm")
            if norm is not None and math.isfinite(float(norm)):
                pre_normalization_norms.append(float(norm))
        post_outcome = {
            "scope": "Habitat ground truth audit only; never policy input",
            "physical_current_to_target_anchor_m_at_first_request": (
                _distance_xz(current, target)
                if current is not None and target is not None else None
            ),
            "physical_current_to_route_start_node_m_at_first_request": (
                _distance_xz(current, start)
                if current is not None and start is not None else None
            ),
            "route_start_equals_target": (
                int(route_start) == int(target_anchor)
                if route_start is not None and target_anchor is not None
                else None
            ),
            "historical_subgoal_bearing_error_deg": ({
                "count": len(bearing_errors),
                "first": bearing_errors[0],
                "median": statistics.median(bearing_errors),
                "maximum": max(bearing_errors),
            } if bearing_errors else None),
            "pre_normalization_pointgoal_norm": ({
                "count": len(pre_normalization_norms),
                "first": pre_normalization_norms[0],
                "median": statistics.median(pre_normalization_norms),
                "minimum": min(pre_normalization_norms),
                "maximum": max(pre_normalization_norms),
            } if pre_normalization_norms else None),
            "controller_fixed_radius_m": (
                next((float(plan["memory_pointgoal_fixed_radius_m"])
                      for plan in historical_subgoals
                      if plan.get("memory_pointgoal_fixed_radius_m") is not None),
                     None)
            ),
        }
    return {
        "requested_plan_count": len(requested),
        "reported_active_plan_count_pre_diagnostic_fix": len(reported_active),
        "historical_subgoal_plan_count": len(historical_subgoals),
        "empty_route_direct_plan_count": len(empty_route_direct),
        "executed_historical_subgoal": bool(historical_subgoals),
        "first_route_start_contract": first.get(
            "certified_graph_route_start_contract"),
        "first_route_start_node": first.get("certified_graph_route_start_node"),
        "first_target_anchor": first.get("certified_graph_target_anchor"),
        "first_temporal_direction": first.get(
            "certified_graph_temporal_direction"),
        "first_route_node_count": first.get("certified_graph_count"),
        "first_reason": first.get("certified_graph_reason"),
        "post_outcome_diagnostic": post_outcome,
    }


def audit(manifest_path: Path, run_root: Path) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text())
    require(
        manifest.get("schema_version")
        == "certified_2leg_stagnation_graph_internal_gate_v1",
        "manifest schema changed",
    )
    entries = manifest["episodes"]
    require(len(entries) == 10, "gate requires ten frozen entries")
    require(
        sum(row["cohort"] == "known_failure" for row in entries) == 5,
        "known-failure denominator changed",
    )
    require(
        sum(row["cohort"] == "success_control" for row in entries) == 5,
        "success-control denominator changed",
    )

    records = []
    for frozen in entries:
        name = (
            f"{int(frozen['index']):02d}_{frozen['scene']}_"
            f"{frozen['episode']}"
        )
        entry_root = run_root / "entries" / name
        episode_audit = json.loads(
            (entry_root / "episode_audit.json").read_text()
        )
        require(episode_audit["entry"] == name, f"{name}: audit identity changed")
        require(
            episode_audit["cohort"] == frozen["cohort"],
            f"{name}: cohort changed",
        )
        require(
            episode_audit["causal_prefix_exact"] is True,
            f"{name}: causal prefix audit failed",
        )
        arms: dict[str, Any] = {}
        for arm, mode in (
            ("direct", "off"),
            ("budget_control", "budget_control"),
            ("rescue", "rescue"),
        ):
            arm_root = entry_root / arm
            summary = json.loads((arm_root / "summary.json").read_text())
            metric = one_metric(arm_root / "metric.csv")
            plans = json.loads(
                (arm_root / f"{frozen['episode']}_plans.json").read_text()
            )
            require(summary["episodes"] == 1, f"{name}/{arm}: incomplete")
            require(
                summary["certified_stagnation_graph"] == mode,
                f"{name}/{arm}: mode changed",
            )
            require(
                metric["episode"] == frozen["episode"],
                f"{name}/{arm}: episode changed",
            )
            arms[arm] = {
                "B": as_bool(metric["reached_B"]),
                "steps_B": int(metric["steps_B"]),
                "termination_B": metric["termination_reason_B"],
                "final_distance_B": float(metric["final_dist_B"]),
                "intervention": as_bool(
                    metric["certified_stagnation_intervention_attempted"]
                ),
                "graph": graph_execution(
                    plans["legB"],
                    memory_trace=plans.get("legA_memory_trace"),
                    rollout_trace=plans.get("legB_rollout_trace"),
                ),
            }

        require(
            arms["direct"]["B"] is frozen["source_B_success"],
            f"{name}: direct outcome did not reproduce",
        )
        require(
            arms["direct"]["termination_B"] == frozen["source_B_termination"],
            f"{name}: direct termination did not reproduce",
        )
        if frozen["cohort"] == "known_failure":
            require(
                arms["budget_control"]["intervention"]
                and arms["rescue"]["intervention"],
                f"{name}: frozen treatment did not trigger",
            )
        else:
            require(
                not any(arms[arm]["intervention"] for arm in arms),
                f"{name}: success control was treated",
            )
            require(
                len({
                    (arms[arm]["B"], arms[arm]["steps_B"],
                     arms[arm]["termination_B"],
                     arms[arm]["final_distance_B"])
                    for arm in arms
                }) == 1,
                f"{name}: success control arms differ",
            )
        records.append({
            "index": frozen["index"],
            "scene": frozen["scene"],
            "episode": frozen["episode"],
            "cohort": frozen["cohort"],
            "source_machine_steps_B": frozen["source_B_steps"],
            "same_process_direct_step_delta": (
                arms["direct"]["steps_B"] - frozen["source_B_steps"]
            ),
            "arms": arms,
        })

    failures = [row for row in records if row["cohort"] == "known_failure"]
    controls = [row for row in records if row["cohort"] == "success_control"]
    rescue_successes = sum(row["arms"]["rescue"]["B"] for row in failures)
    budget_successes = sum(
        row["arms"]["budget_control"]["B"] for row in failures
    )
    gains = sum(
        row["arms"]["rescue"]["B"]
        and not row["arms"]["budget_control"]["B"]
        for row in failures
    )
    losses = sum(
        row["arms"]["budget_control"]["B"]
        and not row["arms"]["rescue"]["B"]
        for row in failures
    )
    true_graph_episodes = sum(
        row["arms"]["rescue"]["graph"]["executed_historical_subgoal"]
        for row in failures
    )
    empty_route_episodes = sum(
        row["arms"]["rescue"]["graph"]["empty_route_direct_plan_count"] > 0
        and not row["arms"]["rescue"]["graph"][
            "executed_historical_subgoal"
        ]
        for row in failures
    )
    control_losses = sum(
        row["arms"]["direct"]["B"] and not row["arms"]["rescue"]["B"]
        for row in controls
    )
    return {
        "schema_version": "certified_2leg_stagnation_graph_gate_report_v1",
        "scope": manifest["scope"],
        "primary_contrast": manifest["frozen_runtime"]["primary_contrast"],
        "known_failure_count": len(failures),
        "success_control_count": len(controls),
        "known_failure_successes": {
            "direct": sum(row["arms"]["direct"]["B"] for row in failures),
            "budget_control": budget_successes,
            "rescue": rescue_successes,
        },
        "rescue_minus_budget_control": {
            "gains": gains,
            "losses": losses,
            "exact_mcnemar_p": exact_mcnemar(gains, losses),
        },
        "success_controls": {
            "direct": sum(row["arms"]["direct"]["B"] for row in controls),
            "budget_control": sum(
                row["arms"]["budget_control"]["B"] for row in controls
            ),
            "rescue": sum(row["arms"]["rescue"]["B"] for row in controls),
            "rescue_losses": control_losses,
            "interventions": sum(
                row["arms"][arm]["intervention"]
                for row in controls for arm in ("budget_control", "rescue")
            ),
        },
        "execution_audit": {
            "failure_episodes_with_historical_subgoals": true_graph_episodes,
            "failure_episodes_with_empty_route_direct_fallback_only": (
                empty_route_episodes
            ),
            "diagnostic_issue": (
                "pre-fix active flag included route_complete_direct_bearing; "
                "historical_subgoal_plan_count is the execution-grounded count"
            ),
        },
        "descriptive_fresh160_counterfactual_not_a_new_full_rerun": {
            "certified_baseline_B_given_A": "112/120",
            "one_observed_rescue_if_all_other_frozen_no_ops_hold": "113/120",
            "absolute_change_pp": 100.0 / 120.0,
            "paired_gains": gains,
            "paired_losses": losses,
            "exact_mcnemar_p": exact_mcnemar(gains, losses),
        },
        "decision": (
            "do_not_expand_two_leg_graph_rescue; retain as a 3-leg-specific "
            "internal mechanism until route-start localization is repaired"
        ),
        "records": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.manifest, args.run_root)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "known_failure_successes": report["known_failure_successes"],
        "contrast": report["rescue_minus_budget_control"],
        "success_controls": report["success_controls"],
        "execution_audit": report["execution_audit"],
        "decision": report["decision"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
