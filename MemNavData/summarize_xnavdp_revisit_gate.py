#!/usr/bin/env python3
"""Audit and summarize the frozen four-arm X-NavDP revisit controller gate."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import random
from typing import Any

try:
    from .summarize_expanded_navdp_router_eval import (
        arm_summary,
        exact_sign_p,
        mean,
        percentile,
        require,
        truth,
    )
    from .xnavdp_revisit_contract import (
        OFFICIAL_XNAVDP_COMMIT,
        OFFICIAL_XNAVDP_POSTTRAIN_SHA256,
        XNAVDP_CHECKPOINT_TENSOR_COUNT,
        XNAVDP_CHECKPOINT_UNEXPECTED_TENSOR_COUNT,
        XNAVDP_MODEL_STATE_TENSOR_COUNT,
    )
except ImportError:  # Script execution from MemNavData/.
    from summarize_expanded_navdp_router_eval import (
        arm_summary,
        exact_sign_p,
        mean,
        percentile,
        require,
        truth,
    )
    from xnavdp_revisit_contract import (
        OFFICIAL_XNAVDP_COMMIT,
        OFFICIAL_XNAVDP_POSTTRAIN_SHA256,
        XNAVDP_CHECKPOINT_TENSOR_COUNT,
        XNAVDP_CHECKPOINT_UNEXPECTED_TENSOR_COUNT,
        XNAVDP_MODEL_STATE_TENSOR_COUNT,
    )


ARMS = (
    "navdp_native",
    "memory_mixed",
    "memory_base_point",
    "memory_xnavdp_point",
)
EXPECTED_CONTROLLERS = {
    "navdp_native": "navdp_mixed",  # inactive for server_backend=navdp
    "memory_mixed": "navdp_mixed",
    "memory_base_point": "navdp_point",
    "memory_xnavdp_point": "xnavdp_point",
}
BOOTSTRAP_SEED = 20260809
BOOTSTRAP_DRAWS = 20_000


def _finite_float(value: Any, label: str) -> float:
    converted = float(value)
    require(math.isfinite(converted), f"{label} is not finite")
    return converted


def _optional_truth(value: Any) -> bool | None:
    return None if value in (None, "") else truth(value)


def load_gate_arm(
    scene_root: Path, arm: str, scene: str,
) -> dict[tuple[str, str], dict[str, Any]]:
    arm_root = scene_root / arm
    metric_path = arm_root / "metric.csv"
    require(metric_path.is_file(), f"missing metric file: {metric_path}")
    with metric_path.open(newline="") as handle:
        metrics = list(csv.DictReader(handle))

    output: dict[tuple[str, str], dict[str, Any]] = {}
    for metric in metrics:
        episode = metric["episode"]
        plans_path = arm_root / f"{episode}_plans.json"
        require(plans_path.is_file(), f"missing plan trace: {plans_path}")
        payload = json.loads(plans_path.read_text())
        leg_a = payload.get("legA", [])
        leg_b = payload.get("legB", [])
        require(isinstance(leg_a, list) and isinstance(leg_b, list),
                f"malformed plan lists: {arm} {scene} {episode}")
        require(truth(metric.get("deterministic_plan_seeds")),
                f"deterministic plan seeds disabled: {arm} {scene} {episode}")
        for plan in leg_a + leg_b:
            requested = plan.get("requested_diffusion_seed")
            echoed = plan.get("diffusion_seed")
            require(requested is not None and echoed is not None,
                    f"missing plan seed: {arm} {scene} {episode}")
            require(int(requested) == int(echoed),
                    f"plan seed mismatch: {arm} {scene} {episode}")

        reported_controller = metric.get("revisit_controller") or "navdp_mixed"
        require(reported_controller == EXPECTED_CONTROLLERS[arm],
                f"wrong controller for {arm}: {reported_controller}")
        if arm == "navdp_native":
            require(metric.get("server_backend") == "navdp",
                    "native_image did not use native NavDP")
        else:
            require(metric.get("server_backend") == "hybrid_pose",
                    f"{arm} did not use the geometry-memory router")

        active_a = [truth(plan.get("router_active")) for plan in leg_a]
        active_b = [truth(plan.get("router_active")) for plan in leg_b]
        trace_sha = (metric.get("leg1_trace_sha256")
                     or payload.get("leg1_trace_sha256"))
        require(isinstance(trace_sha, str) and len(trace_sha) == 64,
                f"missing shared Goal-A trace hash: {arm} {scene} {episode}")
        x_history_valid = _optional_truth(
            metric.get("xnavdp_history_contract_valid"))
        if arm == "memory_xnavdp_point":
            require(x_history_valid is True,
                    f"X history contract failed: {scene} {episode}")
            require(metric.get("xnavdp_official_commit")
                    == OFFICIAL_XNAVDP_COMMIT,
                    f"X source receipt mismatch: {scene} {episode}")
            require(metric.get("xnavdp_checkpoint_sha256")
                    == OFFICIAL_XNAVDP_POSTTRAIN_SHA256,
                    f"X checkpoint receipt mismatch: {scene} {episode}")
            require(truth(metric.get("xnavdp_checkpoint_load_audited")),
                    f"X checkpoint coverage was not audited: {scene} {episode}")
            expected_checkpoint_audit = {
                "xnavdp_model_tensor_count": XNAVDP_MODEL_STATE_TENSOR_COUNT,
                "xnavdp_checkpoint_tensor_count": XNAVDP_CHECKPOINT_TENSOR_COUNT,
                "xnavdp_checkpoint_missing_count": 0,
                "xnavdp_checkpoint_unexpected_count": (
                    XNAVDP_CHECKPOINT_UNEXPECTED_TENSOR_COUNT),
                "xnavdp_checkpoint_shape_mismatch_count": 0,
            }
            for field, expected_value in expected_checkpoint_audit.items():
                require(int(metric.get(field, -1)) == expected_value,
                        f"X checkpoint coverage {field} mismatch: "
                        f"{scene} {episode}")

        verification_ms = []
        selected_ranks = []
        for plan in leg_a + leg_b:
            value = plan.get("router_verification_total_ms")
            if value not in (None, "") and float(value) > 0:
                verification_ms.append(float(value))
            rank = plan.get("router_selected_candidate_rank")
            if rank not in (None, ""):
                selected_ranks.append(int(rank))

        key = (scene, episode)
        require(key not in output, f"duplicate result row: {arm} {key}")
        output[key] = {
            "scene": scene,
            "episode": episode,
            "seed": int(metric["seed"]),
            "recall_gap": int(float(metric["recall_gap"])),
            "reached_a": truth(metric["reached_A"]),
            "reached_b": truth(metric["reached_B"]),
            "joint": truth(metric["reached_A"]) and truth(metric["reached_B"]),
            "spl_a": _finite_float(metric["spl_A"], "spl_A"),
            "spl_b": _finite_float(metric["spl_B"], "spl_B"),
            "geo_a": _finite_float(metric["geo_A"], "geo_A"),
            "geo_b": _finite_float(metric["geo_B"], "geo_B"),
            "path_a": _finite_float(metric["len_A"], "len_A"),
            "path_b": _finite_float(metric["len_B"], "len_B"),
            "final_dist_a": _finite_float(metric["final_dist_A"], "final_dist_A"),
            "final_dist_b": _finite_float(
                metric["terminal_final_goal_dist_m"], "final_dist_B"),
            "steps_a": int(metric["steps_A"]),
            "steps_b": int(metric["steps_B"]),
            "termination_reason_b": metric.get("termination_reason_B"),
            "blocked_steps_b": int(float(metric.get("blocked_steps_B") or 0)),
            "router_plans_a": len(active_a),
            "router_plans_b": len(active_b),
            "router_active_plans_a": sum(active_a),
            "router_active_plans_b": sum(active_b),
            "router_active_episode_a": any(active_a),
            "router_active_episode_b": any(active_b),
            "geometry_verification_ms": verification_ms,
            "selected_candidate_ranks": selected_ranks,
            "leg1_trace_sha256": trace_sha,
            "deterministic_plan_seeds": True,
            "xnavdp_history_contract_valid": x_history_valid,
            "xnavdp_checkpoint_coverage_valid": (
                True if arm == "memory_xnavdp_point" else None),
        }
    return output


def _validate_pair(
    left_name: str, right_name: str,
    left: dict[str, Any], right: dict[str, Any], key: tuple[str, str],
) -> None:
    require(left["seed"] == right["seed"],
            f"seed mismatch: {left_name} {right_name} {key}")
    require(left["recall_gap"] == right["recall_gap"],
            f"recall-gap mismatch: {left_name} {right_name} {key}")
    for field in ("geo_a", "geo_b", "spl_a", "path_a", "final_dist_a"):
        require(math.isclose(left[field], right[field], abs_tol=1e-9),
                f"paired {field} mismatch: {left_name} {right_name} {key}")
    require(left["leg1_trace_sha256"] == right["leg1_trace_sha256"],
            f"Goal-A trace mismatch: {left_name} {right_name} {key}")
    require(left["reached_a"] == right["reached_a"],
            f"Goal-A outcome mismatch: {left_name} {right_name} {key}")
    require(left["steps_a"] == right["steps_a"],
            f"Goal-A step mismatch: {left_name} {right_name} {key}")


def scene_cluster_interval(
    episode_differences: dict[str, list[float]],
    *, draws: int = BOOTSTRAP_DRAWS, seed: int = BOOTSTRAP_SEED,
) -> list[float | None]:
    scenes = sorted(episode_differences)
    if not scenes:
        return [None, None]
    rng = random.Random(seed)
    samples = []
    for _ in range(draws):
        selected = [scenes[rng.randrange(len(scenes))]
                    for _ in range(len(scenes))]
        values = [value for scene in selected
                  for value in episode_differences[scene]]
        samples.append(sum(values) / len(values))
    return [percentile(samples, 0.025), percentile(samples, 0.975)]


def paired_revisit_summary(
    left_name: str,
    right_name: str,
    left: dict[tuple[str, str], dict[str, Any]],
    right: dict[tuple[str, str], dict[str, Any]],
    expected: set[tuple[str, str]],
) -> dict[str, Any]:
    outcomes = {
        "both_success": 0,
        "left_only_success": 0,
        "right_only_success": 0,
        "neither_success": 0,
    }
    common_active_outcomes = dict(outcomes)
    eligible = []
    per_scene_differences: dict[str, list[float]] = {}
    activation_divergence = {"left_only": 0, "right_only": 0}
    new_stuck_failures = []
    episodes = []
    for key in sorted(expected):
        left_row, right_row = left[key], right[key]
        _validate_pair(left_name, right_name, left_row, right_row, key)
        if not left_row["reached_a"]:
            continue
        eligible.append(key)
        left_success = left_row["reached_b"]
        right_success = right_row["reached_b"]
        if left_success and right_success:
            outcome = "both_success"
        elif left_success:
            outcome = "left_only_success"
        elif right_success:
            outcome = "right_only_success"
        else:
            outcome = "neither_success"
        outcomes[outcome] += 1
        difference = float(right_success) - float(left_success)
        per_scene_differences.setdefault(key[0], []).append(difference)

        left_active = left_row["router_active_episode_b"]
        right_active = right_row["router_active_episode_b"]
        if left_active and right_active:
            common_active_outcomes[outcome] += 1
        elif left_active:
            activation_divergence["left_only"] += 1
        elif right_active:
            activation_divergence["right_only"] += 1
        if (right_row["termination_reason_b"] == "stuck"
                and left_row["termination_reason_b"] != "stuck"):
            new_stuck_failures.append({"scene": key[0], "episode": key[1]})
        episodes.append({
            "scene": key[0],
            "episode": key[1],
            "outcome": outcome,
            "left_active": left_active,
            "right_active": right_active,
            "left_termination": left_row["termination_reason_b"],
            "right_termination": right_row["termination_reason_b"],
        })

    discordant = outcomes["left_only_success"] + outcomes["right_only_success"]
    net_gain = outcomes["right_only_success"] - outcomes["left_only_success"]
    return {
        "left": left_name,
        "right": right_name,
        "eligible_goal_a_successes": len(eligible),
        "outcomes": outcomes,
        "paired_risk_difference": net_gain / len(eligible) if eligible else None,
        "scene_cluster_bootstrap_95": scene_cluster_interval(
            per_scene_differences),
        "mcnemar_exact_two_sided_p": exact_sign_p(
            outcomes["right_only_success"], discordant),
        "mean_spl_delta_right_minus_left": mean([
            right[key]["spl_b"] - left[key]["spl_b"] for key in eligible]),
        "mean_path_delta_m_right_minus_left": mean([
            right[key]["path_b"] - left[key]["path_b"] for key in eligible]),
        "common_activation": {
            "eligible": sum(common_active_outcomes.values()),
            "outcomes": common_active_outcomes,
        },
        "activation_divergence": activation_divergence,
        "safety": {
            "new_stuck_failure_count": len(new_stuck_failures),
            "new_stuck_failures": new_stuck_failures,
            "left_blocked_steps": sum(left[key]["blocked_steps_b"]
                                      for key in eligible),
            "right_blocked_steps": sum(right[key]["blocked_steps_b"]
                                       for key in eligible),
        },
        "episodes": episodes,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text())
    scenes = manifest["selection"]["selected_scenes"]
    expected = {
        (scene, record["episode"])
        for scene in scenes for record in manifest["episodes"][scene]
    }
    rows = {arm: {} for arm in ARMS}
    for index, scene in enumerate(scenes):
        scene_root = args.run_root / "scenes" / f"{index:02d}_{scene}"
        for arm in ARMS:
            rows[arm].update(load_gate_arm(scene_root, arm, scene))
    for arm in ARMS:
        require(set(rows[arm]) == expected,
                f"{arm} result keys differ from the frozen manifest")

    comparisons = {
        "x_vs_mixed": paired_revisit_summary(
            "memory_mixed", "memory_xnavdp_point",
            rows["memory_mixed"], rows["memory_xnavdp_point"], expected),
        "x_vs_base_point": paired_revisit_summary(
            "memory_base_point", "memory_xnavdp_point",
            rows["memory_base_point"], rows["memory_xnavdp_point"], expected),
        "mixed_vs_native": paired_revisit_summary(
            "navdp_native", "memory_mixed",
            rows["navdp_native"], rows["memory_mixed"], expected),
    }
    x_vs_mixed = comparisons["x_vs_mixed"]
    x_outcomes = x_vs_mixed["outcomes"]
    x_history_valid = all(
        rows["memory_xnavdp_point"][key][
            "xnavdp_history_contract_valid"] is True
        for key in expected)
    x_checkpoint_coverage_valid = all(
        rows["memory_xnavdp_point"][key][
            "xnavdp_checkpoint_coverage_valid"] is True
        for key in expected)
    net_gain = (x_outcomes["right_only_success"]
                - x_outcomes["left_only_success"])
    no_new_safety_failure = (
        x_vs_mixed["safety"]["new_stuck_failure_count"] == 0
        and x_history_valid
        and x_checkpoint_coverage_valid)

    print(json.dumps({
        "audit": {
            "status": "ok",
            "scenes": len(scenes),
            "episodes": len(expected),
            "shared_goal_a_trace_match": True,
            "deterministic_seed_echo_match": True,
            "xnavdp_history_contract_valid": x_history_valid,
            "xnavdp_checkpoint_coverage_valid": x_checkpoint_coverage_valid,
        },
        "arms": {
            arm: arm_summary([rows[arm][key] for key in sorted(expected)])
            for arm in ARMS
        },
        "pairwise": comparisons,
        "frozen_g1_decision": {
            "criterion": (
                "x_vs_mixed net paired Revisit-B gain > 0, no new paired "
                "stuck termination, and zero X history/response/checkpoint-"
                "coverage invalidity"),
            "net_gain_episodes": net_gain,
            "no_new_safety_failure": no_new_safety_failure,
            "advance_to_scene_disjoint_g2": (
                net_gain > 0 and no_new_safety_failure),
        },
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
