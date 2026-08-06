#!/usr/bin/env python3
"""Summarize paired native, top-1, and temporal top-K Habitat results."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any


ARMS = ("navdp_native", "geometry_top1", "geometry_router")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def mean(values: list[float]) -> float | None:
    return float(statistics.fmean(values)) if values else None


def percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = quantile * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    weight = position - lower
    return float(ordered[lower] * (1.0 - weight) + ordered[upper] * weight)


def wilson(successes: int, total: int, z: float = 1.959963984540054) -> list[float | None]:
    if total == 0:
        return [None, None]
    rate = successes / total
    denominator = 1.0 + z * z / total
    center = (rate + z * z / (2.0 * total)) / denominator
    radius = z * math.sqrt(
        rate * (1.0 - rate) / total + z * z / (4.0 * total * total)
    ) / denominator
    return [center - radius, center + radius]


def exact_sign_p(successes: int, trials: int) -> float:
    if trials == 0:
        return 1.0
    tail = min(successes, trials - successes)
    return min(1.0, 2.0 * sum(
        math.comb(trials, index) for index in range(tail + 1)
    ) / (2 ** trials))


def truth(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None or value == "":
        return False
    return bool(int(float(value)))


def load_arm(scene_root: Path, arm: str, scene: str) -> dict[tuple[str, str], dict]:
    arm_root = scene_root / arm
    metric_path = arm_root / "metric.csv"
    require(metric_path.is_file(), f"missing metric file: {metric_path}")
    with metric_path.open(newline="") as handle:
        metrics = list(csv.DictReader(handle))
    output = {}
    for metric in metrics:
        episode = metric["episode"]
        plans_path = arm_root / f"{episode}_plans.json"
        require(plans_path.is_file(), f"missing plan trace: {plans_path}")
        plans = json.loads(plans_path.read_text())
        leg_a = plans.get("legA", [])
        leg_b = plans.get("legB", [])
        deterministic_plan_seeds = truth(
            metric.get("deterministic_plan_seeds"))
        if deterministic_plan_seeds:
            for plan in leg_a + leg_b:
                requested = plan.get("requested_diffusion_seed")
                echoed = plan.get("diffusion_seed")
                require(requested is not None and echoed is not None,
                        f"missing deterministic plan seed: {arm} {scene} {episode}")
                require(int(requested) == int(echoed),
                        f"diffusion seed echo mismatch: {arm} {scene} {episode}")
        active_a = [truth(plan.get("router_active")) for plan in leg_a]
        active_b = [truth(plan.get("router_active")) for plan in leg_b]
        verify_ms = []
        selected_ranks = []
        for plan in leg_a + leg_b:
            value = plan.get("router_verification_total_ms")
            if value in (None, ""):
                value = plan.get("router_overlap_verification_ms")
            if value not in (None, "") and float(value) > 0.0:
                verify_ms.append(float(value))
            rank = plan.get("router_selected_candidate_rank")
            if rank not in (None, ""):
                selected_ranks.append(int(rank))
        key = (scene, episode)
        require(key not in output, f"duplicate metric row: {arm} {key}")
        output[key] = {
            "scene": scene,
            "episode": episode,
            "seed": int(metric["seed"]),
            "recall_gap": int(float(metric["recall_gap"])),
            "reached_a": truth(metric["reached_A"]),
            "reached_b": truth(metric["reached_B"]),
            "joint": truth(metric["reached_A"]) and truth(metric["reached_B"]),
            "spl_a": float(metric["spl_A"]),
            "spl_b": float(metric["spl_B"]),
            "geo_a": float(metric["geo_A"]),
            "geo_b": float(metric["geo_B"]),
            "path_a": float(metric["len_A"]),
            "path_b": float(metric["len_B"]),
            "final_dist_a": float(metric["final_dist_A"]),
            "final_dist_b": float(metric["terminal_final_goal_dist_m"]),
            "steps_a": int(metric["steps_A"]),
            "steps_b": int(metric["steps_B"]),
            "router_plans_a": len(active_a),
            "router_plans_b": len(active_b),
            "router_active_plans_a": sum(active_a),
            "router_active_plans_b": sum(active_b),
            "router_active_episode_a": any(active_a),
            "router_active_episode_b": any(active_b),
            "geometry_verification_ms": verify_ms,
            "selected_candidate_ranks": selected_ranks,
            "leg1_trace_sha256": (
                metric.get("leg1_trace_sha256")
                or plans.get("leg1_trace_sha256")
                or None),
            "deterministic_plan_seeds": deterministic_plan_seeds,
        }
    return output


def arm_summary(rows: list[dict]) -> dict:
    novel_success = sum(row["reached_a"] for row in rows)
    conditional = [row for row in rows if row["reached_a"]]
    revisit_success = sum(row["reached_b"] for row in conditional)
    joint_success = sum(row["joint"] for row in rows)
    verification = [value for row in rows for value in row["geometry_verification_ms"]]
    selected_ranks = [
        value for row in rows for value in row.get("selected_candidate_ranks", [])
    ]
    return {
        "episodes": len(rows),
        "novel": {
            "successes": novel_success,
            "sr": novel_success / len(rows),
            "wilson_95": wilson(novel_success, len(rows)),
            "mean_spl": mean([row["spl_a"] for row in rows]),
            "mean_final_distance_m": mean([row["final_dist_a"] for row in rows]),
        },
        "revisit_given_novel_success": {
            "eligible": len(conditional),
            "successes": revisit_success,
            "sr": revisit_success / len(conditional) if conditional else None,
            "wilson_95": wilson(revisit_success, len(conditional)),
            "mean_spl": mean([row["spl_b"] for row in conditional]),
            "mean_final_distance_m": mean([row["final_dist_b"] for row in conditional]),
        },
        "joint": {
            "successes": joint_success,
            "sr": joint_success / len(rows),
            "wilson_95": wilson(joint_success, len(rows)),
        },
        "router": {
            "novel_false_activation_episodes": sum(
                row["router_active_episode_a"] for row in rows),
            "novel_false_activation_rate": mean([
                float(row["router_active_episode_a"]) for row in rows
            ]),
            "revisit_activation_episodes": sum(
                row["router_active_episode_b"] for row in conditional),
            "revisit_activation_rate": mean([
                float(row["router_active_episode_b"]) for row in conditional
            ]),
            "mean_geometry_verification_ms": mean(verification),
            "p50_geometry_verification_ms": percentile(verification, 0.50),
            "p95_geometry_verification_ms": percentile(verification, 0.95),
            "selected_candidate_rank_p50": percentile(selected_ranks, 0.50),
            "selected_candidate_rank_p95": percentile(selected_ranks, 0.95),
            "selected_candidate_rank_max": max(selected_ranks, default=None),
        },
    }


def paired_summary(left_name: str, right_name: str,
                   left: dict[tuple[str, str], dict],
                   right: dict[tuple[str, str], dict],
                   expected: set[tuple[str, str]]) -> dict:
    outcomes = {
        "both_joint_success": 0,
        "left_only_joint_success": 0,
        "right_only_joint_success": 0,
        "neither_joint_success": 0,
    }
    episodes = []
    for key in sorted(expected):
        left_row = left[key]
        right_row = right[key]
        require(left_row["seed"] == right_row["seed"],
                f"paired seed mismatch: {left_name} {right_name} {key}")
        require(left_row["recall_gap"] == right_row["recall_gap"],
                f"gap mismatch: {left_name} {right_name} {key}")
        require(math.isclose(left_row["geo_a"], right_row["geo_a"], abs_tol=1e-9),
                f"Goal-A geodesic mismatch: {left_name} {right_name} {key}")
        require(math.isclose(left_row["geo_b"], right_row["geo_b"], abs_tol=1e-9),
                f"Goal-B geodesic mismatch: {left_name} {right_name} {key}")
        left_trace = left_row.get("leg1_trace_sha256")
        right_trace = right_row.get("leg1_trace_sha256")
        if left_trace is not None or right_trace is not None:
            require(left_trace is not None and right_trace is not None,
                    f"shared Goal-A trace missing: {left_name} {right_name} {key}")
            require(left_trace == right_trace,
                    f"shared Goal-A trace mismatch: {left_name} {right_name} {key}")
            require(left_row.get("deterministic_plan_seeds") is True
                    and right_row.get("deterministic_plan_seeds") is True,
                    f"paired plan seeding disabled: {left_name} {right_name} {key}")
            require(left_row["reached_a"] == right_row["reached_a"],
                    f"shared Goal-A outcome mismatch: {left_name} {right_name} {key}")
            require(left_row["steps_a"] == right_row["steps_a"],
                    f"shared Goal-A step mismatch: {left_name} {right_name} {key}")
            for field in ("spl_a", "path_a", "final_dist_a"):
                require(math.isclose(
                    left_row[field], right_row[field], abs_tol=1e-9),
                    f"shared Goal-A {field} mismatch: "
                    f"{left_name} {right_name} {key}")
        if left_row["joint"] and right_row["joint"]:
            outcome = "both_joint_success"
        elif left_row["joint"]:
            outcome = "left_only_joint_success"
        elif right_row["joint"]:
            outcome = "right_only_joint_success"
        else:
            outcome = "neither_joint_success"
        outcomes[outcome] += 1
        episodes.append({
            "scene": key[0],
            "episode": key[1],
            "recall_gap": left_row["recall_gap"],
            "outcome": outcome,
            "left_reached_a": left_row["reached_a"],
            "right_reached_a": right_row["reached_a"],
            "left_reached_b": left_row["reached_b"],
            "right_reached_b": right_row["reached_b"],
            "left_router_active_b": left_row["router_active_episode_b"],
            "right_router_active_b": right_row["router_active_episode_b"],
        })
    discordant = (
        outcomes["left_only_joint_success"]
        + outcomes["right_only_joint_success"]
    )
    return {
        "left": left_name,
        "right": right_name,
        "outcomes": outcomes,
        "joint_sr_delta_right_minus_left": (
            outcomes["right_only_joint_success"]
            - outcomes["left_only_joint_success"]
        ) / len(expected),
        "mcnemar_exact_two_sided_p": exact_sign_p(
            outcomes["right_only_joint_success"], discordant
        ),
        "episodes_by_recall_gap": sorted(
            episodes, key=lambda row: row["recall_gap"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text())
    scenes = manifest["selection"]["selected_scenes"]
    episode_ids = {
        scene: [record["episode"] for record in manifest["episodes"][scene]]
        for scene in scenes
    }
    expected = {
        (scene, episode) for scene in scenes for episode in episode_ids[scene]
    }
    rows = {arm: {} for arm in ARMS}
    for index, scene in enumerate(scenes):
        scene_root = args.run_root / "scenes" / f"{index:02d}_{scene}"
        for arm in ARMS:
            rows[arm].update(load_arm(scene_root, arm, scene))
    for arm in ARMS:
        require(set(rows[arm]) == expected, f"{arm} result keys differ from manifest")

    comparisons = {
        "top1_vs_native": paired_summary(
            "navdp_native", "geometry_top1",
            rows["navdp_native"], rows["geometry_top1"], expected),
        "topk_vs_native": paired_summary(
            "navdp_native", "geometry_router",
            rows["navdp_native"], rows["geometry_router"], expected),
        "topk_vs_top1": paired_summary(
            "geometry_top1", "geometry_router",
            rows["geometry_top1"], rows["geometry_router"], expected),
    }
    per_scene = {}
    for scene in scenes:
        scene_keys = [(scene, episode) for episode in episode_ids[scene]]
        per_scene[scene] = {
            arm: {
                "novel_sr": mean([float(rows[arm][key]["reached_a"]) for key in scene_keys]),
                "joint_sr": mean([float(rows[arm][key]["joint"]) for key in scene_keys]),
            }
            for arm in ARMS
        }

    print(json.dumps({
        "audit": {
            "status": "ok",
            "scenes": len(scenes),
            "episodes": len(expected),
            "policy_training_overlap": sorted(
                set(scenes) & set(manifest["training_scenes"])),
            "paired_seed_gap_geodesic_match": True,
        },
        "arms": {
            arm: arm_summary([rows[arm][key] for key in sorted(expected)])
            for arm in ARMS
        },
        "pairwise": comparisons,
        "per_scene": per_scene,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
