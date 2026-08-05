#!/usr/bin/env python3
"""Summarize paired official-NavDP and automatic-memory Habitat results."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any


ARMS = ("navdp_native", "geometry_router")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def mean(values: list[float]) -> float | None:
    return float(statistics.fmean(values)) if values else None


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
        active_a = [truth(plan.get("router_active")) for plan in leg_a]
        active_b = [truth(plan.get("router_active")) for plan in leg_b]
        verify_ms = []
        for plan in leg_a + leg_b:
            value = plan.get("router_verification_total_ms")
            if value in (None, ""):
                value = plan.get("router_overlap_verification_ms")
            if value not in (None, "") and float(value) > 0.0:
                verify_ms.append(float(value))
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
        }
    return output


def arm_summary(rows: list[dict]) -> dict:
    novel_success = sum(row["reached_a"] for row in rows)
    conditional = [row for row in rows if row["reached_a"]]
    revisit_success = sum(row["reached_b"] for row in conditional)
    joint_success = sum(row["joint"] for row in rows)
    verification = [value for row in rows for value in row["geometry_verification_ms"]]
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
        },
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

    paired = []
    outcomes = {
        "both_joint_success": 0,
        "navdp_only_joint_success": 0,
        "geometry_only_joint_success": 0,
        "neither_joint_success": 0,
    }
    for key in sorted(expected):
        native = rows["navdp_native"][key]
        geometry = rows["geometry_router"][key]
        require(native["seed"] == geometry["seed"], f"paired seed mismatch: {key}")
        require(native["recall_gap"] == geometry["recall_gap"], f"gap mismatch: {key}")
        require(math.isclose(native["geo_a"], geometry["geo_a"], abs_tol=1e-9),
                f"Goal-A geodesic mismatch: {key}")
        require(math.isclose(native["geo_b"], geometry["geo_b"], abs_tol=1e-9),
                f"Goal-B geodesic mismatch: {key}")
        if native["joint"] and geometry["joint"]:
            outcome = "both_joint_success"
        elif native["joint"]:
            outcome = "navdp_only_joint_success"
        elif geometry["joint"]:
            outcome = "geometry_only_joint_success"
        else:
            outcome = "neither_joint_success"
        outcomes[outcome] += 1
        paired.append({
            "scene": key[0],
            "episode": key[1],
            "recall_gap": native["recall_gap"],
            "outcome": outcome,
            "navdp_reached_a": native["reached_a"],
            "geometry_reached_a": geometry["reached_a"],
            "navdp_reached_b": native["reached_b"],
            "geometry_reached_b": geometry["reached_b"],
            "geometry_router_active_a": geometry["router_active_episode_a"],
            "geometry_router_active_b": geometry["router_active_episode_b"],
        })

    discordant = (
        outcomes["navdp_only_joint_success"]
        + outcomes["geometry_only_joint_success"]
    )
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
        "paired": {
            "outcomes": outcomes,
            "joint_sr_delta_geometry_minus_navdp": (
                outcomes["geometry_only_joint_success"]
                - outcomes["navdp_only_joint_success"]
            ) / len(expected),
            "mcnemar_exact_two_sided_p": exact_sign_p(
                outcomes["geometry_only_joint_success"], discordant
            ),
        },
        "per_scene": per_scene,
        "episodes_by_recall_gap": sorted(paired, key=lambda row: row["recall_gap"]),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
