#!/usr/bin/env python3
"""Summarize paired NavDP/router start-A-B-C Habitat evaluations."""

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
        active = {}
        verification_ms = []
        for leg in "ABC":
            leg_plans = plans.get(f"leg{leg}", [])
            active[leg] = [
                truth(plan.get("router_active"))
                for plan in leg_plans
                if plan.get("router_active") is not None
            ]
            for plan in leg_plans:
                value = plan.get("router_verification_total_ms")
                if value in (None, ""):
                    value = plan.get("router_overlap_verification_ms")
                if value not in (None, "") and float(value) > 0.0:
                    verification_ms.append(float(value))
        key = (scene, episode)
        require(key not in output, f"duplicate metric row: {arm} {key}")
        output[key] = {
            "scene": scene,
            "episode": episode,
            "seed": int(metric["seed"]),
            "recall_gap": int(float(metric["c_recall_gap"])),
            "reached_a": truth(metric["reached_A"]),
            "reached_b": truth(metric["reached_B"]),
            "reached_c": truth(metric["reached_C"]),
            "joint": truth(metric["joint_success"]),
            "spl_a": float(metric["spl_A"]),
            "spl_b": float(metric["spl_B"]),
            "spl_c": float(metric["spl_C"]),
            "joint_spl": float(metric["joint_spl"]),
            "geo_a": float(metric["geo_A"]),
            "geo_b": float(metric["geo_B"]),
            "geo_c": float(metric["geo_C"]),
            "final_dist_a": float(metric["final_dist_A"]),
            "final_dist_b": float(metric["final_dist_B"]),
            "final_dist_c": float(metric["final_dist_C"]),
            "router_active_episode_a": any(active["A"]),
            "router_active_episode_b": any(active["B"]),
            "router_active_episode_c": any(active["C"]),
            "router_active_plans_a": sum(active["A"]),
            "router_active_plans_b": sum(active["B"]),
            "router_active_plans_c": sum(active["C"]),
            "geometry_verification_ms": verification_ms,
        }
    return output


def success_block(rows: list[dict], key: str, spl_key: str, dist_key: str) -> dict:
    successes = sum(row[key] for row in rows)
    return {
        "eligible": len(rows),
        "successes": successes,
        "sr": successes / len(rows) if rows else None,
        "wilson_95": wilson(successes, len(rows)),
        "mean_spl": mean([row[spl_key] for row in rows]),
        "mean_final_distance_m": mean([row[dist_key] for row in rows]),
    }


def arm_summary(rows: list[dict]) -> dict:
    after_a = [row for row in rows if row["reached_a"]]
    after_ab = [row for row in after_a if row["reached_b"]]
    verification = [value for row in rows for value in row["geometry_verification_ms"]]
    joint_successes = sum(row["joint"] for row in rows)
    novel_trials = len(rows) + len(after_a)
    novel_successes = sum(row["reached_a"] for row in rows) + sum(
        row["reached_b"] for row in after_a
    )
    return {
        "episodes": len(rows),
        "leg_A_novel": success_block(rows, "reached_a", "spl_a", "final_dist_a"),
        "leg_B_novel_given_A": success_block(
            after_a, "reached_b", "spl_b", "final_dist_b"
        ),
        "novel_legs_combined_conditional": {
            "eligible": novel_trials,
            "successes": novel_successes,
            "sr": novel_successes / novel_trials if novel_trials else None,
            "wilson_95": wilson(novel_successes, novel_trials),
        },
        "leg_C_revisit_given_AB": success_block(
            after_ab, "reached_c", "spl_c", "final_dist_c"
        ),
        "joint": {
            "successes": joint_successes,
            "sr": joint_successes / len(rows),
            "wilson_95": wilson(joint_successes, len(rows)),
            "mean_spl": mean([row["joint_spl"] for row in rows]),
        },
        "router": {
            "novel_A_false_activation_rate": mean([
                float(row["router_active_episode_a"]) for row in rows
            ]),
            "novel_B_false_activation_rate_given_A": mean([
                float(row["router_active_episode_b"]) for row in after_a
            ]),
            "revisit_C_activation_rate_given_AB": mean([
                float(row["router_active_episode_c"]) for row in after_ab
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

    outcomes = {
        "both_joint_success": 0,
        "navdp_only_joint_success": 0,
        "geometry_only_joint_success": 0,
        "neither_joint_success": 0,
    }
    paired = []
    for key in sorted(expected):
        native = rows["navdp_native"][key]
        geometry = rows["geometry_router"][key]
        require(native["seed"] == geometry["seed"], f"paired seed mismatch: {key}")
        require(native["recall_gap"] == geometry["recall_gap"],
                f"recall-gap mismatch: {key}")
        for leg in "abc":
            require(
                math.isclose(native[f"geo_{leg}"], geometry[f"geo_{leg}"], abs_tol=1e-9),
                f"Goal-{leg.upper()} geodesic mismatch: {key}",
            )
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
            "c_recall_gap": native["recall_gap"],
            "outcome": outcome,
            "navdp": {
                "A": native["reached_a"],
                "B": native["reached_b"],
                "C": native["reached_c"],
            },
            "geometry": {
                "A": geometry["reached_a"],
                "B": geometry["reached_b"],
                "C": geometry["reached_c"],
                "router_active_A": geometry["router_active_episode_a"],
                "router_active_B": geometry["router_active_episode_b"],
                "router_active_C": geometry["router_active_episode_c"],
            },
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
                "A": mean([float(rows[arm][key]["reached_a"]) for key in scene_keys]),
                "B": mean([float(rows[arm][key]["reached_b"]) for key in scene_keys]),
                "C": mean([float(rows[arm][key]["reached_c"]) for key in scene_keys]),
                "joint": mean([float(rows[arm][key]["joint"]) for key in scene_keys]),
            }
            for arm in ARMS
        }

    print(json.dumps({
        "audit": {
            "status": "ok",
            "scenes": len(scenes),
            "episodes": len(expected),
            "protocol": "start->A Novel->B Novel->C revisit(A)",
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
        "episodes_by_c_recall_gap": sorted(
            paired, key=lambda row: row["c_recall_gap"]
        ),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
