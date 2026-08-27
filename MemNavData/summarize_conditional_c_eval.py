#!/usr/bin/env python3
"""Aggregate the five-arm conditional-C diagnostic across frozen scenes."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

try:
    from MemNavData.summarize_expanded_navdp_router_eval import (
        exact_sign_p,
        mean,
        wilson,
    )
except ModuleNotFoundError:  # direct ``python MemNavData/...py`` execution
    from summarize_expanded_navdp_router_eval import exact_sign_p, mean, wilson


ARMS = (
    "navdp_native",
    "geometry_top1",
    "geometry_router",
    "oracle_anchor",
    "oracle_point",
)
EXPECTED_MODES = {
    "navdp_native": "native",
    "geometry_top1": "geometry_top1",
    "geometry_router": "geometry_topk",
    "oracle_anchor": "oracle_anchor",
    "oracle_point": "oracle_point",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def truth(value) -> bool:
    """Parse the explicit boolean encodings emitted by CSV writers.

    Python's ``csv`` module returns strings, and different evaluators have
    historically written either ``0/1`` or ``False/True``.  Accept only those
    exact semantic forms; values such as ``2``, ``yes`` or NaN must not be
    silently treated as true in an audited summary.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        require(math.isfinite(float(value)) and float(value) in (0.0, 1.0),
                f"invalid boolean value: {value!r}")
        return bool(int(value))
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in ("true", "1"):
            return True
        if normalized in ("false", "0"):
            return False
    raise RuntimeError(f"invalid boolean value: {value!r}")


def load_arm(scene_root: Path, scene: str, arm: str) -> dict:
    metric_path = scene_root / arm / "metric.csv"
    require(metric_path.is_file(), f"missing metric file: {metric_path}")
    with metric_path.open(newline="") as handle:
        records = list(csv.DictReader(handle))
    output = {}
    for record in records:
        require(record["mode"] == EXPECTED_MODES[arm],
                f"conditional mode mismatch: {scene} {arm}")
        episode = record["episode"]
        plan_path = scene_root / arm / f"{episode}_plans.json"
        require(plan_path.is_file(), f"missing plans: {plan_path}")
        plan = json.loads(plan_path.read_text())
        require(plan.get("protocol") ==
                "conditional_C_after_causal_source_AB_replay",
                f"protocol mismatch: {plan_path}")
        deterministic_plan_seeds = truth(
            record.get("deterministic_plan_seeds"))
        if deterministic_plan_seeds:
            for decision in plan.get("legC", []):
                requested = decision.get("requested_diffusion_seed")
                echoed = decision.get("diffusion_seed")
                require(requested is not None and echoed is not None,
                        f"missing conditional-C plan seed: {plan_path}")
                require(int(requested) == int(echoed),
                        f"conditional-C seed echo mismatch: {plan_path}")
        key = (scene, episode)
        require(key not in output, f"duplicate conditional-C row: {key}")
        output[key] = {
            "scene": scene,
            "episode": episode,
            "seed": int(record["seed"]),
            "success": truth(record["reached_C"]),
            "spl": float(record["spl_C"]),
            "geodesic": float(record["geo_C"]),
            "path": float(record["len_C"]),
            "steps": int(record["steps_C"]),
            "final_distance": float(record["final_dist_C"]),
            "prefix_last_frame": int(record["prefix_last_source_frame"]),
            "prefix_source_frames": int(record["prefix_source_frames"]),
            "memory_prefix_frames": int(record["memory_prefix_frames"]),
            "navdp_prefix_decision_frames": int(
                record.get("navdp_prefix_decision_frames") or 0),
            "recall_gap": int(record["c_recall_gap"]),
            "gt_anchor": int(record["c_gt_covis_anchor"]),
            "router_active": truth(record["router_active_episode"]),
            "deterministic_plan_seeds": deterministic_plan_seeds,
            "candidate_gap": (
                None if record.get("retrieval_candidate_min_gap") in (None, "")
                else int(record["retrieval_candidate_min_gap"])),
            "graph_spacing_m": float(
                record.get("graph_subgoal_spacing_m") or 0.0),
            "graph_arrival_m": (
                None if record.get("graph_subgoal_arrival_m") in (None, "")
                else float(record["graph_subgoal_arrival_m"])),
        }
    return output


def summarize_rows(rows: list[dict]) -> dict:
    successes = sum(row["success"] for row in rows)
    return {
        "episodes": len(rows),
        "successes": successes,
        "conditional_C_SR": successes / len(rows),
        "wilson_95": wilson(successes, len(rows)),
        "mean_SPL_C": mean([row["spl"] for row in rows]),
        "mean_final_distance_m": mean([
            row["final_distance"] for row in rows]),
        "mean_steps": mean([float(row["steps"]) for row in rows]),
        "router_activation_rate": mean([
            float(row["router_active"]) for row in rows]),
    }


def compare(left_name: str, right_name: str, left: dict, right: dict,
            expected: set[tuple[str, str]]) -> dict:
    outcomes = {"both": 0, "left_only": 0, "right_only": 0, "neither": 0}
    for key in expected:
        a, b = left[key], right[key]
        for field in ("seed", "prefix_last_frame", "prefix_source_frames",
                      "navdp_prefix_decision_frames", "recall_gap", "gt_anchor"):
            require(a[field] == b[field],
                    f"paired {field} mismatch: {left_name} {right_name} {key}")
        require(math.isclose(a["geodesic"], b["geodesic"], abs_tol=1e-9),
                f"paired geodesic mismatch: {left_name} {right_name} {key}")
        if a["success"] and b["success"]:
            outcomes["both"] += 1
        elif a["success"]:
            outcomes["left_only"] += 1
        elif b["success"]:
            outcomes["right_only"] += 1
        else:
            outcomes["neither"] += 1
    discordant = outcomes["left_only"] + outcomes["right_only"]
    return {
        "left": left_name,
        "right": right_name,
        "outcomes": outcomes,
        "conditional_C_SR_delta_right_minus_left": (
            outcomes["right_only"] - outcomes["left_only"]
        ) / len(expected),
        "mcnemar_exact_two_sided_p": exact_sign_p(
            outcomes["right_only"], discordant),
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
            rows[arm].update(load_arm(scene_root, scene, arm))
    for arm in ARMS:
        require(set(rows[arm]) == expected,
                f"{arm} result keys differ from manifest")

    comparisons = {
        "top1_vs_native": compare(
            "navdp_native", "geometry_top1",
            rows["navdp_native"], rows["geometry_top1"], expected),
        "topk_vs_native": compare(
            "navdp_native", "geometry_router",
            rows["navdp_native"], rows["geometry_router"], expected),
        "topk_vs_top1": compare(
            "geometry_top1", "geometry_router",
            rows["geometry_top1"], rows["geometry_router"], expected),
        "oracle_anchor_vs_topk": compare(
            "geometry_router", "oracle_anchor",
            rows["geometry_router"], rows["oracle_anchor"], expected),
        "oracle_point_vs_oracle_anchor": compare(
            "oracle_anchor", "oracle_point",
            rows["oracle_anchor"], rows["oracle_point"], expected),
    }
    report = {
        "audit": {
            "status": "ok",
            "protocol": "conditional_C_after_causal_source_AB_replay",
            "diagnostic_not_end_to_end_sr": True,
            "scenes": len(scenes),
            "episodes": len(expected),
            "parameters_frozen_before_run": True,
        },
        "arms": {
            arm: summarize_rows([rows[arm][key] for key in sorted(expected)])
            for arm in ARMS
        },
        "pairwise": comparisons,
        "per_scene": {
            scene: {
                arm: int(rows[arm][(scene, record["episode"])]["success"])
                for arm in ARMS
            }
            for scene in scenes
            for record in manifest["episodes"][scene]
        },
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
