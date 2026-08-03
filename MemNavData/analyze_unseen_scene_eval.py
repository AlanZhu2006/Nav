#!/usr/bin/env python3
"""Audit and summarize a paired unseen-scene MemNav Habitat evaluation."""

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


def mean(values: list[float]) -> float:
    return float(statistics.fmean(values)) if values else float("nan")


def median(values: list[float]) -> float:
    return float(statistics.median(values)) if values else float("nan")


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> list[float]:
    if total == 0:
        return [float("nan"), float("nan")]
    rate = successes / total
    denominator = 1.0 + z * z / total
    center = (rate + z * z / (2.0 * total)) / denominator
    radius = (
        z
        * math.sqrt(rate * (1.0 - rate) / total + z * z / (4.0 * total * total))
        / denominator
    )
    return [center - radius, center + radius]


def exact_two_sided_binomial(successes: int, trials: int) -> float:
    """Two-sided sign/binomial p-value for p=0.5, capped at one."""
    if trials == 0:
        return 1.0
    tail = min(successes, trials - successes)
    probability = sum(math.comb(trials, k) for k in range(tail + 1)) / (2**trials)
    return min(1.0, 2.0 * probability)


def load_metrics(run_root: Path, label: str) -> dict[tuple[str, str], dict[str, str]]:
    rows: dict[tuple[str, str], dict[str, str]] = {}
    result_root = run_root / "results" / label
    require(result_root.is_dir(), f"missing result directory: {result_root}")
    for metric_path in sorted(result_root.glob("*/metric.csv")):
        scene = metric_path.parent.name
        with metric_path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                key = (scene, row["episode"])
                require(key not in rows, f"duplicate metric row for {label}: {key}")
                rows[key] = row
    return rows


def load_trace(
    run_root: Path, label: str, scene: str, episode: str
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    plan_path = run_root / "results" / label / scene / f"{episode}_plans.json"
    meta_path = run_root / "episodes" / scene / episode / "meta" / "gen_meta.json"
    require(plan_path.is_file(), f"missing plan trace: {plan_path}")
    require(meta_path.is_file(), f"missing episode metadata: {meta_path}")
    trace = json.loads(plan_path.read_text())
    metadata = json.loads(meta_path.read_text())
    require(isinstance(trace.get("legB"), list), f"missing legB trace: {plan_path}")
    return trace["legB"], metadata


def model_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    successes = sum(int(row["success"]) for row in rows)
    total = len(rows)
    return {
        "episodes": total,
        "successes": successes,
        "sr": successes / total,
        "sr_wilson_95": wilson_interval(successes, total),
        "mean_spl": mean([row["spl"] for row in rows]),
        "mean_final_distance_m": mean([row["final_distance_m"] for row in rows]),
        "median_final_distance_m": median([row["final_distance_m"] for row in rows]),
        "mean_path_m": mean([row["path_m"] for row in rows]),
        "mean_steps": mean([row["steps"] for row in rows]),
        "mean_gate": mean([row["gate"] for row in rows]),
        "mean_plans": mean([row["plan_count"] for row in rows]),
        "median_initial_anchor_abs_error": median(
            [row["initial_anchor_abs_error"] for row in rows]
        ),
        "post_switch_anchor_episodes": sum(row["has_post_switch_anchor"] for row in rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--baseline", default="flowgate2600")
    parser.add_argument("--candidate", default="gatecurr600")
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text())
    scenes = manifest["selection"]["selected_scenes"]
    expected_per_scene = int(manifest["episode_generation"]["episodes_per_scene"])
    expected_keys = {
        (scene, f"episode_{index:04d}")
        for scene in scenes
        for index in range(expected_per_scene)
    }

    raw = {
        args.baseline: load_metrics(args.run_root, args.baseline),
        args.candidate: load_metrics(args.run_root, args.candidate),
    }
    for label, rows in raw.items():
        require(set(rows) == expected_keys, f"{label}: result keys do not match manifest")

    normalized: dict[str, dict[tuple[str, str], dict[str, Any]]] = {
        args.baseline: {},
        args.candidate: {},
    }
    for key in sorted(expected_keys):
        scene, episode = key
        baseline_metric = raw[args.baseline][key]
        candidate_metric = raw[args.candidate][key]
        for field in ("seed", "recall_gap"):
            require(
                baseline_metric[field] == candidate_metric[field],
                f"paired {field} mismatch for {key}",
            )
        require(
            math.isclose(
                float(baseline_metric["geo_B"]),
                float(candidate_metric["geo_B"]),
                rel_tol=0.0,
                abs_tol=1e-9,
            ),
            f"paired geo_B mismatch for {key}",
        )

        for label in (args.baseline, args.candidate):
            metric = raw[label][key]
            plans, metadata = load_trace(args.run_root, label, scene, episode)
            anchors = [int(plan["anchor"]) for plan in plans if plan.get("anchor") is not None]
            require(anchors, f"empty anchor trace for {label} {key}")
            goal = metadata["goals"][0]
            gt_anchor = int(goal["covis_argmax"])
            switch_index = int(metadata["switch_idx"])
            normalized[label][key] = {
                "scene": scene,
                "episode": episode,
                "seed": int(metric["seed"]),
                "recall_gap": int(metric["recall_gap"]),
                "n_frames": int(metadata["n_frames"]),
                "success": bool(int(float(metric["reached_B"]))),
                "spl": float(metric["spl_B"]),
                "final_distance_m": float(metric["terminal_final_goal_dist_m"]),
                "path_m": float(metric["len_B"]),
                "steps": int(metric["steps_B"]),
                "gate": float(metric["gate_B_mean"]),
                "plan_count": len(plans),
                "initial_anchor": anchors[0],
                "gt_covis_argmax": gt_anchor,
                "initial_anchor_abs_error": abs(anchors[0] - gt_anchor),
                "anchor_switches": sum(a != b for a, b in zip(anchors, anchors[1:])),
                "unique_anchors": sorted(set(anchors)),
                "has_post_switch_anchor": any(anchor >= switch_index for anchor in anchors),
            }

    baseline_rows = [normalized[args.baseline][key] for key in sorted(expected_keys)]
    candidate_rows = [normalized[args.candidate][key] for key in sorted(expected_keys)]
    paired_rows = []
    categories = {"both_success": 0, "baseline_only": 0, "candidate_only": 0, "neither": 0}
    candidate_closer = 0
    candidate_farther = 0
    same_initial_anchor = 0
    candidate_only_same_initial_anchor = 0
    candidate_only_gate_deltas = []

    for key in sorted(expected_keys):
        baseline = normalized[args.baseline][key]
        candidate = normalized[args.candidate][key]
        if baseline["success"] and candidate["success"]:
            category = "both_success"
        elif baseline["success"]:
            category = "baseline_only"
        elif candidate["success"]:
            category = "candidate_only"
        else:
            category = "neither"
        categories[category] += 1

        initial_anchor_equal = baseline["initial_anchor"] == candidate["initial_anchor"]
        same_initial_anchor += int(initial_anchor_equal)
        if category == "candidate_only":
            candidate_only_same_initial_anchor += int(initial_anchor_equal)
            candidate_only_gate_deltas.append(candidate["gate"] - baseline["gate"])

        delta_distance = candidate["final_distance_m"] - baseline["final_distance_m"]
        candidate_closer += int(delta_distance < 0.0)
        candidate_farther += int(delta_distance > 0.0)
        paired_rows.append(
            {
                "scene": key[0],
                "episode": key[1],
                "recall_gap": baseline["recall_gap"],
                "baseline_success": baseline["success"],
                "candidate_success": candidate["success"],
                "outcome": category,
                "baseline_spl": baseline["spl"],
                "candidate_spl": candidate["spl"],
                "baseline_final_distance_m": baseline["final_distance_m"],
                "candidate_final_distance_m": candidate["final_distance_m"],
                "final_distance_delta_m": delta_distance,
                "baseline_path_m": baseline["path_m"],
                "candidate_path_m": candidate["path_m"],
                "baseline_gate": baseline["gate"],
                "candidate_gate": candidate["gate"],
                "same_initial_anchor": initial_anchor_equal,
                "baseline_initial_anchor": baseline["initial_anchor"],
                "candidate_initial_anchor": candidate["initial_anchor"],
                "gt_covis_argmax": baseline["gt_covis_argmax"],
                "baseline_post_switch_anchor": baseline["has_post_switch_anchor"],
                "candidate_post_switch_anchor": candidate["has_post_switch_anchor"],
            }
        )

    discordant = categories["baseline_only"] + categories["candidate_only"]
    non_tied_distance = candidate_closer + candidate_farther
    per_scene = {}
    for scene in scenes:
        scene_keys = [key for key in sorted(expected_keys) if key[0] == scene]
        per_scene[scene] = {
            args.baseline: sum(normalized[args.baseline][key]["success"] for key in scene_keys)
            / len(scene_keys),
            args.candidate: sum(normalized[args.candidate][key]["success"] for key in scene_keys)
            / len(scene_keys),
        }

    summary = {
        "audit": {
            "status": "ok",
            "scenes": len(scenes),
            "episodes": len(expected_keys),
            "scene_overlap_with_training": sorted(
                set(scenes) & set(manifest["training_scenes"])
            ),
            "paired_seed_gap_and_geo_match": True,
        },
        "models": {
            args.baseline: model_summary(baseline_rows),
            args.candidate: model_summary(candidate_rows),
        },
        "paired": {
            "outcomes": categories,
            "sr_delta_candidate_minus_baseline": (
                categories["candidate_only"] - categories["baseline_only"]
            )
            / len(expected_keys),
            "mcnemar_exact_two_sided_p": exact_two_sided_binomial(
                categories["candidate_only"], discordant
            ),
            "candidate_closer_final_distance": candidate_closer,
            "candidate_farther_final_distance": candidate_farther,
            "final_distance_sign_exact_two_sided_p": exact_two_sided_binomial(
                candidate_closer, non_tied_distance
            ),
            "mean_final_distance_delta_m": mean(
                [row["final_distance_delta_m"] for row in paired_rows]
            ),
            "median_final_distance_delta_m": median(
                [row["final_distance_delta_m"] for row in paired_rows]
            ),
            "same_initial_anchor": same_initial_anchor,
            "candidate_only_same_initial_anchor": candidate_only_same_initial_anchor,
            "candidate_only_count": categories["candidate_only"],
            "candidate_only_mean_gate_delta": mean(candidate_only_gate_deltas),
        },
        "per_scene_sr": per_scene,
        "episodes_by_recall_gap": sorted(paired_rows, key=lambda row: row["recall_gap"]),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
