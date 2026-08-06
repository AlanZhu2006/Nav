"""Replay and score shadow-arrival thresholds from saved planning traces."""

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import List, Optional

import numpy as np

try:
    from .arrival_shadow import ArrivalShadowConfig, ArrivalShadowDetector
except ImportError:  # direct script execution
    from arrival_shadow import ArrivalShadowConfig, ArrivalShadowDetector


def plan_paths(root: Path, arm: Optional[str] = None) -> List[Path]:
    paths = sorted(root.glob("**/episode_*_plans.json"))
    if arm is not None:
        paths = [path for path in paths if path.parent.name == arm]
    return paths


def _ratio(numerator: int, denominator: int):
    return float(numerator / denominator) if denominator else None


def _mode_summary(records: List[dict], key: str) -> dict:
    triggered = [row for row in records if row[key] is not None]
    correct = [row for row in triggered if row[key]["gt_arrived"]]
    reached = [row for row in records if row["reached"]]
    correct_reached = [row for row in reached
                       if row[key] is not None and row[key]["gt_arrived"]]
    distances = [row[key]["gt_distance_m"] for row in triggered
                 if row[key]["gt_distance_m"] is not None]
    return {
        "triggered_episodes": len(triggered),
        "correct_first_triggers": len(correct),
        "false_stop_episodes": len(triggered) - len(correct),
        "precision": _ratio(len(correct), len(triggered)),
        "recall_given_gt_arrival": _ratio(len(correct_reached), len(reached)),
        "first_trigger_gt_distance_median": (
            float(np.median(distances)) if distances else None),
        "first_trigger_gt_distance_max": (
            float(np.max(distances)) if distances else None),
    }


def summarize(paths: List[Path], config: ArrivalShadowConfig) -> dict:
    if not paths:
        raise ValueError("no arrival-shadow plan traces found")
    records = []
    reason_counts = Counter()
    critic_available = critic_total = 0
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        plans = payload.get("legB") or []
        detector = ArrivalShadowDetector(config)
        first_pose = first_strict = None
        reached = False
        for plan in plans:
            diagnostic = detector.update(plan, step=plan.get("step"))
            gt_arrived = bool(plan.get("evaluation_gt_arrived"))
            gt_distance = plan.get("evaluation_gt_goal_distance_m")
            gt_distance = (float(gt_distance)
                           if gt_distance is not None else None)
            reached = reached or gt_arrived
            critic_total += 1
            critic_available += int(
                diagnostic["arrival_shadow_critic_available"])
            reason_counts.update(diagnostic["arrival_shadow_reason"].split(","))
            event = {
                "step": plan.get("step"),
                "gt_arrived": gt_arrived,
                "gt_distance_m": gt_distance,
            }
            if diagnostic["arrival_shadow_pose_ready"] and first_pose is None:
                first_pose = event
            if (diagnostic["arrival_shadow_strict_ready"]
                    and first_strict is None):
                first_strict = event
        records.append({
            "path": str(path),
            "reached": reached,
            "pose": first_pose,
            "strict": first_strict,
        })
    return {
        "schema": "memnav.arrival_shadow.v1",
        "gt_usage": "scoring_only",
        "config": {
            "window_plans": config.window_plans,
            "distance_m": config.distance_m,
            "max_distance_mad_m": config.max_distance_mad_m,
            "max_distance_growth_m": config.max_distance_growth_m,
        },
        "episodes": len(records),
        "gt_arrived_episodes": sum(row["reached"] for row in records),
        "critic_availability_rate": _ratio(critic_available, critic_total),
        "pose_consensus": _mode_summary(records, "pose"),
        "pose_plus_navdp_critic": _mode_summary(records, "strict"),
        "reason_plan_counts": dict(sorted(reason_counts.items())),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--arm", default=None)
    parser.add_argument("--window", type=int, default=3)
    parser.add_argument("--distance-m", type=float, default=0.75)
    parser.add_argument("--max-mad-m", type=float, default=0.20)
    parser.add_argument("--max-growth-m", type=float, default=0.15)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    config = ArrivalShadowConfig(
        window_plans=args.window,
        distance_m=args.distance_m,
        max_distance_mad_m=args.max_mad_m,
        max_distance_growth_m=args.max_growth_m,
    )
    report = summarize(plan_paths(args.root, args.arm), config)
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
