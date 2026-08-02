"""Aggregate paired terminal-U-turn evaluator CSVs.

The evaluator records the distance-only baseline at first reach and the extra
terminal maneuver in the same rollout.  This script keeps their denominators
explicit so navigation failures are not mislabeled as terminal failures.
"""

import argparse
import csv
import json
import math
import statistics
from pathlib import Path


def _float(row, key):
    value = row.get(key, "")
    return None if value in (None, "", "None") else float(value)


def _bool(row, key):
    return str(row.get(key, "")).lower() == "true"


def _mean(values):
    values = [v for v in values if v is not None]
    return statistics.fmean(values) if values else None


def _median(values):
    values = [v for v in values if v is not None]
    return statistics.median(values) if values else None


def _percentile(values, q):
    values = sorted(v for v in values if v is not None)
    if not values:
        return None
    at = (len(values) - 1) * q
    lo, hi = math.floor(at), math.ceil(at)
    if lo == hi:
        return values[lo]
    return values[lo] * (hi - at) + values[hi] * (at - lo)


def _wilson(successes, total, z=1.959963984540054):
    if total == 0:
        return None
    p = successes / total
    den = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / den
    half = z * math.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total * total)) / den
    return [center - half, center + half]


def _completed(row):
    if row.get("terminal_completed", "") not in ("", None):
        return _bool(row, "terminal_completed")
    return bool(row.get("terminal_path_type")) and not bool(row.get("terminal_failure"))


def _pose_success(row, success_dist, yaw_tol_deg):
    """Final-pose metric, intentionally independent of path completion."""
    if not _bool(row, "terminal_attempted"):
        return False
    distance = _float(row, "terminal_final_goal_dist_m")
    yaw_error = _float(row, "post_turn_yaw_err_deg")
    if distance is not None and yaw_error is not None:
        return distance < success_dist and yaw_error <= yaw_tol_deg
    # Backward-compatible fallback for result files that predate final-pose
    # fields. New files must use the two numerical criteria above.
    return _bool(row, "terminal_success")


def summarize(rows, success_dist=1.0, yaw_tol_deg=15.0):
    reached = [r for r in rows if _float(r, "reached_B") == 1.0]
    attempted = [r for r in reached if _bool(r, "terminal_attempted")]
    completed = [r for r in attempted if _completed(r)]
    pose_success = [r for r in attempted
                    if _pose_success(r, success_dist, yaw_tol_deg)]
    failed_nav = [r for r in rows if _float(r, "reached_B") == 0.0]

    pre_yaw = [_float(r, "pre_turn_yaw_err_deg") for r in completed]
    post_yaw = [_float(r, "post_turn_yaw_err_deg") for r in completed]
    pre_cos = [_float(r, "pre_turn_goal_cos") for r in completed]
    post_cos = [_float(r, "post_turn_goal_cos") for r in completed]
    cosine_delta = [b - a for a, b in zip(pre_cos, post_cos)
                    if a is not None and b is not None]
    extra_path = [_float(r, "terminal_path_m") for r in completed]

    return {
        "episodes": len(rows),
        "success_dist_m": success_dist,
        "yaw_tolerance_deg": yaw_tol_deg,
        "navigation_reached": len(reached),
        "navigation_sr": len(reached) / len(rows) if rows else None,
        "navigation_sr_wilson95": _wilson(len(reached), len(rows)),
        "official_spl_mean": _mean([_float(r, "spl_B") for r in rows]),
        "terminal_spl_mean": _mean([_float(r, "spl_B_with_terminal") for r in rows]),
        "terminal_attempted": len(attempted),
        "terminal_completed": len(completed),
        "terminal_completion_given_attempt": (
            len(completed) / len(attempted) if attempted else None),
        "terminal_pose_success": len(pose_success),
        "terminal_pose_success_given_attempt": (
            len(pose_success) / len(attempted) if attempted else None),
        "terminal_pose_success_wilson95": _wilson(len(pose_success), len(attempted)),
        "terminal_pose_success_overall": (
            len(pose_success) / len(rows) if rows else None),
        "extra_path_m_mean": _mean(extra_path),
        "extra_path_m_median": _median(extra_path),
        "extra_path_m_range": ([min(extra_path), max(extra_path)] if extra_path else None),
        "pre_yaw_err_deg_median": _median(pre_yaw),
        "post_yaw_err_deg_mean": _mean(post_yaw),
        "post_yaw_err_deg_median": _median(post_yaw),
        "post_yaw_err_deg_p90": _percentile(post_yaw, 0.90),
        "pre_goal_cos_mean": _mean(pre_cos),
        "post_goal_cos_mean": _mean(post_cos),
        "goal_cos_delta_mean": _mean(cosine_delta),
        "goal_cos_improved_count": sum(d > 0.0 for d in cosine_delta),
        "goal_cos_compared_count": len(cosine_delta),
        "gate_mean_reached": _mean([_float(r, "gate_B_mean") for r in reached]),
        "gate_mean_navigation_failed": _mean(
            [_float(r, "gate_B_mean") for r in failed_nav]),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True,
                        help="directory whose scene subdirs contain metric.csv")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--success_dist", type=float, default=1.0)
    parser.add_argument("--yaw_tol_deg", type=float, default=15.0)
    args = parser.parse_args()

    by_scene = {}
    pooled = []
    for metric_path in sorted(args.root.glob("*/metric.csv")):
        rows = list(csv.DictReader(metric_path.open()))
        by_scene[metric_path.parent.name] = summarize(
            rows, args.success_dist, args.yaw_tol_deg)
        pooled.extend(rows)

    result = {
        "root": str(args.root),
        "pooled": summarize(pooled, args.success_dist, args.yaw_tol_deg),
        "scenes": by_scene,
    }
    out = args.out or args.root / "aggregate_summary.json"
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
