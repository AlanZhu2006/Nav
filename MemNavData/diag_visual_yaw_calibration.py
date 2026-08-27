"""Calibrate direct visual-yaw confidence on generated episodes.

This diagnostic renders known yaw residuals from navigable points within one
metre of each ImageGoal.  It stores raw SIFT/essential-matrix statistics, then
sweeps only confidence thresholds; no LingBot or navigation checkpoint is used.

Example (Habitat environment):

  python MemNavData/diag_visual_yaw_calibration.py \
    --episode_root memnav_viz/validate_gated/mp3d_2leg \
    --scene_dir /home/asus/Research/datasets/mp3d \
    --out /tmp/visual_yaw_calibration.json
"""

import argparse
import io
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image

from generate_twoleg import M_W, make_sim, render
from visual_yaw_refinement import estimate_visual_yaw


CAMERA_HEIGHT = 0.5
RESIDUAL_DEG = (-35.0, -25.0, -15.0, -8.0, 0.0, 8.0, 15.0, 25.0, 35.0)
RADII_M = (0.15, 0.35, 0.60, 0.85)


def wrap_angle(angle):
    return (float(angle) + math.pi) % (2.0 * math.pi) - math.pi


def jpg_bytes(rgb):
    output = io.BytesIO()
    Image.fromarray(rgb).save(output, format="JPEG", quality=95)
    return output.getvalue()


def camera_intrinsic(ep_dir):
    import pandas as pd
    parquet = ep_dir / "data/chunk-000/episode_000000.parquet"
    raw = pd.read_parquet(parquet).iloc[0]["observation.camera_intrinsic"]
    return np.stack([np.asarray(row, dtype=np.float64) for row in raw])


def nearby_point(pathfinder, goal_floor, radius, bearing):
    desired = np.asarray(goal_floor, dtype=np.float64).copy()
    desired[0] += radius * math.cos(bearing)
    desired[2] += radius * math.sin(bearing)
    snapped = np.asarray(pathfinder.snap_point(desired), dtype=np.float64)
    if not np.isfinite(snapped).all():
        return None
    if np.linalg.norm(snapped[[0, 2]] - desired[[0, 2]]) > 0.15:
        return None
    if np.linalg.norm(snapped[[0, 2]] - goal_floor[[0, 2]]) >= 0.98:
        return None
    return snapped


def percentile(values, q):
    if not values:
        return None
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def threshold_sweep(rows):
    result = []
    for min_inliers in (8, 12, 16, 20, 24, 30):
        for min_ratio in (0.40, 0.50, 0.60, 0.70):
            for max_consensus in (3.0, 5.0, 8.0, 12.0):
                accepted = [
                    row for row in rows
                    if row["yaw_est_deg"] is not None
                    and row["matches"] >= 8
                    and row["inliers"] >= min_inliers
                    and row["inlier_ratio"] >= min_ratio
                    and row["off_axis_deg"] is not None
                    and row["off_axis_deg"] <= 15.0
                    and row["consensus_error_deg"] is not None
                    and row["consensus_error_deg"] <= max_consensus
                ]
                errors = [row["abs_error_deg"] for row in accepted]
                result.append({
                    "min_inliers": min_inliers,
                    "min_inlier_ratio": min_ratio,
                    "max_consensus_error_deg": max_consensus,
                    "accepted": len(accepted),
                    "total": len(rows),
                    "coverage": len(accepted) / len(rows) if rows else 0.0,
                    "mae_deg": float(np.mean(errors)) if errors else None,
                    "p95_abs_error_deg": percentile(errors, 95),
                    "max_abs_error_deg": max(errors) if errors else None,
                    "gross_error_gt5_count": sum(error > 5.0 for error in errors),
                })
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episode_root", type=Path, required=True)
    parser.add_argument("--scene_dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--episodes_per_scene", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260802)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    rows = []
    for scene_root in sorted(path for path in args.episode_root.iterdir()
                             if path.is_dir()):
        scene_path = args.scene_dir / f"{scene_root.name}.glb"
        if not scene_path.is_file():
            continue
        episodes = sorted(scene_root.glob("episode_*"))
        if args.episodes_per_scene:
            episodes = episodes[:args.episodes_per_scene]
        sim = make_sim(str(scene_path), "", agent_radius=0.30)
        try:
            for ep_dir in episodes:
                meta_path = ep_dir / "meta/gen_meta.json"
                goal_path = ep_dir / "goal_1.jpg"
                if not meta_path.is_file() or not goal_path.is_file():
                    continue
                meta = json.loads(meta_path.read_text())
                if meta.get("n_legs", 2) != 2:
                    continue
                goal = meta["goals"][0]
                goal_floor = M_W.T @ np.asarray(goal["pos"], dtype=np.float64)
                goal_yaw = float(goal["yaw_habitat"])
                intrinsic = camera_intrinsic(ep_dir)
                goal_jpg = goal_path.read_bytes()
                for index, residual_deg in enumerate(RESIDUAL_DEG):
                    radius = RADII_M[index % len(RADII_M)]
                    current_floor = None
                    for _attempt in range(12):
                        current_floor = nearby_point(
                            sim.pathfinder, goal_floor, radius,
                            float(rng.uniform(-math.pi, math.pi)))
                        if current_floor is not None:
                            break
                    if current_floor is None:
                        continue
                    current_yaw = wrap_angle(goal_yaw + math.radians(residual_deg))
                    current_rgb, _ = render(
                        sim,
                        current_floor + np.array([0.0, CAMERA_HEIGHT, 0.0]),
                        current_yaw,
                    )
                    estimate = estimate_visual_yaw(
                        jpg_bytes(current_rgb), goal_jpg, intrinsic,
                        min_inliers=0,
                        min_inlier_ratio=0.0,
                        max_off_axis_deg=180.0,
                        max_consensus_error_deg=180.0,
                    )
                    expected = wrap_angle(goal_yaw - current_yaw)
                    measured = estimate.yaw_correction_rad
                    error = (None if measured is None else math.degrees(abs(
                        wrap_angle(measured - expected))))
                    rows.append({
                        "scene": scene_root.name,
                        "episode": ep_dir.name,
                        "radius_m": radius,
                        "residual_deg": residual_deg,
                        "expected_correction_deg": math.degrees(expected),
                        "yaw_est_deg": estimate.yaw_correction_deg,
                        "bearing_est_deg": estimate.bearing_correction_deg,
                        "abs_error_deg": error,
                        "matches": estimate.matches,
                        "inliers": estimate.inliers,
                        "inlier_ratio": estimate.inlier_ratio,
                        "off_axis_deg": estimate.off_axis_deg,
                        "bearing_mad_deg": estimate.bearing_mad_deg,
                        "consensus_error_deg": estimate.consensus_error_deg,
                        "raw_reason": estimate.reason,
                        "current_xz": current_floor[[0, 2]].tolist(),
                    })
        finally:
            sim.close()

    sweep = threshold_sweep(rows)
    safe = [entry for entry in sweep
            if entry["accepted"] > 0
            and entry["gross_error_gt5_count"] == 0
            and entry["p95_abs_error_deg"] <= 2.0]
    recommended = (max(safe, key=lambda entry: (
        entry["coverage"], entry["min_inlier_ratio"], entry["min_inliers"]))
                   if safe else None)
    report = {
        "episode_root": str(args.episode_root),
        "seed": args.seed,
        "rows": rows,
        "threshold_sweep": sweep,
        "recommended": recommended,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({
        "samples": len(rows),
        "scenes": sorted({row["scene"] for row in rows}),
        "recommended": recommended,
    }, indent=2))


if __name__ == "__main__":
    main()
