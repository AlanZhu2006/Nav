#!/usr/bin/env python3
"""Classify generated-episode batches as pre/post the camera-facing fix.

The old generate_twoleg pursuit_track bug rendered every frame with a yaw
~90 deg off the travel direction (theta-as-yaw).  Batches generated before
the fix have systematically wrong RGB history, covis labels and goal views
and must not be reused for visual-memory experiments.  gen_meta.json does
not record the generator commit, so this audit recovers the property from
the data itself: it compares the camera forward axis stored in the parquet
pose with the frame-to-frame travel direction, using only numpy/pandas
(no Habitat), so it runs directly on HPC login/dtn nodes.

Usage:
  python audit_camera_facing_batches.py ROOT [ROOT ...]

Each ROOT is scanned recursively for episode dirs containing
meta/gen_meta.json + data/chunk-000/episode_000000.parquet.  Verdict per
episode and per root: median absolute yaw-vs-travel offset < 30 deg =>
"post_fix"; within 30 deg of 90 => "pre_fix_90deg"; otherwise "suspect".
"""

from __future__ import annotations

import json
import math
import os
import sys

import numpy as np
import pandas as pd

MIN_STEP_M = 0.03          # ignore near-stationary frames (turn-in-place)
POST_FIX_MAX_DEG = 30.0
PRE_FIX_CENTER_DEG = 90.0
PRE_FIX_TOL_DEG = 30.0


def episode_offset_deg(parquet_path: str) -> float | None:
    rows = pd.read_parquet(parquet_path, columns=["action"])
    poses = np.stack([
        np.stack([np.asarray(r, dtype=np.float64) for r in row])
        for row in rows["action"]])
    if poses.shape[0] < 3 or poses.shape[1:] != (4, 4):
        return None
    positions = poses[:, :3, 3]
    # data frame is Z-up; horizontal plane is (x, y)
    steps = np.diff(positions[:, :2], axis=0)
    lengths = np.linalg.norm(steps, axis=1)
    # camera looks along -Z in its own frame; forward in world = R @ (0,0,-1)
    forward = -poses[:-1, :3, 2][:, :2]
    forward_norm = np.linalg.norm(forward, axis=1)
    keep = (lengths >= MIN_STEP_M) & (forward_norm > 1e-6)
    if keep.sum() < 5:
        return None
    travel_yaw = np.arctan2(steps[keep, 1], steps[keep, 0])
    camera_yaw = np.arctan2(forward[keep, 1], forward[keep, 0])
    offset = np.degrees(
        (camera_yaw - travel_yaw + np.pi) % (2.0 * np.pi) - np.pi)
    return float(np.median(np.abs(offset)))


def verdict(offset_deg: float) -> str:
    if offset_deg < POST_FIX_MAX_DEG:
        return "post_fix"
    if abs(offset_deg - PRE_FIX_CENTER_DEG) <= PRE_FIX_TOL_DEG:
        return "pre_fix_90deg"
    return "suspect"


def scan_root(root: str) -> dict:
    episodes = []
    for dirpath, _dirnames, filenames in os.walk(root):
        if "gen_meta.json" not in filenames:
            continue
        episode_dir = os.path.dirname(dirpath)  # .../episode_x/meta
        parquet = os.path.join(
            episode_dir, "data", "chunk-000", "episode_000000.parquet")
        if not os.path.isfile(parquet):
            continue
        try:
            offset = episode_offset_deg(parquet)
        except Exception as error:  # noqa: BLE001 - batch audit must not abort
            episodes.append({"episode": episode_dir, "offset_deg": None,
                             "verdict": f"read_error:{type(error).__name__}"})
            continue
        if offset is None:
            episodes.append({"episode": episode_dir, "offset_deg": None,
                             "verdict": "too_short"})
            continue
        episodes.append({"episode": episode_dir,
                         "offset_deg": round(offset, 2),
                         "verdict": verdict(offset)})
    counts: dict[str, int] = {}
    for episode in episodes:
        counts[episode["verdict"]] = counts.get(episode["verdict"], 0) + 1
    measured = [e["offset_deg"] for e in episodes
                if e["offset_deg"] is not None]
    usable = (bool(measured)
              and counts.get("pre_fix_90deg", 0) == 0
              and counts.get("suspect", 0) == 0)
    return {
        "root": root,
        "episodes": len(episodes),
        "verdict_counts": counts,
        "median_offset_deg": (round(float(np.median(measured)), 2)
                              if measured else None),
        "batch_usable_for_visual_memory": usable,
        "per_episode": episodes,
    }


def main() -> None:
    roots = sys.argv[1:]
    if not roots:
        print(__doc__)
        raise SystemExit(2)
    report = [scan_root(root) for root in roots]
    for entry in report:
        print(f"{entry['root']}: episodes={entry['episodes']} "
              f"median_offset={entry['median_offset_deg']} "
              f"counts={entry['verdict_counts']} "
              f"usable={entry['batch_usable_for_visual_memory']}")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
