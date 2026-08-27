"""Evaluation-only diagnostics for the strict double-Revisit protocol."""

from __future__ import annotations

import math
from typing import Iterable, Mapping

import numpy as np


def online_path_nearest_anchor(
    memory_trace: Iterable[Mapping[str, object]],
    goal_xz: np.ndarray,
    *,
    candidate_ceiling: int,
) -> dict[str, float | int]:
    """Return the causal online-A frame spatially nearest to a later goal.

    This is a privileged evaluation oracle: ``goal_xz`` comes from Habitat.
    The explicit ceiling prevents an intervening rollout from entering the
    candidate set when measuring retention of the initial leg's memory.
    """
    target = np.asarray(goal_xz, dtype=np.float64)
    if target.shape != (2,) or not np.isfinite(target).all():
        raise ValueError("goal_xz must be one finite planar coordinate")
    if int(candidate_ceiling) != candidate_ceiling or candidate_ceiling < 0:
        raise ValueError("candidate_ceiling must be a non-negative integer")

    eligible = []
    seen_frames = set()
    for item in memory_trace:
        if item.get("frame_idx") is None:
            continue
        frame_idx = int(item["frame_idx"])
        if frame_idx in seen_frames:
            raise ValueError("memory_trace contains duplicate frame indices")
        seen_frames.add(frame_idx)
        point = np.asarray([item["x"], item["z"]], dtype=np.float64)
        if point.shape != (2,) or not np.isfinite(point).all():
            raise ValueError("memory_trace contains a non-finite position")
        if frame_idx <= candidate_ceiling:
            distance_m = float(np.linalg.norm(point - target))
            eligible.append((distance_m, frame_idx, point))

    if not eligible:
        raise ValueError("no causal online-path anchor is under the ceiling")
    distance_m, frame_idx, point = min(
        eligible, key=lambda candidate: (candidate[0], candidate[1]))
    if not math.isfinite(distance_m):
        raise ValueError("nearest online-path distance is non-finite")
    return {
        "frame_idx": frame_idx,
        "distance_m": distance_m,
        "x": float(point[0]),
        "z": float(point[1]),
    }
