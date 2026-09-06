"""Pure contracts for the consumed long-range local-tangent diagnosis.

This module intentionally consumes evaluator geometry.  It is not a CEC
method variant and is never paper-result eligible.  Its only purpose is to
separate three possible causes of one already-consumed long-range failure:

* a 2.5 m geodesic chord can point through a wall at a local corner;
* the first traversable tangent can supply the missing local turn direction;
* the tangent may be needed only to align heading before native ImageGoal is
  allowed to resume control.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np


TANGENT_SCHEMA_VERSION = "longrange_local_tangent_attribution_v2_20260903"
TANGENT_ARMS = (
    "oracle_chord_mixed_realized",
    "oracle_tangent_mixed_realized",
    "oracle_tangent_then_native_realized",
)
TANGENT_MIN_PLANAR_M = 0.30
TANGENT_ALIGNMENT_THRESHOLD_DEG = 20.0


def _finite_xyz_path(path_points: Sequence[Sequence[float]]) -> np.ndarray:
    points = np.asarray(path_points, dtype=np.float64)
    if points.ndim != 2 or points.shape[0] < 2 or points.shape[1] < 3:
        raise ValueError("geodesic path must contain at least two 3-D points")
    points = points[:, :3]
    if not np.isfinite(points).all():
        raise ValueError("geodesic path contains a non-finite point")
    return points


def first_local_tangent_world(
    path_points: Sequence[Sequence[float]],
    *,
    min_planar_m: float = TANGENT_MIN_PLANAR_M,
) -> tuple[np.ndarray, dict[str, float | int]]:
    """Return the first path point with enough planar baseline.

    The target is selected from the pathfinder's ordered 3-D polyline, but the
    heading is defined in the navigable x/z plane.  We deliberately do not
    integrate to a long lookahead: the returned vector is the local route
    tangent needed to negotiate the next corner.
    """

    points = _finite_xyz_path(path_points)
    threshold = float(min_planar_m)
    if not math.isfinite(threshold) or threshold <= 0.0:
        raise ValueError("min_planar_m must be positive and finite")
    origin = points[0]
    planar = np.linalg.norm(points[:, [0, 2]] - origin[[0, 2]], axis=1)
    eligible = np.flatnonzero(planar >= threshold)
    if eligible.size:
        index = int(eligible[0])
    else:
        nonzero = np.flatnonzero(planar > 1e-8)
        if not nonzero.size:
            raise ValueError("geodesic path has no planar tangent")
        index = int(nonzero[-1])
    target = points[index].copy()
    return target, {
        "tangent_path_index": index,
        "tangent_planar_baseline_m": float(planar[index]),
        "tangent_3d_baseline_m": float(np.linalg.norm(target - origin)),
        "tangent_vertical_delta_m": float(target[1] - origin[1]),
    }


def signed_pointgoal_heading_deg(pointgoal: Sequence[float]) -> float:
    """Heading residual for local NavDP coordinates [forward, left]."""

    point = np.asarray(pointgoal, dtype=np.float64)
    if point.shape != (2,) or not np.isfinite(point).all():
        raise ValueError("pointgoal must be one finite [forward,left] pair")
    if float(np.linalg.norm(point)) <= 1e-8:
        raise ValueError("pointgoal direction is degenerate")
    return float(np.degrees(math.atan2(float(point[1]), float(point[0]))))


def tangent_requires_alignment(
    pointgoal: Sequence[float],
    *,
    threshold_deg: float = TANGENT_ALIGNMENT_THRESHOLD_DEG,
) -> bool:
    threshold = float(threshold_deg)
    if not math.isfinite(threshold) or not 0.0 < threshold < 180.0:
        raise ValueError("threshold_deg must lie strictly between 0 and 180")
    return abs(signed_pointgoal_heading_deg(pointgoal)) > threshold


def realized_planar_translation_m(
    previous_position: Sequence[float], current_position: Sequence[float],
) -> float:
    """Measure executed x/z displacement after pathfinder snapping."""

    previous = np.asarray(previous_position, dtype=np.float64)
    current = np.asarray(current_position, dtype=np.float64)
    if (previous.shape != (3,) or current.shape != (3,)
            or not np.isfinite(previous).all()
            or not np.isfinite(current).all()):
        raise ValueError("realized motion endpoints must be finite xyz poses")
    return float(np.linalg.norm(current[[0, 2]] - previous[[0, 2]]))


def floor_aware_goal_distance_m(
    position: Sequence[float], goal_position: Sequence[float],
) -> tuple[float, float, float]:
    """Return benchmark distance, planar distance, and vertical error."""

    current = np.asarray(position, dtype=np.float64)
    goal = np.asarray(goal_position, dtype=np.float64)
    if (current.shape != (3,) or goal.shape != (3,)
            or not np.isfinite(current).all() or not np.isfinite(goal).all()):
        raise ValueError("goal-distance endpoints must be finite xyz poses")
    planar = float(np.linalg.norm(current[[0, 2]] - goal[[0, 2]]))
    vertical = float(abs(current[1] - goal[1]))
    return float(np.linalg.norm(current - goal)), planar, vertical
