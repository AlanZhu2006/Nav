#!/usr/bin/env python3
"""Evaluation-only diagnostics for scale-free navigation bearings.

Nothing in this module is a controller input.  It converts a deployed
``[forward, left]`` vector and Habitat's first shortest-path segment into an
angular error after a policy response has already been produced.
"""

from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np


def _wrap_angle(angle: float) -> float:
    return (float(angle) + math.pi) % (2.0 * math.pi) - math.pi


def _yaw_facing(delta_xz: Sequence[float]) -> float:
    delta = np.asarray(delta_xz, dtype=np.float64)
    return float(np.arctan2(-delta[0], -delta[1]))


def bearing_error_deg_from_world_delta(
        predicted_forward_left: Sequence[float],
        delta_xz: Sequence[float],
        robot_yaw: float) -> float:
    """Return angle error against one world-frame direction segment."""
    predicted = np.asarray(predicted_forward_left, dtype=np.float64)
    delta = np.asarray(delta_xz, dtype=np.float64)
    yaw = float(robot_yaw)
    if (predicted.shape != (2,) or delta.shape != (2,)
            or not np.isfinite(predicted).all()
            or not np.isfinite(delta).all() or not math.isfinite(yaw)):
        raise ValueError("bearing diagnostic inputs must be finite 2-vectors")
    predicted_norm = float(np.linalg.norm(predicted))
    delta_norm = float(np.linalg.norm(delta))
    if predicted_norm <= 1e-12 or delta_norm <= 1e-12:
        raise ValueError("bearing diagnostic vectors must be non-zero")
    relative_yaw = _wrap_angle(_yaw_facing(delta) - yaw)
    target = np.asarray(
        [math.cos(relative_yaw), math.sin(relative_yaw)], dtype=np.float64)
    cosine = float(np.clip(
        np.dot(predicted / predicted_norm, target), -1.0, 1.0))
    return float(math.degrees(math.acos(cosine)))


def evaluation_geodesic_bearing_error_deg(
        pathfinder: Any,
        position_xyz: Sequence[float],
        robot_yaw: float,
        goal_xz: Sequence[float],
        predicted_forward_left: Sequence[float] | None) -> float | None:
    """Score a bearing against Habitat's first non-degenerate path segment.

    Ground truth is queried only after the policy response. ``None`` means no
    valid segment was available; formal verification rejects a missing score
    on an accepted learned plan.
    """
    if predicted_forward_left is None:
        return None
    position = np.asarray(position_xyz, dtype=np.float64)
    goal = np.asarray(goal_xz, dtype=np.float64)
    if (position.shape != (3,) or goal.shape != (2,)
            or not np.isfinite(position).all()
            or not np.isfinite(goal).all()):
        return None
    try:
        import habitat_sim

        path = habitat_sim.ShortestPath()
        path.requested_start = position
        path.requested_end = np.asarray(
            [goal[0], position[1], goal[1]], dtype=np.float64)
        if not pathfinder.find_path(path):
            return None
        points = path.points
    except Exception:
        return None
    start_xz = position[[0, 2]]
    for point in points[1:]:
        delta = np.asarray(point, dtype=np.float64)[[0, 2]] - start_xz
        if float(np.linalg.norm(delta)) > 1e-6:
            try:
                return bearing_error_deg_from_world_delta(
                    predicted_forward_left, delta, robot_yaw)
            except ValueError:
                return None
    return None
