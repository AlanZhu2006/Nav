"""Pure helpers for the causal conditional-C three-leg diagnostic."""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np


CONDITIONAL_C_MODES = (
    "auto",
    "native",
    "geometry_top1",
    "geometry_topk",
    "oracle_anchor",
    "oracle_point",
)


def infer_mode(requested: str, server_backend: str,
               router_verify_top_k: int) -> str:
    """Resolve an explicit mode or infer the paired arm from CLI settings."""
    if requested not in CONDITIONAL_C_MODES:
        raise ValueError(f"unknown conditional-C mode: {requested}")
    if requested != "auto":
        return requested
    if server_backend == "navdp":
        return "native"
    if server_backend != "hybrid_pose":
        raise ValueError(
            "automatic conditional-C mode requires navdp or hybrid_pose")
    if router_verify_top_k == 1:
        return "geometry_top1"
    if router_verify_top_k > 1:
        return "geometry_topk"
    raise ValueError("router_verify_top_k must be positive")


def prefix_last_frame(switches: Sequence[int], n_frames: int) -> int:
    """Return the inclusive last A/B-prefix frame immediately before Goal C."""
    if len(switches) != 2:
        raise ValueError("conditional-C requires exactly two goal switches")
    switch_b = int(switches[1])
    if not 0 < switch_b < int(n_frames):
        raise ValueError("Goal-C switch lies outside the episode")
    return switch_b - 1


def world_goal_to_local(goal_xz: Sequence[float], current_xz: Sequence[float],
                        yaw: float) -> np.ndarray:
    """Convert a Habitat world goal to NavDP's ``[forward, left]`` frame.

    This is the exact inverse of ``eval_2leg_habitat.waypoints_to_world``:

    ``dx = -forward*sin(yaw) - left*cos(yaw)``
    ``dz = -forward*cos(yaw) + left*sin(yaw)``.
    """
    goal = np.asarray(goal_xz, dtype=np.float64)
    current = np.asarray(current_xz, dtype=np.float64)
    if goal.shape != (2,) or current.shape != (2,):
        raise ValueError("goal_xz and current_xz must both have shape (2,)")
    if not np.isfinite(goal).all() or not np.isfinite(current).all():
        raise ValueError("goal/current coordinates must be finite")
    if not math.isfinite(float(yaw)):
        raise ValueError("yaw must be finite")
    dx, dz = goal - current
    sine, cosine = math.sin(float(yaw)), math.cos(float(yaw))
    return np.asarray([
        -sine * dx - cosine * dz,
        -cosine * dx + sine * dz,
    ], dtype=np.float64)
