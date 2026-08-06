"""Pure geometry helpers for privileged global-subgoal diagnostics."""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np


def polyline_subgoal(points: Sequence[Sequence[float]], distance_m: float) -> np.ndarray:
    """Interpolate a point ``distance_m`` along a 3-D path polyline.

    The function is intentionally independent of Habitat so the privileged
    A*/geodesic upper bound can be unit-tested without a simulator.  Repeated
    vertices are ignored and a requested distance beyond the path returns the
    final endpoint.
    """
    path = np.asarray(points, dtype=np.float64)
    if path.ndim != 2 or path.shape[1] != 3 or len(path) == 0:
        raise ValueError("points must have shape [N, 3] with N >= 1")
    if not np.isfinite(path).all():
        raise ValueError("polyline points must be finite")
    distance = float(distance_m)
    if not math.isfinite(distance) or distance <= 0:
        raise ValueError("distance_m must be finite and positive")
    if len(path) == 1:
        return path[0].copy()

    traversed = 0.0
    for start, end in zip(path[:-1], path[1:]):
        delta = end - start
        segment_m = float(np.linalg.norm(delta))
        if segment_m <= 1e-12:
            continue
        if traversed + segment_m >= distance:
            fraction = (distance - traversed) / segment_m
            return start + fraction * delta
        traversed += segment_m
    return path[-1].copy()
