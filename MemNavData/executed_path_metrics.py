"""Offline planar path measurement from per-action simulator positions.

These evaluators record the pose BEFORE each action.  The terminal position
is therefore required for an exact total; a missing final action is never
silently treated as zero.  This module does not change success labels.
"""

from __future__ import annotations

import math


def planar_path_length(poses, *, steps, end_position=None):
    """Measure actions [0, steps), including a separately recorded endpoint.

    An observation at ``step == steps`` also supplies the endpoint, e.g. when
    a longer diagnostic rollout is truncated at its first successful step.
    Returns None if the only missing observation is the terminal position.
    """
    if steps < 0 or steps != int(steps):
        raise ValueError("steps must be a nonnegative integer")
    prefix = [pose for pose in poses if int(pose["step"]) <= steps]
    if [pose["step"] for pose in prefix] not in (
        list(range(steps)), list(range(steps + 1))
    ):
        raise ValueError("trace must contain every action's pre-action pose")
    if steps == 0:
        return 0.0
    points = [(float(p["x"]), float(p["z"])) for p in prefix]
    if len(prefix) == steps:
        if end_position is None:
            return None
        points.append((float(end_position[0]), float(end_position[2])))
    if not all(math.isfinite(c) for point in points for c in point):
        raise ValueError("non-finite trajectory position")
    return math.fsum(math.dist(a, b) for a, b in zip(points, points[1:]))


def spl(success, geodesic_m, path_m):
    if success not in (0, 1):
        raise ValueError("success must be binary")
    if not all(math.isfinite(v) and v >= 0 for v in (geodesic_m, path_m)):
        raise ValueError("distances must be finite and nonnegative")
    if not success:
        return 0.0
    denominator = max(geodesic_m, path_m)
    return geodesic_m / denominator if denominator else 1.0


def spl_from_trace(success, geodesic_m, poses, *, steps,
                   end_position=None, missing_action_bound_m=None):
    """Exact SPL, or a declared interval if the final position was not saved.

    The optional bound is a physical upper bound on ONE final displacement,
    not an estimated endpoint.  The caller must justify it from the frozen
    executor.  No interval is fabricated when more than one pose is missing.
    """
    measured = planar_path_length(
        poses, steps=steps, end_position=end_position)
    if measured is not None:
        value = spl(success, geodesic_m, measured)
        return {"path_m": measured, "spl": value,
                "spl_lower": value, "spl_upper": value,
                "terminal_position_available": True}
    observed = planar_path_length(poses, steps=steps - 1)
    if missing_action_bound_m is None:
        lower = upper = None
    else:
        if not math.isfinite(missing_action_bound_m) or missing_action_bound_m < 0:
            raise ValueError("missing-action bound must be finite and nonnegative")
        lower = spl(success, geodesic_m, observed + missing_action_bound_m)
        upper = spl(success, geodesic_m, observed)
    return {"path_m": None, "observed_path_m": observed,
            "spl": 0.0 if not success else None,
            "spl_lower": 0.0 if not success else lower,
            "spl_upper": 0.0 if not success else upper,
            "terminal_position_available": False}
