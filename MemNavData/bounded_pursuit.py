"""Finite-displacement pursuit commands, independent of maps and goal labels.

The original steering law and limits are retained. The requested displacement
cannot exceed the distance to its selected reference, and a zero displacement
reference does not invent motion. Neither condition declares goal arrival.
Units follow the Habitat evaluator: max_step_m is metres per control tick.
"""
from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class PursuitLimits:
    max_step_m: float = 0.0376
    lookahead_m: float = 0.7
    min_radius_m: float = 0.40
    max_turn_rad: float = math.radians(4.5)


@dataclass(frozen=True)
class PursuitCommand:
    position: np.ndarray
    yaw: float
    displacement_m: float
    nominal_displacement_m: float
    reference_distance_m: float
    reference_index: int
    reason: str


def command(position, yaw, world_path, limits=PursuitLimits()):
    """Return a pre-collision request; never inspect scene geometry.

    This is a bounded version of the existing forward pursuit, not a new
    exploration policy, reverse controller, obstacle planner or STOP detector.
    A reference behind the robot is not silently treated as a reverse command.
    """
    pos = np.asarray(position, dtype=float)
    path = np.asarray(world_path, dtype=float)
    values = [yaw, limits.max_step_m, limits.lookahead_m,
              limits.min_radius_m, limits.max_turn_rad]
    if (pos.shape != (3,) or path.ndim != 2 or path.shape[1:] != (2,)
            or len(path) == 0 or not np.isfinite(pos).all()
            or not np.isfinite(path).all() or not np.isfinite(values).all()):
        raise ValueError("Expected finite pose and a nonempty [K,2] world polyline")
    if min(values[1:]) <= 0:
        raise ValueError("Pursuit motion limits must be positive")
    distance = np.linalg.norm(path - pos[[0, 2]][None, :], axis=1)
    ahead = np.flatnonzero(distance >= limits.lookahead_m)
    index = int(ahead[0]) if len(ahead) else len(path) - 1
    d = float(distance[index])
    # Arithmetic tolerance only, not a learned confidence or arrival threshold.
    if d <= 1e-8:
        return PursuitCommand(pos.copy(), float(yaw), 0.0, 0.0, d, index,
                              "zero_reference_hold_not_arrival")
    delta = path[index] - pos[[0, 2]]
    desired = np.arctan2(-delta[0], -delta[1])
    alpha = (desired - yaw + np.pi) % (2 * np.pi) - np.pi
    nominal = limits.max_step_m * (0.48 + 0.52 * (1 + np.cos(alpha)) / 2)
    travel = min(float(nominal), d)
    curvature = np.clip(2 * alpha / limits.lookahead_m,
                        -1 / limits.min_radius_m, 1 / limits.min_radius_m)
    next_yaw = float(yaw + np.clip(curvature * travel,
                                  -limits.max_turn_rad, limits.max_turn_rad))
    forward = np.array([-np.sin(next_yaw), 0.0, -np.cos(next_yaw)])
    return PursuitCommand(pos + travel * forward, next_yaw, travel,
                          float(nominal), d, index,
                          "reference_limited" if travel < nominal else "tracking")


def limits_from_evaluator(args):
    return PursuitLimits(float(args.v_max), float(args.lookahead),
                         float(args.r_min), math.radians(args.max_turn_deg))


def apply_collision(position, request, step_filter):
    """Delegate one request to the simulator; no snap, retries or escape moves.

    step_filter is explicitly Habitat's NavMesh collision approximation in
    this experiment. This does NOT make the simulator NavMesh-free.
    """
    pos = np.asarray(position, dtype=float)
    if request.displacement_m == 0.0:
        return pos.copy(), request.yaw, 0.0
    actual = np.asarray(step_filter(pos, request.position), dtype=float)
    if actual.shape != (3,) or not np.isfinite(actual).all():
        raise ValueError("Invalid simulator collision response")
    return actual, request.yaw, float(np.linalg.norm((actual - pos)[[0, 2]]))
