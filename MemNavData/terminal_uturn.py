"""Forward-only terminal pose alignment for the MemNav Habitat evaluator.

The generated trajectories use a coupled forward/turning controller with a
minimum turning radius; an in-place yaw correction at an ImageGoal endpoint is
therefore out of distribution.  This module connects the current camera pose to
the goal-image pose with a Dubins path instead.  For the common same-position,
opposite-heading case, the two shortest candidates are mirrored RLR/LRL
"teardrop" U-turns: every yaw change is accompanied by forward translation and
the path returns to the goal position with the desired image orientation.

The implementation is dependency-light (NumPy only).  Habitat's pathfinder is
accepted optionally to reject candidates that leave the navigable surface and
to choose the safer mirrored turn.
"""

from dataclasses import dataclass
import math

import numpy as np


_TAU = 2.0 * math.pi


def wrap_angle(angle):
    """Wrap a scalar angle to ``[-pi, pi)``."""
    return (float(angle) + math.pi) % _TAU - math.pi


def habitat_yaw_to_planar(yaw):
    """Habitat camera yaw -> standard planar heading.

    Standard planar kinematics use forward ``[cos(theta), sin(theta)]`` in
    ``(x,z)``.  Habitat camera forward is ``[-sin(yaw), -cos(yaw)]``.
    """
    return wrap_angle(-math.pi / 2.0 - float(yaw))


def planar_to_habitat_yaw(theta):
    """Inverse of :func:`habitat_yaw_to_planar`."""
    return wrap_angle(-math.pi / 2.0 - float(theta))


def relative_xy_to_world(local_xy, origin_xz, origin_yaw):
    """NavDP ``[forward,left]`` metres -> Habitat world ``[x,z]``."""
    forward, left = np.asarray(local_xy, dtype=np.float64)
    sine, cosine = math.sin(origin_yaw), math.cos(origin_yaw)
    delta = np.array(
        [-forward * sine - left * cosine,
         -forward * cosine + left * sine],
        dtype=np.float64,
    )
    return np.asarray(origin_xz, dtype=np.float64) + delta


def _mod2pi(angle):
    return float(angle) % _TAU


def _lsl(alpha, beta, distance):
    p2 = (2.0 + distance**2 - 2.0 * math.cos(alpha - beta)
          + 2.0 * distance * (math.sin(alpha) - math.sin(beta)))
    if p2 < 0.0:
        return None
    tmp = math.atan2(
        math.cos(beta) - math.cos(alpha),
        distance + math.sin(alpha) - math.sin(beta),
    )
    return (_mod2pi(-alpha + tmp), math.sqrt(p2),
            _mod2pi(beta - tmp), "LSL")


def _rsr(alpha, beta, distance):
    p2 = (2.0 + distance**2 - 2.0 * math.cos(alpha - beta)
          + 2.0 * distance * (math.sin(beta) - math.sin(alpha)))
    if p2 < 0.0:
        return None
    tmp = math.atan2(
        math.cos(alpha) - math.cos(beta),
        distance - math.sin(alpha) + math.sin(beta),
    )
    return (_mod2pi(alpha - tmp), math.sqrt(p2),
            _mod2pi(-beta + tmp), "RSR")


def _lsr(alpha, beta, distance):
    p2 = (-2.0 + distance**2 + 2.0 * math.cos(alpha - beta)
          + 2.0 * distance * (math.sin(alpha) + math.sin(beta)))
    if p2 < 0.0:
        return None
    straight = math.sqrt(p2)
    tmp = (math.atan2(
        -math.cos(alpha) - math.cos(beta),
        distance + math.sin(alpha) + math.sin(beta),
    ) - math.atan2(-2.0, straight))
    return (_mod2pi(-alpha + tmp), straight,
            _mod2pi(-_mod2pi(beta) + tmp), "LSR")


def _rsl(alpha, beta, distance):
    p2 = (distance**2 - 2.0 + 2.0 * math.cos(alpha - beta)
          - 2.0 * distance * (math.sin(alpha) + math.sin(beta)))
    if p2 < 0.0:
        return None
    straight = math.sqrt(p2)
    tmp = (math.atan2(
        math.cos(alpha) + math.cos(beta),
        distance - math.sin(alpha) - math.sin(beta),
    ) - math.atan2(2.0, straight))
    return (_mod2pi(alpha - tmp), straight,
            _mod2pi(beta - tmp), "RSL")


def _rlr(alpha, beta, distance):
    tmp = (6.0 - distance**2 + 2.0 * math.cos(alpha - beta)
           + 2.0 * distance * (math.sin(alpha) - math.sin(beta))) / 8.0
    if abs(tmp) > 1.0 + 1e-12:
        return None
    tmp = float(np.clip(tmp, -1.0, 1.0))
    middle = _mod2pi(_TAU - math.acos(tmp))
    first = _mod2pi(
        alpha - math.atan2(
            math.cos(alpha) - math.cos(beta),
            distance - math.sin(alpha) + math.sin(beta),
        ) + middle / 2.0
    )
    return first, middle, _mod2pi(alpha - beta - first + middle), "RLR"


def _lrl(alpha, beta, distance):
    tmp = (6.0 - distance**2 + 2.0 * math.cos(alpha - beta)
           + 2.0 * distance * (-math.sin(alpha) + math.sin(beta))) / 8.0
    if abs(tmp) > 1.0 + 1e-12:
        return None
    tmp = float(np.clip(tmp, -1.0, 1.0))
    middle = _mod2pi(_TAU - math.acos(tmp))
    first = _mod2pi(
        -alpha - math.atan2(
            math.cos(alpha) - math.cos(beta),
            distance + math.sin(alpha) - math.sin(beta),
        ) + middle / 2.0
    )
    return (first, middle,
            _mod2pi(_mod2pi(beta) - alpha - first + _mod2pi(middle)),
            "LRL")


_PLANNERS = (_lsl, _rsr, _lsr, _rsl, _rlr, _lrl)


@dataclass(frozen=True)
class TerminalManeuver:
    """A sampled forward-only target-pose path in Habitat ground coordinates."""

    points_xz: np.ndarray
    yaws: np.ndarray
    mode: str
    length_m: float
    min_clearance_m: float


def _candidate_words(start, goal, radius):
    dx, dz = goal[0] - start[0], goal[1] - start[1]
    distance = math.hypot(dx, dz) / radius
    direction = _mod2pi(math.atan2(dz, dx))
    alpha = _mod2pi(start[2] - direction)
    beta = _mod2pi(goal[2] - direction)
    return [candidate for planner in _PLANNERS
            if (candidate := planner(alpha, beta, distance)) is not None]


def _advance(x, z, heading, mode, distance_m, radius):
    if mode == "S":
        return (x + distance_m * math.cos(heading),
                z + distance_m * math.sin(heading), heading)
    curvature = (1.0 / radius) if mode == "L" else (-1.0 / radius)
    new_heading = heading + curvature * distance_m
    return (
        x + (math.sin(new_heading) - math.sin(heading)) / curvature,
        z + (-math.cos(new_heading) + math.cos(heading)) / curvature,
        new_heading,
    )


def _sample_candidate(start, goal, word, radius, turn_step_m, straight_step_m):
    lengths = word[:3]
    modes = word[3]
    x, z, heading = map(float, start)
    points = [[x, z]]
    headings = [heading]
    for normalized_length, mode in zip(lengths, modes):
        segment_m = float(normalized_length) * radius
        if segment_m < 1e-10:
            continue
        max_step = straight_step_m if mode == "S" else turn_step_m
        count = max(1, int(math.ceil(segment_m / max_step)))
        step_m = segment_m / count
        for _ in range(count):
            x, z, heading = _advance(x, z, heading, mode, step_m, radius)
            points.append([x, z])
            headings.append(heading)

    # Closed-form words end at the requested pose.  Pin the last sample to remove
    # accumulated floating-point noise (normally <1e-12).
    points[-1] = [float(goal[0]), float(goal[1])]
    headings[-1] = float(goal[2])
    return np.asarray(points), np.asarray(headings), sum(lengths) * radius


def _path_clearance(pathfinder, points_xz, floor_y, snap_tolerance):
    if pathfinder is None:
        return math.inf
    minimum = math.inf
    for x, z in points_xz:
        point = np.array([x, floor_y, z], dtype=np.float64)
        snapped = np.asarray(pathfinder.snap_point(point), dtype=np.float64)
        if (not np.isfinite(snapped).all()
                or not pathfinder.is_navigable(snapped)
                or np.linalg.norm(snapped[[0, 2]] - point[[0, 2]]) > snap_tolerance):
            return None
        if hasattr(pathfinder, "distance_to_closest_obstacle"):
            minimum = min(
                minimum,
                float(pathfinder.distance_to_closest_obstacle(snapped)),
            )
    return minimum


def plan_terminal_maneuver(
        current_xz,
        current_yaw,
        goal_xz,
        goal_yaw,
        radius=0.40,
        turn_step_m=0.018,
        straight_step_m=0.038,
        pathfinder=None,
        floor_y=0.0,
        snap_tolerance=0.08):
    """Return the shortest collision-free Dubins connection to a goal-image pose.

    All six Dubins words are considered.  Candidates within 5 cm of the shortest
    feasible length are tie-broken by navmesh clearance, which naturally chooses
    the safer side for a mirrored 180-degree U-turn.
    """
    if radius <= 0.0 or turn_step_m <= 0.0 or straight_step_m <= 0.0:
        raise ValueError("radius and sampling steps must be positive")

    start = np.array([
        float(np.asarray(current_xz)[0]),
        float(np.asarray(current_xz)[1]),
        habitat_yaw_to_planar(current_yaw),
    ])
    goal = np.array([
        float(np.asarray(goal_xz)[0]),
        float(np.asarray(goal_xz)[1]),
        habitat_yaw_to_planar(goal_yaw),
    ])

    feasible = []
    for word in _candidate_words(start, goal, radius):
        points, headings, length_m = _sample_candidate(
            start, goal, word, radius, turn_step_m, straight_step_m)
        clearance = _path_clearance(
            pathfinder, points, floor_y, snap_tolerance)
        if clearance is None:
            continue
        feasible.append(TerminalManeuver(
            points_xz=points,
            yaws=np.asarray([planar_to_habitat_yaw(h) for h in headings]),
            mode=word[3],
            length_m=float(length_m),
            min_clearance_m=float(clearance),
        ))

    if not feasible:
        return None
    shortest = min(path.length_m for path in feasible)
    near_shortest = [path for path in feasible
                     if path.length_m <= shortest + 0.05]
    return max(near_shortest, key=lambda path: path.min_clearance_m)


def _join_maneuvers(maneuvers, mode):
    points = [maneuvers[0].points_xz]
    yaws = [maneuvers[0].yaws]
    for maneuver in maneuvers[1:]:
        points.append(maneuver.points_xz[1:])
        yaws.append(maneuver.yaws[1:])
    return TerminalManeuver(
        points_xz=np.concatenate(points, axis=0),
        yaws=np.concatenate(yaws, axis=0),
        mode=mode,
        length_m=float(sum(path.length_m for path in maneuvers)),
        min_clearance_m=float(min(path.min_clearance_m for path in maneuvers)),
    )


def plan_staged_terminal_maneuver(
        current_xz,
        current_yaw,
        goal_yaw,
        radius=0.40,
        stage_min_m=0.60,
        stage_max_m=2.50,
        stage_step_m=0.20,
        turn_step_m=0.018,
        straight_step_m=0.038,
        pathfinder=None,
        floor_y=0.0,
        snap_tolerance=0.08,
        stage_snap_tolerance=0.15):
    """Move to nearby free space, U-turn there, then return to this position.

    A compact same-position Dubins loop can be impossible at a goal beside a
    wall.  The leg-transition U-turn does not have that restriction: it keeps
    moving into free space before curving.  This fallback reproduces that
    behavior without using the noisy metric goal translation:

    ``current -> staging (arrival heading) -> U-turn -> current (goal yaw)``.
    """
    if pathfinder is None:
        raise ValueError("staged terminal planning requires a pathfinder")
    if stage_min_m <= 0.0 or stage_max_m < stage_min_m or stage_step_m <= 0.0:
        raise ValueError("invalid staging distance range")

    current_xz = np.asarray(current_xz, dtype=np.float64)
    forward = np.array(
        [-math.sin(current_yaw), -math.cos(current_yaw)], dtype=np.float64)
    candidates = []
    distances = np.arange(
        stage_min_m, stage_max_m + 0.5 * stage_step_m, stage_step_m)
    for distance in distances:
        requested = current_xz + float(distance) * forward
        snapped3 = np.asarray(pathfinder.snap_point(
            [requested[0], floor_y, requested[1]]), dtype=np.float64)
        if (not np.isfinite(snapped3).all()
                or not pathfinder.is_navigable(snapped3)
                or np.linalg.norm(snapped3[[0, 2]] - requested) > stage_snap_tolerance):
            continue
        staging = snapped3[[0, 2]]
        outbound = plan_terminal_maneuver(
            current_xz, current_yaw, staging, current_yaw,
            radius=radius, turn_step_m=turn_step_m,
            straight_step_m=straight_step_m, pathfinder=pathfinder,
            floor_y=floor_y, snap_tolerance=snap_tolerance)
        turn = plan_terminal_maneuver(
            staging, current_yaw, staging, goal_yaw,
            radius=radius, turn_step_m=turn_step_m,
            straight_step_m=straight_step_m, pathfinder=pathfinder,
            floor_y=floor_y, snap_tolerance=snap_tolerance)
        inbound = plan_terminal_maneuver(
            staging, goal_yaw, current_xz, goal_yaw,
            radius=radius, turn_step_m=turn_step_m,
            straight_step_m=straight_step_m, pathfinder=pathfinder,
            floor_y=floor_y, snap_tolerance=snap_tolerance)
        if outbound is None or turn is None or inbound is None:
            continue
        candidates.append(_join_maneuvers(
            [outbound, turn, inbound],
            f"STAGED-{outbound.mode}-{turn.mode}-{inbound.mode}",
        ))

    if not candidates:
        return None
    shortest = min(path.length_m for path in candidates)
    near_shortest = [path for path in candidates
                     if path.length_m <= shortest + 0.05]
    return max(near_shortest, key=lambda path: path.min_clearance_m)


class TerminalManeuverExecutor:
    """Execute a sampled maneuver without ever rotating in place."""

    def __init__(self, maneuver, snap_tolerance=0.08):
        self.maneuver = maneuver
        self.snap_tolerance = float(snap_tolerance)
        self.index = 1  # sample zero is the pose from which the path was planned
        self.failed = False

    @property
    def done(self):
        return not self.failed and self.index >= len(self.maneuver.points_xz)

    def step(self, position, yaw, pathfinder, floor_y):
        """Advance one coupled-motion sample.

        Returns ``(position, yaw, travelled_m)``.  A rejected navmesh step keeps
        both position *and yaw* unchanged, so it cannot silently become an
        in-place turn.
        """
        if self.failed or self.done:
            return np.asarray(position, dtype=np.float64), float(yaw), 0.0

        desired_xz = self.maneuver.points_xz[self.index]
        desired_yaw = float(self.maneuver.yaws[self.index])
        desired = np.array([desired_xz[0], floor_y, desired_xz[1]], dtype=np.float64)
        snapped = np.asarray(pathfinder.snap_point(desired), dtype=np.float64)
        old = np.asarray(position, dtype=np.float64)
        travelled = float(np.linalg.norm(snapped[[0, 2]] - old[[0, 2]]))
        yaw_change = abs(wrap_angle(desired_yaw - yaw))
        valid = (
            np.isfinite(snapped).all()
            and pathfinder.is_navigable(snapped)
            and np.linalg.norm(snapped[[0, 2]] - desired_xz) <= self.snap_tolerance
            and not (travelled < 1e-6 and yaw_change > 1e-6)
        )
        if not valid:
            self.failed = True
            return old, float(yaw), 0.0

        self.index += 1
        return snapped, desired_yaw, travelled
