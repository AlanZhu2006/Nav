"""Pure geometry for the consumed long-range oracle attribution fork.

Nothing in this module is deployable: both readouts consume evaluator pose.
They exist only to separate route-state estimation, route choice, and frozen
controller conditioning after CEC has already certified a historical target.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Sequence

import numpy as np


ORACLE_SCHEMA_VERSION = "longrange_oracle_attribution_v1_20260903"
ORACLE_ARMS = (
    "action_coordinate_mixed",
    "oracle_route_mixed",
    "oracle_geodesic_mixed",
    "oracle_geodesic_point",
)


def _finite_xz(value: Sequence[float], label: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (2,) or not np.isfinite(result).all():
        raise ValueError(f"{label} must be one finite x/z pair")
    return result


def world_delta_to_local(delta_xz: Sequence[float], yaw_rad: float) -> np.ndarray:
    """Habitat world x/z displacement -> camera [forward, left]."""

    dx, dz = _finite_xz(delta_xz, "world displacement")
    yaw = float(yaw_rad)
    if not math.isfinite(yaw):
        raise ValueError("yaw must be finite")
    sine, cosine = math.sin(yaw), math.cos(yaw)
    return np.asarray([
        -sine * dx - cosine * dz,
        -cosine * dx + sine * dz,
    ], dtype=np.float64)


def fixed_radius_pointgoal(
    world_target_xz: Sequence[float],
    current_xz: Sequence[float],
    yaw_rad: float,
    radius_m: float = 2.5,
) -> np.ndarray:
    """Return a scale-free local direction projected to a fixed radius."""

    target = _finite_xz(world_target_xz, "world target")
    current = _finite_xz(current_xz, "current position")
    radius = float(radius_m)
    if not math.isfinite(radius) or radius <= 0.0:
        raise ValueError("radius_m must be positive and finite")
    local = world_delta_to_local(target - current, yaw_rad)
    norm = float(np.linalg.norm(local))
    if not math.isfinite(norm) or norm <= 1e-8:
        raise ValueError("target direction is degenerate")
    return local * (radius / norm)


def _deduplicate_route(route_xz: Sequence[Sequence[float]]) -> np.ndarray:
    route = np.asarray(route_xz, dtype=np.float64)
    if route.ndim != 2 or route.shape[1] != 2 or route.shape[0] < 2:
        raise ValueError("route must have shape [N,2] with N >= 2")
    if not np.isfinite(route).all():
        raise ValueError("route contains a non-finite position")
    kept = [route[0]]
    for point in route[1:]:
        if float(np.linalg.norm(point - kept[-1])) > 1e-8:
            kept.append(point)
    if len(kept) < 2:
        raise ValueError("route has zero spatial extent")
    return np.stack(kept)


def interpolate_polyline(
    route_xz: np.ndarray, cumulative_m: np.ndarray, arc_m: float,
) -> tuple[np.ndarray, int]:
    target_arc = float(np.clip(arc_m, 0.0, cumulative_m[-1]))
    segment = int(np.searchsorted(cumulative_m, target_arc, side="right") - 1)
    segment = min(max(segment, 0), len(route_xz) - 2)
    length = float(cumulative_m[segment + 1] - cumulative_m[segment])
    fraction = ((target_arc - float(cumulative_m[segment])) / length
                if length > 1e-12 else 0.0)
    return (
        route_xz[segment]
        + fraction * (route_xz[segment + 1] - route_xz[segment]),
        segment,
    )


@dataclass(frozen=True)
class RouteProjection:
    pointgoal: np.ndarray
    progress_m: float
    remaining_m: float
    cross_track_m: float
    projection_segment: int
    reference_arc_m: float
    reference_world_xz: np.ndarray
    progress_jump_m: float


class OracleRouteProjector:
    """Monotone GT-pose projection onto one observed historical route."""

    def __init__(
        self,
        route_xz: Sequence[Sequence[float]],
        *,
        lookahead_m: float = 2.5,
        radius_m: float = 2.5,
    ) -> None:
        self.route = _deduplicate_route(route_xz)
        lengths = np.linalg.norm(np.diff(self.route, axis=0), axis=1)
        self.cumulative = np.concatenate(([0.0], np.cumsum(lengths)))
        self.lookahead_m = float(lookahead_m)
        self.radius_m = float(radius_m)
        if (not math.isfinite(self.lookahead_m) or self.lookahead_m <= 0.0
                or not math.isfinite(self.radius_m) or self.radius_m <= 0.0):
            raise ValueError("lookahead and radius must be positive and finite")
        self.progress_m = 0.0

    @property
    def extent_m(self) -> float:
        return float(self.cumulative[-1])

    def _project_monotone(
        self, current_xz: np.ndarray,
    ) -> tuple[float, float, int]:
        best: tuple[float, float, int] | None = None
        for index, (start, end) in enumerate(
                zip(self.route[:-1], self.route[1:])):
            if float(self.cumulative[index + 1]) + 1e-9 < self.progress_m:
                continue
            segment = end - start
            length_sq = float(segment @ segment)
            lower = 0.0
            if self.progress_m > float(self.cumulative[index]):
                lower = min(
                    1.0,
                    (self.progress_m - float(self.cumulative[index]))
                    / math.sqrt(length_sq),
                )
            fraction = float(np.clip(
                ((current_xz - start) @ segment) / length_sq,
                lower,
                1.0,
            ))
            projected = start + fraction * segment
            distance = float(np.linalg.norm(current_xz - projected))
            arc = float(
                self.cumulative[index] + fraction * math.sqrt(length_sq))
            candidate = (distance, arc, index)
            if best is None or candidate[:2] < best[:2]:
                best = candidate
        if best is None:
            raise RuntimeError("route projection found no eligible segment")
        return best

    def update(
        self, current_xz: Sequence[float], yaw_rad: float,
    ) -> RouteProjection:
        current = _finite_xz(current_xz, "current position")
        cross_track, projected_arc, segment = self._project_monotone(current)
        previous = self.progress_m
        self.progress_m = max(previous, projected_arc)
        reference_arc = min(
            self.extent_m, self.progress_m + self.lookahead_m)
        reference, _ = interpolate_polyline(
            self.route, self.cumulative, reference_arc)
        if float(np.linalg.norm(reference - current)) <= 1e-8:
            # A terminal route point can coincide exactly with the robot.  The
            # caller should score arrival before requesting another action.
            raise ValueError("route reference coincides with current position")
        pointgoal = fixed_radius_pointgoal(
            reference, current, yaw_rad, self.radius_m)
        return RouteProjection(
            pointgoal=pointgoal,
            progress_m=float(self.progress_m),
            remaining_m=float(max(0.0, self.extent_m - self.progress_m)),
            cross_track_m=float(cross_track),
            projection_segment=int(segment),
            reference_arc_m=float(reference_arc),
            reference_world_xz=reference,
            progress_jump_m=float(self.progress_m - previous),
        )


class OracleRuntimeState:
    """Explicit bindings for one privileged attribution query.

    The production evaluator materializes a runtime query before replaying its
    causal RGB prefix.  Binding that prefix must therefore preserve the query
    goal.  A new query, in contrast, invalidates every history-, pathfinder-,
    and route-specific object so stale privileged state cannot cross queries.
    """

    def __init__(self) -> None:
        self.trace: dict[str, Any] | None = None
        self.goal_floor: np.ndarray | None = None
        self.pathfinder: Any | None = None
        self.route_projector: OracleRouteProjector | None = None
        self.route_anchor: int | None = None
        self.route_goal_start: int | None = None

    def _clear_route(self) -> None:
        self.route_projector = None
        self.route_anchor = None
        self.route_goal_start = None

    def bind_query(self, goal_floor: Sequence[float]) -> None:
        floor = np.asarray(goal_floor, dtype=np.float64)
        if floor.shape != (3,) or not np.isfinite(floor).all():
            raise ValueError("runtime query goal floor must be one finite xyz")
        self.goal_floor = floor.copy()
        self.trace = None
        self.pathfinder = None
        self._clear_route()

    def bind_history(self, trace: dict[str, Any]) -> None:
        if not isinstance(trace, dict) or not isinstance(trace.get("poses"), list):
            raise ValueError("causal history trace must contain a pose list")
        # Deliberately preserve goal_floor: runtime_query(query) is called
        # before replay_prefix(frozen) by eval_shared_online_role_pairs.py.
        self.trace = trace
        self.pathfinder = None
        self._clear_route()

    def bind_pathfinder(self, pathfinder: Any) -> None:
        if pathfinder is None:
            raise ValueError("runtime pathfinder is missing")
        self.pathfinder = pathfinder


def historical_reverse_route(
    poses: Sequence[dict],
    *,
    query_start_xz: Sequence[float],
    goal_start_frame: int,
    target_anchor: int,
) -> np.ndarray:
    """Build query-start -> certified-anchor route from causal history poses."""

    if not poses:
        raise ValueError("causal history contains no poses")
    start = int(goal_start_frame) - 1
    anchor = int(target_anchor)
    if not (0 <= anchor <= start < len(poses)):
        raise ValueError("anchor/goal-start interval lies outside history")
    route = [_finite_xz(query_start_xz, "query start")]
    route.extend(
        np.asarray([poses[index]["x"], poses[index]["z"]], dtype=np.float64)
        for index in range(start, anchor - 1, -1)
    )
    return _deduplicate_route(route)


def geodesic_lookahead_world(
    path_points: Sequence[Sequence[float]], lookahead_m: float = 2.5,
) -> tuple[np.ndarray, float]:
    """Interpolate a world x/z lookahead on a current shortest path."""

    points = np.asarray(path_points, dtype=np.float64)
    if points.ndim != 2 or points.shape[0] < 2 or points.shape[1] < 3:
        raise ValueError("geodesic path must contain at least two 3-D points")
    route = _deduplicate_route(points[:, [0, 2]])
    lengths = np.linalg.norm(np.diff(route, axis=0), axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(lengths)))
    lookahead = min(float(lookahead_m), float(cumulative[-1]))
    if not math.isfinite(lookahead) or lookahead <= 1e-8:
        raise ValueError("geodesic path has no positive lookahead")
    target, _ = interpolate_polyline(route, cumulative, lookahead)
    return target, float(cumulative[-1])
