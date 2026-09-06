"""Local SE(2) route projection after one CEC authorization.

The initial CEC proof selects one causal historical anchor.  This module then
reconstructs the intervening route from frame-bound executor receipts and
tracks the live executor in the same local coordinate system.  Route progress
is the monotone orthogonal projection of the estimated live position onto the
historical polyline; total travelled distance is never treated as progress.

Only relative translation/yaw receipts are consumed.  Simulator/world pose,
metric depth, success labels, visual thresholds, and distance classes are not
part of this state estimator.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import numpy as np


SE2_ROUTE_COMPASS_SCHEMA_VERSION = (
    "se2_projected_route_compass_v2_20260902"
)
SE2_EXECUTOR_MOTION_MODEL = "post_yaw_forward_translation_v1"
SE2_LOCAL_ODOMETRY_MODEL = "previous_body_delta_se2_v1"


def _finite_nonnegative(value: float, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return result


def _finite_yaw(value: float, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _finite_positive(value: float, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return result


def _wrap_angle(value: float) -> float:
    return float((float(value) + math.pi) % (2.0 * math.pi) - math.pi)


def _heading(yaw_rad: float) -> np.ndarray:
    """Unit forward vector in the initial query frame (forward, left)."""

    return np.asarray(
        [math.cos(float(yaw_rad)), math.sin(float(yaw_rad))],
        dtype=np.float64,
    )


def _rotate(vector: np.ndarray, yaw_rad: float) -> np.ndarray:
    c, s = math.cos(float(yaw_rad)), math.sin(float(yaw_rad))
    forward, left = (float(value) for value in vector)
    return np.asarray([
        c * forward - s * left,
        s * forward + c * left,
    ], dtype=np.float64)


def reconstruct_reverse_route(
    translations_m: Iterable[float],
    yaw_deltas_rad: Iterable[float],
) -> tuple[np.ndarray, np.ndarray]:
    """Reconstruct newest-to-oldest poses from chronological motion receipts.

    Inputs must be ordered from the newest edge to the oldest edge.  Receipt
    ``i`` describes the original motion from historical frame ``i-1`` to
    frame ``i``: the executor changes yaw by ``yaw_delta`` and then translates
    forward by ``translation``.  The newest pose is the origin with zero yaw,
    so the output is expressed entirely in the current query body frame.

    This inversion is exact for the frozen Habitat pure-pursuit executor and
    for any deployed executor that reports the same post-yaw-forward contract.
    """

    translations = np.asarray(list(translations_m), dtype=np.float64)
    yaw_deltas = np.asarray(list(yaw_deltas_rad), dtype=np.float64)
    if (translations.ndim != 1 or yaw_deltas.ndim != 1
            or translations.shape != yaw_deltas.shape
            or translations.size < 2
            or not np.isfinite(translations).all()
            or not np.isfinite(yaw_deltas).all()
            or np.any(translations < 0.0)):
        raise ValueError(
            "reverse route requires at least two finite non-negative "
            "translation receipts with aligned finite yaw receipts"
        )

    positions = [np.zeros(2, dtype=np.float64)]
    yaws = [0.0]
    position = positions[0].copy()
    yaw = 0.0
    for translation, yaw_delta in zip(translations, yaw_deltas):
        # Original action: theta_i = theta_{i-1} + delta, followed by a
        # translation along theta_i.  We stand at frame i and invert it.
        position = position - float(translation) * _heading(yaw)
        yaw = _wrap_angle(yaw - float(yaw_delta))
        positions.append(position.copy())
        yaws.append(yaw)

    points = np.stack(positions, axis=0)
    headings = np.asarray(yaws, dtype=np.float64)
    if float(np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1))) <= 1e-6:
        raise ValueError("reverse route has no translational extent")
    return points, headings


def reconstruct_reverse_route_se2(
    delta_forwards_m: Iterable[float],
    delta_lefts_m: Iterable[float],
    yaw_deltas_rad: Iterable[float],
) -> tuple[np.ndarray, np.ndarray]:
    """Invert local SE(2) receipts ordered from newest to oldest.

    Each translation is expressed in the *previous* camera/body frame, i.e.
    the standard relative transform from frame ``i-1`` to frame ``i``.  This
    preserves realized sideways displacement from collision handling, wheel
    slip, or navmesh projection without revealing either frame's global pose.
    """

    forwards = np.asarray(list(delta_forwards_m), dtype=np.float64)
    lefts = np.asarray(list(delta_lefts_m), dtype=np.float64)
    yaw_deltas = np.asarray(list(yaw_deltas_rad), dtype=np.float64)
    if (forwards.ndim != 1 or lefts.ndim != 1 or yaw_deltas.ndim != 1
            or forwards.shape != lefts.shape
            or forwards.shape != yaw_deltas.shape
            or forwards.size < 2
            or not np.isfinite(forwards).all()
            or not np.isfinite(lefts).all()
            or not np.isfinite(yaw_deltas).all()):
        raise ValueError(
            "reverse SE(2) route requires at least two aligned finite "
            "forward, left, and yaw receipts"
        )

    positions = [np.zeros(2, dtype=np.float64)]
    yaws = [0.0]
    position = positions[0].copy()
    yaw = 0.0
    for forward, left, yaw_delta in zip(forwards, lefts, yaw_deltas):
        previous_yaw = _wrap_angle(yaw - float(yaw_delta))
        previous_to_current = _rotate(
            np.asarray([forward, left], dtype=np.float64), previous_yaw)
        position = position - previous_to_current
        yaw = previous_yaw
        positions.append(position.copy())
        yaws.append(yaw)

    points = np.stack(positions, axis=0)
    headings = np.asarray(yaws, dtype=np.float64)
    if float(np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1))) <= 1e-6:
        raise ValueError("reverse SE(2) route has no translational extent")
    return points, headings


@dataclass(frozen=True)
class SE2RouteCompassReadout:
    schema_version: str
    motion_model: str
    projected_progress_m: float
    progress_increment_m: float
    progress_budget_m: float
    route_progress_fraction: float
    route_state_index: int
    reference_arc_m: float
    query_path_length_m: float
    cross_track_error_m: float
    estimated_position: tuple[float, float]
    estimated_yaw_rad: float
    projected_position: tuple[float, float]
    reference_position: tuple[float, float]
    terminal_tangent_held: bool
    unit_bearing: tuple[float, float]
    controller_pointgoal: tuple[float, float]

    def audit_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "motion_model": self.motion_model,
            "projected_progress_m": self.projected_progress_m,
            "progress_increment_m": self.progress_increment_m,
            "progress_budget_m": self.progress_budget_m,
            "route_progress_fraction": self.route_progress_fraction,
            "route_state_index": self.route_state_index,
            "reference_arc_m": self.reference_arc_m,
            "query_path_length_m": self.query_path_length_m,
            "cross_track_error_m": self.cross_track_error_m,
            "estimated_position": list(self.estimated_position),
            "estimated_yaw_rad": self.estimated_yaw_rad,
            "projected_position": list(self.projected_position),
            "reference_position": list(self.reference_position),
            "terminal_tangent_held": self.terminal_tangent_held,
            "unit_bearing": list(self.unit_bearing),
            "controller_pointgoal": list(self.controller_pointgoal),
        }


class SE2ProjectedRouteCompass:
    """Follow one certified route using local SE(2) dead reckoning.

    The route and live query pose share the initial query body frame.  Each
    action updates the live 2-D pose; progress is then re-estimated by a
    monotone orthogonal projection onto the causally reachable route interval.
    The route coordinate cannot advance farther than the current local motion.
    The
    controller receives a unit bearing from the *estimated live position* to
    a fixed arc-length lookahead point, so lateral deviation produces an
    explicit cross-track correction.

    The estimator is total after construction.  It contains no learned gate,
    distance regime, stuck detector, endpoint controller, or native fallback.
    """

    def __init__(
        self,
        route_positions_m: np.ndarray,
        *,
        lookahead_m: float = 2.5,
        controller_radius_m: float = 2.5,
        motion_model: str = SE2_EXECUTOR_MOTION_MODEL,
    ) -> None:
        positions = np.asarray(route_positions_m, dtype=np.float64)
        if (positions.ndim != 2 or positions.shape[0] < 3
                or positions.shape[1] != 2
                or not np.isfinite(positions).all()):
            raise ValueError(
                "route_positions_m must be finite with shape [N>=3,2]")
        edge_lengths = np.linalg.norm(np.diff(positions, axis=0), axis=1)
        cumulative = np.concatenate([
            np.zeros(1, dtype=np.float64), np.cumsum(edge_lengths),
        ])
        if float(cumulative[-1]) <= 1e-6:
            raise ValueError("route_positions_m has no translational extent")

        self._positions = positions.copy()
        self._edge_lengths = edge_lengths
        self._arc = cumulative
        self._lookahead = _finite_positive(lookahead_m, "lookahead_m")
        self._controller_radius = _finite_positive(
            controller_radius_m, "controller_radius_m")
        if motion_model not in (
                SE2_EXECUTOR_MOTION_MODEL, SE2_LOCAL_ODOMETRY_MODEL):
            raise ValueError("unsupported SE(2) executor motion model")
        self._motion_model = str(motion_model)
        self._position = np.zeros(2, dtype=np.float64)
        self._yaw = 0.0
        self._progress = 0.0
        self._query_path_length = 0.0

    @classmethod
    def from_reverse_executor_receipts(
        cls,
        translations_m: Iterable[float],
        yaw_deltas_rad: Iterable[float],
        *,
        lookahead_m: float = 2.5,
        controller_radius_m: float = 2.5,
    ) -> "SE2ProjectedRouteCompass":
        route, _ = reconstruct_reverse_route(
            translations_m, yaw_deltas_rad)
        return cls(
            route,
            lookahead_m=lookahead_m,
            controller_radius_m=controller_radius_m,
            motion_model=SE2_EXECUTOR_MOTION_MODEL,
        )

    @classmethod
    def from_reverse_local_se2_receipts(
        cls,
        delta_forwards_m: Iterable[float],
        delta_lefts_m: Iterable[float],
        yaw_deltas_rad: Iterable[float],
        *,
        lookahead_m: float = 2.5,
        controller_radius_m: float = 2.5,
    ) -> "SE2ProjectedRouteCompass":
        route, _ = reconstruct_reverse_route_se2(
            delta_forwards_m, delta_lefts_m, yaw_deltas_rad)
        return cls(
            route,
            lookahead_m=lookahead_m,
            controller_radius_m=controller_radius_m,
            motion_model=SE2_LOCAL_ODOMETRY_MODEL,
        )

    @property
    def route_extent_m(self) -> float:
        return float(self._arc[-1])

    @property
    def projected_progress_m(self) -> float:
        return float(self._progress)

    @property
    def estimated_position(self) -> np.ndarray:
        return self._position.copy()

    @property
    def estimated_yaw_rad(self) -> float:
        return float(self._yaw)

    def _sample(self, arc_m: float) -> np.ndarray:
        arc = float(np.clip(arc_m, 0.0, self.route_extent_m))
        if arc >= self.route_extent_m:
            return self._positions[-1].copy()
        segment = int(np.searchsorted(self._arc, arc, side="right") - 1)
        segment = min(max(segment, 0), len(self._positions) - 2)
        while (segment < len(self._positions) - 2
               and self._edge_lengths[segment] <= 1e-12):
            segment += 1
        length = float(self._edge_lengths[segment])
        if length <= 1e-12:
            return self._positions[segment].copy()
        alpha = (arc - float(self._arc[segment])) / length
        return ((1.0 - alpha) * self._positions[segment]
                + alpha * self._positions[segment + 1])

    def _state_index(self, arc_m: float) -> int:
        state = int(np.searchsorted(self._arc, arc_m, side="right") - 1)
        return min(max(state, 0), len(self._positions) - 1)

    def _monotone_projection(
            self, *, maximum_progress_m: float,
    ) -> tuple[float, np.ndarray, float]:
        """Project into the causally reachable route interval.

        The admissible coordinate is

        ``previous_s <= s <= previous_s + ||delta_position||``.

        Thus a single local motion receipt cannot teleport the route state
        across a self-intersection or a spatially adjacent later corridor.
        This is a kinematic invariant, not a confidence threshold or fallback.
        Earliest arc length wins exact geometric ties.
        """

        maximum = float(np.clip(
            maximum_progress_m, self._progress, self.route_extent_m))

        best_distance_sq = math.inf
        best_arc = float(self._progress)
        best_point = self._sample(best_arc)
        numerical_tie = 1e-12

        for segment, length_value in enumerate(self._edge_lengths):
            length = float(length_value)
            if length <= 1e-12:
                continue
            low = float(self._arc[segment])
            high = float(self._arc[segment + 1])
            if high < self._progress - numerical_tie:
                continue
            if low > maximum + numerical_tie:
                break
            minimum_alpha = max(0.0, (self._progress - low) / length)
            maximum_alpha = min(1.0, (maximum - low) / length)
            if minimum_alpha > 1.0 + numerical_tie:
                continue
            if maximum_alpha < -numerical_tie:
                continue
            if minimum_alpha > maximum_alpha + numerical_tie:
                continue
            start = self._positions[segment]
            vector = self._positions[segment + 1] - start
            alpha = float(np.dot(self._position - start, vector)
                          / (length * length))
            alpha = float(np.clip(alpha, minimum_alpha, maximum_alpha))
            point = start + alpha * vector
            distance_sq = float(np.dot(
                self._position - point, self._position - point))
            arc = low + alpha * length
            if (distance_sq < best_distance_sq - numerical_tie
                    or (abs(distance_sq - best_distance_sq) <= numerical_tie
                        and arc < best_arc)):
                best_distance_sq = distance_sq
                best_arc = arc
                best_point = point

        if not math.isfinite(best_distance_sq):
            raise RuntimeError("route projection produced no finite candidate")
        return best_arc, best_point, math.sqrt(max(0.0, best_distance_sq))

    def advance(
        self,
        *,
        executed_translation_m: float,
        executed_yaw_rad: float,
    ) -> SE2RouteCompassReadout:
        translation = _finite_nonnegative(
            executed_translation_m, "executed_translation_m")
        yaw_delta = _finite_yaw(executed_yaw_rad, "executed_yaw_rad")

        # The frozen executor changes heading before translating.  Integrating
        # the same atomic contract is what recovers 2-D displacement from the
        # existing frame-bound scalar translation/yaw receipt.
        delta_previous_body = translation * _heading(yaw_delta)
        return self._advance_delta(
            delta_previous_body=delta_previous_body,
            yaw_delta=yaw_delta,
            motion_model=SE2_EXECUTOR_MOTION_MODEL,
        )

    def advance_local_se2(
        self,
        *,
        executed_forward_m: float,
        executed_left_m: float,
        executed_yaw_rad: float,
    ) -> SE2RouteCompassReadout:
        forward = _finite_yaw(executed_forward_m, "executed_forward_m")
        left = _finite_yaw(executed_left_m, "executed_left_m")
        yaw_delta = _finite_yaw(executed_yaw_rad, "executed_yaw_rad")
        return self._advance_delta(
            delta_previous_body=np.asarray([forward, left], dtype=np.float64),
            yaw_delta=yaw_delta,
            motion_model=SE2_LOCAL_ODOMETRY_MODEL,
        )

    def _advance_delta(
        self,
        *,
        delta_previous_body: np.ndarray,
        yaw_delta: float,
        motion_model: str,
    ) -> SE2RouteCompassReadout:
        if motion_model != self._motion_model:
            raise ValueError(
                "runtime motion receipt differs from frozen route motion model")
        displacement = _rotate(delta_previous_body, self._yaw)
        progress_before = float(self._progress)
        progress_budget = float(np.linalg.norm(delta_previous_body))
        self._position = self._position + displacement
        self._yaw = _wrap_angle(self._yaw + yaw_delta)
        self._query_path_length += progress_budget

        progress, projected, cross_track = self._monotone_projection(
            maximum_progress_m=min(
                self.route_extent_m, progress_before + progress_budget))
        if progress + 1e-9 < self._progress:
            raise RuntimeError("SE(2) route progress regressed")
        self._progress = max(self._progress, float(progress))
        progress_increment = float(self._progress - progress_before)
        if progress_increment > progress_budget + 1e-9:
            raise RuntimeError("SE(2) route progress exceeded local motion")
        reference_arc = min(
            self.route_extent_m, self._progress + self._lookahead)
        reference = self._sample(reference_arc)

        delta = reference - self._position
        terminal_hold = float(np.linalg.norm(delta)) <= 1e-12
        if terminal_hold:
            start_arc = max(0.0, self.route_extent_m - self._lookahead)
            delta = self._positions[-1] - self._sample(start_arc)

        c, s = math.cos(self._yaw), math.sin(self._yaw)
        # R(theta)^T maps the initial query frame into the current body frame.
        local = np.asarray([
            c * delta[0] + s * delta[1],
            -s * delta[0] + c * delta[1],
        ], dtype=np.float64)
        norm = float(np.linalg.norm(local))
        if not math.isfinite(norm) or norm <= 1e-12:
            raise RuntimeError("SE(2) route has no finite guidance direction")
        local /= norm
        unit = (float(local[0]), float(local[1]))

        return SE2RouteCompassReadout(
            schema_version=SE2_ROUTE_COMPASS_SCHEMA_VERSION,
            motion_model=self._motion_model,
            projected_progress_m=float(self._progress),
            progress_increment_m=progress_increment,
            progress_budget_m=progress_budget,
            route_progress_fraction=float(
                self._progress / self.route_extent_m),
            route_state_index=self._state_index(self._progress),
            reference_arc_m=float(reference_arc),
            query_path_length_m=float(self._query_path_length),
            cross_track_error_m=float(cross_track),
            estimated_position=tuple(float(v) for v in self._position),
            estimated_yaw_rad=float(self._yaw),
            projected_position=tuple(float(v) for v in projected),
            reference_position=tuple(float(v) for v in reference),
            terminal_tangent_held=terminal_hold,
            unit_bearing=unit,
            controller_pointgoal=tuple(
                self._controller_radius * value for value in unit),
        )
