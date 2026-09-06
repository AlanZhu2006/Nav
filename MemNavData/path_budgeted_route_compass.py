"""Causal route projection with a conserved cumulative path budget.

The earlier per-step projection discarded any motion budget that could not be
used immediately at a corner. A small scale or cross-track error therefore
became irreversible and could leave the projection on an old segment forever.
This variant preserves the unused budget:

    previous_progress <= progress_t <= min(route_extent, query_path_length_t)

Two-dimensional projection still decides *where* on the route the state lies;
travelled distance is only an upper bound. No distance regime, confidence
gate, endpoint branch, or fallback is introduced.
"""

from __future__ import annotations

import math

import numpy as np

from MemNavData.se2_projected_route_compass import (
    _finite_positive,
    _rotate,
    _wrap_angle,
    SE2_LOCAL_ODOMETRY_MODEL,
    SE2ProjectedRouteCompass,
    SE2RouteCompassReadout,
)


PATH_BUDGETED_ROUTE_COMPASS_SCHEMA_VERSION = (
    "path_budgeted_route_compass_v1_20260903"
)
TANGENT_PATH_BUDGETED_ROUTE_COMPASS_SCHEMA_VERSION = (
    "tangent_path_budgeted_route_compass_v1_20260903"
)
FIRST_FEASIBLE_TANGENT_ROUTE_COMPASS_SCHEMA_VERSION = (
    "first_feasible_local_tangent_route_compass_v1_20260903"
)
PATH_BUDGET_MODEL = "cumulative_query_path_upper_bound_v1"
FIRST_FEASIBLE_TANGENT_BASELINE_M = 0.30


class PathBudgetedRouteCompass(SE2ProjectedRouteCompass):
    """Use cumulative realized visual motion as a causal projection budget."""

    schema_version = PATH_BUDGETED_ROUTE_COMPASS_SCHEMA_VERSION

    def _guidance_reference_arc(self, default_arc: float) -> float:
        """Return the route coordinate used by the guidance readout.

        Existing pose-chord and 2.5 m route-chord experiments keep their
        historical lookahead exactly.  A subclass may change only this
        readout coordinate without changing projection or its path budget.
        """

        return float(default_arc)

    def _guidance_delta(
            self, reference_arc: float,
    ) -> tuple[np.ndarray, np.ndarray, bool]:
        """Return the historical lookahead chord from the live pose."""

        reference = self._sample(reference_arc)
        delta = reference - self._position
        terminal_hold = float(np.linalg.norm(delta)) <= 1e-12
        if terminal_hold:
            start_arc = max(0.0, self.route_extent_m - self._lookahead)
            delta = self._positions[-1] - self._sample(start_arc)
        return delta, reference, terminal_hold

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
        current_path_increment = float(np.linalg.norm(delta_previous_body))
        self._position = self._position + displacement
        self._yaw = _wrap_angle(self._yaw + float(yaw_delta))
        self._query_path_length += current_path_increment

        maximum_progress = min(
            self.route_extent_m, float(self._query_path_length))
        available_budget = max(0.0, maximum_progress - progress_before)
        progress, projected, cross_track = self._monotone_projection(
            maximum_progress_m=maximum_progress)
        if progress + 1e-9 < self._progress:
            raise RuntimeError("path-budgeted route progress regressed")
        self._progress = max(self._progress, float(progress))
        progress_increment = float(self._progress - progress_before)
        if progress_increment > available_budget + 1e-9:
            raise RuntimeError(
                "route progress exceeded cumulative query-path budget")

        default_reference_arc = min(
            self.route_extent_m, self._progress + self._lookahead)
        reference_arc = self._guidance_reference_arc(default_reference_arc)
        if (not math.isfinite(reference_arc)
                or reference_arc < self._progress - 1e-9
                or reference_arc > self.route_extent_m + 1e-9):
            raise RuntimeError("guidance reference escaped the forward route")
        delta, reference, terminal_hold = self._guidance_delta(reference_arc)

        cosine, sine = math.cos(self._yaw), math.sin(self._yaw)
        local = np.asarray([
            cosine * delta[0] + sine * delta[1],
            -sine * delta[0] + cosine * delta[1],
        ], dtype=np.float64)
        norm = float(np.linalg.norm(local))
        if not math.isfinite(norm) or norm <= 1e-12:
            raise RuntimeError("route has no finite guidance direction")
        local /= norm
        unit = (float(local[0]), float(local[1]))

        return SE2RouteCompassReadout(
            schema_version=self.schema_version,
            motion_model=self._motion_model,
            projected_progress_m=float(self._progress),
            progress_increment_m=progress_increment,
            progress_budget_m=available_budget,
            route_progress_fraction=float(
                self._progress / self.route_extent_m),
            route_state_index=self._state_index(self._progress),
            reference_arc_m=float(reference_arc),
            query_path_length_m=float(self._query_path_length),
            cross_track_error_m=float(cross_track),
            estimated_position=tuple(float(value) for value in self._position),
            estimated_yaw_rad=float(self._yaw),
            projected_position=tuple(float(value) for value in projected),
            reference_position=tuple(float(value) for value in reference),
            terminal_tangent_held=terminal_hold,
            unit_bearing=unit,
            controller_pointgoal=tuple(
                self._controller_radius * value for value in unit),
        )


class TangentPathBudgetedRouteCompass(PathBudgetedRouteCompass):
    """Read route direction without using the drifted live position.

    The live position still participates in monotone route projection, where
    it prevents lateral travel from automatically becoming route progress.
    Once progress is fixed, guidance is the forward historical route chord
    from ``s`` to ``s + lookahead``.  Only the accumulated relative yaw is
    needed to express that direction in the current camera frame.  Thus
    transverse visual-odometry drift cannot rotate the bearing merely by
    moving the estimated query origin away from the route.
    """

    schema_version = TANGENT_PATH_BUDGETED_ROUTE_COMPASS_SCHEMA_VERSION

    def _guidance_delta(
            self, reference_arc: float,
    ) -> tuple[np.ndarray, np.ndarray, bool]:
        route_position = self._sample(self._progress)
        reference = self._sample(reference_arc)
        delta = reference - route_position
        terminal_hold = float(np.linalg.norm(delta)) <= 1e-12
        if terminal_hold:
            start_arc = max(0.0, self.route_extent_m - self._lookahead)
            delta = self._positions[-1] - self._sample(start_arc)
        return delta, reference, terminal_hold


class FirstFeasibleLocalTangentRouteCompass(
        TangentPathBudgetedRouteCompass):
    """Read the first observable forward direction of the authorized route.

    The legacy ``TangentPathBudgetedRouteCompass`` is intentionally retained
    because its published development receipts used an arc-ahead 2.5 m chord.
    This class implements the distinct mechanism isolated by the frozen
    chord-versus-tangent experiment: starting at the current monotone route
    coordinate, select the earliest future route vertex whose planar baseline
    reaches one local-motion observation (0.30 m by default).  The selected
    vector is normalized before control, so neither route length nor endpoint
    distance enters NavDP.

    The baseline is an observability primitive, not a distance regime or an
    intervention gate.  Every valid route coordinate yields exactly one
    direction; at the endpoint the last observable tangent is held.
    """

    schema_version = FIRST_FEASIBLE_TANGENT_ROUTE_COMPASS_SCHEMA_VERSION

    def __init__(
        self,
        route_positions_m: np.ndarray,
        *,
        tangent_baseline_m: float = FIRST_FEASIBLE_TANGENT_BASELINE_M,
        controller_radius_m: float = 2.5,
        motion_model: str = SE2_LOCAL_ODOMETRY_MODEL,
    ) -> None:
        self._tangent_baseline = _finite_positive(
            tangent_baseline_m, "tangent_baseline_m")
        super().__init__(
            route_positions_m,
            lookahead_m=self._tangent_baseline,
            controller_radius_m=controller_radius_m,
            motion_model=motion_model,
        )

    @property
    def tangent_baseline_m(self) -> float:
        return float(self._tangent_baseline)

    def _guidance_reference_arc(self, default_arc: float) -> float:
        del default_arc
        if self._progress >= self.route_extent_m - 1e-12:
            return self.route_extent_m
        origin = self._sample(self._progress)
        # Include the end of the currently occupied segment, then every later
        # historical vertex.  Selecting the earliest qualifying vertex keeps
        # the route's temporal order and does not skip the first corner.
        first_vertex = int(np.searchsorted(
            self._arc, self._progress, side="right"))
        for index in range(first_vertex, len(self._positions)):
            candidate = self._positions[index]
            if float(np.linalg.norm(candidate - origin)) + 1e-12 >= (
                    self._tangent_baseline):
                return float(self._arc[index])
        return self.route_extent_m

    def _guidance_delta(
            self, reference_arc: float,
    ) -> tuple[np.ndarray, np.ndarray, bool]:
        route_position = self._sample(self._progress)
        reference = self._sample(reference_arc)
        delta = reference - route_position
        terminal_hold = float(np.linalg.norm(delta)) <= 1e-12
        if terminal_hold:
            endpoint = self._positions[-1]
            # Walk backwards from the endpoint and retain the closest route
            # vertex that supplies the same observable baseline.  This is the
            # boundary value of the continuous route field, not a new endpoint
            # controller or fallback.
            delta = None
            for index in range(len(self._positions) - 2, -1, -1):
                candidate = endpoint - self._positions[index]
                if float(np.linalg.norm(candidate)) + 1e-12 >= (
                        self._tangent_baseline):
                    delta = candidate
                    break
            if delta is None:
                delta = endpoint - self._positions[0]
            reference = endpoint.copy()
        return np.asarray(delta, dtype=np.float64), reference, terminal_hold


def from_local_route(
    route_positions_m: np.ndarray,
    *,
    lookahead_m: float = 2.5,
    controller_radius_m: float = 2.5,
) -> PathBudgetedRouteCompass:
    return PathBudgetedRouteCompass(
        route_positions_m,
        lookahead_m=lookahead_m,
        controller_radius_m=controller_radius_m,
        motion_model=SE2_LOCAL_ODOMETRY_MODEL,
    )


def tangent_from_local_route(
    route_positions_m: np.ndarray,
    *,
    lookahead_m: float = 2.5,
    controller_radius_m: float = 2.5,
) -> TangentPathBudgetedRouteCompass:
    return TangentPathBudgetedRouteCompass(
        route_positions_m,
        lookahead_m=lookahead_m,
        controller_radius_m=controller_radius_m,
        motion_model=SE2_LOCAL_ODOMETRY_MODEL,
    )


def first_feasible_tangent_from_local_route(
    route_positions_m: np.ndarray,
    *,
    tangent_baseline_m: float = FIRST_FEASIBLE_TANGENT_BASELINE_M,
    controller_radius_m: float = 2.5,
) -> FirstFeasibleLocalTangentRouteCompass:
    return FirstFeasibleLocalTangentRouteCompass(
        route_positions_m,
        tangent_baseline_m=tangent_baseline_m,
        controller_radius_m=controller_radius_m,
        motion_model=SE2_LOCAL_ODOMETRY_MODEL,
    )
