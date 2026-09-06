"""Causal one-dimensional filtering on an already-authorized RGB route.

The target certificate identifies one historical endpoint.  This module does
not decide whether memory should be used and it does not plan in a metric map.
It tracks only *where along the temporally ordered, authorized history* the
live camera is likely to be.  The transition prior is calibrated from the
history's own monocular pose stream; DINO similarity is a soft observation.

The filter is deliberately monotone and total: every observation produces a
route-coordinate readout.  There is no distance regime, observation threshold,
endpoint chord, or native-policy fallback in this state estimator.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


ROUTE_FILTER_SCHEMA_VERSION = "episodic_route_filter_v1_20260902"
ACTION_COORDINATE_SCHEMA_VERSION = (
    "action_coordinate_route_compass_v1_20260902"
)


def _finite_positive(value: float, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return result


def _route_positions(value: np.ndarray) -> np.ndarray:
    positions = np.asarray(value, dtype=np.float64)
    if (positions.ndim != 2 or positions.shape[0] < 3
            or positions.shape[1] != 2 or not np.isfinite(positions).all()):
        raise ValueError("route_positions_m must be finite with shape [N>=3,2]")
    return positions


@dataclass(frozen=True)
class RouteTransitionReceipt:
    """Frozen motion prior derived only from the authorized causal route."""

    schema_version: str
    nominal_motion_m: float
    mean_advance_frames: int
    sigma_advance_frames: float
    maximum_advance_frames: int
    calibration_search_frames: int
    local_lag_q25: float
    local_lag_q50: float
    local_lag_q75: float
    calibrated_pair_count: int
    mean_chord_m_by_lag: tuple[float, ...]

    def audit_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "nominal_motion_m": self.nominal_motion_m,
            "mean_advance_frames": self.mean_advance_frames,
            "sigma_advance_frames": self.sigma_advance_frames,
            "maximum_advance_frames": self.maximum_advance_frames,
            "calibration_search_frames": self.calibration_search_frames,
            "local_lag_q25": self.local_lag_q25,
            "local_lag_q50": self.local_lag_q50,
            "local_lag_q75": self.local_lag_q75,
            "calibrated_pair_count": self.calibrated_pair_count,
            "mean_chord_m_by_lag": list(self.mean_chord_m_by_lag),
        }


def derive_route_transition(
    route_positions_m: np.ndarray,
    *,
    nominal_motion_m: float,
    maximum_search_frames: int = 64,
) -> RouteTransitionReceipt:
    """Derive one temporal transition kernel without evaluator geometry.

    A route state is one historical RGB address.  For each temporal lag we
    measure the mean monocular-pose chord on the frozen route, then select the
    lag whose expected chord best matches one controller update.  The kernel
    width is half the interquartile range of local first-passage lags.  Both
    quantities therefore come from the causal history itself rather than an
    outcome-selected route or a short/long distance class.
    """

    positions = _route_positions(route_positions_m)
    motion = _finite_positive(nominal_motion_m, "nominal_motion_m")
    search = int(maximum_search_frames)
    if search < 2:
        raise ValueError("maximum_search_frames must be at least two")
    search = min(search, len(positions) - 1)

    mean_chords = []
    for lag in range(1, search + 1):
        chords = np.linalg.norm(positions[lag:] - positions[:-lag], axis=1)
        mean_chords.append(float(np.mean(chords)))
    mean_advance = 1 + int(np.argmin(np.abs(
        np.asarray(mean_chords, dtype=np.float64) - motion)))

    local_lags: list[int] = []
    for start in range(len(positions) - 1):
        stop = min(len(positions), start + search + 1)
        chords = np.linalg.norm(
            positions[start + 1:stop] - positions[start], axis=1)
        reached = np.flatnonzero(chords >= motion)
        if len(reached):
            local_lags.append(1 + int(reached[0]))
    if len(local_lags) < max(3, len(positions) // 4):
        raise ValueError("causal route cannot calibrate one control transition")
    q25, q50, q75 = np.percentile(
        np.asarray(local_lags, dtype=np.float64), [25.0, 50.0, 75.0])
    sigma = max(1.0, 0.5 * float(q75 - q25))
    maximum_advance = min(
        search,
        max(mean_advance, int(math.ceil(mean_advance + 4.0 * sigma))),
    )
    return RouteTransitionReceipt(
        schema_version=ROUTE_FILTER_SCHEMA_VERSION,
        nominal_motion_m=motion,
        mean_advance_frames=mean_advance,
        sigma_advance_frames=sigma,
        maximum_advance_frames=maximum_advance,
        calibration_search_frames=search,
        local_lag_q25=float(q25),
        local_lag_q50=float(q50),
        local_lag_q75=float(q75),
        calibrated_pair_count=len(local_lags),
        mean_chord_m_by_lag=tuple(mean_chords),
    )


@dataclass(frozen=True)
class RouteCoordinateReadout:
    schema_version: str
    state_index: int
    observation_index: int
    source_index: int | None
    observation_standard_deviation: float

    def audit_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "state_index": self.state_index,
            "observation_index": self.observation_index,
            "source_index": self.source_index,
            "observation_standard_deviation": (
                self.observation_standard_deviation),
        }


class MonotoneRouteFilter:
    """Max-product filter over one temporally ordered visual route.

    The observation potential is a per-frame standardized DINO cosine vector.
    Standardization fixes the evidence scale without a similarity threshold.
    A zero-motion update (for an in-place action) retains route state exactly;
    a translational update uses the frozen causal transition receipt.
    """

    def __init__(
        self,
        state_count: int,
        transition: RouteTransitionReceipt,
        *,
        source_indices: np.ndarray | None = None,
    ) -> None:
        count = int(state_count)
        if count < 3:
            raise ValueError("state_count must be at least three")
        if transition.schema_version != ROUTE_FILTER_SCHEMA_VERSION:
            raise ValueError("transition receipt schema is incompatible")
        self._count = count
        self._transition = transition
        self._scores = np.full(count, -np.inf, dtype=np.float64)
        self._scores[0] = 0.0
        self._observation_index = 0
        self._last_state = 0
        if source_indices is None:
            self._source_indices = None
        else:
            indices = np.asarray(source_indices, dtype=np.int64)
            if indices.shape != (count,):
                raise ValueError("source_indices must have shape [state_count]")
            self._source_indices = indices.copy()

    @property
    def state_index(self) -> int:
        return self._last_state

    def _translation_prediction(self) -> np.ndarray:
        receipt = self._transition
        prediction = np.full(self._count, -np.inf, dtype=np.float64)
        for delta in range(receipt.maximum_advance_frames + 1):
            penalty = -0.5 * (
                (delta - receipt.mean_advance_frames)
                / receipt.sigma_advance_frames
            ) ** 2
            prediction[delta:] = np.maximum(
                prediction[delta:], self._scores[:self._count - delta]
                + penalty)
        return prediction

    def update(
        self,
        similarity: np.ndarray,
        *,
        translated: bool,
    ) -> RouteCoordinateReadout:
        observation = np.asarray(similarity, dtype=np.float64)
        if (observation.shape != (self._count,)
                or not np.isfinite(observation).all()):
            raise ValueError("similarity must be finite with shape [state_count]")
        standard_deviation = float(np.std(observation))
        if standard_deviation <= 1e-12:
            potential = np.zeros_like(observation)
        else:
            potential = (
                observation - float(np.mean(observation)))
            potential /= standard_deviation

        moved = bool(translated)
        prediction = (
            self._translation_prediction() if moved
            else self._scores.copy())
        self._scores = prediction + potential
        # The emitted coordinate is part of the causal state.  A later global
        # MAP path may not retroactively revise it to an earlier route address.
        # Nor may a low-scoring hypothesis already far ahead manufacture a
        # physically impossible jump.  This is the defining one-way route
        # dynamics, not an evidence gate.
        self._scores[:self._last_state] = -np.inf
        advance = self._transition.maximum_advance_frames if moved else 0
        self._scores[min(
            self._count, self._last_state + advance + 1):] = -np.inf
        maximum = float(np.max(self._scores))
        if not math.isfinite(maximum):
            raise RuntimeError("route-coordinate posterior became invalid")
        self._scores -= maximum
        state = int(np.argmax(self._scores))
        self._last_state = state
        source = (
            None if self._source_indices is None
            else int(self._source_indices[state]))
        readout = RouteCoordinateReadout(
            schema_version=ROUTE_FILTER_SCHEMA_VERSION,
            state_index=state,
            observation_index=self._observation_index,
            source_index=source,
            observation_standard_deviation=standard_deviation,
        )
        self._observation_index += 1
        return readout


def _camera_to_world_rotation(pose9: np.ndarray) -> np.ndarray:
    pose = np.asarray(pose9, dtype=np.float64)
    if pose.shape != (9,) or not np.isfinite(pose).all():
        raise ValueError("initial_pose9 must be a finite pose9 vector")
    x, y, z, w = pose[3:7]
    norm = float(np.linalg.norm([x, y, z, w]))
    if norm <= 1e-12:
        raise ValueError("initial_pose9 contains a zero quaternion")
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    return np.asarray([
        [1.0 - 2.0 * (y * y + z * z),
         2.0 * (x * y - z * w),
         2.0 * (x * z + y * w)],
        [2.0 * (x * y + z * w),
         1.0 - 2.0 * (x * x + z * z),
         2.0 * (y * z - x * w)],
        [2.0 * (x * z - y * w),
         2.0 * (y * z + x * w),
         1.0 - 2.0 * (x * x + y * y)],
    ], dtype=np.float64)


def _yaw_rotation(angle_rad: float) -> np.ndarray:
    cosine, sine = math.cos(angle_rad), math.sin(angle_rad)
    return np.asarray([
        [cosine, 0.0, sine],
        [0.0, 1.0, 0.0],
        [-sine, 0.0, cosine],
    ], dtype=np.float64)


@dataclass(frozen=True)
class RouteBearingReadout:
    schema_version: str
    state_index: int
    reference_arc_m: float
    terminal_tangent_held: bool
    cumulative_executor_yaw_rad: float
    unit_bearing: tuple[float, float]
    controller_pointgoal: tuple[float, float]

    def audit_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "state_index": self.state_index,
            "reference_arc_m": self.reference_arc_m,
            "terminal_tangent_held": self.terminal_tangent_held,
            "cumulative_executor_yaw_rad": self.cumulative_executor_yaw_rad,
            "unit_bearing": list(self.unit_bearing),
            "controller_pointgoal": list(self.controller_pointgoal),
        }


class ActionIntegratedRouteBearing:
    """Turn a filtered route coordinate into one fixed-radius local bearing.

    Long-horizon LingBot orientation is not reused after the goal switch.  The
    initial camera frame is known at the causal history tail; subsequent yaw
    is the executor's own efference copy.  This removes the opposite-view pose
    drift diagnosed in the pure-odometry audit without adding a sensor or a
    second controller.
    """

    def __init__(
        self,
        route_positions_m: np.ndarray,
        initial_pose9: np.ndarray,
        *,
        lookahead_m: float = 2.5,
    ) -> None:
        self._positions = _route_positions(route_positions_m).copy()
        self._initial_rotation = _camera_to_world_rotation(initial_pose9)
        self._lookahead = _finite_positive(lookahead_m, "lookahead_m")
        edges = np.linalg.norm(
            np.diff(self._positions, axis=0), axis=1)
        self._cumulative = np.concatenate([
            np.zeros(1, dtype=np.float64), np.cumsum(edges),
        ])
        if float(self._cumulative[-1]) <= 1e-6:
            raise ValueError("route has no spatial extent")
        self._executor_yaw = 0.0
        self._last_unit: tuple[float, float] | None = None

    def advance_executor_yaw(self, delta_yaw_rad: float) -> None:
        delta = float(delta_yaw_rad)
        if not math.isfinite(delta):
            raise ValueError("delta_yaw_rad must be finite")
        self._executor_yaw += delta

    def _sample(self, arc_m: float) -> np.ndarray:
        arc = float(np.clip(arc_m, 0.0, float(self._cumulative[-1])))
        if arc >= float(self._cumulative[-1]):
            return self._positions[-1].copy()
        segment = int(np.searchsorted(
            self._cumulative, arc, side="right") - 1)
        segment = min(max(segment, 0), len(self._positions) - 2)
        while (segment < len(self._positions) - 2
               and self._cumulative[segment + 1]
               <= self._cumulative[segment] + 1e-12):
            segment += 1
        low = float(self._cumulative[segment])
        high = float(self._cumulative[segment + 1])
        if high <= low + 1e-12:
            return self._positions[segment].copy()
        alpha = (arc - low) / (high - low)
        return ((1.0 - alpha) * self._positions[segment]
                + alpha * self._positions[segment + 1])

    def guidance(self, state_index: int) -> RouteBearingReadout:
        state = int(state_index)
        if not 0 <= state < len(self._positions):
            raise ValueError("state_index is outside the authorized route")
        current_arc = float(self._cumulative[state])
        reference_arc = min(
            float(self._cumulative[-1]), current_arc + self._lookahead)
        reference = self._sample(reference_arc)
        delta = reference - self._positions[state]
        terminal_hold = float(np.linalg.norm(delta)) <= 1e-9
        if terminal_hold:
            # The external ImageGoal success contract decides arrival.  The
            # route field therefore keeps its last finite tangent at the
            # boundary instead of introducing an endpoint mode or fallback.
            if self._last_unit is None:
                start = self._sample(max(
                    0.0, float(self._cumulative[-1]) - self._lookahead))
                delta = self._positions[-1] - start
            else:
                unit = self._last_unit
                return RouteBearingReadout(
                    schema_version=ROUTE_FILTER_SCHEMA_VERSION,
                    state_index=state,
                    reference_arc_m=reference_arc,
                    terminal_tangent_held=True,
                    cumulative_executor_yaw_rad=self._executor_yaw,
                    unit_bearing=unit,
                    controller_pointgoal=tuple(
                        self._lookahead * value for value in unit),
                )
        world = np.asarray([delta[0], 0.0, delta[1]], dtype=np.float64)
        # Habitat's executor yaw increases in the opposite handed direction
        # to LingBot's camera-to-world Y rotation.
        current_rotation = (
            _yaw_rotation(-self._executor_yaw) @ self._initial_rotation)
        local = current_rotation.T @ world
        vector = np.asarray([local[2], -local[0]], dtype=np.float64)
        norm = float(np.linalg.norm(vector))
        if not math.isfinite(norm) or norm <= 1e-12:
            raise RuntimeError("route tangent produced no finite bearing")
        vector /= norm
        unit = (float(vector[0]), float(vector[1]))
        self._last_unit = unit
        return RouteBearingReadout(
            schema_version=ROUTE_FILTER_SCHEMA_VERSION,
            state_index=state,
            reference_arc_m=reference_arc,
            terminal_tangent_held=terminal_hold,
            cumulative_executor_yaw_rad=self._executor_yaw,
            unit_bearing=unit,
            controller_pointgoal=tuple(
                self._lookahead * value for value in unit),
        )


@dataclass(frozen=True)
class ActionCoordinateReadout:
    """One total readout from the scale-free episodic route compass."""

    schema_version: str
    action_progress_m: float
    route_progress_fraction: float
    route_state_index: int
    reference_action_arc_m: float
    terminal_tangent_held: bool
    cumulative_executor_yaw_rad: float
    unit_bearing: tuple[float, float]
    controller_pointgoal: tuple[float, float]

    def audit_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "action_progress_m": self.action_progress_m,
            "route_progress_fraction": self.route_progress_fraction,
            "route_state_index": self.route_state_index,
            "reference_action_arc_m": self.reference_action_arc_m,
            "terminal_tangent_held": self.terminal_tangent_held,
            "cumulative_executor_yaw_rad": (
                self.cumulative_executor_yaw_rad),
            "unit_bearing": list(self.unit_bearing),
            "controller_pointgoal": list(self.controller_pointgoal),
        }


class ActionCoordinateRouteCompass:
    """Follow an authorized history in the executor's own action coordinate.

    Historical route geometry may have arbitrary monocular scale.  Its action
    arc is the cumulative translation already issued while the causal RGB
    history was recorded.  During Revisit, executor translation and yaw
    receipts advance the same coordinate.  Geometry is used only for a
    direction, which is normalized before reaching NavDP.

    Once initialized, every finite action receipt produces guidance.  There
    is no distance class, visual threshold, endpoint mode, or native fallback.
    """

    def __init__(
        self,
        route_positions: np.ndarray,
        route_action_arc_m: np.ndarray,
        initial_pose9: np.ndarray,
        *,
        lookahead_m: float = 2.5,
        controller_radius_m: float = 2.5,
    ) -> None:
        self._positions = _route_positions(route_positions).copy()
        action_arc = np.asarray(route_action_arc_m, dtype=np.float64)
        if (action_arc.shape != (len(self._positions),)
                or not np.isfinite(action_arc).all()
                or abs(float(action_arc[0])) > 1e-9
                or np.any(np.diff(action_arc) < -1e-12)
                or float(action_arc[-1]) <= 1e-6):
            raise ValueError(
                "route_action_arc_m must be finite, nondecreasing, start at "
                "zero, and have positive extent")
        self._action_arc = action_arc.copy()
        self._initial_rotation = _camera_to_world_rotation(initial_pose9)
        self._lookahead = _finite_positive(lookahead_m, "lookahead_m")
        self._controller_radius = _finite_positive(
            controller_radius_m, "controller_radius_m")
        self._progress = 0.0
        self._executor_yaw = 0.0

    @property
    def action_progress_m(self) -> float:
        return self._progress

    @property
    def route_action_extent_m(self) -> float:
        return float(self._action_arc[-1])

    def _sample(self, action_arc_m: float) -> np.ndarray:
        arc = float(np.clip(
            action_arc_m, 0.0, self.route_action_extent_m))
        if arc >= self.route_action_extent_m:
            return self._positions[-1].copy()
        segment = int(np.searchsorted(
            self._action_arc, arc, side="right") - 1)
        segment = min(max(segment, 0), len(self._positions) - 2)
        while (segment < len(self._positions) - 2
               and self._action_arc[segment + 1]
               <= self._action_arc[segment] + 1e-12):
            segment += 1
        low = float(self._action_arc[segment])
        high = float(self._action_arc[segment + 1])
        if high <= low + 1e-12:
            return self._positions[segment].copy()
        alpha = (arc - low) / (high - low)
        return ((1.0 - alpha) * self._positions[segment]
                + alpha * self._positions[segment + 1])

    def _state_index(self, action_arc_m: float) -> int:
        state = int(np.searchsorted(
            self._action_arc, action_arc_m, side="right") - 1)
        return min(max(state, 0), len(self._positions) - 1)

    def advance(
        self,
        *,
        executed_translation_m: float,
        executed_yaw_rad: float,
    ) -> ActionCoordinateReadout:
        translation = float(executed_translation_m)
        yaw = float(executed_yaw_rad)
        if not math.isfinite(translation) or translation < 0.0:
            raise ValueError(
                "executed_translation_m must be finite and non-negative")
        if not math.isfinite(yaw):
            raise ValueError("executed_yaw_rad must be finite")
        self._progress = min(
            self.route_action_extent_m, self._progress + translation)
        self._executor_yaw += yaw

        reference_arc = min(
            self.route_action_extent_m, self._progress + self._lookahead)
        current = self._sample(self._progress)
        reference = self._sample(reference_arc)
        delta = reference - current
        terminal_hold = float(np.linalg.norm(delta)) <= 1e-12
        if terminal_hold:
            # Preserve the terminal direction of the same continuous vector
            # field.  Arrival remains the ImageGoal contract; no new endpoint
            # controller or fallback is introduced.
            start_arc = max(0.0, self.route_action_extent_m - self._lookahead)
            delta = self._positions[-1] - self._sample(start_arc)

        world = np.asarray([delta[0], 0.0, delta[1]], dtype=np.float64)
        current_rotation = (
            _yaw_rotation(-self._executor_yaw) @ self._initial_rotation)
        local = current_rotation.T @ world
        vector = np.asarray([local[2], -local[0]], dtype=np.float64)
        norm = float(np.linalg.norm(vector))
        if not math.isfinite(norm) or norm <= 1e-12:
            raise RuntimeError("action-coordinate route has no finite tangent")
        vector /= norm
        unit = (float(vector[0]), float(vector[1]))
        return ActionCoordinateReadout(
            schema_version=ACTION_COORDINATE_SCHEMA_VERSION,
            action_progress_m=float(self._progress),
            route_progress_fraction=(
                float(self._progress) / self.route_action_extent_m),
            route_state_index=self._state_index(self._progress),
            reference_action_arc_m=float(reference_arc),
            terminal_tangent_held=terminal_hold,
            cumulative_executor_yaw_rad=float(self._executor_yaw),
            unit_bearing=unit,
            controller_pointgoal=tuple(
                self._controller_radius * value for value in unit),
        )
