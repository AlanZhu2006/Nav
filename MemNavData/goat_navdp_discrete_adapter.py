#!/usr/bin/env python3
"""Deterministic NavDP metric-waypoint to GOAT action conversion.

This is the released InternNav pure-pursuit-style discrete conversion adapted
to the GOAT action quantum.  The legacy NavDP HTTP server already returns
cumulative metric waypoints, so this module intentionally does not reconstruct
positions from deltas or apply InternNav's raw-network-output scale factor.

Important semantic boundary: a no-motion NavDP trajectory is an *arrival
proposal*, not Habitat's semantic ``SUBTASK_STOP`` action.  Only an external
arrival verifier may authorize that action.  Keeping the proposal and the
official action in different types prevents the adapter bug found by the first
GOAT runtime pilot from recurring.
"""

from __future__ import annotations

import enum
import math
from dataclasses import dataclass

import numpy as np


class GoatNavAction(enum.IntEnum):
    SUBTASK_STOP = 0
    MOVE_FORWARD = 1
    TURN_LEFT = 2
    TURN_RIGHT = 3


class NavDPAdapterDisposition(str, enum.Enum):
    """Outcome of converting one NavDP trajectory."""

    MOTION = "motion"
    ARRIVAL_PROPOSAL = "arrival_proposal"
    CONVERSION_STALLED = "conversion_stalled"


@dataclass(frozen=True)
class NavDPAdapterDecision:
    """Typed adapter result; ``actions`` never contains ``SUBTASK_STOP``."""

    disposition: NavDPAdapterDisposition
    actions: tuple[GoatNavAction, ...]
    endpoint_norm_m: float
    reason: str
    max_radius_m: float = 0.0
    candidate_index: int | None = None

    @property
    def is_motion(self) -> bool:
        return self.disposition is NavDPAdapterDisposition.MOTION

    @property
    def requires_arrival_certificate(self) -> bool:
        return self.disposition is NavDPAdapterDisposition.ARRIVAL_PROPOSAL


@dataclass(frozen=True)
class DiscreteAdapterConfig:
    forward_step_m: float = 0.25
    turn_angle_deg: float = 30.0
    endpoint_stop_radius_m: float = 0.20
    lookahead_points: int = 4
    lookahead_distance_m: float | None = None
    execution_horizon: int = 8

    def validate(self) -> None:
        if self.forward_step_m <= 0:
            raise ValueError("forward_step_m must be positive")
        if self.turn_angle_deg <= 0 or 360 % self.turn_angle_deg != 0:
            raise ValueError("turn_angle_deg must be a positive divisor of 360")
        if self.endpoint_stop_radius_m < 0:
            raise ValueError("endpoint_stop_radius_m must be non-negative")
        if self.lookahead_points < 1:
            raise ValueError("lookahead_points must be positive")
        if (self.lookahead_distance_m is not None
                and self.lookahead_distance_m <= 0.0):
            raise ValueError("lookahead_distance_m must be positive")
        if self.execution_horizon < 1:
            raise ValueError("execution_horizon must be positive")


def _normalize_angle(angle: float) -> float:
    return (angle + math.pi) % (2 * math.pi) - math.pi


def navdp_waypoints_to_goat_decision(
    waypoints: np.ndarray,
    config: DiscreteAdapterConfig = DiscreteAdapterConfig(),
) -> NavDPAdapterDecision:
    """Convert one ``[K,3]`` cumulative path without authorizing STOP.

    NavDP uses local ``x=forward, y=left`` metric coordinates.  A near-zero
    endpoint returns ``ARRIVAL_PROPOSAL`` with no actions.  A geometrically
    unusable nonzero path returns ``CONVERSION_STALLED``.  Neither case is a
    semantic arrival decision.
    """

    config.validate()
    trajectory = np.asarray(waypoints, dtype=np.float64)
    if trajectory.ndim != 2 or trajectory.shape[1] < 2:
        raise ValueError("waypoints must have shape [K, >=2]")
    if trajectory.shape[0] == 0:
        raise ValueError("waypoints must not be empty")
    trajectory = trajectory[:, :2]
    if not np.isfinite(trajectory).all():
        raise ValueError("waypoints must be finite")

    # InternNav's inner converter expects an explicit origin.  The HTTP server
    # returns only the cumulative future points, so prepend it exactly once.
    if np.linalg.norm(trajectory[0]) > 1e-9:
        trajectory = np.concatenate(
            [np.zeros((1, 2), dtype=np.float64), trajectory], axis=0
        )

    position = trajectory[0].copy()
    original_goal = trajectory[-1]
    endpoint_norm = float(np.linalg.norm(position - original_goal))
    radii = np.linalg.norm(trajectory - position[None, :], axis=1)
    max_radius = float(radii.max())
    if max_radius <= config.endpoint_stop_radius_m:
        return NavDPAdapterDecision(
            disposition=NavDPAdapterDisposition.ARRIVAL_PROPOSAL,
            actions=(),
            endpoint_norm_m=endpoint_norm,
            reason="entire_trajectory_within_native_zero_radius",
            max_radius_m=max_radius,
        )

    # A cumulative path can loop back near the origin.  Its endpoint is not a
    # no-motion proposal.  Execute only the outward prefix instead of either
    # declaring arrival or following the stale return segment.
    control_goal_index = len(trajectory) - 1
    loop_endpoint_truncated = (
        endpoint_norm <= config.endpoint_stop_radius_m)
    if loop_endpoint_truncated:
        control_goal_index = int(np.argmax(radii))
        trajectory = trajectory[:control_goal_index + 1]
    goal = trajectory[-1]

    yaw = 0.0
    turn_angle_rad = math.radians(config.turn_angle_deg)
    actions: list[GoatNavAction] = []

    while np.linalg.norm(position - goal) > config.endpoint_stop_radius_m:
        distances = np.linalg.norm(trajectory - position[None, :], axis=1)
        nearest_index = int(np.argmin(distances))
        if config.lookahead_distance_m is None:
            target_index = min(
                nearest_index + config.lookahead_points,
                len(trajectory) - 1,
            )
        else:
            ahead = np.flatnonzero(
                distances[nearest_index:] >= config.lookahead_distance_m)
            target_index = (
                nearest_index + int(ahead[0])
                if len(ahead) else len(trajectory) - 1)
        target_delta = trajectory[target_index] - position
        if np.linalg.norm(target_delta) < 1e-6:
            break

        target_yaw = math.atan2(target_delta[1], target_delta[0])
        delta_yaw = _normalize_angle(target_yaw - yaw)
        turn_count = int(round(delta_yaw / turn_angle_rad))
        turn_action = (
            GoatNavAction.TURN_LEFT
            if turn_count > 0
            else GoatNavAction.TURN_RIGHT
        )
        actions.extend([turn_action] * abs(turn_count))
        yaw = _normalize_angle(yaw + turn_count * turn_angle_rad)

        next_position = position + config.forward_step_m * np.array(
            [math.cos(yaw), math.sin(yaw)], dtype=np.float64
        )
        if np.linalg.norm(next_position - goal) > np.linalg.norm(
            position - goal
        ):
            break
        actions.append(GoatNavAction.MOVE_FORWARD)
        position = next_position

        # Only actions in the frozen execution chunk are consumed before the
        # next closed-loop NavDP observation and replan.
        if len(actions) >= config.execution_horizon:
            break

    if not actions:
        return NavDPAdapterDecision(
            disposition=NavDPAdapterDisposition.CONVERSION_STALLED,
            actions=(),
            endpoint_norm_m=endpoint_norm,
            reason="pure_pursuit_produced_no_motion",
            max_radius_m=max_radius,
        )
    chunk = tuple(actions[: config.execution_horizon])
    if GoatNavAction.SUBTASK_STOP in chunk:
        raise AssertionError("motion conversion emitted semantic SUBTASK_STOP")
    return NavDPAdapterDecision(
        disposition=NavDPAdapterDisposition.MOTION,
        actions=chunk,
        endpoint_norm_m=endpoint_norm,
        reason=(
            "motion_chunk_loop_endpoint_truncated"
            if loop_endpoint_truncated else "motion_chunk"),
        max_radius_m=max_radius,
    )


def best_scored_motion_candidate(
    all_trajectories: np.ndarray,
    all_values: np.ndarray,
    config: DiscreteAdapterConfig = DiscreteAdapterConfig(),
) -> NavDPAdapterDecision:
    """Choose the highest-critic motion candidate from one frozen batch.

    This is the fail-safe continuation after an arrival proposal is rejected.
    It consumes no extra observation, diffusion sample, or learned module.  If
    every candidate is no-motion, the function fails closed with an empty
    ``CONVERSION_STALLED`` decision instead of fabricating a STOP or motion.
    """

    trajectories = np.asarray(all_trajectories, dtype=np.float64)
    values = np.asarray(all_values, dtype=np.float64)
    if trajectories.ndim == 4 and trajectories.shape[0] == 1:
        trajectories = trajectories[0]
    if values.ndim == 2 and values.shape[0] == 1:
        values = values[0]
    if trajectories.ndim != 3 or trajectories.shape[2] < 2:
        raise ValueError("all_trajectories must have shape [N,K,>=2]")
    if values.shape != (len(trajectories),):
        raise ValueError("all_values must have one score per trajectory")
    if not np.isfinite(trajectories).all() or not np.isfinite(values).all():
        raise ValueError("candidate trajectories and values must be finite")

    # Stable index tie-break makes equal critic scores deterministic.
    order = np.lexsort((np.arange(len(values)), -values))
    for index in order.tolist():
        decision = navdp_waypoints_to_goat_decision(
            trajectories[index], config)
        if decision.is_motion:
            return NavDPAdapterDecision(
                disposition=decision.disposition,
                actions=decision.actions,
                endpoint_norm_m=decision.endpoint_norm_m,
                reason="highest_scored_same_batch_motion",
                max_radius_m=decision.max_radius_m,
                candidate_index=int(index),
            )
    return NavDPAdapterDecision(
        disposition=NavDPAdapterDisposition.CONVERSION_STALLED,
        actions=(),
        endpoint_norm_m=0.0,
        reason="same_batch_contains_no_motion_candidate",
        max_radius_m=0.0,
    )


def navdp_waypoints_to_goat_actions(
    waypoints: np.ndarray,
    config: DiscreteAdapterConfig = DiscreteAdapterConfig(),
) -> tuple[GoatNavAction, ...]:
    """Motion-only compatibility wrapper.

    Empty output means the caller must either run an arrival certificate or
    fail closed.  This function deliberately never returns ``SUBTASK_STOP``.
    New runtime code should use :func:`navdp_waypoints_to_goat_decision` so the
    reason for the empty chunk is explicit.
    """

    return navdp_waypoints_to_goat_decision(waypoints, config).actions
