#!/usr/bin/env python3
"""Fail-closed terminal view alignment using GOAT's native discrete actions.

This module is deliberately not an arrival detector.  A geometric module may
propose a residual camera yaw/pitch only after it has certified a Revisit and a
separate navigation controller has reached the local target.  The output here
contains only TURN/LOOK actions; the frozen GOAT policy retains sole authority
to emit ``subtask_stop``.

Angle convention:

* positive yaw means the desired optical axis lies to the camera's right;
* positive pitch means the desired optical axis lies above the camera.
"""

from __future__ import annotations

from dataclasses import dataclass
import enum
import math
from typing import Sequence

import numpy as np


def _quaternion_xyzw_to_matrix(quaternion: Sequence[float]) -> np.ndarray:
    quaternion = np.asarray(quaternion, dtype=np.float64)
    if quaternion.shape != (4,) or not np.isfinite(quaternion).all():
        raise ValueError("quaternion must be finite xyzw")
    norm = float(np.linalg.norm(quaternion))
    if norm <= 1e-12:
        raise ValueError("quaternion norm is zero")
    x, y, z, w = quaternion / norm
    return np.array([
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


class TerminalAlignmentDisposition(str, enum.Enum):
    NOT_ELIGIBLE = "not_eligible"
    ALREADY_ALIGNED = "already_aligned"
    MOTION = "motion"
    ABSTAIN = "abstain"


@dataclass(frozen=True)
class TerminalAlignmentConfig:
    yaw_quantum_deg: float = 30.0
    pitch_quantum_deg: float = 30.0
    yaw_tolerance_deg: float = 15.0
    pitch_tolerance_deg: float = 15.0
    max_abs_yaw_deg: float = 180.0
    max_abs_pitch_deg: float = 60.0
    max_actions: int = 8

    def validate(self) -> None:
        positive = (
            self.yaw_quantum_deg,
            self.pitch_quantum_deg,
            self.yaw_tolerance_deg,
            self.pitch_tolerance_deg,
            self.max_abs_yaw_deg,
            self.max_abs_pitch_deg,
        )
        if not all(math.isfinite(value) and value > 0.0 for value in positive):
            raise ValueError("terminal-alignment angles must be finite positive")
        if self.yaw_tolerance_deg > self.yaw_quantum_deg / 2.0:
            raise ValueError("yaw tolerance may not exceed half a yaw quantum")
        if self.pitch_tolerance_deg > self.pitch_quantum_deg / 2.0:
            raise ValueError("pitch tolerance may not exceed half a pitch quantum")
        if self.max_actions < 1:
            raise ValueError("terminal-alignment action budget must be positive")


@dataclass(frozen=True)
class TerminalAlignmentDecision:
    disposition: TerminalAlignmentDisposition
    actions: tuple[str, ...]
    requested_yaw_right_deg: float
    requested_pitch_up_deg: float
    residual_yaw_right_deg: float
    residual_pitch_up_deg: float
    reason: str

    @property
    def is_motion(self) -> bool:
        return self.disposition is TerminalAlignmentDisposition.MOTION


def _nearest_step_count(angle_deg: float, quantum_deg: float,
                        tolerance_deg: float) -> int:
    if abs(angle_deg) <= tolerance_deg:
        return 0
    magnitude = int(math.floor(abs(angle_deg) / quantum_deg + 0.5))
    return magnitude if angle_deg > 0.0 else -magnitude


def relative_optical_yaw_pitch_deg(
        current_pose9: Sequence[float], goal_pose9: Sequence[float]
) -> tuple[float, float]:
    """Return goal optical-axis yaw-right and pitch-up in current camera axes."""
    current = np.asarray(current_pose9, dtype=np.float64)
    goal = np.asarray(goal_pose9, dtype=np.float64)
    if (current.shape != (9,) or goal.shape != (9,)
            or not np.isfinite(current).all() or not np.isfinite(goal).all()):
        raise ValueError("current and goal poses must be finite pose9 vectors")
    current_rotation = _quaternion_xyzw_to_matrix(current[3:7])
    goal_rotation = _quaternion_xyzw_to_matrix(goal[3:7])
    goal_forward_current = current_rotation.T @ goal_rotation[:, 2]
    norm = float(np.linalg.norm(goal_forward_current))
    if norm <= 1e-12:
        raise ValueError("relative optical axis is degenerate")
    x_right, y_down, z_forward = goal_forward_current / norm
    yaw_right = math.degrees(math.atan2(x_right, z_forward))
    pitch_up = math.degrees(
        math.atan2(-y_down, math.hypot(x_right, z_forward)))
    return yaw_right, pitch_up


def plan_terminal_alignment(
        yaw_right_deg: float, pitch_up_deg: float, *,
        certificate_accepted: bool, near_goal: bool,
        config: TerminalAlignmentConfig = TerminalAlignmentConfig(),
) -> TerminalAlignmentDecision:
    """Quantize one complete residual or abstain without partial execution."""
    config.validate()
    yaw = float(yaw_right_deg)
    pitch = float(pitch_up_deg)
    if not math.isfinite(yaw) or not math.isfinite(pitch):
        raise ValueError("terminal-alignment proposal must be finite")

    def decision(disposition: TerminalAlignmentDisposition,
                 actions: tuple[str, ...], residual_yaw: float,
                 residual_pitch: float, reason: str) -> TerminalAlignmentDecision:
        return TerminalAlignmentDecision(
            disposition=disposition,
            actions=actions,
            requested_yaw_right_deg=yaw,
            requested_pitch_up_deg=pitch,
            residual_yaw_right_deg=residual_yaw,
            residual_pitch_up_deg=residual_pitch,
            reason=reason,
        )

    if not certificate_accepted or not near_goal:
        return decision(
            TerminalAlignmentDisposition.NOT_ELIGIBLE, (), yaw, pitch,
            "requires_certified_revisit_and_near_goal")
    if (abs(yaw) > config.max_abs_yaw_deg
            or abs(pitch) > config.max_abs_pitch_deg):
        return decision(
            TerminalAlignmentDisposition.ABSTAIN, (), yaw, pitch,
            "angle_outside_safe_envelope")

    yaw_steps = _nearest_step_count(
        yaw, config.yaw_quantum_deg, config.yaw_tolerance_deg)
    pitch_steps = _nearest_step_count(
        pitch, config.pitch_quantum_deg, config.pitch_tolerance_deg)
    residual_yaw = yaw - yaw_steps * config.yaw_quantum_deg
    residual_pitch = pitch - pitch_steps * config.pitch_quantum_deg
    if (abs(residual_yaw) > config.yaw_tolerance_deg + 1e-9
            or abs(residual_pitch) > config.pitch_tolerance_deg + 1e-9):
        return decision(
            TerminalAlignmentDisposition.ABSTAIN, (), yaw, pitch,
            "discrete_quantization_outside_tolerance")

    actions = (
        (("turn_right",) * yaw_steps if yaw_steps > 0
         else ("turn_left",) * (-yaw_steps))
        + (("look_up",) * pitch_steps if pitch_steps > 0
           else ("look_down",) * (-pitch_steps))
    )
    if len(actions) > config.max_actions:
        return decision(
            TerminalAlignmentDisposition.ABSTAIN, (), yaw, pitch,
            "complete_alignment_exceeds_action_budget")
    if not actions:
        return decision(
            TerminalAlignmentDisposition.ALREADY_ALIGNED, (), yaw, pitch,
            "within_terminal_view_tolerance")
    return decision(
        TerminalAlignmentDisposition.MOTION, actions,
        residual_yaw, residual_pitch, "bounded_discrete_alignment")
