import math

import numpy as np
import pytest

from MemNavData.goat_terminal_alignment import (
    TerminalAlignmentConfig,
    TerminalAlignmentDisposition,
    plan_terminal_alignment,
    relative_optical_yaw_pitch_deg,
)


def pose(yaw_deg=0.0, pitch_deg=0.0):
    yaw = math.radians(yaw_deg)
    pitch = math.radians(pitch_deg)
    rotation_y = np.array([
        [math.cos(yaw), 0.0, math.sin(yaw)],
        [0.0, 1.0, 0.0],
        [-math.sin(yaw), 0.0, math.cos(yaw)],
    ])
    rotation_x = np.array([
        [1.0, 0.0, 0.0],
        [0.0, math.cos(pitch), -math.sin(pitch)],
        [0.0, math.sin(pitch), math.cos(pitch)],
    ])
    rotation = rotation_y @ rotation_x
    # Matrix-to-quaternion closed form is already tested in the geometry module.
    from MemNavData.lingbot_colored_registration import (
        matrix_to_quaternion_xyzw,
    )
    return np.r_[
        np.zeros(3), matrix_to_quaternion_xyzw(rotation),
        math.radians(60.0), math.radians(60.0),
    ]


def test_relative_optical_axis_recovers_yaw_right_and_pitch_up():
    yaw, pitch = relative_optical_yaw_pitch_deg(
        pose(), pose(yaw_deg=60.0, pitch_deg=30.0))
    assert yaw == pytest.approx(60.0)
    assert pitch == pytest.approx(30.0)


def test_not_eligible_never_emits_motion():
    result = plan_terminal_alignment(
        90.0, 30.0, certificate_accepted=False, near_goal=True)
    assert result.disposition is TerminalAlignmentDisposition.NOT_ELIGIBLE
    assert result.actions == ()


def test_quantizes_to_native_goat_turn_and_look_actions():
    result = plan_terminal_alignment(
        62.0, -28.0, certificate_accepted=True, near_goal=True)
    assert result.disposition is TerminalAlignmentDisposition.MOTION
    assert result.actions == ("turn_right", "turn_right", "look_down")
    assert result.residual_yaw_right_deg == pytest.approx(2.0)
    assert result.residual_pitch_up_deg == pytest.approx(2.0)
    assert "subtask_stop" not in result.actions


def test_within_tolerance_leaves_stop_authority_to_goat():
    result = plan_terminal_alignment(
        12.0, -10.0, certificate_accepted=True, near_goal=True)
    assert result.disposition is TerminalAlignmentDisposition.ALREADY_ALIGNED
    assert result.actions == ()


def test_out_of_envelope_and_budget_abstain_atomically():
    outside = plan_terminal_alignment(
        0.0, 75.0, certificate_accepted=True, near_goal=True)
    assert outside.disposition is TerminalAlignmentDisposition.ABSTAIN
    assert outside.actions == ()

    budget = plan_terminal_alignment(
        180.0, 60.0, certificate_accepted=True, near_goal=True,
        config=TerminalAlignmentConfig(max_actions=7))
    assert budget.disposition is TerminalAlignmentDisposition.ABSTAIN
    assert budget.actions == ()


def test_nonfinite_visual_proposal_is_rejected():
    with pytest.raises(ValueError):
        plan_terminal_alignment(
            float("nan"), 0.0, certificate_accepted=True, near_goal=True)

