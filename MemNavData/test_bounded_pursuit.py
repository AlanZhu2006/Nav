import math
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from MemNavData.bounded_pursuit import PursuitLimits, apply_collision, command
from MemNavData.habitat_executor_audit import DEFAULTS, commanded_step


@pytest.mark.parametrize("yaw", [0, 0.4, -2.1, math.pi])
def test_zero_reference_has_no_motion_or_invented_heading(yaw):
    pos = np.array([2.0, 0.3, -5.0])
    result = command(pos, yaw, np.repeat(pos[[0, 2]][None], 24, axis=0))
    np.testing.assert_array_equal(result.position, pos)
    assert result.yaw == yaw and result.displacement_m == 0
    assert result.reason == "zero_reference_hold_not_arrival"


def test_short_straight_reference_cannot_be_overrun_and_then_holds():
    pos = np.zeros(3)
    path = np.array([[0., -.01], [0., -.02]])
    first = command(pos, 0, path)
    assert first.displacement_m == pytest.approx(.02)
    np.testing.assert_allclose(first.position, [0, 0, -.02], atol=1e-12)
    second = command(first.position, first.yaw, path)
    assert second.displacement_m == 0


def test_zero_hold_does_not_invoke_map_to_generate_movement():
    def forbidden(*args):
        raise AssertionError("a zero request must not acquire a projected destination")
    pos = np.array([3., .1, 7.])
    result = command(pos, .9, [[3., 7.]])
    p, yaw, distance = apply_collision(pos, result, forbidden)
    np.testing.assert_array_equal(p, pos)
    assert yaw == .9 and distance == 0


def test_one_environment_request_no_short_retry_and_measured_distance():
    calls = []
    def wall(start, end):
        calls.append((start.copy(), end.copy()))
        return np.array([.005, 0, -.008])
    result = command(np.zeros(3), 0, [[0, -2]])
    p, _, distance = apply_collision(np.zeros(3), result, wall)
    assert len(calls) == 1
    assert distance == pytest.approx(math.hypot(.005, .008))
    assert distance != result.displacement_m


def test_nonbinding_reference_preserves_original_command_exactly():
    rng = np.random.default_rng(20260908)
    for _ in range(200):
        pos = rng.normal(size=3)
        yaw = rng.uniform(-math.pi, math.pi)
        path = rng.normal(size=(24, 2)) * 3
        p, next_yaw, travel, _ = commanded_step(
            pos, yaw, path, SimpleNamespace(**DEFAULTS))
        result = command(pos, yaw, path)
        assert result.reason == "tracking"
        np.testing.assert_array_equal(result.position, p)
        assert result.yaw == next_yaw and result.displacement_m == travel


def test_rigid_change_of_world_coordinates_preserves_command():
    pos = np.array([.3, .2, -.5])
    path = np.array([[.25, -.6], [.1, -.8]])
    result = command(pos, .5, path)
    for angle in [.4, -2.1, math.pi]:
        c, s = math.cos(angle), math.sin(angle)
        rotation = np.array([[c, s], [-s, c]])
        shift = np.array([8., -4.])
        changed_pos = pos.copy()
        changed_pos[[0, 2]] = rotation @ pos[[0, 2]] + shift
        transformed = command(changed_pos, .5 + angle, path @ rotation.T + shift)
        np.testing.assert_allclose(transformed.position[[0, 2]],
                                   rotation @ result.position[[0, 2]] + shift, atol=1e-12)
        assert transformed.displacement_m == pytest.approx(result.displacement_m)


@pytest.mark.parametrize("path", [[], [[np.nan, 1]], [[0, 1, 2]], [[np.inf, 0]]])
def test_bad_reference_is_an_error_not_a_fallback_action(path):
    with pytest.raises(ValueError):
        command(np.zeros(3), 0, path)


def test_bad_environment_response_is_not_replaced_by_free_motion():
    result = command(np.zeros(3), 0, [[0, -2]])
    with pytest.raises(ValueError):
        apply_collision(np.zeros(3), result, lambda *a: [np.nan, 0, 0])


def test_motion_limits_are_respected_for_near_and_far_references():
    rng = np.random.default_rng(7)
    for _ in range(500):
        pos = rng.normal(size=3)
        yaw = rng.uniform(-math.pi, math.pi)
        path = pos[[0, 2]] + rng.normal(size=(24, 2)) * 10**rng.uniform(-5, 1)
        result = command(pos, yaw, path)
        assert 0 <= result.displacement_m <= min(.0376, result.reference_distance_m) + 1e-12
        assert abs(result.yaw - yaw) <= math.radians(4.5) + 1e-12
