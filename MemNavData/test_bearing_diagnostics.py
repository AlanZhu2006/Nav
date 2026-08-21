import math

import pytest

from bearing_diagnostics import bearing_error_deg_from_world_delta


def test_habitat_forward_left_convention() -> None:
    assert bearing_error_deg_from_world_delta(
        [1.0, 0.0], [0.0, -2.0], 0.0) == pytest.approx(0.0)
    assert bearing_error_deg_from_world_delta(
        [0.0, 1.0], [0.0, -2.0], 0.0) == pytest.approx(90.0)
    assert bearing_error_deg_from_world_delta(
        [-1.0, 0.0], [0.0, -2.0], 0.0) == pytest.approx(180.0)


def test_rotated_robot_frame() -> None:
    assert bearing_error_deg_from_world_delta(
        [1.0, 0.0], [-3.0, 0.0], math.pi / 2.0
    ) == pytest.approx(0.0)
    assert bearing_error_deg_from_world_delta(
        [0.0, -1.0], [0.0, -3.0], math.pi / 2.0
    ) == pytest.approx(0.0)


def test_invalid_vector_is_rejected() -> None:
    with pytest.raises(ValueError):
        bearing_error_deg_from_world_delta([0.0, 0.0], [0.0, -1.0], 0.0)
