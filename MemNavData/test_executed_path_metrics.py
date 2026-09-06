import math

import pytest

from MemNavData.executed_path_metrics import planar_path_length, spl_from_trace


def poses(*xs):
    return [{"step": i, "x": x, "z": 0} for i, x in enumerate(xs)]


def test_terminal_action_is_included():
    assert planar_path_length(poses(0, 1), steps=2,
                              end_position=[2, 0, 0]) == 2


def test_stationary_commands_add_no_distance():
    assert planar_path_length(poses(0, 0), steps=2,
                              end_position=[0, 0, 0]) == 0


def test_missing_endpoint_is_not_silently_zero():
    assert planar_path_length(poses(0, 1), steps=2) is None
    result = spl_from_trace(1, 1, poses(0, 1), steps=2,
                            missing_action_bound_m=0.1)
    assert result["spl"] is None
    assert result["spl_lower"] == pytest.approx(1 / 1.1)
    assert result["spl_upper"] == 1


def test_first_crossing_excludes_later_diagnostic_actions():
    assert planar_path_length(poses(0, 1, 2, 3), steps=2,
                              end_position=[99, 0, 0]) == 2


def test_failed_episode_spl_is_zero_even_without_endpoint():
    assert spl_from_trace(0, 2, poses(0, 1), steps=2)["spl"] == 0


@pytest.mark.parametrize("trace,steps", [(poses(0, 1)[1:], 2),
                                        (poses(0, 1), 3)])
def test_incomplete_trace_rejected(trace, steps):
    with pytest.raises(ValueError):
        planar_path_length(trace, steps=steps)


def test_nonfinite_rejected():
    with pytest.raises(ValueError):
        planar_path_length(poses(0, math.nan), steps=1)


def test_zero_step():
    assert planar_path_length([], steps=0) == 0


def test_planar_not_vertical_distance():
    assert planar_path_length(poses(0), steps=1,
                              end_position=[3, 50, 4]) == 5
