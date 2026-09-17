import copy

import numpy as np
import pytest

from MemNavData.render_recorded_pose_first_person import (
    camera_position, recorded_states, state_index,
)


def example_payload():
    return {
        "rollout_traces": {"query": [
            dict(step=0, x=1., y=.1, z=3., yaw=0.),
            dict(step=1, x=1.1, y=.1, z=3., yaw=-.2),
        ]},
        "query_result": dict(steps=2, end_position=[1.2, .1, 3.],
                             end_yaw_rad=-.4),
    }


def test_explicit_terminal_state_is_used_without_mutating_input():
    p = example_payload()
    original = copy.deepcopy(p)
    rows = recorded_states(p)
    assert [r["step"] for r in rows] == [0, 1, 2]
    assert rows[-1]["yaw"] == -.4
    assert rows[-1]["x"] == 1.2
    assert rows[-1]["terminal"] is True
    assert p == original


def test_camera_uses_height_once_in_habitat_y_up():
    assert np.allclose(camera_position(dict(x=1., y=.1, z=3.), .5), [1., .6, 3.])


def test_trace_selection_is_causal_and_holds_the_terminal():
    assert [state_index([0, 2, 4], k) for k in range(7)] == [0, 0, 1, 1, 2, 2, 2]


def test_nonmonotone_trace_is_not_silently_reordered():
    p = example_payload()
    p["rollout_traces"]["query"][1]["step"] = 0
    with pytest.raises(ValueError, match="strictly increase"):
        recorded_states(p)


def test_conflicting_terminal_does_not_replace_an_observed_frame():
    p = example_payload()
    p["query_result"]["steps"] = 1
    with pytest.raises(ValueError, match="Conflicting states"):
        recorded_states(p)
