import math

import numpy as np

from MemNavData.generate_hm3d_dense_reverse_route_queries import (
    dense_reverse_states,
    shortest_turn_samples,
)
from MemNavData.generate_hm3d_reverse_route_odometry_queries import wrap
from MemNavData.monocular_route_tangent_contract import (
    SEALED_DENSE_HISTORY_INDICES,
)


def test_shortest_turn_samples_bound_each_step_and_reach_target():
    start = math.radians(170.0)
    target = math.radians(-170.0)
    samples = shortest_turn_samples(start, target, max_step_deg=7.0)
    values = [start, *samples]
    assert samples
    assert abs(wrap(samples[-1] - target)) < 1e-9
    assert all(abs(math.degrees(wrap(second - first))) <= 7.0 + 1e-9
               for first, second in zip(values, values[1:]))


def test_dense_route_never_collapses_turn_and_translation():
    poses = [
        {"x": 0.0, "y": 0.0, "z": 0.0, "yaw": 0.0},
        {"x": 1.0, "y": 0.0, "z": 0.0, "yaw": 0.0},
        {"x": 1.0, "y": 0.0, "z": 1.0, "yaw": 0.0},
    ]
    states = dense_reverse_states(
        poses, [2, 1, 0], target_yaw=math.pi,
        max_turn_step_deg=15.0)
    assert states[0]["kind"] == "route_origin"
    assert states[-1]["kind"] == "terminal_turn"
    translations = 0
    for first, second in zip(states, states[1:]):
        distance = float(np.linalg.norm(
            np.asarray(second["floor"])[[0, 2]]
            - np.asarray(first["floor"])[[0, 2]]))
        yaw_delta = abs(wrap(float(second["yaw"]) - float(first["yaw"])))
        if distance > 1e-6:
            translations += 1
            assert yaw_delta < 1e-9
        else:
            assert math.degrees(yaw_delta) <= 15.0 + 1e-9
    assert translations == 2


def test_confirmation_history_is_sealed_before_generation():
    assert SEALED_DENSE_HISTORY_INDICES == (16, 39, 40)
