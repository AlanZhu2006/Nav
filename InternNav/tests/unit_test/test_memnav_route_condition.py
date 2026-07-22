import importlib.util
from pathlib import Path

import numpy as np


SCRIPT = (
    Path(__file__).resolve().parents[2]
    / 'scripts' / 'eval' / 'diag_memnav_route_condition.py'
)
SPEC = importlib.util.spec_from_file_location('diag_memnav_route_condition', SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_angle_degrees_handles_parallel_opposite_and_quarter_turn():
    x = np.asarray([1.0, 0.0])
    assert MODULE.angle_degrees(x, x) == 0.0
    assert MODULE.angle_degrees(x, -x) == 180.0
    assert MODULE.angle_degrees(x, np.asarray([0.0, 1.0])) == 90.0


def test_angle_degrees_marks_stationary_vector_unknown():
    assert np.isnan(MODULE.angle_degrees(np.zeros(2), np.ones(2)))
