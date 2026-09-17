"""Small adversarial checks at the policy/GT/measurement boundary."""
import ast
import hashlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from MemNavData.executed_path_metrics import planar_path_length, spl


def test_server_selected_trajectory_does_not_touch_goal_or_map():
    source = Path(__file__).with_name("eval_2leg_habitat.py")
    tree = ast.parse(source.read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "select_plan_trajectory")
    class Forbidden:
        def __getattr__(self, name):
            raise AssertionError("server selector attempted map/GT access")
        def __getitem__(self, index):
            raise AssertionError("server selector attempted map/GT access")
        def __array__(self, *args):
            raise AssertionError("server selector attempted map/GT access")
    env = dict(np=np, hashlib=hashlib,
               args=SimpleNamespace(trajectory_selector="server", oracle_selector_horizon=0, exec_horizon=8),
               normalize_navdp_trajectory_candidates=lambda x: np.asarray(x),
               navdp_candidate_diversity=lambda x: {})
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(source), "exec"), env)
    path = np.array([[1., .2, 0], [2., .1, 0]])
    candidates = np.stack([path, path*2])
    selected, _ = env[fn.name]({"trajectory":path, "all_trajectory":candidates},
                               Forbidden(), Forbidden(), Forbidden(), Forbidden())
    np.testing.assert_array_equal(path, selected)


def test_missing_endpoint_is_not_silently_zero_last_motion():
    trace = [dict(step=0, x=0., z=0.), dict(step=1, x=.01, z=0.)]
    assert planar_path_length(trace, steps=2) is None
    assert planar_path_length(trace, steps=2, end_position=[.03, 0., 0.]) == .03


def test_blocked_motion_and_failure_do_not_gain_spl_from_commands():
    trace = [dict(step=i, x=0., z=0.) for i in range(8)]
    actual = planar_path_length(trace, steps=8, end_position=[0, 0, 0])
    assert actual == 0
    assert spl(0, 3., actual) == 0
