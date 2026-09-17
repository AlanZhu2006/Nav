import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from MemNavData.habitat_executor_audit import DEFAULTS, compare_step, summarize_steps


class FlatMesh:
    def snap_point(self, p):
        return np.array(p)

    def try_step(self, start, end):
        return np.array(end)

    def try_step_no_sliding(self, start, end):
        return np.array(end)


def original_executor():
    root = Path(__file__).resolve().parent
    tree = ast.parse((root / 'eval_2leg_habitat.py').read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'pursuit_step')
    env = dict(np=np, args=SimpleNamespace(**DEFAULTS),
               yaw_facing=lambda d: float(np.arctan2(-d[0], -d[1])))
    exec(compile(ast.Module(body=[fn], type_ignores=[]), '<original pursuit_step>', 'exec'), env)
    return env['pursuit_step']


def test_legacy_matches_original_in_free_space():
    rng = np.random.default_rng(20260908)
    original = original_executor()
    for _ in range(100):
        pos = rng.normal(size=3)
        yaw = rng.uniform(-np.pi, np.pi)
        path = rng.normal(size=(24, 2))
        result, new_yaw, record = compare_step(pos, yaw, path, FlatMesh())
        expected, eyaw, distance = original(pos, yaw, path, FlatMesh())
        np.testing.assert_allclose(result['legacy_snap'], expected, atol=1e-12, rtol=0)
        assert abs(eyaw-new_yaw) < 1e-12
        assert record['legacy_vs_try_step_m'] == 0


def test_legacy_retries_short_step_without_changing_standard_request():
    class Mesh(FlatMesh):
        def snap_point(self, p):
            p = np.array(p)
            return p + [1, 0, 0] if abs(p[2]) > .02 else p
    mesh = Mesh()
    result, yaw, record = compare_step(np.zeros(3), 0, [[0, -2]], mesh)
    assert record['legacy_branch'] == 'short_snap'
    assert abs(result['legacy_snap'][2] + .3*.0376) < 1e-12
    assert abs(result['try_step'][2] + .0376) < 1e-12
    expected, _, _ = original_executor()(np.zeros(3), 0, np.array([[0, -2]]), mesh)
    np.testing.assert_array_equal(result['legacy_snap'], expected)


def test_blocked_legacy_keeps_yaw_update():
    class Mesh(FlatMesh):
        def snap_point(self, p):
            return np.array([np.nan, np.nan, np.nan])
    result, yaw, record = compare_step(np.zeros(3), 0, [[-2, -2]], Mesh())
    assert record['legacy_branch'] == 'blocked'
    np.testing.assert_array_equal(result['legacy_snap'], np.zeros(3))
    assert yaw > 0


def test_summary_does_not_call_observed_differences_sr():
    assert summarize_steps([])['actions'] == 0
    assert 'sr' not in summarize_steps([])
