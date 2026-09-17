import ast
import inspect
from pathlib import Path

import pytest

from MemNavData.verify_rebuilt_collision_context import require_matching_mesh


def test_context_requires_identical_serialized_mesh(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    a.write_bytes(b"frozen mesh")
    b.write_bytes(b"frozen mesh")
    assert len(require_matching_mesh(a, b)) == 64
    b.write_bytes(b"different mesh")
    with pytest.raises(ValueError, match="differs"):
        require_matching_mesh(a, b)


def test_verifier_only_accepts_explicit_context_and_keeps_strict_position_check():
    text = Path(__file__).with_name("verify_repaired_fullmono_local.py").read_text()
    tree = ast.parse(text)
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "verify_rollout")
    assert [x.arg for x in fn.args.kwonlyargs] == ["execution_pathfinder"]
    assert ast.literal_eval(fn.args.kw_defaults[0]) is None
    assert 'np.testing.assert_allclose(expected, action["actual_position"], rtol=0, atol=1e-7)' in text
    assert 'pf.load_nav_mesh(str(folder / "execution.navmesh"))' in text


def test_reconstruction_is_headless_and_not_policy_navigation():
    from MemNavData.verify_rebuilt_collision_context import rebuilt_execution_context
    text = inspect.getsource(rebuilt_execution_context)
    assert "config.create_renderer = False" in text
    assert "agent.sensor_specifications = []" in text
    assert "settings = loaded.nav_mesh_settings" in text
    assert "require_matching_mesh(archived_mesh, output_mesh)" in text
    assert "finally:" in text and "sim.close()" in text
