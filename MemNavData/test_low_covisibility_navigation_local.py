import ast
from pathlib import Path

from MemNavData import run_low_covisibility_navigation_local as pilot
from MemNavData.test_repaired_fullmono_local import SOURCE, option


def test_one_explicit_policy_change_and_no_controller_change(monkeypatch, tmp_path):
    monkeypatch.setattr(pilot, 'task', lambda *_: (
        {}, {}, {'analysis_role': 'revisit'}, {}, SOURCE, {}))
    strict = pilot.command(tmp_path, 0, 'cec', tmp_path / 'strict', 21810, 21811)
    changed = pilot.command(tmp_path, 0, 'cec_no_coverage', tmp_path / 'changed', 21810, 21811)
    assert '--certified_authority_policy' not in strict
    assert option(changed, '--certified_authority_policy') == 'certificate_without_coverage'
    for flag in ('--hybrid_route', '--revisit_adapter', '--navdp_depth_source', '--seed',
                 '--exec_horizon', '--max_steps', '--certified_cdec_rescue', '--revisit_controller'):
        assert option(changed, flag) == option(strict, flag)


def test_complete_fixed_roster_and_position_balanced_arm_order():
    assert pilot.HISTORIES == (2, 12)
    assert pilot.QUERIES == ('c10_30', 'natural_novel')
    orders = [pilot.ARMS[i:] + pilot.ARMS[:i] for i in range(4)]
    for slot in zip(*orders):
        assert set(slot) == set(pilot.ARMS)


def test_original_cli_default_is_strict_and_finite_pnp_keeps_its_policy():
    source = Path(pilot.ROOT / 'MemNavData/eval_2leg_habitat.py').read_text()
    tree = ast.parse(source)
    arg = next(n for n in ast.walk(tree) if isinstance(n, ast.Call) and n.args
               and isinstance(n.args[0], ast.Constant) and n.args[0].value == '--certified_authority_policy')
    assert next(k.value.value for k in arg.keywords if k.arg == 'default') == 'strict_certificate'
    assignment = next(n for n in ast.walk(tree) if isinstance(n, ast.Assign)
                      and any(isinstance(t, ast.Name) and t.id == 'expected_authority_policy' for t in n.targets))
    assert assignment.value.body.value == 'pnp_pose_available'
    assert ast.unparse(assignment.value.orelse) == 'args.certified_authority_policy'
