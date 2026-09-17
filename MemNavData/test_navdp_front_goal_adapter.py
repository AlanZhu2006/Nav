import ast
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from MemNavData.navdp_front_goal_adapter import (
    EXECUTOR_ODOMETRY_FIELDS, FrontGoalAdapter, install_loop_hook,
    rgb_only_memory_form, wrap,
)


def request(angle, **extra):
    beta = math.radians(angle)
    return dict(revisit_adapter_takeover=True,
                memory_controller_pointgoal=[2.5*math.cos(beta), 2.5*math.sin(beta)], **extra)


@pytest.mark.parametrize("angle", [-180, -165, -91, 91, 175, 180])
def test_rearward_request_becomes_forward_without_translation_or_excess_turn(angle):
    adapter = FrontGoalAdapter(enabled=True, max_turn_rad=math.radians(4.5))
    yaw, pos = 2.9, np.array([1., 0., -3.])
    original = yaw
    assert adapter.consider(request(angle), yaw, 0)
    for step in range(41):
        if not adapter.active:
            break
        command = adapter.command(pos, yaw)
        np.testing.assert_array_equal(command.position, pos)
        assert command.displacement_m == 0.
        assert abs(command.yaw-yaw) <= math.radians(4.5) + 1e-12
        yaw = command.yaw
        adapter.observe(yaw, step)
    assert not adapter.active
    assert adapter.actions <= 40
    assert abs(wrap(yaw-original-math.radians(angle))) < 1e-8
    assert adapter.events[-1]["fresh_replan_required"]
    # Re-express the ORIGINAL target after the executed turn, not as new input RGB.
    delta = yaw-original
    x,y = request(angle)["memory_controller_pointgoal"]
    np.testing.assert_allclose([math.cos(delta)*x+math.sin(delta)*y,
                               -math.sin(delta)*x+math.cos(delta)*y], [2.5,0.], atol=1e-8)


@pytest.mark.parametrize("angle", [-90, -20, 0, 65, 90])
def test_forward_domain_is_unchanged(angle):
    adapter = FrontGoalAdapter(enabled=True, max_turn_rad=.1)
    assert not adapter.consider(request(angle), 1., 0)
    assert adapter.receipt()["events"] == []


def test_no_issued_pointgoal_no_direction_invention():
    adapter = FrontGoalAdapter(enabled=True, max_turn_rad=.1)
    assert not adapter.consider({"memory_controller_pointgoal": [-2.5,0.]}, 0., 0)
    assert not adapter.active


def test_raw_and_cec_requests_use_the_same_domain_rule_without_scores_or_roles():
    outputs = []
    for extras in ({"source":"raw", "critic":-.9, "role":"novel"},
                   {"source":"cec", "critic":.9, "role":"revisit"}):
        adapter = FrontGoalAdapter(enabled=True, max_turn_rad=.1)
        adapter.consider(request(160, **extras), 0., 0)
        outputs.append(adapter.receipt())
    assert outputs[0] == outputs[1]


def test_disabled_arm_reports_eligibility_without_actuating():
    adapter = FrontGoalAdapter(enabled=False, max_turn_rad=.1)
    assert not adapter.consider(request(160), .3, 8)
    assert adapter.actions == 0 and not adapter.active
    assert adapter.events[0]["activated"] is False


def test_active_heading_is_held_without_resampling():
    adapter = FrontGoalAdapter(enabled=True, max_turn_rad=.1)
    adapter.consider(request(160), .3, 8)
    with pytest.raises(RuntimeError):
        adapter.consider(request(-160), .3, 9)


def test_simulator_odometry_does_not_cross_endpoint_memory_request_boundary():
    sentinel = object()
    data = dict(materialize_monocular_depth="1", graph_rescue="0", candidates="[]")
    data.update({field: sentinel for field in EXECUTOR_ODOMETRY_FIELDS})
    result = rgb_only_memory_form(data)
    assert result == dict(materialize_monocular_depth="1", graph_rescue="0", candidates="[]")
    assert all(data[field] is sentinel for field in EXECUTOR_ODOMETRY_FIELDS)


def test_filter_covers_the_actual_evaluator_serialization_including_contract_tag():
    source = Path(__file__).with_name("eval_2leg_habitat.py")
    function = next(n for n in ast.parse(source.read_text()).body
                    if isinstance(n, ast.FunctionDef) and n.name == "executor_motion_form")
    namespace = {"np": np}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"), namespace)
    emitted = namespace[function.name](.02, .05, .019, .001)
    assert emitted["executor_local_se2_contract"] == "frame_bound_local_se2_v1"
    assert rgb_only_memory_form(emitted) == {}


def test_loop_hook_retains_original_source_and_adds_only_the_pending_turn_branch(tmp_path):
    source = Path(__file__).with_name("eval_2leg_habitat.py")
    before = source.read_bytes()
    tree = ast.parse(before)
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "run_policy_leg")
    namespace = {}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"), namespace)
    changed = install_loop_hook(SimpleNamespace(), namespace[function.name],
                                FrontGoalAdapter(enabled=True, max_turn_rad=.1), tmp_path / "loop.py")
    newtree = ast.parse((tmp_path/"loop.py").read_text())
    loop = next(n for n in ast.walk(newtree) if isinstance(n, ast.For)
                and any(isinstance(k, ast.If) and ast.unparse(k.test)=="_front_goal_adapter.active" for k in n.body))
    inserted = next(n for n in loop.body if isinstance(n, ast.If) and ast.unparse(n.test)=="_front_goal_adapter.active")
    loop.body.remove(inserted)
    assert ast.dump(newtree.body[0]) == ast.dump(function)
    assert changed.__name__ == function.name and before == source.read_bytes()
    calls = [ast.unparse(n.func) for n in ast.walk(inserted) if isinstance(n, ast.Call)]
    assert calls.count("srv_memory") == 1
    assert calls.count("srv_navdp_memory_replay") == 1
    assert "srv_plan" not in calls


def test_private_loop_is_materialized_only_after_evaluator_output_initialization():
    source = Path(__file__).with_name("run_habitat_minimal_repair_local.py")
    evaluate = next(n for n in ast.parse(source.read_text()).body
                    if isinstance(n, ast.FunctionDef) and n.name == "evaluate")
    leg = next(n for n in evaluate.body if isinstance(n, ast.FunctionDef) and n.name == "leg")
    calls = [n for n in ast.walk(evaluate) if isinstance(n, ast.Call)
             and ast.unparse(n.func) == "install_loop_hook"]
    assert len(calls) == 1 and calls[0] in list(ast.walk(leg))


@pytest.mark.parametrize("previous_x,expected", [(0., (4, "stuck")), (1., None)])
def test_pending_turn_keeps_the_original_position_stagnation_exit(tmp_path, previous_x, expected):
    source = Path(__file__).with_name("eval_2leg_habitat.py")
    original = next(n for n in ast.parse(source.read_text()).body
                    if isinstance(n, ast.FunctionDef) and n.name == "run_policy_leg")
    namespace = {}
    exec(compile(ast.Module(body=[original], type_ignores=[]), str(source), "exec"), namespace)
    output = tmp_path / "loop.py"
    install_loop_hook(SimpleNamespace(), namespace[original.name], None, output)
    inserted = next(n for n in ast.walk(ast.parse(output.read_text())) if isinstance(n, ast.If)
                    and ast.unparse(n.test) == "_front_goal_adapter.active")
    wrapper = ast.parse("def tick():\n    for _ in range(1):\n        pass\n")
    wrapper.body[0].body[0].body = inserted.body
    ast.fix_missing_locations(wrapper)
    env = dict(np=np, frame=b"rgb", args=SimpleNamespace(exec_horizon=8, stuck_window=3, stuck_dist=.08),
               step=3, memory_trace=[], history=[np.array([previous_x, 0.]) for _ in range(4)],
               pos=np.zeros(3), psi=0., pf=None, path_len=0., frame_pose_position=np.zeros(3), frame_pose_yaw=0.,
               srv_memory=lambda _: {"frame_idx": 3}, srv_navdp_memory_replay=lambda _: None,
               pursuit_step=lambda *a: (np.zeros(3), .05, 0.), bind_next_executor_receipt=lambda *a: None,
               result=lambda step, termination_reason: (step, termination_reason))
    # Assign the normally enclosing-loop locals before executing the extracted
    # branch; all calls are CPU stubs, not a parallel navigation implementation.
    initial = ast.parse("pos=np.zeros(3)\npsi=0.\npath_len=0.\n").body
    wrapper.body[0].body = initial + wrapper.body[0].body
    ast.fix_missing_locations(wrapper)
    exec(compile(wrapper, "<turn-stagnation-test>", "exec"), env)
    assert env["tick"]() == expected
