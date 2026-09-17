import ast
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pytest

from MemNavData.run_repaired_fullmono_local import (
    ARMS, evaluator_command, execution_environment, sources,
)


SOURCE = {"scene": "test_scene", "episode": "episode_0000", "asset": "/data/scene.glb",
          "source_episode": "/data/episodes/test_scene/episode_0000", "seed": 0}


def option(command, key):
    return command[command.index(key)+1]


def test_actual_a_cannot_use_old_trace_or_replay_or_memory_takeover(tmp_path):
    cmd = evaluator_command(SOURCE, tmp_path, 21710, 21711)
    assert cmd[3] == "goal_a"
    assert option(cmd, "--leg1_mode") == "policy"
    assert option(cmd, "--hybrid_route") == "native_sidecar"
    assert option(cmd, "--navdp_depth_source") == "monocular_sidecar"
    assert "--write_leg1_trace" in cmd and "--stop_after_leg1" in cmd
    assert "--shared_leg1_trace_root" not in cmd and "--role_pair_query_role" not in cmd
    with pytest.raises(ValueError):
        evaluator_command(SOURCE, tmp_path, 21710, 21711, arm="cec")


@pytest.mark.parametrize("arm", ARMS)
@pytest.mark.parametrize("role", ("novel", "revisit"))
def test_queries_share_one_profile_and_new_constructed_history(tmp_path, arm, role):
    cmd = evaluator_command(SOURCE, tmp_path, 21710, 21711, arm=arm, role=role,
                            benchmark=tmp_path / "new_benchmark")
    assert cmd[3] == "eval"
    assert option(cmd, "--leg1_mode") == "shared_trace"
    assert option(cmd, "--episode_root") == str(tmp_path / "new_benchmark/test_scene")
    assert option(cmd, "--navdp_depth_source") == "monocular_sidecar"
    assert option(cmd, "--max_steps") == "600" and option(cmd, "--exec_horizon") == "8"
    assert option(cmd, "--role_pair_query_role") == role  # evaluator argument, not model request
    assert option(cmd, "--retrieval_override") == "off"
    assert option(cmd, "--cec_initial_bearing_alignment") == "off"  # no second aligner


def test_execution_profile_has_no_legacy_color_raster_or_snap():
    env = execution_environment()
    assert env["MINIMAL_EXECUTOR"] == "bounded_standard"
    assert env["MINIMAL_ACTOR_COLOR"] == "rgb_v1"
    assert env["MINIMAL_DEPTH_RASTER"] == "source_rgb"
    assert env["MINIMAL_FRONT_GOAL"] == "heading_on"


def test_habitat_extra_is_explicit_without_importing_other_interpreter_packages(monkeypatch):
    monkeypatch.setenv("REPAIRED_HAB_VENDOR", "/pinned/habitat/pip/_vendor")
    monkeypatch.setenv("REPAIRED_HAB_EXTRA", "/pinned/opencv_abi3")
    monkeypatch.setenv("PYTHONPATH", "/wrong/python3.10/site-packages")
    paths = execution_environment()["PYTHONPATH"].split(":")
    assert paths[-2:] == ["/pinned/habitat/pip/_vendor", "/pinned/opencv_abi3"]
    assert "/wrong/python3.10/site-packages" not in paths


def test_all_collection_precedes_construction_and_all_construction_precedes_query():
    path = Path(__file__).with_name("run_repaired_fullmono_local.py")
    tree = ast.parse(path.read_text())
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "local")
    blocks = [n for n in ast.walk(function) if isinstance(n, ast.With)
              and "private_servers" in ast.unparse(n.items[0].context_expr)]
    assert len(blocks) == 1
    statements = [ast.unparse(n) for n in blocks[0].body]
    collect_index = next(i for i,s in enumerate(statements) if s.startswith("for source in source_rows:"))
    construct_index = next(i for i,s in enumerate(statements) if s.startswith("run_child(") and "construct" in s)
    query_index = next(i for i,n in enumerate(blocks[0].body)
                       if isinstance(n, ast.For) and isinstance(n.target, ast.Tuple)
                       and [e.id for e in n.target.elts] == ["index", "report"])
    assert collect_index < construct_index < query_index


def test_independent_motion_reconstructs_bounded_reference():
    from MemNavData.bounded_pursuit import command, apply_collision
    path = Path(__file__).with_name("verify_repaired_fullmono_local.py")
    tree = ast.parse(path.read_text())
    f = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "motion")
    namespace = {"np": np, "math": math}
    exec(compile(ast.Module(body=[f], type_ignores=[]), str(path), "exec"), namespace)

    class Collision:
        def try_step(self, start, target):
            return np.asarray(start) + .6 * (np.asarray(target)-start)

    rng = np.random.default_rng(20260908)
    pf = Collision()
    for scale in (0., .001, .02, 1., 5.):
        for _ in range(30):
            p = rng.normal(size=3)
            yaw = rng.uniform(-math.pi, math.pi)
            reference = p[[0, 2]] + scale * rng.normal(size=(24, 2))
            request = command(p, yaw, reference)
            actual, angle, travel = namespace["motion"](p, yaw, reference, pf)
            target = apply_collision(p, request, pf.try_step)
            np.testing.assert_allclose(actual, target[0], atol=1e-10, rtol=0)
            assert abs(angle-target[1]) < 1e-10 and abs(travel-request.displacement_m) < 1e-10


def test_fixed_source_order_and_assets_are_bound():
    rows = sources()
    assert [r["scene"] for r in rows] == ["gxdoqLR6rwA", "pLe4wQe7qrG"]
    assert all(r["episode"] == "episode_0000" and r["seed"] == 0 for r in rows)
    assert all(len(r["source_files"]) == 4 for r in rows)


def test_wrapper_dry_run_exits_before_output_or_execution_setup():
    module = Path(__file__).with_name("run_habitat_minimal_repair_local.py")
    tree = ast.parse(module.read_text())
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "evaluate")
    statements = [ast.unparse(n) for n in function.body]
    dry_index = next(i for i, text in enumerate(statements) if text.startswith("if base.args.contract_dry_run:"))
    output_index = next(i for i, text in enumerate(statements) if text.startswith("out.mkdir("))
    assert dry_index < output_index
    assert isinstance(function.body[dry_index].body[-1], ast.Return)
    base = ast.parse(Path(__file__).with_name("eval_2leg_habitat.py").read_text())
    main = next(n for n in base.body if isinstance(n, ast.FunctionDef) and n.name == "main")
    statements = [ast.unparse(n) for n in main.body]
    assert next(i for i, t in enumerate(statements) if t.startswith("if args.contract_dry_run:")) < next(
        i for i, t in enumerate(statements) if t.startswith("os.makedirs("))


def test_hpc_gate_selection_does_not_read_navigation_outcomes():
    from MemNavData.build_repaired_hm3d_gate_sources import SCENES, PARENT_SHA
    assert SCENES == ("rJhMRvNn4DS", "6D36GQHuP8H") and len(PARENT_SHA) == 64
    path = Path(__file__).with_name("build_repaired_hm3d_gate_sources.py")
    tree = ast.parse(path.read_text())
    text = ast.unparse(tree)
    assert 'parent[\'episodes\'][scene][0]' in text
    for forbidden in ("goal_a_results", "query_results", "shared_trace", "evaluation/"):
        assert forbidden not in text


def test_fixed_extension_uses_next_four_parent_scenes_and_disjoint_shards():
    from MemNavData.build_repaired_hm3d_gate_sources import (
        SCENES, EXTENSION_SCENES, selected_sources,
    )
    parent = {"scenes": list(SCENES + EXTENSION_SCENES)}
    assert selected_sources(parent) == tuple(enumerate(SCENES))
    first = selected_sources(parent, "extension", 0)
    second = selected_sources(parent, "extension", 1)
    assert first == ((2, EXTENSION_SCENES[0]), (3, EXTENSION_SCENES[1]))
    assert second == ((4, EXTENSION_SCENES[2]), (5, EXTENSION_SCENES[3]))
    assert not set(first) & set(second)
    with pytest.raises(ValueError):
        selected_sources(parent, "initial", 1)
    with pytest.raises(ValueError):
        selected_sources(parent, "extension", 2)
    parent["scenes"][2:4] = reversed(parent["scenes"][2:4])
    with pytest.raises(ValueError):
        selected_sources(parent, "extension", 0)


def test_construction_and_arm_order_preserve_parent_rank_across_shards():
    from MemNavData.final14_role_pair_contract import assigned_direction_stratum
    assert [assigned_direction_stratum(rank, 0) for rank in range(2, 6)] == [
        "side", "front", "rear", "side",
    ]
    module = ast.parse(Path(__file__).with_name("run_repaired_fullmono_local.py").read_text())
    function = next(n for n in module.body if isinstance(n, ast.FunctionDef) and n.name == "construct")
    calls = [n for n in ast.walk(function) if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Name) and n.func.id == "build"]
    assert len(calls) == 1
    rank = next(k.value for k in calls[0].keywords if k.arg == "scene_rank")
    assert ast.unparse(rank) == "int(source.get('source_scene_rank', rank))"
    function = next(n for n in module.body if isinstance(n, ast.FunctionDef) and n.name == "local")
    assignments = [n for n in ast.walk(function) if isinstance(n, ast.Assign)
                   and any(isinstance(t, ast.Name) and t.id == "order_start" for t in n.targets)]
    assert len(assignments) == 1
    assert "source_scene_rank" in ast.unparse(assignments[0].value)


@pytest.mark.parametrize("case", ("failed_a", "no_geometric_pair", "lost_attrition", "bad_manifest_hash", "false_retained_count"))
def test_empty_construction_is_accounted_for_not_treated_as_a_query(tmp_path, monkeypatch, case):
    module = Path(__file__).with_name("verify_repaired_fullmono_local.py")
    monkeypatch.syspath_prepend(str(module.parent))
    tree = ast.parse(module.read_text())
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "constructed_histories")
    digest = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    namespace = {"Path": Path, "json": json, "digest": digest}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(module), "exec"), namespace)
    online = tmp_path / "online"
    benchmark = tmp_path / "benchmark"
    online.mkdir()
    benchmark.mkdir()
    (online / "manifest.json").write_text(json.dumps({"episodes": []}))
    trace_path = tmp_path / "trace.json"
    trace = {"reached": case == "no_geometric_pair"}
    trace_path.write_text(json.dumps(trace))
    manifest = {"episodes": [], "source_online_root": str(online),
                "source_online_manifest_sha256": digest(online / "manifest.json")}
    (benchmark / "manifest.json").write_text(json.dumps(manifest))
    (benchmark / "manifest.json.sha256").write_text(
        "wrong" if case == "bad_manifest_hash" else digest(benchmark / "manifest.json"))
    report = {"benchmark": str(benchmark),
              "materialization": {"source_traces": 1, "goal_a_successes": int(trace["reached"]),
                  "materialized": int(trace["reached"]), "manifest_sha256": digest(online / "manifest.json"),
                  "attrition": [{"reason": "mono_a_failed", "trace_sha256": digest(trace_path)}]},
              "construction": {"retained_standard_natural_histories": 0, "attempts": []}}
    if case == "no_geometric_pair":
        report["materialization"]["attrition"] = []
        report["construction"]["attempts"] = [{"retained": False}]
    elif case == "lost_attrition":
        report["materialization"]["attrition"] = []
    elif case == "false_retained_count":
        report["construction"]["retained_standard_natural_histories"] = 1
    if case in ("failed_a", "no_geometric_pair"):
        assert namespace["constructed_histories"](report, trace_path, trace) == []
    else:
        with pytest.raises(AssertionError):
            namespace["constructed_histories"](report, trace_path, trace)
