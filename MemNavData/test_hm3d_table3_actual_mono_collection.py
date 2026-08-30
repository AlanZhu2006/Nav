from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "MemNavData/hm3d_table3_actual_mono_execution_protocol_20260830.json"
RUNNER = ROOT / "MemNavData/run_hm3d_fullmono_server_scene.sh"
SBATCH = ROOT / "MemNavData/slurm_hm3d_table3_actual_mono_a.sbatch"
COLLECTOR = ROOT / "MemNavData/collect_hm3d_table3_actual_mono_a.py"
SUBMITTER = ROOT / "MemNavData/submit_hm3d_table3_actual_mono_a_hpc.sh"
PAIR_SBATCH = ROOT / "MemNavData/slurm_hm3d_table3_actual_mono_pair.sbatch"
DOWNSTREAM_SUBMITTER = (
    ROOT / "MemNavData/submit_hm3d_table3_actual_mono_downstream_hpc.sh"
)


def test_final_goal_yaw_uses_the_last_nontrivial_path_segment():
    text = COLLECTOR.read_text()
    assert "for point in reversed(points[:-1]):" in text
    assert "delta = goal_xz - np.asarray(point" in text
    assert "return float(yaw_facing(delta))" in text


def test_execution_protocol_keeps_full_reserves_and_no_fallback():
    protocol = json.loads(PROTOCOL.read_text())
    assert protocol["candidate_plan"]["candidate_count"] == 125
    assert protocol["collection"]["all_frozen_reserves_are_collected"] is True
    assert protocol["guards"]["fallback_completion_allowed"] is False
    assert protocol["factual_A"]["metric_depth_for_control"] is False
    assert protocol["runtime_geometry"] == {
        "mode": "content_addressed_pinned_navmesh",
        "navmesh_path_source": "candidate_plan.asset.navmesh_path",
        "navmesh_sha256_source": "candidate_plan.asset.navmesh_sha256",
        "runtime_recomputation": False,
        "reason": ("the capacity bins and every rollout must use the "
                   "identical frozen Habitat path graph"),
    }


def test_server_runner_has_an_explicit_table3_actual_history_mode():
    text = RUNNER.read_text()
    assert '"${MODE}" == table3_a' in text
    assert 'collect_hm3d_table3_actual_mono_a.py' in text
    assert '--candidate-plan "${TABLE3_PLAN}"' in text
    assert '--source-root "${WRAPPER_ROOT}"' in text
    assert 'PYTHONPATH_PREFIX=${WRAPPER_ROOT}:${WRAPPER_ROOT}/MemNavData' in text
    assert 'PYTHONPATH_PREFIX+=:${RUNTIME_CLOSURE_ROOT}' in text
    assert 'SCENE_RANK_FIELD=${SCENE_RANK_FIELD:-final14_scene_rank}' in text
    assert '"${SCENE_RANK_FIELD}" == scene_index' in text
    assert 'QUERY_SOURCE_ROOT=${QUERY_SOURCE_ROOT:-${TASK_ROOT}}' in text
    assert '"${QUERY_SOURCE_ROOT}/MemNavData/run_hm3d_fullmono_query_history.py"' in text
    assert 'all("scene_index" in row for row in p["episodes"])' in text


def test_slurm_task_is_one_frozen_candidate_and_not_a_short_smoke():
    text = SBATCH.read_text()
    assert 'SCENE_INDEX="${SLURM_ARRAY_TASK_ID}"' in text
    assert 'MODE=table3_a' in text
    assert 'MAX_STEPS=' not in text
    assert 'TABLE3_EXECUTION_PROTOCOL=' in text


def test_table3_collection_uses_the_capacity_navmesh_verbatim():
    text = COLLECTOR.read_text()
    assert "recompute_navmesh=False" in text
    assert '"--pinned_navmesh", row["asset"]["navmesh_path"]' in text
    assert '"--expected_pinned_navmesh_sha256"' in text
    assert '"runtime_geometry": "content_addressed_pinned_navmesh"' in text


def test_capacity_receipt_is_checked_in_its_original_query_direction():
    text = COLLECTOR.read_text()
    assert "capacity_ok, capacity_distance, _ = geodesic(" in text
    assert "simulator.pathfinder, goal, start" in text
    assert 'float(geometry["first_goal_geodesic_m"])' in text
    assert '"capacity query-direction geodesic changed"' in text
    assert '"capacity/factual Goal-A geodesic changed"' not in text


def test_factual_a_bundle_import_tests_the_local_runtime_closure():
    text = SUBMITTER.read_text()
    assert "MemNavData/cec_handoff_contract.py" in text
    assert "MemNavData/deterministic_eval_protocol.py" in text
    assert "eval_2leg_habitat.py' --help" in text


def test_formal_pair_uses_the_same_lifetime_port_allocator_as_collection():
    pair = PAIR_SBATCH.read_text()
    downstream = DOWNSTREAM_SUBMITTER.read_text()
    assert "MEMNAV_PORT=" not in pair
    assert "NAVDP_PORT=" not in pair
    assert "MemNavData/slurm_port_pair.sh" in downstream
