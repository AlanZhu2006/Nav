from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "MemNavData/hm3d_table3_actual_mono_execution_protocol_20260830.json"
RUNNER = ROOT / "MemNavData/run_hm3d_fullmono_server_scene.sh"
SBATCH = ROOT / "MemNavData/slurm_hm3d_table3_actual_mono_a.sbatch"
COLLECTOR = ROOT / "MemNavData/collect_hm3d_table3_actual_mono_a.py"


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


def test_server_runner_has_an_explicit_table3_actual_history_mode():
    text = RUNNER.read_text()
    assert '"${MODE}" == table3_a' in text
    assert 'collect_hm3d_table3_actual_mono_a.py' in text
    assert '--candidate-plan "${TABLE3_PLAN}"' in text


def test_slurm_task_is_one_frozen_candidate_and_not_a_short_smoke():
    text = SBATCH.read_text()
    assert 'SCENE_INDEX="${SLURM_ARRAY_TASK_ID}"' in text
    assert 'MODE=table3_a' in text
    assert 'MAX_STEPS=' not in text
    assert 'TABLE3_EXECUTION_PROTOCOL=' in text
