from pathlib import Path

import pytest

from MemNavData.run_hm3d_fullmono_query_history import audit_history_contract


ROOT = Path(__file__).resolve().parents[1]


def table2_payload():
    receipt = {
        "prefix_receipt_schema": (
            "hm3d_table2_actual_mono_ab_prefix_v1_20260829"
        ),
        "prefix_semantics": "actual_mono_Novel_A_then_Novel_B",
        "prefix_A_steps": 2,
        "prefix_B_steps": 3,
    }
    trace = {
        "poses": [{"step": value} for value in range(5)],
        "prefix_semantics": (
            "exact_actual_mono_A_then_B_observation_concat"
        ),
    }
    return receipt, trace


def test_actual_ab_history_contract_is_explicit():
    receipt, trace = table2_payload()
    assert audit_history_contract(receipt, trace, "actual_ab") == (
        2,
        3,
        "actual_mono_navdp_novel_A_then_novel_B_rgb_replay",
    )


def test_actual_ab_rejects_segment_mismatch():
    receipt, trace = table2_payload()
    receipt["prefix_B_steps"] = 4
    with pytest.raises(RuntimeError, match="segment lengths"):
        audit_history_contract(receipt, trace, "actual_ab")


def test_goal_a_cannot_silently_consume_table2_prefix():
    receipt, trace = table2_payload()
    with pytest.raises(RuntimeError, match="ordinary Goal-A"):
        audit_history_contract(receipt, trace, "goal_a")


def test_server_runner_forwards_history_contract():
    script = (ROOT / "MemNavData/run_hm3d_fullmono_server_scene.sh").read_text()
    assert "HISTORY_CONTRACT=${HISTORY_CONTRACT:-goal_a}" in script
    assert '--history-contract "${HISTORY_CONTRACT}"' in script


def test_table2_slurm_never_falls_back_to_goal_a_contract():
    pair = (
        ROOT / "MemNavData/slurm_hm3d_table2_leg3_navdp_pair.sbatch"
    ).read_text()
    analysis = (
        ROOT / "MemNavData/slurm_hm3d_table2_leg3_navdp_analysis.sbatch"
    ).read_text()
    assert "HISTORY_CONTRACT=actual_ab" in pair
    assert "formal_policy_evaluation_authorized" in pair
    assert "policy_outcomes_read" in pair
    assert "--dataset HM3D_TABLE2" in analysis
    assert "unconditional_three_leg_joint_sr_reported" in analysis
