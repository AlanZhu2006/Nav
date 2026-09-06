import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MEM = ROOT / "MemNavData"


def test_quota_retry_is_exact_and_does_not_change_scientific_execution():
    protocol = json.loads(
        (MEM / "hm3d_action_coordinate_quota_retry_protocol_20260902.json")
        .read_text(encoding="utf-8")
    )
    submit = (
        MEM / "submit_hm3d_action_coordinate_compass_quota_retry_remote.sh"
    ).read_text(encoding="utf-8")
    wrapper = (
        MEM / "slurm_hm3d_action_coordinate_compass_quota_retry.sbatch"
    ).read_text(encoding="utf-8")

    frozen = protocol["frozen_evaluation"]
    assert frozen["population_size"] == 48
    assert frozen["retained_indices"] == list(range(18))
    assert frozen["exact_retry_indices"] == "18-47"
    assert frozen["source_receipt_sha256"].startswith("b7de41263049e415")
    assert protocol["guards"]["method_changed"] is False
    assert protocol["guards"]["completed_indices_rerun"] is False
    assert protocol["guards"]["partial_outcomes_read"] is False
    assert "--array=18-47%3" in submit
    assert "retained set changed" in submit
    assert "partial != [18,19,20]" in submit
    assert "navigation_outcomes_read_for_repair':False" in submit
    assert "bash \"${ORIGINAL_PAIR}\"" in wrapper
    assert "rm -rf -- \"${buffer}\"" in wrapper
    assert "completion sidecar mismatch" in wrapper
    assert "--dependency=afterok:${retry}" in submit
    assert "--dependency=afterok:${summary}" in submit


def test_quota_cleanup_is_limited_to_regenerable_buffers():
    submit = (
        MEM / "submit_hm3d_action_coordinate_compass_quota_retry_remote.sh"
    ).read_text(encoding="utf-8")
    wrapper = (
        MEM / "slurm_hm3d_action_coordinate_compass_quota_retry.sbatch"
    ).read_text(encoding="utf-8")

    assert "eval_*_actCoordPair{i:03d}/buffer" in submit
    assert "target.parent.parent != runtime" in submit
    assert "data_class':'regenerable_runtime_rgb_buffers'" in submit
    assert "partial_navigation_outcomes_read':False" in submit
    assert "successful_task_buffer_cleanup" in wrapper
    assert "data_class':'regenerable_runtime_rgb_buffer'" in wrapper
