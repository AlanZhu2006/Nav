from unittest.mock import patch

import pytest

from MemNavData.table2_final_postprocess import next_dependency, submit, dependency_arguments


def test_handoff_follows_the_reducer_not_a_gpu_gate():
    b = dict(TABLE2_STAGE="B", EXPECTED_PLAN_SHA="sha", TABLE2_REDUCE_JOB="123")
    assert next_dependency("after_A", b, "sha") == (123, "after_B")
    c = dict(TABLE2_STAGE="C", EXPECTED_PLAN_SHA="sha", TABLE2_REDUCE_JOB="456")
    assert next_dependency("after_B", c, "sha") == (456, "verify")
    with pytest.raises(ValueError):
        next_dependency("after_A", c, "sha")
    with pytest.raises(ValueError):
        next_dependency("after_A", b, "other-plan")


def test_postprocessing_requests_cpu_only_and_preserves_real_dependency(tmp_path):
    (tmp_path/"plan.json").write_text("{}")
    (tmp_path/"POSTPROCESSING.sha256").write_text("test fixture")
    with patch("MemNavData.table2_final_postprocess.subprocess.check_output", side_effect=["PENDING|0:0|\n", "789;torch\n"]) as call:
        assert submit(tmp_path, tmp_path/"bundle", "verify", 456) == 789
    argv = call.call_args.args[0]
    assert "--dependency=afterok:456" in argv
    assert "--partition=cpu_short" in argv and "--time=01:00:00" in argv
    assert not any(arg.startswith("--gres") for arg in argv)
    assert "safe_sbatch" in argv[2]


def test_completed_parent_can_expire_from_squeue_without_blocking_postprocessing():
    with patch("MemNavData.table2_final_postprocess.subprocess.check_output", return_value="COMPLETED|0:0|\n"):
        assert dependency_arguments(456) == []
    with patch("MemNavData.table2_final_postprocess.subprocess.check_output", return_value="FAILED|1:0|\n"):
        with pytest.raises(ValueError,match="did not complete successfully"):
            dependency_arguments(456)
