from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "MemNavData/submit_hm3d_table3_actual_mono_downstream_hpc.sh"


def test_downstream_dag_is_full_power_and_fail_closed():
    text = SCRIPT.read_text()
    assert "--array=0-124%4" in text
    assert "--array=0-47%4" in text
    assert "--dependency='afterok:${factual_gate}:${factual_array}'" in text
    assert "--dependency='afterok:${population_verify_job}'" in text
    assert "'powered_histories':48,'formal_queries':96" in text
    assert "'partial_results_allowed':False" in text
    assert "'fallback_completion_allowed':False" in text
    assert "'threshold_relaxation':False,'smoke_substitution':False" in text
    assert text.count("--partition=cpu_short") >= 8
    assert "CONSTRUCTION_JOB_OVERRIDE" in text


def test_all_runtime_overlays_are_receipt_bound():
    text = SCRIPT.read_text()
    assert "SOURCE_BUNDLE.sha256" in text
    assert "sha256sum -c --quiet" in text
    assert "eval_2leg_habitat.py" in text
    assert "eval_shared_online_role_pairs.py" in text
    assert "run_hm3d_fullmono_query_history.py" in text
