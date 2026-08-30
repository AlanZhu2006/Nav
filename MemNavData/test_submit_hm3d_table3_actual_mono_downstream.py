from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "MemNavData/submit_hm3d_table3_actual_mono_downstream_hpc.sh"


def test_downstream_dag_is_full_power_and_fail_closed():
    text = SCRIPT.read_text()
    assert "--array=0-124%4" in text
    assert "--array=0-47%4" in text
    assert "factual_dependency=${repair_finish}" in text
    assert "completion_receipts_verified']==125" in text
    assert "directed repair completion receipt did not verify" in text
    assert "directed_repair_completion_receipt_sha256" in text
    assert "--dependency='afterok:${population_verify_job}'" in text
    assert "'powered_histories':48,'formal_queries':96" in text
    assert "'partial_results_allowed':False" in text
    assert "'fallback_completion_allowed':False" in text
    assert "'threshold_relaxation':False,'smoke_substitution':False" in text
    assert text.count("--partition=cpu_short") >= 8
    assert "CONSTRUCTION_JOB_OVERRIDE" in text
    assert "FACTUAL_DEPENDENCY_OVERRIDE" in text
    assert "HM3D_TABLE3_ACTUAL_MONO_A_SIGABRT_REPAIR_SUBMISSION_20260830.json" in text
    assert "HM3D_TABLE3_ACTUAL_MONO_A_DIRECTED_GEODESIC_REPAIR_SUBMISSION_20260830.json" in text
    assert "factual-A receipt and directed repair bind different arrays" in text
    assert "factual dependency override is not the receipt-bound repair verifier" in text
    assert "hm3d_table3_actual_mono_downstream_submission_v2_20260830" in text
    assert "directed_repair_submission_receipt_sha256" in text


def test_all_runtime_overlays_are_receipt_bound():
    text = SCRIPT.read_text()
    assert "SOURCE_BUNDLE.sha256" in text
    assert "sha256sum -c --quiet" in text
    assert "eval_2leg_habitat.py" in text
    assert "eval_shared_online_role_pairs.py" in text
    assert "run_hm3d_fullmono_query_history.py" in text
