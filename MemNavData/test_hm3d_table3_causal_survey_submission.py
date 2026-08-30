from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PAIR = ROOT / "MemNavData/slurm_hm3d_table3_causal_survey_pair.sbatch"
ANALYSIS = ROOT / "MemNavData/slurm_hm3d_table3_causal_survey_analysis.sbatch"
SUBMITTER = ROOT / "MemNavData/submit_hm3d_table3_causal_survey_queries_hpc.sh"
MERGED_SUBMITTER = ROOT / (
    "MemNavData/submit_hm3d_table3_causal_survey_merged_queries_hpc.sh"
)
AGGREGATOR = ROOT / "MemNavData/analyze_hm3d_table3_causal_survey.py"
VERIFIER = ROOT / "MemNavData/independent_verify_hm3d_table3_causal_survey_result.py"
DEFERRED = ROOT / "MemNavData/slurm_hm3d_table3_causal_survey_deferred.sbatch"
DEFERRED_SUBMITTER = ROOT / (
    "MemNavData/submit_hm3d_table3_causal_survey_deferred_hpc.sh"
)


def test_formal_pair_binds_the_independently_sealed_population():
    text = PAIR.read_text()
    assert "EXPECTED_POPULATION_VERIFICATION_SHA" in text
    assert "EXPECTED_BENCHMARK_MANIFEST_SHA" in text
    assert "HISTORY_CONTRACT=causal_survey" in text
    assert "QUERY_ARMS=mono_native,mono_cec" in text
    assert "EXPECTED_RUNTIME_CLOSURE_RECEIPT_SHA" in text
    assert 'RUNTIME_CLOSURE_ROOT="${RUNTIME_CLOSURE_ROOT}"' in text


def test_result_chain_rechecks_population_hashes():
    text = ANALYSIS.read_text()
    assert "EXPECTED_POPULATION_VERIFICATION_SHA" in text
    assert "EXPECTED_BENCHMARK_MANIFEST_SHA" in text
    assert "independent_verify_hm3d_table3_causal_survey_result.py" in text
    assert "analyze_hm3d_table3_causal_survey.py" in text


def test_submitter_has_no_smoke_or_partial_fallback_path():
    text = SUBMITTER.read_text()
    assert "--array='${gate_index}'" in text
    assert "--array='${remaining_array}'" in text
    assert "--dependency='afterok:${population_verify_job}'" in text
    assert "--dependency='afterok:${gate_job}'" in text
    assert "'formal_gate_retained_in_final_population':True" in text
    assert "'powered_histories':48" in text
    assert "'formal_queries':96" in text
    assert "'raw_arm_role_rows':192" in text
    assert "'partial_results_allowed':False" in text
    assert "'fallback_completion_allowed':False" in text
    assert "'smoke_substitution':False" in text
    assert "test ! -e '${run_root}/evaluation'" in text
    assert "EXPECTED_RUNTIME_CLOSURE_RECEIPT_SHA" in text
    assert "runtime_provenance_verified=true" in text


def test_result_artifacts_disclose_survey_history_and_reaudit_raw_rows():
    aggregate = AGGREGATOR.read_text()
    verifier = VERIFIER.read_text()
    assert "not actual NavDP Goal-A history" in aggregate
    assert "hm3d_table3_causal_survey_result_v1_20260830" in aggregate
    assert "raw_arm_role_rows\": 192" in aggregate
    assert "raw_metric_rows" in verifier
    assert "metric_depth_sensor_consumed_any" in verifier
    assert "analysis_role_not_forwarded" in verifier
    assert "exact_reject_fallback" in verifier
    assert "audit_raw_outcome" in verifier
    assert "success was not reproduced from raw final distance" in verifier
    assert 'len(completion_artifacts) == 48 and len(raw_artifacts) == 288' in verifier
    assert "completion_artifact_set_sha256" in verifier


def test_merged_query_chain_binds_v2_population_without_fallback():
    submitter = MERGED_SUBMITTER.read_text()
    pair = PAIR.read_text()
    analysis = ANALYSIS.read_text()
    aggregate = AGGREGATOR.read_text()
    verifier = VERIFIER.read_text()
    assert "hm3d_table3_causal_survey_population_verification_v2_20260831" in submitter
    assert "POPULATION_RELATIVE_ROOT=merged_query_population" in submitter
    assert "candidate_plan=${manifest}" in submitter
    assert "--array='${gate_index}'" in submitter
    assert "--array='${remaining_array}'" in submitter
    assert "--dependency='afterok:${population_verify_job}'" in submitter
    assert "--dependency='afterok:${gate_job}'" in submitter
    assert "'formal_gate_retained_in_final_population':True" in submitter
    assert "'powered_histories':48" in submitter
    assert "'formal_queries':96" in submitter
    assert "'raw_arm_role_rows':192" in submitter
    assert "'partial_results_allowed':False" in submitter
    assert "'fallback_completion_allowed':False" in submitter
    assert "'smoke_substitution':False" in submitter
    assert "test ! -e '${run_root}/evaluation'" in submitter
    assert "EXPECTED_RUNTIME_CLOSURE_RECEIPT_SHA" in submitter
    assert "runtime_provenance_verified=true" in submitter
    assert "merged_query_population" in pair
    assert "merged_query_population" in analysis
    assert "hm3d_table3_causal_survey_result_v2_20260831" in aggregate
    assert "hm3d_table3_causal_survey_result_verification_v2_20260831" in verifier


def test_deferred_chain_preserves_the_powered_population_and_formal_gate():
    text = DEFERRED.read_text()
    assert '"${DEFERRED_MODE}" == query_base' in text
    assert '"${DEFERRED_MODE}" == query_merged' in text
    assert 'if [[ "${candidate_count}" -eq 0 ]]' in text
    assert "'base_population_used_directly':True" in text
    assert 'population_relative_root=query_population' in text
    assert 'population_relative_root=merged_query_population' in text
    assert 'fail "base construction is not 125/125"' in text
    assert "'0_to_20_m':16" in text
    assert "'20_to_30_m':16" in text
    assert "'30_to_50_m':16" in text
    assert "'powered_histories':48" in text
    assert "'formal_queries':96" in text
    assert "'raw_arm_role_rows':192" in text
    assert "'formal_gate_retained_in_final_population':True" in text
    assert "'query_policy_outcomes_read_at_submission':False" in text
    assert "'partial_results_allowed':False" in text
    assert "'fallback_completion_allowed':False" in text
    assert "'threshold_relaxation':False" in text
    assert "'smoke_substitution':False" in text
    assert '--dependency="afterok:${verify}"' in text
    assert '--dependency="afterok:${gate}"' in text
    assert "controlled_causal_rgb_geodesic_survey" in text


def test_deferred_submitter_is_identity_gated_and_result_blind():
    text = DEFERRED_SUBMITTER.read_text()
    assert "EXPECTED_SSH_USER" in text
    assert "shared SSH identity" in text
    assert "BASE_POPULATION_VERIFY_JOB=${base_population_verify_job}" in text
    assert "bundle_selftest.sh" in text
    assert "--dependency='afterok:${plan_verify_job}'" in text
    assert "'query_policy_jobs_submitted':False" in text
    assert "'query_policy_outcomes_read':False" in text
    assert "'base_candidates_deleted_or_replaced':False" in text
    assert "'threshold_relaxation':False" in text
    assert "'fallback_completion_allowed':False" in text
    assert "'smoke_substitution':False" in text
