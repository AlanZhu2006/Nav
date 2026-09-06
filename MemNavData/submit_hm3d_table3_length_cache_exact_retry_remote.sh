#!/usr/bin/env bash
# Run on the authenticated yz11502 Torch login node.
set -euo pipefail
umask 0022
export PYTHONDONTWRITEBYTECODE=1

WRAPPER_ROOT=${WRAPPER_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
WRAPPER_RECEIPT=${WRAPPER_ROOT}/SOURCE_BUNDLE.sha256
WRAPPER_SHA=$(sha256sum "${WRAPPER_RECEIPT}" | awk '{print $1}')
RUN=/scratch/yz11502/Research/Nav-axis-uturn-results/hm3d_table3_causal_survey_expansion_20260831/expansion_20260830T165729Z_a8dd232e
SURVEY=/scratch/yz11502/Research/Nav-axis-uturn-source-bundles/hm3d_table3_query_api_closure_e226a4b456846283
SURVEY_SHA=e226a4b4568462831e726a4f486f485233dc4c90bfef72012898cc34ba4744e6
TASK=/scratch/yz11502/Research/Nav-axis-uturn-source-bundles/hm3d_lifelong_natural_b_expansion_execution_1f4979a7fd37d467
TASK_SHA=1f4979a7fd37d46700011558063be34a8fba0a0b8746668469dba7e7955f4282
OLD_SERVER=/scratch/yz11502/Research/Nav-axis-uturn-source-bundles/hm3d_table1_navdp_authority_transaction_718661db1733d5de
OLD_SERVER_SHA=718661db1733d5de16cd86687eec880a8d02fc5ae5ca982e1ab7d5bde5e96f7d
SERVER=/scratch/yz11502/Research/Nav-axis-uturn-source-bundles/mp3d_table1_controller_portability_eb7cdf82477f6aa1
SERVER_SHA=eb7cdf82477f6aa192b5becf42c85e93d490de7bfe7564132bbff285910a32c4
BASE=/scratch/yz11502/Research/Nav-axis-uturn-source-bundles/final14_mono_factorial_5690569a4373f2d2
BASE_RECEIPT=${BASE}/source_inputs.sha256
BASE_SHA=5690569a4373f2d2768671418f0c604c4a03aa4b0ffe01baf70b288af03ba216
RUNTIME=/scratch/yz11502/Research/Nav-axis-uturn-source-bundles/hm3d_table1_navdp_authority_transaction_repair_82e71f19ee7f4e52
RUNTIME_SHA=82e71f19ee7f4e5233fae499633ce5a233c9c036bb41b9e2bf7d4f0f18effd7d
POP=${RUN}/merged_query_population
MANIFEST=${POP}/role_pairs/manifest.json
MANIFEST_SHA=cbc518cea991fd252893f97fd5e730c277e4d899369932536a745351d47e7451
POP_VERIFY=${POP}/independent_verification.json
POP_VERIFY_SHA=8398456ec71503698b3cef60f3b49be6917111d7c76b5e038279dff44b0b4ace
PROTOCOL=${SURVEY}/MemNavData/hm3d_table3_causal_survey_protocol_20260830.json
PAIR=${SURVEY}/MemNavData/slurm_hm3d_table3_causal_survey_pair.sbatch
ANALYSIS=${SURVEY}/MemNavData/slurm_hm3d_table3_causal_survey_analysis.sbatch
REPAIR_TAG=table3_length_cache_exact_retry_20260902
REPAIR_ROOT=${RUN}/repairs/${REPAIR_TAG}
ARCHIVE=${RUN}/failed_attempts/${REPAIR_TAG}
PY=/scratch/lg154/conda-envs/memnav/bin/python
DRY_RUN=${DRY_RUN:-0}

fail() { echo "ABORT: $*" >&2; exit 2; }
job_id() { awk -F';' 'NR==1{print $1}'; }
assert_job() {
  local id=$1 expected_state=$2 expected_exit=$3
  sacct -X -j "${id}" -n -o State,ExitCode | \
    awk -v s="${expected_state}" -v e="${expected_exit}" \
      '$1==s && $2==e{ok=1} END{exit !ok}' || \
    fail "job ${id} is not ${expected_state} ${expected_exit}"
}
verify_bundle() {
  local root=$1 receipt=$2 expected=$3 label=$4
  [[ -d "${root}" && -f "${receipt}" ]] || fail "missing ${label} bundle"
  [[ "$(sha256sum "${receipt}" | awk '{print $1}')" == "${expected}" ]] || \
    fail "${label} receipt changed"
  (cd "${root}" && sha256sum -c --quiet "${receipt}") || \
    fail "${label} bundle content changed"
}

[[ "${DRY_RUN}" =~ ^[01]$ ]] || fail "DRY_RUN must be 0 or 1"
[[ "$(id -un)" == yz11502 ]] || fail "wrong HPC identity"
verify_bundle "${WRAPPER_ROOT}" "${WRAPPER_RECEIPT}" "${WRAPPER_SHA}" wrapper
verify_bundle "${SURVEY}" "${SURVEY}/SOURCE_BUNDLE.sha256" "${SURVEY_SHA}" survey
verify_bundle "${TASK}" "${TASK}/SOURCE_BUNDLE.sha256" "${TASK_SHA}" task
verify_bundle "${OLD_SERVER}" "${OLD_SERVER}/SOURCE_BUNDLE.sha256" \
  "${OLD_SERVER_SHA}" old_server
verify_bundle "${SERVER}" "${SERVER}/SOURCE_BUNDLE.sha256" "${SERVER_SHA}" server
verify_bundle "${BASE}" "${BASE_RECEIPT}" "${BASE_SHA}" base
verify_bundle "${RUNTIME}" "${RUNTIME}/SOURCE_BUNDLE.sha256" \
  "${RUNTIME_SHA}" runtime

assert_job 16697907_35 FAILED 1:0
assert_job 16697907_41 FAILED 1:0
assert_job 16697907_44 FAILED 1:0
assert_job 16697908 CANCELLED 0:0
assert_job 16697909 CANCELLED 0:0
[[ "$(sha256sum "${MANIFEST}" | awk '{print $1}')" == "${MANIFEST_SHA}" ]] || \
  fail "benchmark manifest changed"
[[ "$(sha256sum "${POP_VERIFY}" | awk '{print $1}')" == "${POP_VERIFY_SHA}" ]] || \
  fail "population verifier changed"
(cd "${POP}" && sha256sum -c --quiet independent_verification.json.sha256) || \
  fail "population verifier sidecar changed"

declare -A expected_server_files=(
  [NavDP/baselines/memnav/memnav_server.py]=edd670749bc99d457ba68991f3cf6009378a64f379318ff2ba154aa04bcfee61
  [NavDP/baselines/memnav/policy_agent.py]=5c0c4046692f2cf7c14ccd661389dd8fe5f810c45cd6afea9ebbb322a6df134e
  [NavDP/baselines/memnav/router_candidates.py]=dd4aca0e9db4fbc5c4d43221d6b6e495eca2186f67cb37fd3a94db10c26233c5
  [NavDP/baselines/navdp/navdp_server.py]=222f1be19c1edffb79b9bc67cbbb60acb6076be07e043370b8c010a1a0003529
)
for path in "${!expected_server_files[@]}"; do
  [[ "$(sha256sum "${SERVER}/${path}" | awk '{print $1}')" == \
     "${expected_server_files[$path]}" ]] || \
    fail "replacement server component changed: ${path}"
done
for path in NavDP/baselines/memnav/memnav_server.py \
            NavDP/baselines/memnav/policy_agent.py \
            NavDP/baselines/memnav/router_candidates.py; do
  cmp -s "${OLD_SERVER}/${path}" "${SERVER}/${path}" || \
    fail "strict-authority component drifted: ${path}"
done
grep -q 'The cache is keyed by image digest' \
  "${SERVER}/NavDP/baselines/navdp/navdp_server.py" || \
  fail "identical-frame cache correction missing"
grep -q 'authority_policy = request.form.get' \
  "${SERVER}/NavDP/baselines/memnav/memnav_server.py" || \
  fail "strict authority endpoint missing"

"${PY}" - "${RUN}" "${MANIFEST}" "${POP_VERIFY}" <<'PY'
import hashlib, json, pathlib, sys
run=pathlib.Path(sys.argv[1]); manifest=json.load(open(sys.argv[2])); verify=json.load(open(sys.argv[3]))
assert verify['verified'] is True and verify['formal_policy_evaluation_authorized'] is True
assert len(manifest['episodes']) == 48
missing=[]
for i,row in enumerate(manifest['episodes']):
    label=f"{i:03d}_{row['scene']}_{row['episode']}"
    root=run/'evaluation/natural_direction'/label
    receipt=root/'completion.json'; sidecar=root/'completion.json.sha256'
    if not receipt.is_file():
        missing.append(i); continue
    if not sidecar.is_file(): raise SystemExit(f'missing sidecar: {i}')
    expected=sidecar.read_text().split()[0]
    actual=hashlib.sha256(receipt.read_bytes()).hexdigest()
    if actual != expected: raise SystemExit(f'completion hash mismatch: {i}')
if missing != [35,41,44]: raise SystemExit(f'missing set changed: {missing}')
for path in (run/'table3_result/summary.json',
             run/'table3_result/independent_verification.json'):
    if path.exists(): raise SystemExit('downstream output already exists: '+str(path))
PY

declare -A eval_dir=(
  [35]=035_p53SfW6mjZe_episode_table3_survey_090
  [41]=041_rsggHU7g7dh_episode_table3_survey_459
  [44]=044_k1cupFYWXJ6_episode_table3_survey_114
)
declare -A runtime_dir=(
  [35]=eval_87_table3Survey035
  [41]=eval_92_table3Survey041
  [44]=eval_83_table3Survey044
)
for index in 35 41 44; do
  partial=${RUN}/evaluation/natural_direction/${eval_dir[$index]}
  runtime=${RUN}/runtime/${runtime_dir[$index]}
  [[ -d "${partial}" && ! -e "${partial}/completion.json" ]] || \
    fail "partial state changed for index ${index}"
  [[ -d "${runtime}" ]] || fail "runtime state missing for index ${index}"
  grep -q 'cached monocular depth belongs to a different transaction' \
    "${runtime}/logs/server_navdp.log" || \
    fail "failure signature changed for index ${index}"
done
[[ ! -e "${REPAIR_ROOT}" && ! -e "${ARCHIVE}" ]] || \
  fail "repair/archive root already exists"

source "${WRAPPER_ROOT}/MemNavData/slurm_safe_submit.sh"
lint_sbatch_template "${PAIR}" || fail "pair sbatch lint failed"
lint_sbatch_template "${ANALYSIS}" || fail "analysis sbatch lint failed"
common="ALL,SURVEY_SOURCE_ROOT=${SURVEY},SURVEY_SOURCE_RECEIPT=${SURVEY}/SOURCE_BUNDLE.sha256,EXPECTED_SURVEY_SOURCE_RECEIPT_SHA=${SURVEY_SHA},TASK_ROOT=${TASK},TASK_RECEIPT=${TASK}/SOURCE_BUNDLE.sha256,EXPECTED_TASK_RECEIPT_SHA=${TASK_SHA},SERVER_SOURCE_ROOT=${SERVER},SERVER_SOURCE_RECEIPT=${SERVER}/SOURCE_BUNDLE.sha256,EXPECTED_SERVER_SOURCE_RECEIPT_SHA=${SERVER_SHA},BASE_SOURCE_ROOT=${BASE},BASE_RECEIPT=${BASE_RECEIPT},EXPECTED_BASE_RECEIPT_SHA=${BASE_SHA},RUNTIME_CLOSURE_ROOT=${RUNTIME},RUNTIME_CLOSURE_RECEIPT=${RUNTIME}/SOURCE_BUNDLE.sha256,EXPECTED_RUNTIME_CLOSURE_RECEIPT_SHA=${RUNTIME_SHA},RUN_ROOT=${RUN},CANDIDATE_PLAN=${MANIFEST},EXPECTED_CANDIDATE_PLAN_SHA=${MANIFEST_SHA},SURVEY_PROTOCOL=${PROTOCOL},POPULATION_RELATIVE_ROOT=merged_query_population,EXPECTED_POPULATION_VERIFICATION_SHA=${POP_VERIFY_SHA},EXPECTED_BENCHMARK_MANIFEST_SHA=${MANIFEST_SHA}"

safe_sbatch --lint-fatal --test-only \
  --partition=h100_tandon,a100_tandon \
  --account=torch_pr_769_tandon_advanced --qos=gpu48 --gres=gpu:1 \
  --cpus-per-task=10 --mem=72G --time=02:00:00 --array=35,41,44%3 \
  --export="${common}" "${PAIR}" >/dev/null
for mode in analyze verify; do
  safe_sbatch --lint-fatal --test-only --partition=cpu_short \
    --account=torch_pr_769_tandon_advanced --cpus-per-task=2 --mem=12G \
    --time=00:20:00 --export="${common},MODE=${mode}" "${ANALYSIS}" >/dev/null
done
if [[ "${DRY_RUN}" == 1 ]]; then
  echo "DRY_RUN_OK exact_indices=35,41,44 retained=45 server=${SERVER_SHA}"
  exit 0
fi

mkdir -p "${REPAIR_ROOT}" "${ARCHIVE}/evaluation/natural_direction" \
  "${ARCHIVE}/runtime"
for index in 35 41 44; do
  mv -- "${RUN}/evaluation/natural_direction/${eval_dir[$index]}" \
    "${ARCHIVE}/evaluation/natural_direction/${eval_dir[$index]}"
  mv -- "${RUN}/runtime/${runtime_dir[$index]}" \
    "${ARCHIVE}/runtime/${runtime_dir[$index]}"
done
"${PY}" - "${ARCHIVE}" <<'PY'
import hashlib, json, os, pathlib, sys
root=pathlib.Path(sys.argv[1]); files=[]
for path in sorted(root.rglob('*')):
    if path.is_file():
        files.append({'path':path.relative_to(root).as_posix(),
                      'bytes':path.stat().st_size,
                      'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
payload={'schema_version':'hm3d_table3_length_cache_failed_attempt_v1_20260902',
         'failed_indices':[35,41,44], 'files':files, 'deleted':False,
         'exception_text_inspected':True,
         'navigation_success_or_distance_read':False,
         'partial_aggregate_or_sr_computed':False}
path=root/'archive_manifest.json'
fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o444)
with os.fdopen(fd,'w') as f:
    json.dump(payload,f,indent=2,sort_keys=True); f.write('\n')
PY
sha256sum "${ARCHIVE}/archive_manifest.json" \
  >"${ARCHIVE}/archive_manifest.json.sha256"
chmod -R a-w "${ARCHIVE}"

repair=$(safe_sbatch --lint-fatal --parsable --job-name=h3T3LenCacheR \
  --partition=h100_tandon,a100_tandon \
  --account=torch_pr_769_tandon_advanced --qos=gpu48 --gres=gpu:1 \
  --cpus-per-task=10 --mem=72G --time=02:00:00 --array=35,41,44%3 \
  --export="${common}" "${PAIR}" | job_id)
summary=$(safe_sbatch --lint-fatal --parsable --partition=cpu_short \
  --account=torch_pr_769_tandon_advanced --cpus-per-task=2 --mem=12G \
  --time=00:20:00 --dependency=afterok:${repair} --kill-on-invalid-dep=yes \
  --export="${common},MODE=analyze" "${ANALYSIS}" | job_id)
verify=$(safe_sbatch --lint-fatal --parsable --partition=cpu_short \
  --account=torch_pr_769_tandon_advanced --cpus-per-task=2 --mem=12G \
  --time=00:20:00 --dependency=afterok:${summary} --kill-on-invalid-dep=yes \
  --export="${common},MODE=verify" "${ANALYSIS}" | job_id)
for id in "${repair}" "${summary}" "${verify}"; do
  [[ "${id}" =~ ^[0-9]+$ ]] || fail "invalid submitted job id: ${id}"
done

archive_sha=$(sha256sum "${ARCHIVE}/archive_manifest.json" | awk '{print $1}')
"${PY}" - "${REPAIR_ROOT}/repair_submission.json" "${WRAPPER_ROOT}" \
  "${WRAPPER_SHA}" "${repair}" "${summary}" "${verify}" \
  "${ARCHIVE}" "${archive_sha}" <<'PY'
import json, os, sys
path,bundle,bundle_sha,repair,summary,verify,archive,archive_sha=sys.argv[1:]
payload={
 'schema_version':'hm3d_table3_length_cache_exact_retry_submission_v1_20260902',
 'failure_class':'identical_jpeg_new_valid_transaction_cache_collision',
 'run_root':'/scratch/yz11502/Research/Nav-axis-uturn-results/hm3d_table3_causal_survey_expansion_20260831/expansion_20260830T165729Z_a8dd232e',
 'retained_completion_count':45, 'exact_retry_indices':[35,41,44],
 'jobs':{'exact_retry':int(repair),'summary':int(summary),
         'independent_verifier':int(verify)},
 'replacement_server_root':'/scratch/yz11502/Research/Nav-axis-uturn-source-bundles/mp3d_table1_controller_portability_eb7cdf82477f6aa1',
 'replacement_server_receipt_sha256':'eb7cdf82477f6aa192b5becf42c85e93d490de7bfe7564132bbff285910a32c4',
 'wrapper_bundle':bundle, 'wrapper_receipt_sha256':bundle_sha,
 'archive_root':archive, 'archive_manifest_sha256':archive_sha,
 'method_or_population_changed':False, 'completed_indices_rerun':False,
 'exception_text_inspected':True,
 'navigation_success_or_distance_read':False,
 'partial_aggregate_or_sr_computed':False,
 'fallback_completion_allowed':False,
}
fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o444)
with os.fdopen(fd,'w') as f:
    json.dump(payload,f,indent=2,sort_keys=True); f.write('\n')
PY
sha256sum "${REPAIR_ROOT}/repair_submission.json" \
  >"${REPAIR_ROOT}/repair_submission.json.sha256"
chmod a-w "${REPAIR_ROOT}/repair_submission.json" \
  "${REPAIR_ROOT}/repair_submission.json.sha256"
printf 'EXACT_RETRY=%s SUMMARY=%s VERIFY=%s\n' \
  "${repair}" "${summary}" "${verify}"
