#!/usr/bin/env bash
# HPC-login-node exact retry for the MP3D NavDP authority/cache composition.
set -euo pipefail
umask 0022
export PYTHONDONTWRITEBYTECODE=1

WRAPPER_ROOT=${WRAPPER_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
WRAPPER_RECEIPT=${WRAPPER_ROOT}/SOURCE_BUNDLE.sha256
WRAPPER_SHA=$(sha256sum "${WRAPPER_RECEIPT}" | awk '{print $1}')
TASK=/scratch/yz11502/Research/Nav-axis-uturn-source-bundles/mp3d_table1_controller_portability_eb7cdf82477f6aa1
TASK_SHA=eb7cdf82477f6aa192b5becf42c85e93d490de7bfe7564132bbff285910a32c4
RUN=/scratch/yz11502/Research/Nav-axis-uturn-results/mp3d_table1_controller_portability_20260829/formal_20260829T085025Z
BENCH=/scratch/yz11502/Research/Nav-axis-uturn-results/mp3d_table1_fullmono_source_expansion_20260829/source_expansion_20260829T060541Z_f3e7c3e5/population/natural_direction
CONSTRUCTION=/scratch/yz11502/Research/Nav-axis-uturn-results/mp3d_table1_fullmono_source_expansion_20260829/source_expansion_20260829T060541Z_f3e7c3e5/mp3d_table1_new_query_verification.json
CONSTRUCTION_SHA=618c409f7c7c62ad739687935cdd6f2e564e96aed6ccf6059d887d795c3e953e
BASE=/scratch/yz11502/Research/Nav-axis-uturn-source-bundles/final14_mono_factorial_5690569a4373f2d2
BASE_RECEIPT=${BASE}/source_inputs.sha256
BASE_SHA=5690569a4373f2d2768671418f0c604c4a03aa4b0ffe01baf70b288af03ba216
SOURCE_RUN=/scratch/yz11502/Research/Nav-axis-uturn-results/mdtec_monocular_cec_composition_20260819/formal_20260819T055600Z_624f9fa9
PARENT=/scratch/yz11502/Research/Nav-axis-uturn-results/mp3d_table1_fullmono_source_expansion_20260829/source_expansion_20260829T060541Z_f3e7c3e5/population/parent_manifest.json
PROTOCOL=${TASK}/MemNavData/mp3d_table1_new_query_protocol_20260829.json
REPAIR_TAG=mp3d_table1_navdp_authority_cache_exact_retry2_20260829
REPAIR_ROOT=${RUN}/repairs/${REPAIR_TAG}
ARCHIVE=${REPAIR_ROOT}/failed_attempts
PARTIAL=${RUN}/formal/navdp/evaluation/natural_direction/029_kEZ7cmS4wCh_episode_0004
FAILED_RUNTIME=${RUN}/formal/navdp/runtime/eval_27_mp3d_t1_navdp_exact_retry1_20260829
WRAP=${WRAPPER_ROOT}/MemNavData/slurm_hm3d_table1_navdp_pair.sbatch
ANALYSIS=${TASK}/MemNavData/slurm_hm3d_table1_navdp_analysis.sbatch
SEAL=${TASK}/MemNavData/slurm_hm3d_table1_controller_seal.sbatch
PY=/scratch/lg154/conda-envs/memnav/bin/python
VINT_VERIFY=16558669
VINT_VERIFICATION=${RUN}/formal/vint/vint_table1_independent_verification.json
DRY_RUN=${DRY_RUN:-0}
fail() { echo "ABORT: $*" >&2; exit 2; }
job_id() { awk -F';' 'NR==1{print $1}'; }
assert_job() {
  local id=$1 state=$2 exit_code=$3
  sacct -X -j "${id}" -n -o State,ExitCode | \
    awk -v s="${state}" -v e="${exit_code}" \
      '$1==s && $2==e{ok=1} END{exit !ok}' || \
    fail "job ${id} is not ${state} ${exit_code}"
}

[[ "${DRY_RUN}" =~ ^[01]$ ]] || fail "DRY_RUN must be 0 or 1"
[[ "$(id -un)" == yz11502 ]] || fail "wrong HPC identity"
[[ "$(sha256sum "${WRAPPER_RECEIPT}" | awk '{print $1}')" == \
   "${WRAPPER_SHA}" ]] || fail "wrapper receipt changed"
(cd "${WRAPPER_ROOT}" && sha256sum -c --quiet SOURCE_BUNDLE.sha256) || \
  fail "wrapper bundle changed"
[[ "$(sha256sum "${TASK}/SOURCE_BUNDLE.sha256" | awk '{print $1}')" == \
   "${TASK_SHA}" ]] || fail "task receipt changed"
(cd "${TASK}" && sha256sum -c --quiet SOURCE_BUNDLE.sha256) || \
  fail "task bundle changed"
[[ "$(sha256sum "${BASE_RECEIPT}" | awk '{print $1}')" == "${BASE_SHA}" ]] || \
  fail "base receipt changed"
(cd "${BASE}" && sha256sum -c --quiet "${BASE_RECEIPT}") || \
  fail "base bundle changed"
[[ "$(sha256sum "${CONSTRUCTION}" | awk '{print $1}')" == \
   "${CONSTRUCTION_SHA}" ]] || fail "construction verifier changed"

assert_job 16558664_27 FAILED 1:0
for id in 16558666 16558667 16558670; do assert_job "${id}" CANCELLED 0:0; done
assert_job "${VINT_VERIFY}" COMPLETED 0:0
[[ -f "${VINT_VERIFICATION}" ]] || fail "retained ViNT verifier output missing"
VINT_VERIFICATION_SHA=$(sha256sum "${VINT_VERIFICATION}" | awk '{print $1}')
grep -q 'certificate runtime failure is not a valid policy outcome' \
  "${FAILED_RUNTIME}/logs/query_29.log" || fail "attempt-1 failure changed"
grep -q 'authority_policy = request.form.get' \
  "${TASK}/NavDP/baselines/memnav/memnav_server.py" || \
  fail "strict authority endpoint missing"
grep -q 'The cache is keyed by image digest' \
  "${TASK}/NavDP/baselines/navdp/navdp_server.py" || \
  fail "identical-frame cache repair missing"

declare -A expected=(
  [NavDP/baselines/memnav/memnav_server.py]=edd670749bc99d457ba68991f3cf6009378a64f379318ff2ba154aa04bcfee61
  [NavDP/baselines/memnav/policy_agent.py]=5c0c4046692f2cf7c14ccd661389dd8fe5f810c45cd6afea9ebbb322a6df134e
  [NavDP/baselines/memnav/router_candidates.py]=dd4aca0e9db4fbc5c4d43221d6b6e495eca2186f67cb37fd3a94db10c26233c5
  [NavDP/baselines/navdp/navdp_server.py]=222f1be19c1edffb79b9bc67cbbb60acb6076be07e043370b8c010a1a0003529
)
for path in "${!expected[@]}"; do
  [[ "$(sha256sum "${TASK}/${path}" | awk '{print $1}')" == \
     "${expected[$path]}" ]] || fail "server component changed: ${path}"
done

"${PY}" - "${RUN}" "${BENCH}/manifest.json" <<'PY'
import json,pathlib,sys
run=pathlib.Path(sys.argv[1]); manifest=json.load(open(sys.argv[2])); missing=[]
for i,row in enumerate(manifest['episodes']):
    label=f"{i:03d}_{row['scene']}_{row['episode']}"
    receipt=run/'formal/navdp/evaluation/natural_direction'/label/'completion.json'
    if not receipt.is_file(): missing.append(i)
if missing != [29,30]: raise SystemExit(f'NavDP missing set changed: {missing}')
for path in (run/'formal/navdp/navdp_table1_summary.json',
             run/'formal/navdp/navdp_table1_independent_verification.json',
             run/'mp3d_table1_controller_portability_receipt.json'):
    if path.exists(): raise SystemExit('downstream output already exists: '+str(path))
PY
[[ -d "${PARTIAL}" && -d "${FAILED_RUNTIME}" ]] || \
  fail "attempt-1 partial/runtime missing"
[[ ! -e "${REPAIR_ROOT}" ]] || fail "repair root already exists"

source "${WRAPPER_ROOT}/MemNavData/slurm_safe_submit.sh"
lint_sbatch_template "${WRAP}" || fail "NavDP wrapper lint failed"
wrapper="WRAPPER_ROOT=${WRAPPER_ROOT},WRAPPER_RECEIPT=${WRAPPER_RECEIPT},EXPECTED_WRAPPER_RECEIPT_SHA=${WRAPPER_SHA}"
common="ALL,TASK_ROOT=${TASK},TASK_RECEIPT=${TASK}/SOURCE_BUNDLE.sha256,EXPECTED_TASK_RECEIPT_SHA=${TASK_SHA},FORMAL_RUN_ROOT=${RUN},BENCH_ROOT=${BENCH},CONSTRUCTION_VERIFICATION=${CONSTRUCTION},EXPECTED_CONSTRUCTION_VERIFICATION_SHA=${CONSTRUCTION_SHA},DATASET=MP3D,CLAIM_SCOPE=conference_table_mp3d_reused_scene_history_new_query_replication,BASE_SOURCE_ROOT=${BASE},BASE_RECEIPT=${BASE_RECEIPT},EXPECTED_BASE_RECEIPT_SHA=${BASE_SHA},SERVER_SOURCE_ROOT=${TASK},SERVER_SOURCE_RECEIPT=${TASK}/SOURCE_BUNDLE.sha256,EXPECTED_SERVER_SOURCE_RECEIPT_SHA=${TASK_SHA},SOURCE_RUN_ROOT=${SOURCE_RUN},PARENT_MANIFEST=${PARENT},PROTOCOL=${PROTOCOL},ROLE_PAIR_SCOPE=paper_replication,${wrapper}"
analysis_common="ALL,TASK_ROOT=${TASK},TASK_RECEIPT=${TASK}/SOURCE_BUNDLE.sha256,EXPECTED_TASK_RECEIPT_SHA=${TASK_SHA},FORMAL_RUN_ROOT=${RUN},BENCH_ROOT=${BENCH},CONSTRUCTION_VERIFICATION=${CONSTRUCTION},EXPECTED_CONSTRUCTION_VERIFICATION_SHA=${CONSTRUCTION_SHA},DATASET=MP3D,CLAIM_SCOPE=conference_table_mp3d_reused_scene_history_new_query_replication"
seal_common="ALL,FORMAL_RUN_ROOT=${RUN},CONSTRUCTION_VERIFICATION=${CONSTRUCTION},EXPECTED_CONSTRUCTION_VERIFICATION_SHA=${CONSTRUCTION_SHA},DATASET=MP3D,CLAIM_SCOPE=conference_table_mp3d_reused_scene_history_new_query_replication,VINT_VERIFICATION_OVERRIDE=${VINT_VERIFICATION},EXPECTED_VINT_VERIFICATION_SHA=${VINT_VERIFICATION_SHA}"

sbatch --test-only --array=27 \
  --export="${common},PHASE=formal,EXACT_REPAIR=1,RUNTIME_ATTEMPT=mp3d_t1_navdp_exact_retry2_authority_cache_20260829,FORMAL_INDICES_SPEC=29:30" \
  "${WRAP}" >/dev/null
sbatch --test-only --export="${analysis_common},MODE=aggregate" \
  "${ANALYSIS}" >/dev/null
sbatch --test-only --export="${seal_common}" "${SEAL}" >/dev/null
if [[ "${DRY_RUN}" == 1 ]]; then
  echo "DRY_RUN_OK navdp=29,30 server_source=${TASK} vint_verify=${VINT_VERIFY}"
  exit 0
fi

mkdir -p "${ARCHIVE}/navdp"
mv "${PARTIAL}" "${ARCHIVE}/navdp/029_kEZ7cmS4wCh_episode_0004"
chmod -R a-w "${FAILED_RUNTIME}"
"${PY}" - "${ARCHIVE}" "${FAILED_RUNTIME}" <<'PY'
import hashlib,json,os,sys
from pathlib import Path
root=Path(sys.argv[1]); runtime=Path(sys.argv[2]); files=[]
for base,kind in ((root,'archived_partial'),(runtime,'failed_runtime')):
    for path in sorted(base.rglob('*')):
        if path.is_file():
            files.append({'kind':kind,'path':str(path),'bytes':path.stat().st_size,
                          'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
payload={'schema_version':'mp3d_table1_navdp_authority_cache_failed_attempt_v1_20260829',
         'files':files,'deleted':False,'success_or_distance_read':False,
         'aggregate_or_sr_computed':False}
path=root/'archive_manifest.json'
fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o444)
with os.fdopen(fd,'w') as handle:
    json.dump(payload,handle,indent=2,sort_keys=True); handle.write('\n')
PY
sha256sum "${ARCHIVE}/archive_manifest.json" >"${ARCHIVE}/archive_manifest.json.sha256"
chmod -R a-w "${ARCHIVE}"

repair=$(safe_sbatch --lint-fatal --parsable --job-name=m3T1NavR2 --array=27 \
  --export="${common},PHASE=formal,EXACT_REPAIR=1,RUNTIME_ATTEMPT=mp3d_t1_navdp_exact_retry2_authority_cache_20260829,FORMAL_INDICES_SPEC=29:30" \
  "${WRAP}" | job_id)
aggregate=$(sbatch --parsable --dependency=afterok:${repair} \
  --kill-on-invalid-dep=yes --export="${analysis_common},MODE=aggregate" \
  "${ANALYSIS}" | job_id)
verify=$(sbatch --parsable --dependency=afterok:${aggregate} \
  --kill-on-invalid-dep=yes --export="${analysis_common},MODE=verify" \
  "${ANALYSIS}" | job_id)
seal=$(sbatch --parsable --dependency=afterok:${verify} \
  --kill-on-invalid-dep=yes --export="${seal_common}" "${SEAL}" | job_id)
for id in "${repair}" "${aggregate}" "${verify}" "${seal}"; do
  [[ "${id}" =~ ^[0-9]+$ ]] || fail "invalid job id: ${id}"
done

"${PY}" - "${REPAIR_ROOT}/repair_submission.json" "${WRAPPER_ROOT}" \
  "${WRAPPER_SHA}" "${repair}" "${aggregate}" "${verify}" "${seal}" \
  "${VINT_VERIFY}" "${ARCHIVE}" "${VINT_VERIFICATION_SHA}" <<'PY'
import hashlib,json,os,sys
path,bundle,bundle_sha,repair,aggregate,verify,seal,vint_verify,archive,vint_sha=sys.argv[1:]
manifest=os.path.join(archive,'archive_manifest.json')
payload={
 'schema_version':'mp3d_table1_navdp_authority_cache_repair_submission_v1_20260829',
 'failure_class':'cache_fixed_but_obsolete_memnav_authority_endpoint',
 'original_exact_retry':16558664,'exact_histories':[29,30],
 'jobs':{'navdp_exact_retry2':int(repair),'navdp_aggregate':int(aggregate),
         'navdp_verify':int(verify),'retained_vint_verify':int(vint_verify),
         'replacement_joint_seal':int(seal)},
 'server_source':'/scratch/yz11502/Research/Nav-axis-uturn-source-bundles/mp3d_table1_controller_portability_eb7cdf82477f6aa1',
 'server_source_receipt_sha256':'eb7cdf82477f6aa192b5becf42c85e93d490de7bfe7564132bbff285910a32c4',
 'wrapper_bundle':bundle,'wrapper_receipt_sha256':bundle_sha,
 'archive_root':archive,
 'archive_manifest_sha256':hashlib.sha256(open(manifest,'rb').read()).hexdigest(),
 'retained_vint_verification_sha256':vint_sha,
 'seal_dependency_submission_incident':False,
 'replacement_seal_depends_on_navdp_verify_only':True,
 'retained_vint_verifier_was_completed_and_sha_pinned_before_seal_submission':True,
 'method_or_population_changed':False,
 'exception_and_runtime_failure_fields_inspected':True,
 'navigation_success_or_distance_read':False,'partial_aggregate_or_sr_computed':False,
}
fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o444)
with os.fdopen(fd,'w') as handle:
    json.dump(payload,handle,indent=2,sort_keys=True); handle.write('\n')
PY
sha256sum "${REPAIR_ROOT}/repair_submission.json" \
  >"${REPAIR_ROOT}/repair_submission.json.sha256"
chmod a-w "${REPAIR_ROOT}/repair_submission.json" \
  "${REPAIR_ROOT}/repair_submission.json.sha256"
printf 'NAV_REPAIR2=%s NAV_AGG=%s NAV_VERIFY=%s RETAINED_VINT_VERIFY=%s SEAL=%s\n' \
  "${repair}" "${aggregate}" "${verify}" "${VINT_VERIFY}" "${seal}"
