#!/usr/bin/env bash
# Run on the authenticated yz11502 Torch login node from the immutable bundle.
set -euo pipefail
umask 0022
export PYTHONDONTWRITEBYTECODE=1

SOURCE_ROOT=${SOURCE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
SOURCE_RECEIPT=${SOURCE_ROOT}/SOURCE_BUNDLE.sha256
EXPECTED_SOURCE_RECEIPT_SHA=${EXPECTED_SOURCE_RECEIPT_SHA:?}
RUN_ROOT=${RUN_ROOT:-/scratch/yz11502/Research/Nav-axis-uturn-results/hm3d_longrange_rear_alignment_20260904/consumed_${EXPECTED_SOURCE_RECEIPT_SHA:0:16}}
CONCURRENCY=${CONCURRENCY:-4}
PROTOCOL=${SOURCE_ROOT}/MemNavData/hm3d_longrange_rear_alignment_protocol_20260904.json
PROTOCOL_SHA=f61dbd3021df5ff8603b68666ead03863d77a0f91d5f760301f28b2a596dbe4f
SELECTED=3,4,5,8,10,11,13,15,21
fail() { echo "ABORT: $*" >&2; exit 2; }
job_id() { tr -d '\r' | awk -F';' '/^[0-9]+(;|$)/ {print $1; exit}'; }

[[ "$(id -un)" == yz11502 ]] || fail "wrong HPC identity"
[[ "${CONCURRENCY}" =~ ^[1-9][0-9]*$ ]] || fail "bad concurrency"
[[ -f "${SOURCE_RECEIPT}" ]] || fail "source receipt missing"
[[ "$(sha256sum "${SOURCE_RECEIPT}" | awk '{print $1}')" == \
   "${EXPECTED_SOURCE_RECEIPT_SHA}" ]] || fail "source receipt changed"
(cd "${SOURCE_ROOT}" && sha256sum -c --quiet "${SOURCE_RECEIPT}") || \
  fail "source bundle content changed"
[[ "$(sha256sum "${PROTOCOL}" | awk '{print $1}')" == "${PROTOCOL_SHA}" ]] || \
  fail "rear-alignment protocol changed"

FORMAL_PARENT=/scratch/yz11502/Research/Nav-axis-uturn-results/hm3d_longrange_route_tangent_20260903/fresh_751efdcbb436c99b
test "$(sha256sum "${FORMAL_PARENT}/fresh_population/role_pairs/manifest.json" | awk '{print $1}')" = 0491632af1925fb14fe0bf69d9aaf8d769590552f343df38e2e30a0380655a47
test "$(sha256sum "${FORMAL_PARENT}/result/summary.json" | awk '{print $1}')" = b3f55c8bf90f1897556931f05795dd94787864a0682ec4b6ba59995b7bea421a
test "$(sha256sum "${FORMAL_PARENT}/result/independent_verification.json" | awk '{print $1}')" = 3cb3d3c77d9de2a013fe223183750ca3588c595b83dbf33f3b67f18c26e2c2d5

PY=/scratch/lg154/conda-envs/memnav/bin/python
HAB_PY=/scratch/lg154/conda-envs/habitat/bin/python
HAB_SITE=$("${HAB_PY}" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
HAB_VENDOR=${HAB_SITE}/pip/_vendor
[[ -r "${HAB_VENDOR}/requests/__init__.py" ]] || \
  fail "Habitat vendored requests is missing"
export PYTHONPATH=${SOURCE_ROOT}:${SOURCE_ROOT}/MemNavData
"${PY}" -m pytest -q -p no:cacheprovider \
  "${SOURCE_ROOT}/MemNavData/test_hm3d_longrange_rear_alignment.py" \
  "${SOURCE_ROOT}/MemNavData/test_analyze_hm3d_longrange_rear_alignment.py" \
  "${SOURCE_ROOT}/MemNavData/test_monocular_route_tangent_runtime.py" \
  "${SOURCE_ROOT}/MemNavData/test_cec_bearing_alignment.py" \
  "${SOURCE_ROOT}/MemNavData/test_policy_agent_monocular_route_tangent.py"

dry_common=(
  --contract_dry_run --episode_root /tmp/cec_contract_dry_run
  --scene /tmp/cec_contract_dry_run.glb
  --port 1 --novel_port 2
  --server_backend hybrid_pose --leg1_mode shared_trace
  --leg1_goal_source own --deterministic_plan_seeds
  --trajectory_selector server --trajectory_selector_scope all
  --terminal_uturn off --terminal_visual_refine off
  --retrieval_override off --certified_cdec_rescue off
  --certified_stagnation_graph off --revisit_controller navdp_mixed
  --role_pair_scope table3_longrange_route_tangent
  --role_pair_query_role revisit --navdp_depth_source monocular_sidecar
  --hybrid_route certified_relocalization
  --revisit_adapter verified_bearing_v1
  --certified_guidance_mode monocular_route_tangent)
PYTHONPATH=${SOURCE_ROOT}:${SOURCE_ROOT}/MemNavData:${HAB_VENDOR} \
  "${HAB_PY}" "${SOURCE_ROOT}/MemNavData/eval_shared_online_role_pairs.py" \
  "${dry_common[@]}" --cec_initial_bearing_alignment off >/dev/null
PYTHONPATH=${SOURCE_ROOT}:${SOURCE_ROOT}/MemNavData:${HAB_VENDOR} \
  "${HAB_PY}" "${SOURCE_ROOT}/MemNavData/eval_shared_online_role_pairs.py" \
  "${dry_common[@]}" \
  --cec_initial_bearing_alignment first_route_tangent_rear_bounded >/dev/null

source "${SOURCE_ROOT}/MemNavData/slurm_safe_submit.sh"
EVAL=${SOURCE_ROOT}/MemNavData/slurm_hm3d_longrange_rear_alignment.sbatch
RESULT=${SOURCE_ROOT}/MemNavData/slurm_hm3d_longrange_rear_alignment_result.sbatch
lint_sbatch_template "${EVAL}" || fail "evaluation template lint failed"
lint_sbatch_template "${RESULT}" || fail "result template lint failed"
common="ALL,SOURCE_ROOT=${SOURCE_ROOT},SOURCE_RECEIPT=${SOURCE_RECEIPT},EXPECTED_SOURCE_RECEIPT_SHA=${EXPECTED_SOURCE_RECEIPT_SHA}"
safe_sbatch --lint-fatal --test-only --array=8 \
  --export="${common},PHASE=gate,RUN_ROOT=${RUN_ROOT}/technical_gate" \
  "${EVAL}" >/dev/null
safe_sbatch --lint-fatal --test-only --array="${SELECTED}%${CONCURRENCY}" \
  --export="${common},PHASE=formal,RUN_ROOT=${RUN_ROOT}" \
  "${EVAL}" >/dev/null
sbatch --test-only \
  --export="${common},MODE=analyze,RUN_ROOT=${RUN_ROOT}" \
  "${RESULT}" >/dev/null

[[ ! -e "${RUN_ROOT}" ]] || fail "run root already exists"
mkdir -p "${RUN_ROOT}/sealed_inputs" \
  /scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs
cp "${PROTOCOL}" \
  "${FORMAL_PARENT}/result/summary.json" \
  "${FORMAL_PARENT}/result/independent_verification.json" \
  "${FORMAL_PARENT}/fresh_population/role_pairs/manifest.json" \
  "${RUN_ROOT}/sealed_inputs/"
sha256sum "${RUN_ROOT}/sealed_inputs/"* > \
  "${RUN_ROOT}/sealed_inputs/inputs.sha256"
chmod -R a-w "${RUN_ROOT}/sealed_inputs"

gate_job=$(safe_sbatch --lint-fatal --parsable --array=8 \
  --export="${common},PHASE=gate,RUN_ROOT=${RUN_ROOT}/technical_gate" \
  "${EVAL}" | job_id)
formal_job=$(safe_sbatch --lint-fatal --parsable \
  --array="${SELECTED}%${CONCURRENCY}" \
  --dependency="afterok:${gate_job}" --kill-on-invalid-dep=yes \
  --export="${common},PHASE=formal,RUN_ROOT=${RUN_ROOT}" \
  "${EVAL}" | job_id)
analysis_job=$(sbatch --parsable \
  --dependency="afterok:${formal_job}" --kill-on-invalid-dep=yes \
  --export="${common},MODE=analyze,RUN_ROOT=${RUN_ROOT}" \
  "${RESULT}" | job_id)
verify_job=$(sbatch --parsable \
  --dependency="afterok:${analysis_job}" --kill-on-invalid-dep=yes \
  --export="${common},MODE=verify,RUN_ROOT=${RUN_ROOT}" \
  "${RESULT}" | job_id)
for value in "${gate_job}" "${formal_job}" "${analysis_job}" "${verify_job}"; do
  [[ "${value}" =~ ^[0-9]+$ ]] || fail "bad Slurm job id: ${value}"
done

"${PY}" - "${RUN_ROOT}/submission.json" "${gate_job}" "${formal_job}" \
  "${analysis_job}" "${verify_job}" "${SOURCE_ROOT}" \
  "${EXPECTED_SOURCE_RECEIPT_SHA}" "${PROTOCOL_SHA}" <<'PY'
import datetime, json, os, sys
path,gate,formal,analysis,verify,source,source_sha,protocol_sha=sys.argv[1:]
payload={
 'schema_version':'hm3d_longrange_rear_alignment_submission_v1_20260904',
 'submitted_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),
 'technical_gate_job':int(gate), 'technical_gate_history_index':8,
 'formal_evaluation_job':int(formal), 'analysis_job':int(analysis),
 'result_verification_job':int(verify),
 'selected_history_indices':[3,4,5,8,10,11,13,15,21],
 'array_concurrency':4, 'partition':['h100_tandon','a100_tandon'],
 'qos':'gpu48', 'requested_time_per_evaluation_element':'01:00:00',
 'source_root':source, 'source_receipt_sha256':source_sha,
 'protocol_sha256':protocol_sha,
 'claim_scope':'consumed controller-support mechanism only',
 'outcomes_read_before_design':True,
 'navigation_claim_allowed':False,
 'fresh_confirmation_required':True,
 'threshold_search_allowed':False,
}
fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o444)
with os.fdopen(fd,'w') as handle:
    json.dump(payload,handle,indent=2,sort_keys=True); handle.write('\n')
PY
sha256sum "${RUN_ROOT}/submission.json" > \
  "${RUN_ROOT}/submission.json.sha256"
chmod a-w "${RUN_ROOT}/submission.json" "${RUN_ROOT}/submission.json.sha256"
printf 'GATE_JOB=%s\nFORMAL_JOB=%s\nANALYSIS_JOB=%s\nVERIFY_JOB=%s\nRUN_ROOT=%s\n' \
  "${gate_job}" "${formal_job}" "${analysis_job}" "${verify_job}" \
  "${RUN_ROOT}"
