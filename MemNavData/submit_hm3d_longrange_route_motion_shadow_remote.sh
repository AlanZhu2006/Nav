#!/usr/bin/env bash
# Run on the authenticated yz11502 Torch login node from the immutable bundle.
set -euo pipefail
umask 0022
export PYTHONDONTWRITEBYTECODE=1

SOURCE_ROOT=${SOURCE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
SOURCE_RECEIPT=${SOURCE_ROOT}/SOURCE_BUNDLE.sha256
EXPECTED_SOURCE_RECEIPT_SHA=${EXPECTED_SOURCE_RECEIPT_SHA:?}
SOURCE_SHA=$(sha256sum "${SOURCE_RECEIPT}" | awk '{print $1}')
PROTOCOL_SHA=589af0882063bca63b47892938887c8a93e74477ccb68f2f9cfb02778da99943
RUN_ROOT=${RUN_ROOT:-/scratch/yz11502/Research/Nav-axis-uturn-results/hm3d_longrange_route_motion_shadow_20260903/consumed_${SOURCE_SHA:0:16}}
DRY_RUN=${DRY_RUN:-0}
PAIR_TIME=${PAIR_TIME:-01:00:00}
SELECTED=0,1,2,6,7,9,12,14,16,17,18,19,20,22
REMAINDER=1,2,6,7,9,12,14,16,17,18,19,20,22

fail() { echo "ABORT: $*" >&2; exit 2; }
job_id() { tr -d '\r' | awk -F';' '/^[0-9]+(;|$)/ {print $1; exit}'; }
[[ "$(id -un)" == yz11502 ]] || fail "wrong HPC identity"
[[ "${DRY_RUN}" =~ ^[01]$ ]] || fail "DRY_RUN must be 0 or 1"
[[ "${SOURCE_SHA}" == "${EXPECTED_SOURCE_RECEIPT_SHA}" ]] || \
  fail "source receipt changed"
(cd "${SOURCE_ROOT}" && sha256sum -c --quiet "${SOURCE_RECEIPT}") || \
  fail "source bundle content changed"
PROTOCOL=${SOURCE_ROOT}/MemNavData/hm3d_longrange_route_motion_shadow_protocol_20260903.json
[[ "$(sha256sum "${PROTOCOL}" | awk '{print $1}')" == "${PROTOCOL_SHA}" ]] || \
  fail "diagnostic protocol changed"
[[ ! -e "${RUN_ROOT}" ]] || fail "run root already exists: ${RUN_ROOT}"

PAIR_SBATCH=${SOURCE_ROOT}/MemNavData/slurm_hm3d_longrange_route_motion_shadow.sbatch
RESULT_SBATCH=${SOURCE_ROOT}/MemNavData/slurm_hm3d_longrange_route_motion_shadow_result.sbatch
source "${SOURCE_ROOT}/MemNavData/slurm_safe_submit.sh"
for template in "${PAIR_SBATCH}" "${RESULT_SBATCH}"; do
  lint_sbatch_template "${template}" || fail "Slurm template lint failed"
done
common="ALL,SOURCE_ROOT=${SOURCE_ROOT},SOURCE_RECEIPT=${SOURCE_RECEIPT},EXPECTED_SOURCE_RECEIPT_SHA=${SOURCE_SHA},RUN_ROOT=${RUN_ROOT}"

safe_sbatch --lint-fatal --test-only --partition=h100_tandon,a100_tandon \
  --account=torch_pr_769_tandon_advanced --qos=gpu48 --gres=gpu:1 \
  --cpus-per-task=10 --mem=72G --time="${PAIR_TIME}" --array=0 \
  --export="${common}" "${PAIR_SBATCH}" >/dev/null
safe_sbatch --lint-fatal --test-only --partition=h100_tandon,a100_tandon \
  --account=torch_pr_769_tandon_advanced --qos=gpu48 --gres=gpu:1 \
  --cpus-per-task=10 --mem=72G --time="${PAIR_TIME}" --array="${REMAINDER}%4" \
  --export="${common}" "${PAIR_SBATCH}" >/dev/null
safe_sbatch --lint-fatal --test-only --partition=cpu_short --time=00:10:00 \
  --export="${common},MODE=analyze" "${RESULT_SBATCH}" >/dev/null
safe_sbatch --lint-fatal --test-only --partition=cpu_short --time=00:10:00 \
  --export="${common},MODE=verify" "${RESULT_SBATCH}" >/dev/null
if [[ "${DRY_RUN}" == 1 ]]; then
  echo "DRY_RUN_OK selected=${SELECTED} pair_time=${PAIR_TIME}"
  exit 0
fi

mkdir -p "${RUN_ROOT}"
gate_raw=$(safe_sbatch --lint-fatal --parsable \
  --partition=h100_tandon,a100_tandon \
  --account=torch_pr_769_tandon_advanced --qos=gpu48 --gres=gpu:1 \
  --cpus-per-task=10 --mem=72G --time="${PAIR_TIME}" --array=0 \
  --export="${common}" "${PAIR_SBATCH}")
gate_job=$(printf '%s\n' "${gate_raw}" | job_id)
[[ "${gate_job}" =~ ^[0-9]+$ ]] || fail "invalid technical gate job id"
remaining_raw=$(safe_sbatch --lint-fatal --parsable \
  --partition=h100_tandon,a100_tandon \
  --account=torch_pr_769_tandon_advanced --qos=gpu48 --gres=gpu:1 \
  --cpus-per-task=10 --mem=72G --time="${PAIR_TIME}" \
  --array="${REMAINDER}%4" --dependency="afterok:${gate_job}" \
  --kill-on-invalid-dep=yes --export="${common}" "${PAIR_SBATCH}")
remaining_job=$(printf '%s\n' "${remaining_raw}" | job_id)
[[ "${remaining_job}" =~ ^[0-9]+$ ]] || fail "invalid remainder job id"
analysis_raw=$(safe_sbatch --lint-fatal --parsable --partition=cpu_short \
  --time=00:10:00 --dependency="afterok:${remaining_job}" \
  --kill-on-invalid-dep=yes --export="${common},MODE=analyze" \
  "${RESULT_SBATCH}")
analysis_job=$(printf '%s\n' "${analysis_raw}" | job_id)
[[ "${analysis_job}" =~ ^[0-9]+$ ]] || fail "invalid analysis job id"
verify_raw=$(safe_sbatch --lint-fatal --parsable --partition=cpu_short \
  --time=00:10:00 --dependency="afterok:${analysis_job}" \
  --kill-on-invalid-dep=yes --export="${common},MODE=verify" \
  "${RESULT_SBATCH}")
verify_job=$(printf '%s\n' "${verify_raw}" | job_id)
[[ "${verify_job}" =~ ^[0-9]+$ ]] || fail "invalid verifier job id"

/scratch/lg154/conda-envs/memnav/bin/python - \
  "${RUN_ROOT}/submission.json" "${gate_job}" "${remaining_job}" \
  "${analysis_job}" "${verify_job}" "${SOURCE_ROOT}" "${SOURCE_SHA}" \
  "${PROTOCOL_SHA}" "${PAIR_TIME}" "${SELECTED}" <<'PY'
import datetime, json, os, sys
(path, gate, remaining, analysis, verify, source, source_sha,
 protocol_sha, pair_time, selected)=sys.argv[1:]
payload={
 'schema_version':'hm3d_longrange_route_motion_shadow_submission_v1_20260903',
 'submitted_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),
 'technical_gate_job':int(gate), 'technical_gate_index':0,
 'remaining_evaluation_job':int(remaining),
 'analysis_job':int(analysis), 'result_verification_job':int(verify),
 'selected_history_indices':[int(x) for x in selected.split(',')],
 'array_concurrency':4, 'partition':['h100_tandon','a100_tandon'],
 'qos':'gpu48', 'requested_time_per_evaluation_element':pair_time,
 'source_root':source, 'source_receipt_sha256':source_sha,
 'protocol_sha256':protocol_sha,
 'claim_scope':'consumed same-edge mechanism diagnostic only',
 'primary_estimator':'fundamental_magsac_then_depth_pnp_ransac',
 'shadow_estimator':'direct_depth_pnp_ransac',
 'shadow_control_authority':False,
 'navigation_claim_allowed':False,
 'fresh_confirmation_required':True,
 'outcomes_read_before_submission':True,
 'threshold_search_allowed':False,
}
fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o444)
with os.fdopen(fd,'w') as handle:
    json.dump(payload,handle,indent=2,sort_keys=True); handle.write('\n')
PY
sha256sum "${RUN_ROOT}/submission.json" >"${RUN_ROOT}/submission.json.sha256"
chmod a-w "${RUN_ROOT}/submission.json" "${RUN_ROOT}/submission.json.sha256"
printf 'GATE_JOB=%s\nREMAINING_JOB=%s\nANALYSIS_JOB=%s\nVERIFY_JOB=%s\nRUN_ROOT=%s\n' \
  "${gate_job}" "${remaining_job}" "${analysis_job}" "${verify_job}" \
  "${RUN_ROOT}"
