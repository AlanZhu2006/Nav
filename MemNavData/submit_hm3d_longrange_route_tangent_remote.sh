#!/usr/bin/env bash
# Run on the authenticated yz11502 Torch login node from the immutable bundle.
set -euo pipefail
umask 0022
export PYTHONDONTWRITEBYTECODE=1

SOURCE_ROOT=${SOURCE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
SOURCE_RECEIPT=${SOURCE_ROOT}/SOURCE_BUNDLE.sha256
EXPECTED_SOURCE_RECEIPT_SHA=${EXPECTED_SOURCE_RECEIPT_SHA:?}
SOURCE_SHA=$(sha256sum "${SOURCE_RECEIPT}" | awk '{print $1}')
PROTOCOL_SHA=435fe26646a8c42229230213ff70d9fc07c3d7b746b44641251d4eb4ce84227e
RUN_ROOT=${RUN_ROOT:-/scratch/yz11502/Research/Nav-axis-uturn-results/hm3d_longrange_route_tangent_20260903/fresh_${SOURCE_SHA:0:16}}
DRY_RUN=${DRY_RUN:-0}
PAIR_TIME=${PAIR_TIME:-02:00:00}

fail() { echo "ABORT: $*" >&2; exit 2; }
job_id() { tr -d '\r' | awk -F';' '/^[0-9]+(;|$)/ {print $1; exit}'; }
[[ "$(id -un)" == yz11502 ]] || fail "wrong HPC identity"
[[ "${DRY_RUN}" =~ ^[01]$ ]] || fail "DRY_RUN must be 0 or 1"
[[ "${SOURCE_SHA}" == "${EXPECTED_SOURCE_RECEIPT_SHA}" ]] || \
  fail "source receipt changed"
(cd "${SOURCE_ROOT}" && sha256sum -c --quiet "${SOURCE_RECEIPT}") || \
  fail "source bundle content changed"
PROTOCOL=${SOURCE_ROOT}/MemNavData/hm3d_longrange_route_tangent_freeze_protocol_v2_20260903.json
[[ "$(sha256sum "${PROTOCOL}" | awk '{print $1}')" == "${PROTOCOL_SHA}" ]] || \
  fail "frozen protocol changed"
[[ ! -e "${RUN_ROOT}" ]] || fail "run root already exists: ${RUN_ROOT}"

POP_SBATCH=${SOURCE_ROOT}/MemNavData/slurm_hm3d_longrange_route_tangent_population.sbatch
PAIR_SBATCH=${SOURCE_ROOT}/MemNavData/slurm_hm3d_longrange_route_tangent_pair.sbatch
RESULT_SBATCH=${SOURCE_ROOT}/MemNavData/slurm_hm3d_longrange_route_tangent_result.sbatch
source "${SOURCE_ROOT}/MemNavData/slurm_safe_submit.sh"
for template in "${POP_SBATCH}" "${PAIR_SBATCH}" "${RESULT_SBATCH}"; do
  lint_sbatch_template "${template}" || fail "Slurm template lint failed"
done
common="ALL,SOURCE_ROOT=${SOURCE_ROOT},SOURCE_RECEIPT=${SOURCE_RECEIPT},EXPECTED_SOURCE_RECEIPT_SHA=${SOURCE_SHA},RUN_ROOT=${RUN_ROOT}"

# Validate every submission shape before creating any run state.  The gate
# and remainder are submitted in one transaction; no gate outcome is read.
safe_sbatch --lint-fatal --test-only --partition=cpu_short --time=00:30:00 \
  --export="${common},MODE=freeze" "${POP_SBATCH}" >/dev/null
safe_sbatch --lint-fatal --test-only --partition=cpu_short --time=00:30:00 \
  --export="${common},MODE=verify" "${POP_SBATCH}" >/dev/null
safe_sbatch --lint-fatal --test-only --partition=h100_tandon,a100_tandon \
  --account=torch_pr_769_tandon_advanced --qos=gpu48 --gres=gpu:1 \
  --cpus-per-task=10 --mem=72G --time="${PAIR_TIME}" --array=0 \
  --export="${common}" "${PAIR_SBATCH}" >/dev/null
safe_sbatch --lint-fatal --test-only --partition=h100_tandon,a100_tandon \
  --account=torch_pr_769_tandon_advanced --qos=gpu48 --gres=gpu:1 \
  --cpus-per-task=10 --mem=72G --time="${PAIR_TIME}" --array=1-22%4 \
  --export="${common}" "${PAIR_SBATCH}" >/dev/null
safe_sbatch --lint-fatal --test-only --partition=cpu_short --time=00:20:00 \
  --export="${common},MODE=analyze" "${RESULT_SBATCH}" >/dev/null
safe_sbatch --lint-fatal --test-only --partition=cpu_short --time=00:20:00 \
  --export="${common},MODE=verify" "${RESULT_SBATCH}" >/dev/null
if [[ "${DRY_RUN}" == 1 ]]; then
  echo "DRY_RUN_OK histories=23 scenes=8 pair_time=${PAIR_TIME}"
  exit 0
fi

mkdir -p "${RUN_ROOT}"
freeze_raw=$(safe_sbatch --lint-fatal --parsable --partition=cpu_short \
  --time=00:30:00 --export="${common},MODE=freeze" "${POP_SBATCH}")
freeze_job=$(printf '%s\n' "${freeze_raw}" | job_id)
[[ "${freeze_job}" =~ ^[0-9]+$ ]] || fail "invalid population freeze job id"
verify_raw=$(safe_sbatch --lint-fatal --parsable --partition=cpu_short \
  --time=00:30:00 --dependency="afterok:${freeze_job}" \
  --kill-on-invalid-dep=yes --export="${common},MODE=verify" "${POP_SBATCH}")
population_verify_job=$(printf '%s\n' "${verify_raw}" | job_id)
[[ "${population_verify_job}" =~ ^[0-9]+$ ]] || \
  fail "invalid population verifier job id"
gate_raw=$(safe_sbatch --lint-fatal --parsable \
  --partition=h100_tandon,a100_tandon \
  --account=torch_pr_769_tandon_advanced --qos=gpu48 --gres=gpu:1 \
  --cpus-per-task=10 --mem=72G --time="${PAIR_TIME}" --array=0 \
  --dependency="afterok:${population_verify_job}" --kill-on-invalid-dep=yes \
  --export="${common}" "${PAIR_SBATCH}")
gate_job=$(printf '%s\n' "${gate_raw}" | job_id)
[[ "${gate_job}" =~ ^[0-9]+$ ]] || fail "invalid technical gate job id"
remaining_raw=$(safe_sbatch --lint-fatal --parsable \
  --partition=h100_tandon,a100_tandon \
  --account=torch_pr_769_tandon_advanced --qos=gpu48 --gres=gpu:1 \
  --cpus-per-task=10 --mem=72G --time="${PAIR_TIME}" --array=1-22%4 \
  --dependency="afterok:${gate_job}" --kill-on-invalid-dep=yes \
  --export="${common}" "${PAIR_SBATCH}")
remaining_job=$(printf '%s\n' "${remaining_raw}" | job_id)
[[ "${remaining_job}" =~ ^[0-9]+$ ]] || fail "invalid remainder job id"
analysis_raw=$(safe_sbatch --lint-fatal --parsable --partition=cpu_short \
  --time=00:20:00 --dependency="afterok:${remaining_job}" \
  --kill-on-invalid-dep=yes --export="${common},MODE=analyze" \
  "${RESULT_SBATCH}")
analysis_job=$(printf '%s\n' "${analysis_raw}" | job_id)
[[ "${analysis_job}" =~ ^[0-9]+$ ]] || fail "invalid analysis job id"
result_verify_raw=$(safe_sbatch --lint-fatal --parsable --partition=cpu_short \
  --time=00:20:00 --dependency="afterok:${analysis_job}" \
  --kill-on-invalid-dep=yes --export="${common},MODE=verify" \
  "${RESULT_SBATCH}")
result_verify_job=$(printf '%s\n' "${result_verify_raw}" | job_id)
[[ "${result_verify_job}" =~ ^[0-9]+$ ]] || \
  fail "invalid result verifier job id"

/scratch/lg154/conda-envs/memnav/bin/python - \
  "${RUN_ROOT}/submission.json" "${freeze_job}" \
  "${population_verify_job}" "${gate_job}" "${remaining_job}" \
  "${analysis_job}" "${result_verify_job}" "${SOURCE_ROOT}" \
  "${SOURCE_SHA}" "${PROTOCOL_SHA}" "${PAIR_TIME}" <<'PY'
import datetime, json, os, sys
(path, freeze_job, population_verify, gate, remaining, analysis,
 result_verify, source, source_sha, protocol_sha, pair_time)=sys.argv[1:]
payload={
 'schema_version':'hm3d_longrange_route_tangent_submission_v1_20260903',
 'submitted_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),
 'population_freeze_job':int(freeze_job),
 'population_verification_job':int(population_verify),
 'technical_gate_job':int(gate), 'technical_gate_index':0,
 'remaining_evaluation_job':int(remaining),
 'remaining_indices':'1-22', 'array_concurrency':4,
 'analysis_job':int(analysis), 'result_verification_job':int(result_verify),
 'partition':['h100_tandon','a100_tandon'], 'qos':'gpu48',
 'requested_time_per_evaluation_element':pair_time,
 'source_root':source, 'source_receipt_sha256':source_sha,
 'protocol_sha256':protocol_sha,
 'population_scope':'fresh-history reused-scene same-floor 20--30 m Revisit',
 'histories':23, 'scene_clusters':8,
 'arms':['mono_native','mono_cec_endpoint','mono_cec_route_tangent'],
 'primary_contrast':['mono_cec_route_tangent','mono_cec_endpoint'],
 'success_contract':'3d_euclidean_strictly_below_1m',
 'query_policy_outcomes_read_before_submission':False,
 'technical_gate_outcome_read_before_remaining_submission':False,
 'threshold_or_radius_sweep_allowed':False,
 'post_accept_fallback_allowed':False,
}
fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o444)
with os.fdopen(fd,'w') as handle:
    json.dump(payload,handle,indent=2,sort_keys=True); handle.write('\n')
PY
sha256sum "${RUN_ROOT}/submission.json" >"${RUN_ROOT}/submission.json.sha256"
chmod a-w "${RUN_ROOT}/submission.json" "${RUN_ROOT}/submission.json.sha256"
printf 'FREEZE_JOB=%s\nPOPULATION_VERIFY_JOB=%s\nGATE_JOB=%s\nREMAINING_JOB=%s\nANALYSIS_JOB=%s\nRESULT_VERIFY_JOB=%s\nRUN_ROOT=%s\n' \
  "${freeze_job}" "${population_verify_job}" "${gate_job}" \
  "${remaining_job}" "${analysis_job}" "${result_verify_job}" "${RUN_ROOT}"
