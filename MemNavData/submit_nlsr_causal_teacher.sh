#!/usr/bin/env bash
# Submit one formal causal-teacher stage from an exact clean commit.
#
# Stage A:
#   RUN_TAG=<tag> ./MemNavData/submit_nlsr_causal_teacher.sh stage-a
# Stage B, after Stage A has completed and its receipt was inspected:
#   RUN_TAG=<tag> STAGE_A_JOB_ID=<id> EXPECTED_EMBEDDING_RECEIPT_SHA=<sha> \
#     ./MemNavData/submit_nlsr_causal_teacher.sh stage-b

set -euo pipefail
umask 0022

MODE="${1:-}"
: "${RUN_TAG:?export the audited NLSR RUN_TAG before submission}"
REPO_ROOT="${REPO_ROOT:-$(pwd -P)}"
EXPECTED_COMMIT="${EXPECTED_COMMIT:-$(git -C "${REPO_ROOT}" rev-parse HEAD)}"
LOG_ROOT="/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs"
RESULT_BASE="${RESULT_BASE:-/scratch/yz11502/Research/Nav-axis-uturn-results/nlsr_gapfill_20260807}"
RUN_ROOT="${RESULT_BASE}/${RUN_TAG}"
OUTPUT_ROOT="${RUN_ROOT}/causal_covisibility_teacher_${EXPECTED_COMMIT:0:12}_bc6bf58536f6"

fail() {
  echo "ABORT: $*" >&2
  exit 2
}

for command_name in git sbatch sha256sum; do
  command -v "${command_name}" >/dev/null || fail "${command_name} is unavailable"
done
[[ "${MODE}" == "stage-a" || "${MODE}" == "stage-b" ]] || \
  fail "usage: $0 {stage-a|stage-b}"
[[ "${RUN_TAG}" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]] || fail "invalid RUN_TAG"
[[ "${EXPECTED_COMMIT}" =~ ^[0-9a-f]{40}$ ]] || fail "invalid commit pin"
[[ -d "${REPO_ROOT}/.git" || -f "${REPO_ROOT}/.git" ]] || fail "not a checkout"
[[ "$(git -C "${REPO_ROOT}" rev-parse HEAD)" == "${EXPECTED_COMMIT}" ]] || \
  fail "checkout HEAD differs from EXPECTED_COMMIT"
if [[ -n "$(git -C "${REPO_ROOT}" status --porcelain)" ]]; then
  git -C "${REPO_ROOT}" status --short >&2
  fail "submission checkout is not clean"
fi

if [[ "${MODE}" == "stage-a" ]]; then
  LAUNCHER_REL="MemNavData/slurm_nlsr_causal_teacher_embeddings.sbatch"
else
  LAUNCHER_REL="MemNavData/slurm_nlsr_causal_teacher_assembly.sbatch"
fi
LAUNCHER="${REPO_ROOT}/${LAUNCHER_REL}"
[[ -f "${LAUNCHER}" && ! -L "${LAUNCHER}" ]] || fail "launcher is absent or symlinked"
launcher_sha="$(sha256sum "${LAUNCHER}" | awk '{print $1}')"
committed_sha="$(git -C "${REPO_ROOT}" show \
  "${EXPECTED_COMMIT}:${LAUNCHER_REL}" | sha256sum | awk '{print $1}')"
[[ "${launcher_sha}" == "${committed_sha}" ]] || fail "launcher differs from commit"
mkdir -p "${LOG_ROOT}"

exports="ALL,REPO_ROOT=${REPO_ROOT},EXPECTED_COMMIT=${EXPECTED_COMMIT}"
exports+=",EXPECTED_LAUNCHER_SHA=${launcher_sha},RUN_TAG=${RUN_TAG}"
dependencies=()
if [[ "${MODE}" == "stage-b" ]]; then
  command -v sacct >/dev/null || fail "sacct is unavailable"
  : "${STAGE_A_JOB_ID:?export the completed Stage-A job ID}"
  : "${EXPECTED_EMBEDDING_RECEIPT_SHA:?externally pin the inspected Stage-A receipt}"
  [[ "${STAGE_A_JOB_ID}" =~ ^[0-9]+$ ]] || fail "invalid Stage-A job ID"
  [[ "${EXPECTED_EMBEDDING_RECEIPT_SHA}" =~ ^[0-9a-f]{64}$ ]] || \
    fail "invalid Stage-A receipt SHA"
  receipt="${OUTPUT_ROOT}/dino_embedding_bundle/embedding_receipt.json"
  sidecar="${receipt}.sha256"
  [[ -f "${receipt}" && ! -L "${receipt}" && -f "${sidecar}" && ! -L "${sidecar}" ]] || \
    fail "completed Stage-A receipt pair is absent"
  actual_receipt_sha="$(sha256sum "${receipt}" | awk '{print $1}')"
  [[ "${actual_receipt_sha}" == "${EXPECTED_EMBEDDING_RECEIPT_SHA}" ]] || \
    fail "Stage-A receipt differs from external pin"
  [[ "$(awk 'NR==1 {print $1}' "${sidecar}")" == "${EXPECTED_EMBEDDING_RECEIPT_SHA}" ]] || \
    fail "Stage-A receipt sidecar differs from external pin"
  stage_a_state="$(sacct -n -X -j "${STAGE_A_JOB_ID}" -o State -P | awk 'NF {print $1; exit}')"
  [[ "${stage_a_state}" == "COMPLETED" ]] || \
    fail "Stage-A job is not completed: ${STAGE_A_JOB_ID} state=${stage_a_state:-unknown}"
  exports+=",EXPECTED_EMBEDDING_RECEIPT_SHA=${EXPECTED_EMBEDDING_RECEIPT_SHA}"
  exports+=",EXPECTED_STAGE_A_JOB_ID=${STAGE_A_JOB_ID}"
  dependencies+=(--dependency="afterok:${STAGE_A_JOB_ID}" --kill-on-invalid-dep=yes)
fi

sbatch --test-only "${dependencies[@]}" --export="${exports}" "${LAUNCHER}"
job_id="$(sbatch --parsable "${dependencies[@]}" --export="${exports}" "${LAUNCHER}")"
job_id="${job_id%%;*}"
[[ "${job_id}" =~ ^[0-9]+$ ]] || fail "unexpected sbatch result: ${job_id}"
printf 'submitted_causal_teacher_%s_job=%s commit=%s launcher_sha256=%s' \
  "${MODE}" "${job_id}" "${EXPECTED_COMMIT}" "${launcher_sha}"
if [[ "${MODE}" == "stage-b" ]]; then
  printf ' dependency=afterok:%s embedding_receipt_sha256=%s' \
    "${STAGE_A_JOB_ID}" "${EXPECTED_EMBEDDING_RECEIPT_SHA}"
fi
printf '\n'
