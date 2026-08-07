#!/usr/bin/env bash
# Submit the complete resumable feature-join -> independent-audit -> pinned
# three-seed training chain from one exact clean commit.

set -euo pipefail
umask 0022

: "${RUN_TAG:?export the audited NLSR RUN_TAG before submission}"
REPO_ROOT="${REPO_ROOT:-$(pwd -P)}"
EXPECTED_COMMIT="${EXPECTED_COMMIT:-$(git -C "${REPO_ROOT}" rev-parse HEAD)}"
TRAIN_PARTITION="${TRAIN_PARTITION:-a100_tandon}"
DEVELOPMENT_PARTITION="${DEVELOPMENT_PARTITION:-h100_tandon}"
LOG_ROOT="/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs"

COLLECT_REL="MemNavData/slurm_nlsr_phase_b_collect.sbatch"
AUDIT_REL="MemNavData/slurm_nlsr_phase_b_audit.sbatch"
STAGE_RELAY_REL="MemNavData/slurm_nlsr_phase_b_stage_relay.sbatch"
RELAY_REL="MemNavData/slurm_nlsr_phase_b_train_relay.sbatch"
COLLECT="${REPO_ROOT}/${COLLECT_REL}"
STAGE_RELAY="${REPO_ROOT}/${STAGE_RELAY_REL}"

fail() { echo "ABORT: $*" >&2; exit 2; }
for command_name in git sbatch sha256sum; do
  command -v "${command_name}" >/dev/null || fail "${command_name} is unavailable"
done
[[ "${RUN_TAG}" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]] || fail "invalid RUN_TAG"
[[ "${EXPECTED_COMMIT}" =~ ^[0-9a-f]{40}$ ]] || fail "invalid commit pin"
[[ "${TRAIN_PARTITION}" =~ ^(a100_tandon|h100_tandon)$ ]] || fail "invalid train partition"
[[ "${DEVELOPMENT_PARTITION}" =~ ^(a100_tandon|h100_tandon)$ ]] || \
  fail "invalid development partition"
[[ "$(git -C "${REPO_ROOT}" rev-parse HEAD)" == "${EXPECTED_COMMIT}" ]] || \
  fail "checkout HEAD differs from EXPECTED_COMMIT"
if [[ -n "$(git -C "${REPO_ROOT}" status --porcelain)" ]]; then
  git -C "${REPO_ROOT}" status --short >&2
  fail "submission checkout is not clean"
fi

for relative in "${COLLECT_REL}" "${AUDIT_REL}" "${STAGE_RELAY_REL}" \
                "${RELAY_REL}" \
                "MemNavData/slurm_nlsr_phase_b_train.sbatch"; do
  path="${REPO_ROOT}/${relative}"
  [[ -f "${path}" && ! -L "${path}" ]] || fail "missing launcher ${relative}"
  actual="$(sha256sum "${path}" | awk '{print $1}')"
  committed="$(git -C "${REPO_ROOT}" show "${EXPECTED_COMMIT}:${relative}" | \
    sha256sum | awk '{print $1}')"
  [[ "${actual}" == "${committed}" ]] || fail "${relative} differs from commit"
done
mkdir -p "${LOG_ROOT}"

COLLECT_SHA="$(sha256sum "${COLLECT}" | awk '{print $1}')"
STAGE_RELAY_SHA="$(sha256sum "${STAGE_RELAY}" | awk '{print $1}')"

submit_collect_once() {
  local role="$1" partition="$2" exports job
  exports="ALL,REPO_ROOT=${REPO_ROOT},EXPECTED_COMMIT=${EXPECTED_COMMIT}"
  exports+=",EXPECTED_LAUNCHER_SHA=${COLLECT_SHA},RUN_TAG=${RUN_TAG}"
  exports+=",PHASE_B_ROLE=${role}"
  sbatch --test-only --partition="${partition}" \
    --export="${exports}" "${COLLECT}" >/dev/null
  job="$(sbatch --parsable --partition="${partition}" \
    --export="${exports}" "${COLLECT}")"
  job="${job%%;*}"
  [[ "${job}" =~ ^[0-9]+$ ]] || fail "unexpected collector submission: ${job}"
  printf '%s' "${job}"
}

# The account admits one GPU request at a time.  Start only train; a committed
# CPU relay submitted with afterany will inspect its audited progress and then
# submit exactly one continuation or the development stage.
TRAIN_JOB="$(submit_collect_once train "${TRAIN_PARTITION}")"
stage_exports="ALL,REPO_ROOT=${REPO_ROOT},EXPECTED_COMMIT=${EXPECTED_COMMIT}"
stage_exports+=",EXPECTED_LAUNCHER_SHA=${STAGE_RELAY_SHA},RUN_TAG=${RUN_TAG}"
stage_exports+=",PHASE_B_ROLE=train,PHASE_B_ATTEMPT=1"
stage_exports+=",PHASE_B_GPU_JOB_ID=${TRAIN_JOB},MAX_ATTEMPTS=3"
stage_exports+=",TRAIN_PARTITION=${TRAIN_PARTITION}"
stage_exports+=",DEVELOPMENT_PARTITION=${DEVELOPMENT_PARTITION}"
sbatch --test-only --dependency="afterany:${TRAIN_JOB}" --kill-on-invalid-dep=yes \
  --export="${stage_exports}" "${STAGE_RELAY}" >/dev/null
STAGE_RELAY_JOB="$(sbatch --parsable --dependency="afterany:${TRAIN_JOB}" \
  --kill-on-invalid-dep=yes --export="${stage_exports}" "${STAGE_RELAY}")"
STAGE_RELAY_JOB="${STAGE_RELAY_JOB%%;*}"
[[ "${STAGE_RELAY_JOB}" =~ ^[0-9]+$ ]] || fail "unexpected stage relay submission"

printf 'submitted_phase_b_pipeline commit=%s train_job=%s stage_relay=%s gpu_serialization=one_request\n' \
  "${EXPECTED_COMMIT}" "${TRAIN_JOB}" "${STAGE_RELAY_JOB}"
