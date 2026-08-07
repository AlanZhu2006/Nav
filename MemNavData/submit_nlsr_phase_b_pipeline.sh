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
RELAY_REL="MemNavData/slurm_nlsr_phase_b_train_relay.sbatch"
COLLECT="${REPO_ROOT}/${COLLECT_REL}"
AUDIT="${REPO_ROOT}/${AUDIT_REL}"
RELAY="${REPO_ROOT}/${RELAY_REL}"

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

for relative in "${COLLECT_REL}" "${AUDIT_REL}" "${RELAY_REL}" \
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
AUDIT_SHA="$(sha256sum "${AUDIT}" | awk '{print $1}')"
RELAY_SHA="$(sha256sum "${RELAY}" | awk '{print $1}')"

submit_collect() {
  local role="$1" partition="$2" dependency="${3:-}" exports args job
  exports="ALL,REPO_ROOT=${REPO_ROOT},EXPECTED_COMMIT=${EXPECTED_COMMIT}"
  exports+=",EXPECTED_LAUNCHER_SHA=${COLLECT_SHA},RUN_TAG=${RUN_TAG}"
  exports+=",PHASE_B_ROLE=${role}"
  args=(--partition="${partition}" --export="${exports}")
  if [[ -n "${dependency}" ]]; then
    args+=(--dependency="${dependency}" --kill-on-invalid-dep=yes)
  fi
  sbatch --test-only "${args[@]}" "${COLLECT}" >/dev/null
  job="$(sbatch --parsable "${args[@]}" "${COLLECT}")"
  job="${job%%;*}"
  [[ "${job}" =~ ^[0-9]+$ ]] || fail "unexpected collector submission: ${job}"
  printf '%s' "${job}"
}

submit_audit() {
  local role="$1" dependency="$2" exports job
  exports="ALL,REPO_ROOT=${REPO_ROOT},EXPECTED_COMMIT=${EXPECTED_COMMIT}"
  exports+=",EXPECTED_LAUNCHER_SHA=${AUDIT_SHA},RUN_TAG=${RUN_TAG}"
  exports+=",PHASE_B_ROLE=${role}"
  sbatch --test-only --dependency="afterok:${dependency}" \
    --kill-on-invalid-dep=yes --export="${exports}" "${AUDIT}" >/dev/null
  job="$(sbatch --parsable --dependency="afterok:${dependency}" \
    --kill-on-invalid-dep=yes --export="${exports}" "${AUDIT}")"
  job="${job%%;*}"
  [[ "${job}" =~ ^[0-9]+$ ]] || fail "unexpected audit submission: ${job}"
  printf '%s' "${job}"
}

TRAIN_INITIAL="$(submit_collect train "${TRAIN_PARTITION}")"
DEVELOPMENT_INITIAL="$(submit_collect development "${DEVELOPMENT_PARTITION}")"
# One continuation per role guarantees that a wall-time timeout loses at most
# the current session.  If the first job already completed, this job performs
# the full preflight and exits before allocating LingBot weights.
TRAIN_CONTINUATION="$(submit_collect train "${TRAIN_PARTITION}" \
  "afterany:${TRAIN_INITIAL}")"
DEVELOPMENT_CONTINUATION="$(submit_collect development "${DEVELOPMENT_PARTITION}" \
  "afterany:${DEVELOPMENT_INITIAL}")"
TRAIN_AUDIT_JOB="$(submit_audit train "${TRAIN_CONTINUATION}")"
DEVELOPMENT_AUDIT_JOB="$(submit_audit development "${DEVELOPMENT_CONTINUATION}")"

relay_exports="ALL,REPO_ROOT=${REPO_ROOT},EXPECTED_COMMIT=${EXPECTED_COMMIT}"
relay_exports+=",EXPECTED_LAUNCHER_SHA=${RELAY_SHA},RUN_TAG=${RUN_TAG}"
relay_exports+=",TRAIN_AUDIT_JOB_ID=${TRAIN_AUDIT_JOB}"
relay_exports+=",DEVELOPMENT_AUDIT_JOB_ID=${DEVELOPMENT_AUDIT_JOB}"
relay_dependency="afterok:${TRAIN_AUDIT_JOB}:${DEVELOPMENT_AUDIT_JOB}"
sbatch --test-only --dependency="${relay_dependency}" --kill-on-invalid-dep=yes \
  --export="${relay_exports}" "${RELAY}" >/dev/null
RELAY_JOB="$(sbatch --parsable --dependency="${relay_dependency}" \
  --kill-on-invalid-dep=yes --export="${relay_exports}" "${RELAY}")"
RELAY_JOB="${RELAY_JOB%%;*}"
[[ "${RELAY_JOB}" =~ ^[0-9]+$ ]] || fail "unexpected relay submission: ${RELAY_JOB}"

printf 'submitted_phase_b_pipeline commit=%s train=%s train_resume=%s dev=%s dev_resume=%s train_audit=%s dev_audit=%s relay=%s\n' \
  "${EXPECTED_COMMIT}" "${TRAIN_INITIAL}" "${TRAIN_CONTINUATION}" \
  "${DEVELOPMENT_INITIAL}" "${DEVELOPMENT_CONTINUATION}" \
  "${TRAIN_AUDIT_JOB}" "${DEVELOPMENT_AUDIT_JOB}" "${RELAY_JOB}"
