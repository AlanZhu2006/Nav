#!/usr/bin/env bash
# Submit the frozen geometry map only from an exact clean checkout.

set -euo pipefail
umask 0022

: "${RUN_TAG:?export the audited NLSR RUN_TAG before submission}"

REPO_ROOT="${REPO_ROOT:-$(pwd -P)}"
EXPECTED_COMMIT="${EXPECTED_COMMIT:-$(git -C "${REPO_ROOT}" rev-parse HEAD)}"
LAUNCHER_REL="MemNavData/slurm_nlsr_frozen_geometry_map.sbatch"
LAUNCHER="${REPO_ROOT}/${LAUNCHER_REL}"
LOG_ROOT="/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs"

fail() {
  echo "ABORT: $*" >&2
  exit 2
}

for command_name in git sbatch sha256sum; do
  command -v "${command_name}" >/dev/null || fail "${command_name} is unavailable"
done
[[ "${RUN_TAG}" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]] || fail "invalid RUN_TAG"
[[ "${EXPECTED_COMMIT}" =~ ^[0-9a-f]{40}$ ]] || fail "invalid commit pin"
[[ -d "${REPO_ROOT}/.git" || -f "${REPO_ROOT}/.git" ]] || fail "not a checkout"
[[ "$(git -C "${REPO_ROOT}" rev-parse HEAD)" == "${EXPECTED_COMMIT}" ]] || \
  fail "checkout HEAD differs from EXPECTED_COMMIT"
if [[ -n "$(git -C "${REPO_ROOT}" status --porcelain)" ]]; then
  git -C "${REPO_ROOT}" status --short >&2
  fail "submission checkout is not clean"
fi
[[ -f "${LAUNCHER}" && ! -L "${LAUNCHER}" ]] || fail "launcher is absent or symlinked"
launcher_sha="$(sha256sum "${LAUNCHER}" | awk '{print $1}')"
committed_sha="$(git -C "${REPO_ROOT}" show \
  "${EXPECTED_COMMIT}:${LAUNCHER_REL}" | sha256sum | awk '{print $1}')"
[[ "${launcher_sha}" == "${committed_sha}" ]] || fail "launcher differs from commit"
mkdir -p "${LOG_ROOT}"
exports="ALL,REPO_ROOT=${REPO_ROOT},EXPECTED_COMMIT=${EXPECTED_COMMIT}"
exports+=",EXPECTED_LAUNCHER_SHA=${launcher_sha},RUN_TAG=${RUN_TAG}"
sbatch --test-only --export="${exports}" "${LAUNCHER}"
job_id="$(sbatch --parsable --export="${exports}" "${LAUNCHER}")"
[[ "${job_id}" =~ ^[0-9]+([;].*)?$ ]] || fail "unexpected sbatch result: ${job_id}"
printf 'submitted_frozen_geometry_job=%s commit=%s launcher_sha256=%s\n' \
  "${job_id}" "${EXPECTED_COMMIT}" "${launcher_sha}"
