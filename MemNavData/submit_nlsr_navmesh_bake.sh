#!/usr/bin/env bash
# Submit the receipt-backed NavMesh bake only from an exact clean checkout.

set -euo pipefail
umask 0022

: "${RUN_TAG:?export the audited NLSR RUN_TAG before submission}"

REPO_ROOT="${REPO_ROOT:-$(pwd -P)}"
RESULT_BASE="${RESULT_BASE:-/scratch/yz11502/Research/Nav-axis-uturn-results/nlsr_gapfill_20260807}"
EXPECTED_COMMIT="${EXPECTED_COMMIT:-$(git -C "${REPO_ROOT}" rev-parse HEAD)}"
LAUNCHER_REL="MemNavData/slurm_nlsr_navmesh_bake.sbatch"
LAUNCHER="${REPO_ROOT}/${LAUNCHER_REL}"
AUDITOR_REL="MemNavData/audit_pinned_navmesh_bake.py"
AUDITOR="${REPO_ROOT}/${AUDITOR_REL}"
LOG_ROOT="/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs"
MANIFEST="${RESULT_BASE}/${RUN_TAG}/manifest_multistage/candidate_manifest_multistage.json"
EXPECTED_MANIFEST_SHA="bc6bf58536f6c159d1898ac03abe365eadba65c22cd246e39401821962abb34c"
EXPECTED_MANIFEST_SIDECAR_SHA="64de7deaba664264aaa74fb566ec9ae866b277b49930b404903f0ea737374d0c"

fail() {
  echo "ABORT: $*" >&2
  exit 2
}

for command_name in git sbatch sha256sum; do
  command -v "${command_name}" >/dev/null || fail "${command_name} is unavailable"
done
[[ "${RUN_TAG}" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]] || \
  fail "invalid RUN_TAG=${RUN_TAG}"
[[ "${EXPECTED_COMMIT}" =~ ^[0-9a-f]{40}$ ]] || \
  fail "EXPECTED_COMMIT is not a full lowercase Git commit"
[[ -d "${REPO_ROOT}/.git" || -f "${REPO_ROOT}/.git" ]] || \
  fail "REPO_ROOT is not a Git checkout"
[[ "$(git -C "${REPO_ROOT}" rev-parse HEAD)" == "${EXPECTED_COMMIT}" ]] || \
  fail "checkout HEAD differs from EXPECTED_COMMIT"
if [[ -n "$(git -C "${REPO_ROOT}" status --porcelain)" ]]; then
  git -C "${REPO_ROOT}" status --short >&2
  fail "submission checkout is not clean"
fi
[[ -f "${LAUNCHER}" && ! -L "${LAUNCHER}" ]] || \
  fail "launcher is absent or symlinked"
launcher_sha="$(sha256sum "${LAUNCHER}" | awk '{print $1}')"
committed_launcher_sha="$(git -C "${REPO_ROOT}" show \
  "${EXPECTED_COMMIT}:${LAUNCHER_REL}" | sha256sum | awk '{print $1}')"
[[ "${launcher_sha}" == "${committed_launcher_sha}" ]] || \
  fail "launcher bytes differ from the pinned commit"
[[ -f "${AUDITOR}" && ! -L "${AUDITOR}" ]] || \
  fail "NavMesh bake auditor is absent or symlinked"
auditor_sha="$(sha256sum "${AUDITOR}" | awk '{print $1}')"
committed_auditor_sha="$(git -C "${REPO_ROOT}" show \
  "${EXPECTED_COMMIT}:${AUDITOR_REL}" | sha256sum | awk '{print $1}')"
[[ "${auditor_sha}" == "${committed_auditor_sha}" ]] || \
  fail "auditor bytes differ from the pinned commit"
launcher_auditor_pin="$(awk -F'"' \
  '$1 == "EXPECTED_AUDITOR_SHA=" {print $2}' "${LAUNCHER}")"
[[ "${launcher_auditor_pin}" =~ ^[0-9a-f]{64}$ ]] || \
  fail "launcher does not contain exactly one valid auditor SHA pin"
[[ "${launcher_auditor_pin}" == "${auditor_sha}" ]] || \
  fail "launcher auditor SHA pin differs from committed auditor"
[[ -f "${MANIFEST}" && -f "${MANIFEST}.sha256" ]] || \
  fail "multistage manifest artifact pair is absent"
[[ "$(sha256sum "${MANIFEST}" | awk '{print $1}')" == \
  "${EXPECTED_MANIFEST_SHA}" ]] || fail "multistage manifest SHA differs"
[[ "$(sha256sum "${MANIFEST}.sha256" | awk '{print $1}')" == \
  "${EXPECTED_MANIFEST_SIDECAR_SHA}" ]] || \
  fail "multistage manifest exact sidecar differs"

mkdir -p "${LOG_ROOT}"
[[ -d "${LOG_ROOT}" && -w "${LOG_ROOT}" ]] || \
  fail "Slurm log directory is not writable"
[[ -d "${RESULT_BASE}/${RUN_TAG}" && -w "${RESULT_BASE}/${RUN_TAG}" ]] || \
  fail "run root is not writable"

exports="ALL,REPO_ROOT=${REPO_ROOT},EXPECTED_COMMIT=${EXPECTED_COMMIT}"
exports+=",EXPECTED_LAUNCHER_SHA=${launcher_sha},RUN_TAG=${RUN_TAG}"
exports+=",RESULT_BASE=${RESULT_BASE}"
sbatch --test-only --export="${exports}" "${LAUNCHER}"
job_id="$(sbatch --parsable --export="${exports}" "${LAUNCHER}")"
[[ "${job_id}" =~ ^[0-9]+([;].*)?$ ]] || fail "unexpected sbatch result: ${job_id}"
printf 'submitted_navmesh_bake_job=%s commit=%s launcher_sha256=%s auditor_sha256=%s\n' \
  "${job_id}" "${EXPECTED_COMMIT}" "${launcher_sha}" "${auditor_sha}"
