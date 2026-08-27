#!/usr/bin/env bash
# Submit a one-episode engineering smoke, the full consumed-pool G1, and its
# read-only paired summarizer.  Run this only on the HPC login node.

set -euo pipefail
umask 0022

ROOT=${ROOT:-$(git rev-parse --show-toplevel)}
RUN_TAG=${RUN_TAG:?export a unique RUN_TAG}
GATE_SBATCH=${ROOT}/MemNavData/slurm_xnavdp_revisit_gate.sbatch
SUMMARY_SBATCH=${ROOT}/MemNavData/slurm_xnavdp_revisit_summary.sbatch
MANIFEST=${ROOT}/MemNavData/expanded_navdp_router_eval_20260805.json
XNAVDP_ASSET_ROOT=${XNAVDP_ASSET_ROOT:-/scratch/yz11502/Research/xnavdp_official_878740a2011856d0}
XNAVDP_OFFICIAL_ROOT=${XNAVDP_OFFICIAL_ROOT:-${XNAVDP_ASSET_ROOT}/NavDP}
XNAVDP_CKPT=${XNAVDP_CKPT:-${XNAVDP_ASSET_ROOT}/x-navdp_posttrain.ckpt}
EXPECTED_XNAVDP_COMMIT=878740a2011856d0e3782dd6ccd880fd2eccd70f
EXPECTED_XNAVDP_CKPT_SHA=267089a81bbbe7a913debda6603f3f1b66a79520370ce953b2d888d793b89f24

[[ "${RUN_TAG}" =~ ^[A-Za-z0-9._-]+$ ]] || {
  echo "ABORT: RUN_TAG contains unsafe characters" >&2; exit 1; }
for command_name in git sbatch sha256sum; do
  command -v "${command_name}" >/dev/null || {
    echo "ABORT: ${command_name} is unavailable" >&2; exit 1; }
done
for required in "${GATE_SBATCH}" "${SUMMARY_SBATCH}" "${MANIFEST}" \
                "${XNAVDP_OFFICIAL_ROOT}" "${XNAVDP_CKPT}"; do
  test -r "${required}" || {
    echo "ABORT: missing input ${required}" >&2; exit 1; }
done
[[ -z "$(git -C "${ROOT}" status --porcelain --untracked-files=all)" ]] || {
  echo "ABORT: HPC evaluation requires a clean committed worktree" >&2
  git -C "${ROOT}" status --short >&2
  exit 1
}
[[ "$(git -c safe.directory="${XNAVDP_OFFICIAL_ROOT}" \
    -C "${XNAVDP_OFFICIAL_ROOT}" rev-parse HEAD)" == \
    "${EXPECTED_XNAVDP_COMMIT}" ]] || {
  echo "ABORT: official X-NavDP commit mismatch" >&2; exit 1; }
[[ -z "$(git -c safe.directory="${XNAVDP_OFFICIAL_ROOT}" \
    -C "${XNAVDP_OFFICIAL_ROOT}" status --porcelain --untracked-files=no)" ]] || {
  echo "ABORT: official X-NavDP checkout has tracked changes" >&2; exit 1; }
[[ "$(sha256sum "${XNAVDP_CKPT}" | awk '{print $1}')" == \
    "${EXPECTED_XNAVDP_CKPT_SHA}" ]] || {
  echo "ABORT: official X-NavDP checkpoint mismatch" >&2; exit 1; }

expected_commit=$(git -C "${ROOT}" rev-parse HEAD)
expected_manifest_sha=$(sha256sum "${MANIFEST}" | awk '{print $1}')
smoke_tag=${RUN_TAG}_smoke
full_tag=${RUN_TAG}_full
common_exports="ALL,ROOT=${ROOT},EXPECTED_COMMIT=${expected_commit},EXPECTED_MANIFEST_SHA=${expected_manifest_sha},XNAVDP_ASSET_ROOT=${XNAVDP_ASSET_ROOT}"

sbatch --test-only --array=0 \
  --export="${common_exports},RUN_TAG=${smoke_tag},EPISODE_LIMIT=1" \
  "${GATE_SBATCH}" >/dev/null
smoke_result=$(sbatch --parsable --array=0 \
  --export="${common_exports},RUN_TAG=${smoke_tag},EPISODE_LIMIT=1" \
  "${GATE_SBATCH}")
smoke_job=${smoke_result%%;*}
[[ "${smoke_job}" =~ ^[0-9]+$ ]] || {
  echo "ABORT: unexpected smoke submission: ${smoke_result}" >&2; exit 1; }

sbatch --test-only --array=0-19%2 \
  --dependency="afterok:${smoke_job}" --kill-on-invalid-dep=yes \
  --export="${common_exports},RUN_TAG=${full_tag},EPISODE_LIMIT=0" \
  "${GATE_SBATCH}" >/dev/null
full_result=$(sbatch --parsable --array=0-19%2 \
  --dependency="afterok:${smoke_job}" --kill-on-invalid-dep=yes \
  --export="${common_exports},RUN_TAG=${full_tag},EPISODE_LIMIT=0" \
  "${GATE_SBATCH}")
full_job=${full_result%%;*}
[[ "${full_job}" =~ ^[0-9]+$ ]] || {
  echo "ABORT: unexpected full submission: ${full_result}" >&2; exit 1; }

summary_result=$(sbatch --parsable \
  --dependency="afterok:${full_job}" --kill-on-invalid-dep=yes \
  --export="${common_exports},RUN_TAG=${full_tag}" \
  "${SUMMARY_SBATCH}")
summary_job=${summary_result%%;*}
[[ "${summary_job}" =~ ^[0-9]+$ ]] || {
  echo "ABORT: unexpected summary submission: ${summary_result}" >&2; exit 1; }

printf 'SMOKE_TAG=%s\nSMOKE_JOB=%s\nFULL_TAG=%s\nFULL_JOB=%s\nSUMMARY_JOB=%s\n' \
  "${smoke_tag}" "${smoke_job}" "${full_tag}" "${full_job}" "${summary_job}"
