#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/scratch/yz11502/Research/Nav-axis-uturn}
EXPECTED_COMMIT=${EXPECTED_COMMIT:-$(git -C "${ROOT}" rev-parse HEAD)}
RUN_TAG=${RUN_TAG:-bearingA40_$(date +%Y%m%d_%H%M%S)_${EXPECTED_COMMIT:0:8}}
SBATCH=${ROOT}/MemNavData/slurm_novel_a_bearing_gate.sbatch

[[ -z "$(git -C "${ROOT}" status --porcelain --untracked-files=all)" ]] || {
  echo "ABORT: HPC worktree is not clean" >&2; exit 1; }
smoke_output=$(sbatch --parsable --array=0 \
  --export=ALL,RUN_TAG="${RUN_TAG}",EXPECTED_COMMIT="${EXPECTED_COMMIT}" \
  "${SBATCH}")
smoke_job=${smoke_output%%;*}
full_output=$(sbatch --parsable --array=1-19%2 \
  --dependency=afterok:"${smoke_job}" \
  --export=ALL,RUN_TAG="${RUN_TAG}",EXPECTED_COMMIT="${EXPECTED_COMMIT}" \
  "${SBATCH}")
full_job=${full_output%%;*}
printf 'RUN_TAG=%s\nSMOKE_JOB=%s\nFULL_JOB=%s\n' \
  "${RUN_TAG}" "${smoke_job}" "${full_job}"
