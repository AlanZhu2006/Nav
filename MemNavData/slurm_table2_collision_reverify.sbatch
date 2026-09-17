#!/usr/bin/env bash
#SBATCH --job-name=table2_collision_reverify
#SBATCH --output=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/%x_%j.out
#SBATCH --error=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/%x_%j.err
#SBATCH --partition=cpu_short
#SBATCH --account=torch_pr_769_tandon_advanced
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
: "${REPAIRED_BUNDLE:?}" "${TABLE2_PLAN:?}" "${TABLE2_RUN:?}" "${TABLE2_WORK_ROOT:?}"
: "${TABLE2_REVERIFY_BUNDLE:?}" "${TABLE2_STAGE:?}" "${TABLE2_REVERIFY_INDEX:?}"
: "${TABLE2_FAILED_ARCHIVE_SHA:?}" "${TABLE2_PRESERVE_NAME:?}" "${TABLE2_REVERIFY_SHA:?}"
: "${EXPECTED_RUNTIME_SHA:?}" "${EXPECTED_PLAN_SHA:?}"
[[ "$(id -un)" == yz11502 ]]
[[ "${TABLE2_WORK_ROOT}" =~ ^/tmp/memnav_table2_[A-Za-z0-9_-]+$ ]]
[[ "$(sha256sum "${TABLE2_PLAN}" | awk '{print $1}')" == "${EXPECTED_PLAN_SHA}" ]]
[[ "$(sha256sum "${REPAIRED_BUNDLE}/SOURCE_BUNDLE.sha256" | awk '{print $1}')" == "${EXPECTED_RUNTIME_SHA}" ]]
[[ "$(sha256sum "${TABLE2_REVERIFY_BUNDLE}/REVERIFICATION.sha256" | awk '{print $1}')" == "${TABLE2_REVERIFY_SHA}" ]]
(cd "${TABLE2_REVERIFY_BUNDLE}" && sha256sum -c --quiet REVERIFICATION.sha256)
(cd "${REPAIRED_BUNDLE}" && sha256sum -c --quiet SOURCE_BUNDLE.sha256)
TABLE2_VERIFY_TMP=$(mktemp -d "${SLURM_TMPDIR:-/tmp}/table2_reverify_${SLURM_JOB_ID}_XXXXXX")
exec singularity exec -B /scratch/lg154 -B /scratch/yz11502 \
  -B "${TABLE2_VERIFY_TMP}:${TABLE2_WORK_ROOT}" \
  /share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif bash -c '
    set -euo pipefail
    source "${REPAIRED_BUNDLE}/MemNavData/repaired_hm3d_hpc_env.sh"
    export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
    cd "${REPAIRED_BUNDLE}"
    exec "${REPAIRED_HAB_PY}" "${TABLE2_REVERIFY_BUNDLE}/reverify_table2_collision_task.py" \
      --plan "${TABLE2_PLAN}" --run "${TABLE2_RUN}" --stage "${TABLE2_STAGE}" \
      --index "${TABLE2_REVERIFY_INDEX}" --expected-archive-sha "${TABLE2_FAILED_ARCHIVE_SHA}" \
      --preserve-name "${TABLE2_PRESERVE_NAME}"
  '
