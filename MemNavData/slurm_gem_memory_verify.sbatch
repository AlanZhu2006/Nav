#!/usr/bin/env bash
#SBATCH --job-name=gem_memory_verify
#SBATCH --output=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/%x_%A_%a.out
#SBATCH --error=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/%x_%A_%a.err
#SBATCH --time=00:20:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --partition=cpu_short
#SBATCH --account=torch_pr_769_tandon_advanced
set -euo pipefail
: "${REPAIRED_BUNDLE:?}" "${GEM_VERIFY_SCRIPT:?}" "${GEM_VERIFY_SHA:?}" "${GEM_NAV_RUN:?}"
[[ "$(sha256sum "${GEM_VERIFY_SCRIPT}" | awk '{print $1}')" == "${GEM_VERIFY_SHA}" ]]
source "${REPAIRED_BUNDLE}/MemNavData/repaired_hm3d_hpc_env.sh"
export GEM_VERIFY_SOURCE_ROOT="${REPAIRED_BUNDLE}" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
exec singularity exec -B /scratch/lg154 -B /scratch/yz11502 \
  /share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif \
  "${REPAIRED_HAB_PY}" "${GEM_VERIFY_SCRIPT}" task "${GEM_NAV_RUN}/tasks/${SLURM_ARRAY_TASK_ID}"
