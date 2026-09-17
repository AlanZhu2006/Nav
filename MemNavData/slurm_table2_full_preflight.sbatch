#!/usr/bin/env bash
#SBATCH --job-name=table2_full_preflight
#SBATCH --output=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/%x_%j.out
#SBATCH --error=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/%x_%j.err
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --partition=cpu_short
#SBATCH --account=torch_pr_769_tandon_advanced
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
: "${REPAIRED_BUNDLE:?}" "${TABLE2_PLAN:?}" "${TABLE2_RUN:?}" "${EXPECTED_RUNTIME_SHA:?}"
singularity exec -B /scratch/lg154 -B /scratch/yz11502 \
  /share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif \
  bash "${REPAIRED_BUNDLE}/MemNavData/run_table2_full_preflight.sh"
exec bash "${REPAIRED_BUNDLE}/MemNavData/submit_table2_full_hpc.sh" initial
