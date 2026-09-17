#!/usr/bin/env bash
#SBATCH --job-name=nav_cold_archive
#SBATCH --output=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/%x_%A_%a.out
#SBATCH --error=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/%x_%A_%a.err
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --partition=cpu_short
#SBATCH --account=torch_pr_769_tandon_advanced
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
: "${CLEAN_CODE:?}" "${CLEAN_PLAN:?}" "${CLEAN_PLAN_SHA:?}" "${CLEAN_RECEIPTS:?}" "${SLURM_ARRAY_TASK_ID:?}"
[[ "$(id -un)" == yz11502 ]]
exec /scratch/lg154/conda-envs/memnav/bin/python -u "$CLEAN_CODE" execute \
  --plan "$CLEAN_PLAN" --plan-sha "$CLEAN_PLAN_SHA" --index "$SLURM_ARRAY_TASK_ID" --receipts "$CLEAN_RECEIPTS"
