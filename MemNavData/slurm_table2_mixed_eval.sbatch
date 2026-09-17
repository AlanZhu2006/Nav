#!/usr/bin/env bash
#SBATCH --job-name=table2_mixed_pilot
#SBATCH --output=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/%x_%A_%a.out
#SBATCH --error=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/%x_%A_%a.err
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=96G
#SBATCH --partition=a100_tandon
#SBATCH --gres=gpu:1
#SBATCH --account=torch_pr_769_tandon_advanced
#SBATCH --qos=gpu48
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
: "${REPAIRED_BUNDLE:?}" "${TABLE2_WORK_ROOT:?}" "${TABLE2_STAGE:?}"
[[ "${TABLE2_WORK_ROOT}" =~ ^/tmp/memnav_table2_[A-Za-z0-9_-]+$ ]]
TABLE2_HOST_TMP=$(mktemp -d "${SLURM_TMPDIR:-/tmp}/table2_${SLURM_JOB_ID}_XXXXXX")
exec singularity exec --nv -B /scratch/lg154 -B /scratch/yz11502 \
  -B "${TABLE2_HOST_TMP}:${TABLE2_WORK_ROOT}" \
  /share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif \
  bash "${REPAIRED_BUNDLE}/MemNavData/run_table2_mixed_hpc.sh"
