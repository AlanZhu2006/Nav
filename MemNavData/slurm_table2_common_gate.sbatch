#!/usr/bin/env bash
#SBATCH --job-name=table2_common_gate
#SBATCH --output=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/%x_%j.out
#SBATCH --error=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/%x_%j.err
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --partition=a100_tandon
#SBATCH --gres=gpu:1
#SBATCH --account=torch_pr_769_tandon_advanced
#SBATCH --qos=gpu48
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
: "${REPAIRED_BUNDLE:?}" "${TABLE2_COMMON_WORK:?}"
[[ "${TABLE2_COMMON_WORK}" =~ ^/tmp/memnav_table2_common_[A-Za-z0-9_-]+$ ]]
TABLE2_COMMON_HOST_TMP=$(mktemp -d "${SLURM_TMPDIR:-/tmp}/table2_common_${SLURM_JOB_ID}_XXXXXX")
exec singularity exec --nv -B /scratch/lg154 -B /scratch/yz11502 \
  -B "${TABLE2_COMMON_HOST_TMP}:${TABLE2_COMMON_WORK}" \
  /share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif \
  bash "${REPAIRED_BUNDLE}/MemNavData/run_table2_common_hpc.sh"
