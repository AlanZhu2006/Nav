#!/usr/bin/env bash
#SBATCH --job-name=cec_coverage_only
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
: "${REPAIRED_BUNDLE:?}"
exec singularity exec --nv -B /scratch/lg154 -B /scratch/yz11502 \
  /share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif \
  bash "${REPAIRED_BUNDLE}/MemNavData/run_coverage_ablation_hpc.sh"
