#!/usr/bin/env bash
#SBATCH --job-name=h3monoFixA
#SBATCH --output=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/%x_%A_%a.out
#SBATCH --error=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/%x_%A_%a.err
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=72G
#SBATCH --partition=h100_tandon,a100_tandon
#SBATCH --gres=gpu:1
#SBATCH --account=torch_pr_769_tandon_advanced

set -euo pipefail
: "${SLURM_ARRAY_TASK_ID:?scene array index required}"
: "${TASK_ROOT:?}" "${BASE_SOURCE_ROOT:?}" "${RUN_ROOT:?}"
: "${PARENT_MANIFEST:?}" "${PROTOCOL:?}" "${RUNTIME_ATTEMPT:?}"
BASE_SIF=/share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif
mkdir -p /scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs
exec singularity exec --nv -B /scratch/lg154 -B /scratch/yz11502 \
  "${BASE_SIF}" env MODE=collect TASK_ROOT="${TASK_ROOT}" \
  BASE_SOURCE_ROOT="${BASE_SOURCE_ROOT}" RUN_ROOT="${RUN_ROOT}" \
  PARENT_MANIFEST="${PARENT_MANIFEST}" PROTOCOL="${PROTOCOL}" \
  SCENE_INDEX="${SLURM_ARRAY_TASK_ID}" RESUME_INCOMPLETE=1 \
  RUNTIME_ATTEMPT="${RUNTIME_ATTEMPT}" \
  HAB_PY=/scratch/lg154/conda-envs/habitat/bin/python \
  MEMNAV_PY=/scratch/lg154/conda-envs/memnav/bin/python \
  TASK_RECEIPT="${TASK_RECEIPT}" \
  EXPECTED_TASK_RECEIPT_SHA="${EXPECTED_TASK_RECEIPT_SHA}" \
  BASE_RECEIPT="${BASE_RECEIPT}" \
  EXPECTED_BASE_RECEIPT_SHA="${EXPECTED_BASE_RECEIPT_SHA}" \
  bash "${TASK_ROOT}/MemNavData/run_hm3d_fullmono_server_scene.sh"
