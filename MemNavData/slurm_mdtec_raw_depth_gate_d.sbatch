#!/bin/bash
#SBATCH --job-name=mdtecGateD
#SBATCH --output=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/mdtecGateD_%A_%a.out
#SBATCH --error=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/mdtecGateD_%A_%a.err
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH --partition=h100_tandon,a100_tandon,h200_public
#SBATCH --gres=gpu:1
#SBATCH --account=torch_pr_769_tandon_advanced
#SBATCH --export=ALL

set -euo pipefail
: "${SLURM_ARRAY_TASK_ID:?array required}"
: "${SOURCE_ROOT:?set SOURCE_ROOT}"
: "${RUN_ROOT:?set RUN_ROOT}"
: "${SOURCE_RECEIPT:?set SOURCE_RECEIPT}"
: "${EXPECTED_SOURCE_RECEIPT_SHA:?set receipt SHA}"
BASE_SIF=/share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif
mkdir -p "${RUN_ROOT}" /scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs
echo "host=$(hostname) job=${SLURM_JOB_ID} task=${SLURM_ARRAY_TASK_ID}"
nvidia-smi --query-gpu=name,uuid,memory.total --format=csv,noheader
exec singularity exec --nv -B /scratch/lg154 -B /scratch/yz11502 "${BASE_SIF}" \
  env ROOT="${SOURCE_ROOT}" RUN_ROOT="${RUN_ROOT}" \
    SOURCE_RECEIPT="${SOURCE_RECEIPT}" \
    EXPECTED_SOURCE_RECEIPT_SHA="${EXPECTED_SOURCE_RECEIPT_SHA}" \
    SCENE_INDEX="${SLURM_ARRAY_TASK_ID}" \
    HAB_PY=/scratch/lg154/conda-envs/habitat/bin/python \
    MEMNAV_PY=/scratch/lg154/conda-envs/memnav/bin/python \
    bash "${SOURCE_ROOT}/MemNavData/run_mdtec_raw_depth_gate_d_scene.sh"
