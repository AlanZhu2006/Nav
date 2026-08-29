#!/usr/bin/env bash
#SBATCH --job-name=m3T1monoSrc
#SBATCH --output=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/%x_%A_%a.out
#SBATCH --error=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/%x_%A_%a.err
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH --partition=h100_tandon,a100_tandon
#SBATCH --qos=gpu48
#SBATCH --gres=gpu:1
#SBATCH --account=torch_pr_769_tandon_advanced

set -euo pipefail
umask 0022
: "${SLURM_ARRAY_TASK_ID:?submit as an array}"
: "${TASK_ROOT:?}" "${TASK_RECEIPT:?}" "${EXPECTED_TASK_RECEIPT_SHA:?}"
: "${SERVER_SOURCE_ROOT:?}" "${SERVER_SOURCE_RECEIPT:?}"
: "${EXPECTED_SERVER_SOURCE_RECEIPT_SHA:?}"
: "${BASE_SOURCE_ROOT:?}" "${BASE_RECEIPT:?}"
: "${EXPECTED_BASE_RECEIPT_SHA:?}"
: "${RUN_ROOT:?}" "${PARENT_MANIFEST:?}" "${PROTOCOL:?}"

[[ "${SLURM_ARRAY_TASK_ID}" =~ ^([0-9]|1[0-5])$ ]]
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPYCACHEPREFIX=${SLURM_TMPDIR:-/tmp}/m3t1src_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}
mkdir -p "${PYTHONPYCACHEPREFIX}" "${RUN_ROOT}" \
  /scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs
echo "host=$(hostname) job=${SLURM_JOB_ID} task=${SLURM_ARRAY_TASK_ID} smoke=${MP3D_SOURCE_SMOKE:-0}"
nvidia-smi --query-gpu=name,uuid,memory.total --format=csv,noheader

exec singularity exec --nv -B /scratch/lg154 -B /scratch/yz11502 \
  /share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif env \
    MODE=mp3d_collect TASK_ROOT="${TASK_ROOT}" \
    SERVER_SOURCE_ROOT="${SERVER_SOURCE_ROOT}" \
    SERVER_SOURCE_RECEIPT="${SERVER_SOURCE_RECEIPT}" \
    EXPECTED_SERVER_SOURCE_RECEIPT_SHA="${EXPECTED_SERVER_SOURCE_RECEIPT_SHA}" \
    BASE_SOURCE_ROOT="${BASE_SOURCE_ROOT}" \
    RUN_ROOT="${RUN_ROOT}" PARENT_MANIFEST="${PARENT_MANIFEST}" \
    PROTOCOL="${PROTOCOL}" SCENE_INDEX="${SLURM_ARRAY_TASK_ID}" \
    HAB_PY=/scratch/lg154/conda-envs/habitat/bin/python \
    MEMNAV_PY=/scratch/lg154/conda-envs/memnav/bin/python \
    TASK_RECEIPT="${TASK_RECEIPT}" \
    EXPECTED_TASK_RECEIPT_SHA="${EXPECTED_TASK_RECEIPT_SHA}" \
    BASE_RECEIPT="${BASE_RECEIPT}" \
    EXPECTED_BASE_RECEIPT_SHA="${EXPECTED_BASE_RECEIPT_SHA}" \
    MP3D_SOURCE_SMOKE="${MP3D_SOURCE_SMOKE:-0}" \
    MAX_STEPS="${MAX_STEPS:-80}" \
    PYTHONDONTWRITEBYTECODE=1 PYTHONPYCACHEPREFIX="${PYTHONPYCACHEPREFIX}" \
    bash "${TASK_ROOT}/MemNavData/run_hm3d_fullmono_server_scene.sh"
