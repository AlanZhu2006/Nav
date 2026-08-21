#!/bin/bash
#SBATCH --job-name=f14MonoFact
#SBATCH --output=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/f14MonoFact_%A_%a.out
#SBATCH --error=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/f14MonoFact_%A_%a.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=72G
#SBATCH --partition=h100_tandon,a100_tandon
#SBATCH --gres=gpu:1
#SBATCH --account=torch_pr_769_tandon_advanced
#SBATCH --export=ALL

set -euo pipefail
: "${SLURM_ARRAY_TASK_ID:?array required}"
: "${SOURCE_ROOT:?set SOURCE_ROOT}"
: "${RUN_ROOT:?set RUN_ROOT}"
: "${BENCH_ROOT:?set BENCH_ROOT}"
: "${SOURCE_RECEIPT:?set SOURCE_RECEIPT}"
: "${EXPECTED_SOURCE_RECEIPT_SHA:?set receipt SHA}"
: "${SOURCE_OVERLAY:?set immutable episode overlay}"
: "${EXPECTED_SOURCE_OVERLAY_BYTES:?set episode overlay size}"
BASE_SIF=/share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif
[[ -r "${SOURCE_OVERLAY}" ]]
[[ "$(stat -c %s "${SOURCE_OVERLAY}")" -eq "${EXPECTED_SOURCE_OVERLAY_BYTES}" ]]
mkdir -p "${RUN_ROOT}" /scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs
echo "host=$(hostname) job=${SLURM_JOB_ID} task=${SLURM_ARRAY_TASK_ID} smoke=${SMOKE:-0}"
nvidia-smi --query-gpu=name,uuid,memory.total --format=csv,noheader
exec singularity exec --nv --overlay "${SOURCE_OVERLAY}:ro" \
  -B /scratch/lg154 -B /scratch/yz11502 "${BASE_SIF}" \
  env ROOT="${SOURCE_ROOT}" RUN_ROOT="${RUN_ROOT}" BENCH_ROOT="${BENCH_ROOT}" \
    SOURCE_RECEIPT="${SOURCE_RECEIPT}" \
    EXPECTED_SOURCE_RECEIPT_SHA="${EXPECTED_SOURCE_RECEIPT_SHA}" \
    HISTORY_INDEX="${SLURM_ARRAY_TASK_ID}" \
    SMOKE="${SMOKE:-0}" MAX_STEPS="${MAX_STEPS:-600}" \
    HAB_PY=/scratch/lg154/conda-envs/habitat/bin/python \
    MEMNAV_PY=/scratch/lg154/conda-envs/memnav/bin/python \
    bash "${SOURCE_ROOT}/MemNavData/run_final14_mono_factorial_history.sh"
