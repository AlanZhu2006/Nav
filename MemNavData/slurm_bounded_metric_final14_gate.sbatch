#!/bin/bash
#SBATCH --job-name=cecBoundGate
#SBATCH --output=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/cecBoundGate_%j.out
#SBATCH --error=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/cecBoundGate_%j.err
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=72G
#SBATCH --partition=h100_tandon,h200_public,a100_tandon
#SBATCH --gres=gpu:1
#SBATCH --account=torch_pr_769_tandon_advanced
#SBATCH --qos=gpu48
#SBATCH --export=ALL

set -euo pipefail
: "${TASK_ROOT:?set TASK_ROOT}"
: "${BASE_SOURCE_ROOT:?set BASE_SOURCE_ROOT}"
: "${FALLBACK_SOURCE_ROOT:?set FALLBACK_SOURCE_ROOT}"
: "${RUN_ROOT:?set RUN_ROOT}"
: "${BENCH_ROOT:?set BENCH_ROOT}"
: "${SOURCE_OVERLAY:?set MP3D overlay}"
: "${EXPECTED_SOURCE_OVERLAY_BYTES:?set overlay bytes}"

SIF=/share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif
[[ -r "${SOURCE_OVERLAY}" ]]
[[ "$(stat -c %s "${SOURCE_OVERLAY}")" == \
  "${EXPECTED_SOURCE_OVERLAY_BYTES}" ]]
mkdir -p "${RUN_ROOT}" \
  /scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs
echo "host=$(hostname) job=${SLURM_JOB_ID} history=${HISTORY_INDEX:-0}"
nvidia-smi --query-gpu=name,uuid,memory.total --format=csv,noheader
exec singularity exec --nv --overlay "${SOURCE_OVERLAY}:ro" \
  -B /scratch/lg154 -B /scratch/yz11502 "${SIF}" \
  env TASK_ROOT="${TASK_ROOT}" BASE_SOURCE_ROOT="${BASE_SOURCE_ROOT}" \
    FALLBACK_SOURCE_ROOT="${FALLBACK_SOURCE_ROOT}" \
    RUN_ROOT="${RUN_ROOT}" BENCH_ROOT="${BENCH_ROOT}" \
    HISTORY_INDEX="${HISTORY_INDEX:-0}" MAX_STEPS="${MAX_STEPS:-16}" \
    HAB_PY=/scratch/lg154/conda-envs/habitat/bin/python \
    MEMNAV_PY=/scratch/lg154/conda-envs/memnav/bin/python \
    bash "${TASK_ROOT}/MemNavData/run_bounded_metric_final14_gate_hpc.sh"
