#!/usr/bin/env bash
#SBATCH --job-name=f14Authority
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
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPYCACHEPREFIX=${SLURM_TMPDIR:-/tmp}/f14_authority_pycache_${SLURM_JOB_ID}
: "${SLURM_ARRAY_TASK_ID:?}" "${SOURCE_OVERLAY:?}"
: "${EXPECTED_SOURCE_OVERLAY_BYTES:?}" "${REPAIR_ROOT:?}"
: "${BASE_SOURCE_ROOT:?}" "${RUN_ROOT:?}" "${BENCH_ROOT:?}"
[[ "$(stat -c '%s' "${SOURCE_OVERLAY}")" == \
   "${EXPECTED_SOURCE_OVERLAY_BYTES}" ]]
BASE_SIF=/share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif
exec singularity exec --nv --overlay "${SOURCE_OVERLAY}:ro" \
  -B /scratch/lg154 -B /scratch/yz11502 "${BASE_SIF}" \
  env HISTORY_INDEX="${SLURM_ARRAY_TASK_ID}" \
    HAB_PY=/scratch/lg154/conda-envs/habitat/bin/python \
    MEMNAV_PY=/scratch/lg154/conda-envs/memnav/bin/python \
    bash "${REPAIR_ROOT}/MemNavData/run_final14_authority_ablation_history.sh"
