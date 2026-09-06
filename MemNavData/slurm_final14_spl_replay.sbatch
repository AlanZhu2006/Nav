#!/usr/bin/env bash
#SBATCH --job-name=f14ExactSPL
#SBATCH --partition=h100_tandon,a100_tandon
#SBATCH --account=torch_pr_769_tandon_advanced
#SBATCH --qos=gpu48
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=72G
#SBATCH --time=01:00:00
#SBATCH --signal=B:USR1@90
#SBATCH --output=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/f14ExactSPL_%A_%a.out
#SBATCH --error=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/f14ExactSPL_%A_%a.err
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
: "${REPLAY_ROOT:?}" "${RUN_ROOT:?}" "${SLURM_ARRAY_TASK_ID:?}"
source "${REPLAY_ROOT}/MemNavData/final14_spl_replay_env.sh"
[[ "$(stat -c %s "${SOURCE_OVERLAY}")" == 128854888448 ]]
export HISTORY_INDEX=${SLURM_ARRAY_TASK_ID}
exec singularity exec --nv --overlay "${SOURCE_OVERLAY}:ro" -B /scratch/lg154 -B /scratch/yz11502 \
  "${BASE_SIF}" bash "${REPLAY_ROOT}/MemNavData/run_final14_spl_replay_history.sh"
