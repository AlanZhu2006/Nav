#!/usr/bin/env bash
#SBATCH --job-name=f14SPLAudit
#SBATCH --partition=cpu_short
#SBATCH --account=torch_pr_769_tandon_advanced
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/f14SPLAudit_%j.out
#SBATCH --error=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/f14SPLAudit_%j.err
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
: "${REPLAY_ROOT:?}" "${RUN_ROOT:?}" "${MODE:?}"
source "${REPLAY_ROOT}/MemNavData/final14_spl_replay_env.sh"
exec singularity exec --overlay "${SOURCE_OVERLAY}:ro" -B /scratch/lg154 -B /scratch/yz11502 \
  "${BASE_SIF}" bash "${REPLAY_ROOT}/MemNavData/run_final14_spl_replay_cpu.sh"
