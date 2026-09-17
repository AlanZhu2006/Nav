#!/usr/bin/env bash
#SBATCH --job-name=cec_repaired_ext
#SBATCH --output=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/%x_%A_%a.out
#SBATCH --error=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/%x_%A_%a.err
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=96G
#SBATCH --partition=h100_tandon,a100_tandon
#SBATCH --gres=gpu:1
#SBATCH --account=torch_pr_769_tandon_advanced
#SBATCH --qos=gpu48
#SBATCH --array=0-1%2
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
: "${REPAIRED_BUNDLE:?}" "${REPAIRED_EXTENSION_ROOT:?}" "${EXPECTED_REPAIR_SHA:?}"
[[ "${SLURM_ARRAY_TASK_ID}" == 0 || "${SLURM_ARRAY_TASK_ID}" == 1 ]]
export REPAIRED_GATE_PHASE=extension
export REPAIRED_GATE_SHARD=${SLURM_ARRAY_TASK_ID}
export REPAIRED_GATE_PROTOCOL=${REPAIRED_BUNDLE}/MemNavData/REPAIRED_HM3D_GATE_EXTENSION_PROTOCOL_20260908.md
export REPAIRED_RUN=${REPAIRED_EXTENSION_ROOT}/shard_${SLURM_ARRAY_TASK_ID}
exec singularity exec --nv -B /scratch/lg154 -B /scratch/yz11502 \
  /share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif \
  bash "${REPAIRED_BUNDLE}/MemNavData/run_repaired_hm3d_gate.sh"
