#!/usr/bin/env bash
#SBATCH --job-name=table2_expand_A_menu
#SBATCH --output=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/%x_%A_%a.out
#SBATCH --error=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/%x_%A_%a.err
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=24G
#SBATCH --partition=a100_tandon
#SBATCH --gres=gpu:1
#SBATCH --account=torch_pr_769_tandon_advanced
#SBATCH --qos=gpu48
set -euo pipefail
: "${REPAIRED_BUNDLE:?}" "${TABLE2_CONTINUOUS_RUN:?}" "${EXPECTED_BOOTSTRAP_SHA:?}"
: "${SLURM_ARRAY_TASK_ID:?}"
[[ "$(id -un)" == yz11502 ]]
export PYTHONDONTWRITEBYTECODE=1
TABLE2_EXPANSION_TMP=$(mktemp -d "${SLURM_TMPDIR:-/tmp}/table2_expansion_A_${SLURM_JOB_ID}_XXXXXX")
export TMPDIR=${TABLE2_EXPANSION_TMP} LIBFFI_TMPDIR=${TABLE2_EXPANSION_TMP}
cd "${REPAIRED_BUNDLE}"
[[ "$(sha256sum SOURCE_BUNDLE.sha256 | awk '{print $1}')" == "${EXPECTED_BOOTSTRAP_SHA}" ]]
sha256sum -c --quiet SOURCE_BUNDLE.sha256
source MemNavData/repaired_hm3d_hpc_env.sh
exec singularity exec --nv -B /scratch/lg154 -B /scratch/yz11502 \
  -B "${TABLE2_EXPANSION_TMP}:${TABLE2_EXPANSION_TMP}" \
  /share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif \
  "${REPAIRED_HAB_PY}" -u -m MemNavData.table2_continuous_expansion construct \
  --bundle "${REPAIRED_BUNDLE}" --run "${TABLE2_CONTINUOUS_RUN}" \
  --scene-index "${SLURM_ARRAY_TASK_ID}"
