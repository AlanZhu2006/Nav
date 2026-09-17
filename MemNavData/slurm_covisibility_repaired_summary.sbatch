#!/usr/bin/env bash
#SBATCH --job-name=cec_covis_summary
#SBATCH --output=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/%x_%j.out
#SBATCH --error=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/%x_%j.err
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --partition=cpu_short
#SBATCH --account=torch_pr_769_tandon_advanced
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
: "${COVIS_ADDON:?}" "${COVIS_EVAL_ROOT:?}" "${COVIS_POPULATION:?}" "${EXPECTED_ADDON_SHA:?}"
[[ "$(id -un)" == yz11502 ]]
[[ "$(sha256sum "${COVIS_ADDON}/SOURCE_BUNDLE.sha256" | awk '{print $1}')" == "${EXPECTED_ADDON_SHA}" ]]
(cd "${COVIS_ADDON}" && sha256sum -c --quiet SOURCE_BUNDLE.sha256)
exec /scratch/lg154/conda-envs/memnav/bin/python "${COVIS_ADDON}/summarize_covisibility_repaired.py" \
  --population "${COVIS_POPULATION}" --evaluation-root "${COVIS_EVAL_ROOT}" \
  --output "${COVIS_EVAL_ROOT}/paired_summary.json"
