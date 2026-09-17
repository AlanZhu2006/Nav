#!/usr/bin/env bash
#SBATCH --job-name=table1_three_summary
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
: "${REPAIRED_BUNDLE:?}" "${EXPECTED_RUNTIME_SHA:?}" "${TABLE1_EVAL_ROOT:?}" "${TABLE1_PLAN:?}" "${EXPECTED_PLAN_SHA:?}"
[[ "$(sha256sum "${REPAIRED_BUNDLE}/SOURCE_BUNDLE.sha256" | awk '{print $1}')" == "${EXPECTED_RUNTIME_SHA}" ]]
[[ "$(sha256sum "${TABLE1_PLAN}" | awk '{print $1}')" == "${EXPECTED_PLAN_SHA}" ]]
(cd "${REPAIRED_BUNDLE}" && sha256sum -c --quiet SOURCE_BUNDLE.sha256)
export PYTHONPATH=${REPAIRED_BUNDLE}:${REPAIRED_BUNDLE}/MemNavData
exec /scratch/lg154/conda-envs/memnav/bin/python "${REPAIRED_BUNDLE}/MemNavData/summarize_table1_repaired.py" \
  --plan "${TABLE1_PLAN}" --root "${TABLE1_EVAL_ROOT}" --output "${TABLE1_EVAL_ROOT}/paired_summary.json"
