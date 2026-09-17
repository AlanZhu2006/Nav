#!/usr/bin/env bash
#SBATCH --job-name=cec_coverage_summary
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
: "${REPAIRED_BUNDLE:?}" "${EXPECTED_RUNTIME_SHA:?}" "${COVERAGE_EVAL_ROOT:?}" "${COVERAGE_PLAN:?}" "${EXPECTED_PLAN_SHA:?}"
[[ "$(id -un)" == yz11502 ]]
[[ "$(sha256sum "${REPAIRED_BUNDLE}/SOURCE_BUNDLE.sha256" | awk '{print $1}')" == "${EXPECTED_RUNTIME_SHA}" ]]
(cd "${REPAIRED_BUNDLE}" && sha256sum -c --quiet SOURCE_BUNDLE.sha256)
export PYTHONPATH=${REPAIRED_BUNDLE}:${REPAIRED_BUNDLE}/MemNavData
exec /scratch/lg154/conda-envs/memnav/bin/python "${REPAIRED_BUNDLE}/MemNavData/summarize_coverage_ablation.py" \
  --plan "${COVERAGE_PLAN}" --evaluation-root "${COVERAGE_EVAL_ROOT}" \
  --output "${COVERAGE_EVAL_ROOT}/paired_summary.json"
