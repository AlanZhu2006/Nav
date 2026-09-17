#!/usr/bin/env bash
#SBATCH --job-name=table2_all97_summary
#SBATCH --output=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/%x_%j.out
#SBATCH --error=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/%x_%j.err
#SBATCH --time=00:15:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --partition=cpu_short
#SBATCH --account=torch_pr_769_tandon_advanced
# Aggregate the original 12 and the remaining 85 with the same frozen reader.
# This launcher does not modify the inference bundle or the first12 report.
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
: "${REPAIRED_BUNDLE:?}" "${TABLE2_CONTINUOUS_RUN:?}"
: "${EXPECTED_RUNTIME_SHA:?}" "${EXPECTED_POPULATION_SHA:?}"
export PYTHONPATH=${REPAIRED_BUNDLE}
cd "${REPAIRED_BUNDLE}"
[[ "$(sha256sum SOURCE_BUNDLE.sha256 | awk '{print $1}')" == "${EXPECTED_RUNTIME_SHA}" ]]
[[ "$(sha256sum continuous_population/population.json | awk '{print $1}')" == "${EXPECTED_POPULATION_SHA}" ]]
/scratch/lg154/conda-envs/memnav/bin/python -m MemNavData.summarize_table2_continuous \
  --population "${REPAIRED_BUNDLE}/continuous_population/population.json" \
  --run "${TABLE2_CONTINUOUS_RUN}" --count 97 \
  --out "${TABLE2_CONTINUOUS_RUN}/all97_summary_${SLURM_JOB_ID}.json"
