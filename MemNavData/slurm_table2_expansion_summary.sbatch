#!/usr/bin/env bash
#SBATCH --job-name=table2_expand_summary
#SBATCH --output=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/%x_%j.out
#SBATCH --error=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/%x_%j.err
#SBATCH --time=00:15:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --partition=cpu_short
#SBATCH --account=torch_pr_769_tandon_advanced
set -euo pipefail
: "${REPAIRED_BUNDLE:?}" "${TABLE2_CONTINUOUS_RUN:?}"
: "${EXPECTED_RUNTIME_SHA:?}" "${EXPECTED_POPULATION_SHA:?}"
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=${REPAIRED_BUNDLE}
cd "${REPAIRED_BUNDLE}"
[[ "$(sha256sum SOURCE_BUNDLE.sha256 | awk '{print $1}')" == "${EXPECTED_RUNTIME_SHA}" ]]
[[ "$(sha256sum continuous_population/population.json | awk '{print $1}')" == "${EXPECTED_POPULATION_SHA}" ]]
exec /scratch/lg154/conda-envs/memnav/bin/python -u -m MemNavData.table2_continuous_expansion summary \
  --bundle "${REPAIRED_BUNDLE}" --run "${TABLE2_CONTINUOUS_RUN}" \
  --out "${TABLE2_CONTINUOUS_RUN}/expansion_summary_${SLURM_JOB_ID}.json"
