#!/usr/bin/env bash
#SBATCH --job-name=table2_cont_summary
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
export PYTHONDONTWRITEBYTECODE=1
: "${REPAIRED_BUNDLE:?}" "${TABLE2_CONTINUOUS_RUN:?}"
export PYTHONPATH=${REPAIRED_BUNDLE}
cd "${REPAIRED_BUNDLE}"
/scratch/lg154/conda-envs/memnav/bin/python -m MemNavData.summarize_table2_continuous \
  --population "${REPAIRED_BUNDLE}/continuous_population/population.json" \
  --run "${TABLE2_CONTINUOUS_RUN}" --count 12 \
  --out "${TABLE2_CONTINUOUS_RUN}/first12_summary_${SLURM_JOB_ID}.json"
