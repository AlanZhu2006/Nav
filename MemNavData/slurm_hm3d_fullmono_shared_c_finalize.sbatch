#!/usr/bin/env bash
#SBATCH --job-name=h3Cseal
#SBATCH --output=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/%x_%j.out
#SBATCH --error=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/%x_%j.err
#SBATCH --time=00:15:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --partition=cpu_short
#SBATCH --account=torch_pr_769_tandon_advanced
#SBATCH --export=ALL
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
/scratch/lg154/conda-envs/memnav/bin/python -u \
  "${TASK_ROOT}/MemNavData/finalize_lifelong_shared_c_population.py" \
  --source-population "${RUN_ROOT}/population/population.json" \
  --collection-root "${RUN_ROOT}/shared_c_collection" --controller navdp \
  --run-root "${RUN_ROOT}" --out "${RUN_ROOT}/shared_c_population"
chmod -R a-w "${RUN_ROOT}/shared_c_population"
