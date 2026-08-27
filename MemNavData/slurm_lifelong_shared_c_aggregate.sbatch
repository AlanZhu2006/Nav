#!/usr/bin/env bash
#SBATCH --job-name=shared-c-agg
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --partition=cpu_short
#SBATCH --account=torch_pr_769_tandon_advanced
#SBATCH --output=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm-%x-%j.out
#SBATCH --error=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm-%x-%j.err
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
: "${SOURCE_ROOT:?}" "${RUN_ROOT:?}"
MEMNAV_PY=/scratch/lg154/conda-envs/memnav/bin/python
"${MEMNAV_PY}" -u "${SOURCE_ROOT}/MemNavData/aggregate_lifelong_shared_c_b2.py" \
  --population "${RUN_ROOT}/shared_c_population/population.json" \
  --evaluation-root "${RUN_ROOT}/evaluation" \
  --out "${RUN_ROOT}/aggregate"
chmod -R a-w "${RUN_ROOT}/aggregate"
