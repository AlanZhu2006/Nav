#!/usr/bin/env bash
#SBATCH --job-name=h3Canalysis
#SBATCH --output=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/%x_%j.out
#SBATCH --error=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/%x_%j.err
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --partition=cpu_short
#SBATCH --account=torch_pr_769_tandon_advanced
#SBATCH --export=ALL
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
case "${MODE:?aggregate|verify}" in
  aggregate)
    /scratch/lg154/conda-envs/memnav/bin/python -u \
      "${TASK_ROOT}/MemNavData/aggregate_lifelong_shared_c_b2.py" \
      --population "${RUN_ROOT}/shared_c_population/population.json" \
      --evaluation-root "${RUN_ROOT}/shared_c_evaluation" \
      --out "${RUN_ROOT}/shared_c_aggregate"
    chmod -R a-w "${RUN_ROOT}/shared_c_aggregate"
    ;;
  verify)
    /scratch/lg154/conda-envs/memnav/bin/python -u \
      "${TASK_ROOT}/MemNavData/independent_verify_lifelong_shared_c_b2.py" \
      --population "${RUN_ROOT}/shared_c_population/population.json" \
      --aggregate "${RUN_ROOT}/shared_c_aggregate" \
      --fullmono-prefix-audit \
      --out "${RUN_ROOT}/shared_c_independent_verification.json"
    chmod a-w "${RUN_ROOT}/shared_c_independent_verification.json"
    ;;
  *) echo "bad MODE" >&2; exit 2;;
esac
