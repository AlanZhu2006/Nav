#!/usr/bin/env bash
#SBATCH --job-name=f14AuthorityAna
#SBATCH --output=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/%x_%j.out
#SBATCH --error=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/%x_%j.err
#SBATCH --time=00:20:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=cpu_short
#SBATCH --account=torch_pr_769_tandon_advanced

set -euo pipefail
: "${REPAIR_ROOT:?}" "${BASE_SOURCE_ROOT:?}" "${RUN_ROOT:?}"
: "${BENCH_ROOT:?}"
PY=/scratch/lg154/conda-envs/memnav/bin/python
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=${REPAIR_ROOT}:${REPAIR_ROOT}/MemNavData:${BASE_SOURCE_ROOT}:${BASE_SOURCE_ROOT}/MemNavData
mkdir -p "${RUN_ROOT}/POSTHOC"
"${PY}" "${REPAIR_ROOT}/MemNavData/summarize_final14_authority_ablation.py" \
  --run-root "${RUN_ROOT}" --bench-root "${BENCH_ROOT}" \
  --expected-manifest-sha256 7468703a9efbb10e801ffdd226911f696a30fa9432ef9ab486d3134f6e40fe6a \
  --out "${RUN_ROOT}/POSTHOC/final14_cec_authority_ablation_summary.json"
"${PY}" "${REPAIR_ROOT}/MemNavData/independent_verify_final14_authority_ablation.py" \
  --run-root "${RUN_ROOT}" --bench-root "${BENCH_ROOT}" \
  --expected-manifest-sha256 7468703a9efbb10e801ffdd226911f696a30fa9432ef9ab486d3134f6e40fe6a \
  --summary "${RUN_ROOT}/POSTHOC/final14_cec_authority_ablation_summary.json" \
  --out "${RUN_ROOT}/POSTHOC/final14_cec_authority_ablation_independent_verification.json"
