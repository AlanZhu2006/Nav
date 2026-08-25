#!/usr/bin/env bash
#SBATCH --job-name=lifeNNRctlSum
#SBATCH --output=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/%x_%j.out
#SBATCH --error=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/%x_%j.err
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=cpu_short
#SBATCH --account=torch_pr_769_tandon_advanced
#SBATCH --export=ALL

# Three-arm aggregate for one controller-portability lifelong run: all
# three scopes (all_prior, initial_leg_only, forced_reject_native) were run
# fresh under the same RUN_ROOT/evaluation, unlike the navdp addendum which
# joined a prior verified run.
set -euo pipefail
umask 0022
export PYTHONDONTWRITEBYTECODE=1
: "${SOURCE_ROOT:?set immutable task source root}"
: "${SOURCE_RECEIPT:?set source receipt}"
: "${EXPECTED_SOURCE_RECEIPT_SHA:?set source receipt SHA}"
: "${ORIGINAL_RUN_ROOT:?set the navdp-line run root (read-only population source)}"
: "${RUN_ROOT:?set controller-arm run root}"
PY=/scratch/lg154/conda-envs/memnav/bin/python
OUT=${RUN_ROOT}/aggregate
[[ "$(sha256sum "${SOURCE_RECEIPT}" | awk '{print $1}')" == \
   "${EXPECTED_SOURCE_RECEIPT_SHA}" ]]
(cd "${SOURCE_ROOT}" && sha256sum -c --quiet "${SOURCE_RECEIPT}")
(cd "${ORIGINAL_RUN_ROOT}/factual_b_support" && \
  sha256sum -c --quiet population.json.sha256)
export PYTHONPATH=${SOURCE_ROOT}:${SOURCE_ROOT}/MemNavData
export PYTHONPYCACHEPREFIX=${SLURM_TMPDIR:-/tmp}/life_ctlsum_${SLURM_JOB_ID}
"${PY}" -u "${SOURCE_ROOT}/MemNavData/aggregate_lifelong_nnr_expansion.py" \
  --population "${ORIGINAL_RUN_ROOT}/factual_b_support/population.json" \
  --evaluation-root "${RUN_ROOT}/evaluation" \
  --scopes all_prior,initial_leg_only,forced_reject_native \
  --out "${OUT}"
chmod -R a-w "${OUT}"
echo "COMPLETE aggregate=${OUT}"
