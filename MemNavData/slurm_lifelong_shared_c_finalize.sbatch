#!/usr/bin/env bash
#SBATCH --job-name=shared-c-seal
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
: "${SOURCE_ROOT:?}" "${SOURCE_RECEIPT:?}" "${EXPECTED_SOURCE_RECEIPT_SHA:?}"
: "${ORIGINAL_RUN_ROOT:?}" "${RUN_ROOT:?}" "${CONTROLLER:?}"
MEMNAV_PY=/scratch/lg154/conda-envs/memnav/bin/python
[[ "$(sha256sum "${SOURCE_RECEIPT}" | awk '{print $1}')" == \
   "${EXPECTED_SOURCE_RECEIPT_SHA}" ]]
(cd "${SOURCE_ROOT}" && sha256sum -c --quiet "${SOURCE_RECEIPT}")
source_population=${ORIGINAL_RUN_ROOT}/factual_b_support/population.json
"${MEMNAV_PY}" -u "${SOURCE_ROOT}/MemNavData/finalize_lifelong_shared_c_population.py" \
  --source-population "${source_population}" \
  --collection-root "${RUN_ROOT}/collection" --controller "${CONTROLLER}" \
  --run-root "${RUN_ROOT}" --out "${RUN_ROOT}/shared_c_population"
chmod -R a-w "${RUN_ROOT}/shared_c_population"
