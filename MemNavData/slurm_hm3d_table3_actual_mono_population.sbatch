#!/usr/bin/env bash
#SBATCH --job-name=h3T3Population
#SBATCH --output=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/%x_%j.out
#SBATCH --error=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/%x_%j.err
#SBATCH --time=00:20:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --partition=cpu_short
#SBATCH --account=torch_pr_769_tandon_advanced

set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
: "${MODE:?finalize or verify}" "${WRAPPER_ROOT:?}" "${WRAPPER_RECEIPT:?}"
: "${EXPECTED_WRAPPER_RECEIPT_SHA:?}" "${RUN_ROOT:?}"
: "${TABLE3_PLAN:?}" "${EXPECTED_TABLE3_PLAN_SHA:?}"
[[ "$(sha256sum "${WRAPPER_RECEIPT}" | awk '{print $1}')" == \
   "${EXPECTED_WRAPPER_RECEIPT_SHA}" ]]
(cd "${WRAPPER_ROOT}" && sha256sum -c --quiet "${WRAPPER_RECEIPT}")
[[ "$(sha256sum "${TABLE3_PLAN}" | awk '{print $1}')" == \
   "${EXPECTED_TABLE3_PLAN_SHA}" ]]
export PYTHONPATH=${WRAPPER_ROOT}:${WRAPPER_ROOT}/MemNavData
py=/scratch/lg154/conda-envs/memnav/bin/python
population=${RUN_ROOT}/query_population
if [[ "${MODE}" == finalize ]]; then
  [[ "$(find "${RUN_ROOT}/construction_fragments" -mindepth 2 -maxdepth 2 -name completion.json -type f | wc -l)" -eq 125 ]]
  "${py}" -u "${WRAPPER_ROOT}/MemNavData/finalize_hm3d_table3_actual_mono_population.py" \
    --run-root "${RUN_ROOT}" --candidate-plan "${TABLE3_PLAN}" \
    --protocol "${TABLE3_PROTOCOL}" --out "${population}"
elif [[ "${MODE}" == verify ]]; then
  "${py}" -u "${WRAPPER_ROOT}/MemNavData/independent_verify_hm3d_table3_actual_mono_population.py" \
    --population-root "${population}" \
    --out "${population}/independent_verification.json"
  (cd "${population}" && sha256sum -c --quiet independent_verification.json.sha256)
  chmod -R a-w "${population}"
else
  echo "invalid MODE=${MODE}" >&2
  exit 2
fi
