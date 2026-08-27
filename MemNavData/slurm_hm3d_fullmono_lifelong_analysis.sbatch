#!/usr/bin/env bash
#SBATCH --job-name=h3lifeAnalysis
#SBATCH --output=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/%x_%j.out
#SBATCH --error=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/%x_%j.err
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --partition=cpu_short
#SBATCH --account=torch_pr_769_tandon_advanced

set -euo pipefail
: "${TASK_ROOT:?}" "${BASE_SOURCE_ROOT:?}" "${RUN_ROOT:?}" "${MODE:?}"
[[ "$(sha256sum "${TASK_RECEIPT}" | awk '{print $1}')" == \
   "${EXPECTED_TASK_RECEIPT_SHA}" ]]
(cd "${TASK_ROOT}" && sha256sum -c --quiet "${TASK_RECEIPT}")
PY=/scratch/lg154/conda-envs/memnav/bin/python
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=${TASK_ROOT}:${TASK_ROOT}/MemNavData:${BASE_SOURCE_ROOT}:${BASE_SOURCE_ROOT}/MemNavData
POPULATION=${RUN_ROOT}/population/population.json
if [[ "${MODE}" == aggregate ]]; then
  "${PY}" -u "${TASK_ROOT}/MemNavData/aggregate_hm3d_fullmono_lifelong.py" \
    --population "${POPULATION}" --evaluation-root "${RUN_ROOT}/evaluation" \
    --out "${RUN_ROOT}/aggregate"
  chmod -R a-w "${RUN_ROOT}/aggregate"
elif [[ "${MODE}" == verify ]]; then
  "${PY}" -u "${TASK_ROOT}/MemNavData/independent_verify_hm3d_fullmono_lifelong.py" \
    --population "${POPULATION}" --aggregate "${RUN_ROOT}/aggregate" \
    --evaluation-root "${RUN_ROOT}/evaluation" \
    --out "${RUN_ROOT}/independent_verification.json"
  sha256sum "${RUN_ROOT}/independent_verification.json" > \
    "${RUN_ROOT}/independent_verification.json.sha256"
else
  echo "unknown analysis mode ${MODE}" >&2
  exit 2
fi
