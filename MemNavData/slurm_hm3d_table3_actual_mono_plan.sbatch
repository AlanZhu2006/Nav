#!/usr/bin/env bash
#SBATCH --job-name=h3T3Plan
#SBATCH --output=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/%x_%j.out
#SBATCH --error=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/%x_%j.err
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --partition=cpu_short
#SBATCH --account=torch_pr_769_tandon_advanced

set -euo pipefail
umask 0022
export PYTHONDONTWRITEBYTECODE=1
: "${MODE:?}" "${TASK_ROOT:?}" "${TASK_RECEIPT:?}"
: "${EXPECTED_TASK_RECEIPT_SHA:?}" "${RUN_ROOT:?}" "${PROTOCOL:?}"
: "${EXPECTED_PROTOCOL_SHA:?}" "${PARENT_MANIFEST:?}"
: "${EXPECTED_PARENT_MANIFEST_SHA:?}" "${CAPACITY_ROOT:?}"
[[ "${MODE}" == freeze || "${MODE}" == verify ]]
[[ "$(sha256sum "${TASK_RECEIPT}" | awk '{print $1}')" == \
   "${EXPECTED_TASK_RECEIPT_SHA}" ]]
(cd "${TASK_ROOT}" && sha256sum -c --quiet "${TASK_RECEIPT}")
[[ "$(sha256sum "${PROTOCOL}" | awk '{print $1}')" == \
   "${EXPECTED_PROTOCOL_SHA}" ]]
[[ "$(sha256sum "${PARENT_MANIFEST}" | awk '{print $1}')" == \
   "${EXPECTED_PARENT_MANIFEST_SHA}" ]]
py=/scratch/lg154/conda-envs/memnav/bin/python
mkdir -p "${RUN_ROOT}"
if [[ "${MODE}" == freeze ]]; then
  exec env PYTHONPATH="${TASK_ROOT}/MemNavData" "${py}" -u \
    "${TASK_ROOT}/MemNavData/freeze_hm3d_table3_actual_mono_plan.py" \
    --protocol "${PROTOCOL}" --capacity-root "${CAPACITY_ROOT}" \
    --parent-manifest "${PARENT_MANIFEST}" \
    --out "${RUN_ROOT}/candidate_plan.json"
fi
exec env PYTHONPATH="${TASK_ROOT}/MemNavData" "${py}" -u \
  "${TASK_ROOT}/MemNavData/independent_verify_hm3d_table3_actual_mono_plan.py" \
  --protocol "${PROTOCOL}" --plan "${RUN_ROOT}/candidate_plan.json" \
  --out "${RUN_ROOT}/independent_candidate_plan_verification.json"
