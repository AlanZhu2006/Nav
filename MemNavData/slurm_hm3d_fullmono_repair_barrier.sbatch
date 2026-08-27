#!/usr/bin/env bash
#SBATCH --job-name=h3monoFixSeal
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
: "${TASK_ROOT:?}" "${BASE_SOURCE_ROOT:?}" "${RUN_ROOT:?}"
: "${PARENT_MANIFEST:?}" "${PROTOCOL:?}" "${PRE_REPAIR_INVENTORY:?}"
[[ "$(sha256sum "${TASK_RECEIPT}" | awk '{print $1}')" == \
   "${EXPECTED_TASK_RECEIPT_SHA}" ]]
(cd "${TASK_ROOT}" && sha256sum -c --quiet "${TASK_RECEIPT}")
PY=/scratch/lg154/conda-envs/habitat/bin/python
out=${RUN_ROOT}/repair/goal_a_repair_barrier.json
PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH="${TASK_ROOT}:${TASK_ROOT}/MemNavData:${BASE_SOURCE_ROOT}:${BASE_SOURCE_ROOT}/MemNavData" \
  "${PY}" -u "${TASK_ROOT}/MemNavData/hm3d_fullmono_repair_audit.py" \
    barrier --run-root "${RUN_ROOT}" --protocol "${PROTOCOL}" \
    --parent-manifest "${PARENT_MANIFEST}" \
    --pre-repair-inventory "${PRE_REPAIR_INVENTORY}" --out "${out}"
