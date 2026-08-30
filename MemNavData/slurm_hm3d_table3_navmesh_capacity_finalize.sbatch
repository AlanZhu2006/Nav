#!/usr/bin/env bash
#SBATCH --job-name=h3T3CapFin
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
: "${MODE:?finalize or verify}"
: "${TASK_ROOT:?}" "${TASK_RECEIPT:?}" "${EXPECTED_TASK_RECEIPT_SHA:?}"
: "${PARENT_ROOT:?}" "${RUN_ROOT:?}" "${PROTOCOL:?}"
: "${EXPECTED_PROTOCOL_SHA:?}" "${EXPECTED_PARENT_MANIFEST_SHA:?}"

[[ "$(sha256sum "${TASK_RECEIPT}" | awk '{print $1}')" == \
   "${EXPECTED_TASK_RECEIPT_SHA}" ]]
(cd "${TASK_ROOT}" && sha256sum -c --quiet "${TASK_RECEIPT}")
[[ "$(sha256sum "${PROTOCOL}" | awk '{print $1}')" == \
   "${EXPECTED_PROTOCOL_SHA}" ]]
parent=${PARENT_ROOT}/sealed_inputs/parent_manifest.json
[[ "$(sha256sum "${parent}" | awk '{print $1}')" == \
   "${EXPECTED_PARENT_MANIFEST_SHA}" ]]
py=/scratch/lg154/conda-envs/memnav/bin/python
export PYTHONPATH=${TASK_ROOT}/MemNavData

if [[ "${MODE}" == finalize ]]; then
  exec "${py}" -u \
    "${TASK_ROOT}/MemNavData/audit_hm3d_table3_navmesh_capacity.py" finalize \
    --scene-root "${RUN_ROOT}/scenes" --parent-manifest "${parent}" \
    --expected-parent-sha256 "${EXPECTED_PARENT_MANIFEST_SHA}" \
    --protocol "${PROTOCOL}" --out "${RUN_ROOT}/capacity_summary.json"
fi
[[ "${MODE}" == verify ]]
(cd "${RUN_ROOT}" && sha256sum -c --quiet capacity_summary.json.sha256)
exec "${py}" -u \
  "${TASK_ROOT}/MemNavData/independent_verify_hm3d_table3_navmesh_capacity.py" \
  --summary "${RUN_ROOT}/capacity_summary.json" --protocol "${PROTOCOL}" \
  --out "${RUN_ROOT}/independent_capacity_verification.json"
