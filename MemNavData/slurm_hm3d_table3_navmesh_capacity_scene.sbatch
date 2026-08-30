#!/usr/bin/env bash
#SBATCH --job-name=h3T3Cap
#SBATCH --output=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/%x_%A_%a.out
#SBATCH --error=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/%x_%A_%a.err
#SBATCH --time=00:20:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --partition=cpu_short
#SBATCH --account=torch_pr_769_tandon_advanced

set -euo pipefail
umask 0022
export PYTHONDONTWRITEBYTECODE=1
: "${SLURM_ARRAY_TASK_ID:?scene array index required}"
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

py=/scratch/lg154/conda-envs/habitat/bin/python
scene=$(/scratch/lg154/conda-envs/memnav/bin/python - \
  "${parent}" "${SLURM_ARRAY_TASK_ID}" <<'PY'
import json,sys
p=json.load(open(sys.argv[1]))
print(p['scenes'][int(sys.argv[2])])
PY
)
out=${RUN_ROOT}/scenes/$(printf '%02d_%s' "${SLURM_ARRAY_TASK_ID}" "${scene}")/capacity.json
cache=${SLURM_TMPDIR:-/tmp}/h3_t3_capacity_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}
mkdir -p "${cache}"
exec env PYTHONPATH="${TASK_ROOT}/MemNavData" \
  PYTHONPYCACHEPREFIX="${cache}" "${py}" -u \
  "${TASK_ROOT}/MemNavData/audit_hm3d_table3_navmesh_capacity.py" scene \
  --parent-manifest "${parent}" \
  --expected-parent-sha256 "${EXPECTED_PARENT_MANIFEST_SHA}" \
  --protocol "${PROTOCOL}" --scene-index "${SLURM_ARRAY_TASK_ID}" \
  --out "${out}"
