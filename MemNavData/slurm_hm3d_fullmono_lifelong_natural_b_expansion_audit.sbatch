#!/usr/bin/env bash
#SBATCH --job-name=h3lifeNatBX
#SBATCH --output=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/%x_%A_%a.out
#SBATCH --error=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/%x_%A_%a.err
#SBATCH --time=00:45:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=56G
#SBATCH --partition=h100_tandon,a100_tandon
#SBATCH --gres=gpu:1
#SBATCH --account=torch_pr_769_tandon_advanced

set -euo pipefail
: "${SLURM_ARRAY_TASK_ID:?scene array index required}"
: "${AUDIT_ROOT:?}" "${SOURCE_TASK_ROOT:?}" "${BASE_SOURCE_ROOT:?}"
: "${RUN_ROOT:?}" "${PARENT_ROOT:?}" "${SOURCE_RUN_ROOT:?}"
: "${CURRENT_V4_RUN_ROOT:?}"
: "${AUDIT_RECEIPT:?}" "${EXPECTED_AUDIT_RECEIPT_SHA:?}"
: "${SOURCE_TASK_RECEIPT:?}" "${EXPECTED_SOURCE_TASK_RECEIPT_SHA:?}"
: "${BASE_RECEIPT:?}" "${EXPECTED_BASE_RECEIPT_SHA:?}"
: "${ORIGINAL_MANIFEST_SHA:?}" "${SOURCE_PROTOCOL_SHA:?}"
[[ "$(sha256sum "${AUDIT_RECEIPT}" | awk '{print $1}')" == \
   "${EXPECTED_AUDIT_RECEIPT_SHA}" ]]
(cd "${AUDIT_ROOT}" && sha256sum -c --quiet "${AUDIT_RECEIPT}")
[[ "$(sha256sum "${SOURCE_TASK_RECEIPT}" | awk '{print $1}')" == \
   "${EXPECTED_SOURCE_TASK_RECEIPT_SHA}" ]]
(cd "${SOURCE_TASK_ROOT}" && sha256sum -c --quiet "${SOURCE_TASK_RECEIPT}")
[[ "$(sha256sum "${BASE_RECEIPT}" | awk '{print $1}')" == \
   "${EXPECTED_BASE_RECEIPT_SHA}" ]]
(cd "${BASE_SOURCE_ROOT}" && sha256sum -c --quiet "${BASE_RECEIPT}")

source_protocol=${SOURCE_TASK_ROOT}/MemNavData/hm3d_fullmono_lifelong_power_expansion_protocol_20260826.json
expansion_protocol=${AUDIT_ROOT}/MemNavData/hm3d_fullmono_lifelong_natural_b_expansion_audit_protocol_20260828.json
original_manifest=${CURRENT_V4_RUN_ROOT}/ab_population/role_pairs/manifest.json
[[ "$(sha256sum "${source_protocol}" | awk '{print $1}')" == \
   "${SOURCE_PROTOCOL_SHA}" ]]
[[ "$(sha256sum "${original_manifest}" | awk '{print $1}')" == \
   "${ORIGINAL_MANIFEST_SHA}" ]]

BASE_SIF=/share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif
PY=/scratch/lg154/conda-envs/habitat/bin/python
HAB_REQUESTS_VENDOR=/scratch/lg154/conda-envs/habitat/lib/python3.9/site-packages/pip/_vendor
SCENE=$(
  /scratch/lg154/conda-envs/memnav/bin/python - \
    "${PARENT_ROOT}/sealed_inputs/parent_manifest.json" \
    "${SLURM_ARRAY_TASK_ID}" <<'PY'
import json, sys
payload = json.load(open(sys.argv[1]))
print(payload["scenes"][int(sys.argv[2])])
PY
)
OUT=${RUN_ROOT}/expansion_audit/scenes/$(
  printf '%02d_%s' "${SLURM_ARRAY_TASK_ID}" "${SCENE}"
)/natural_b_expansion_audit.json
pycache=${SLURM_TMPDIR:-/tmp}/h3life_natbx_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}
mkdir -p "${pycache}"
exec singularity exec --nv -B /scratch/lg154 -B /scratch/yz11502 \
  "${BASE_SIF}" env PYTHONDONTWRITEBYTECODE=1 \
  PYTHONPYCACHEPREFIX="${pycache}" \
  PYTHONPATH="${AUDIT_ROOT}/MemNavData:${SOURCE_TASK_ROOT}:${SOURCE_TASK_ROOT}/MemNavData:${BASE_SOURCE_ROOT}:${BASE_SOURCE_ROOT}/MemNavData:${HAB_REQUESTS_VENDOR}" \
  "${PY}" -u \
    "${AUDIT_ROOT}/MemNavData/audit_hm3d_fullmono_lifelong_natural_b_expansion.py" \
    scene --parent-root "${PARENT_ROOT}" \
    --source-protocol "${source_protocol}" \
    --expansion-protocol "${expansion_protocol}" \
    --construction-root "${SOURCE_RUN_ROOT}/construct_ab/scenes" \
    --original-manifest "${original_manifest}" \
    --scene-index "${SLURM_ARRAY_TASK_ID}" --out "${OUT}"
