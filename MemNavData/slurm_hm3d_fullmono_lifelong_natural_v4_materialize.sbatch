#!/usr/bin/env bash
#SBATCH --job-name=h3lifeV4Mat
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
: "${V4_ROOT:?}" "${AUDIT_SOURCE_ROOT:?}" "${SOURCE_TASK_ROOT:?}"
: "${BASE_SOURCE_ROOT:?}" "${RUN_ROOT:?}" "${PARENT_ROOT:?}"
: "${PROTOCOL:?}" "${AUDIT_RUN_ROOT:?}" "${SOURCE_CONSTRUCTION_ROOT:?}"
: "${V4_RECEIPT:?}" "${EXPECTED_V4_RECEIPT_SHA:?}"
: "${AUDIT_SOURCE_RECEIPT:?}" "${EXPECTED_AUDIT_SOURCE_RECEIPT_SHA:?}"
: "${SOURCE_TASK_RECEIPT:?}" "${EXPECTED_SOURCE_TASK_RECEIPT_SHA:?}"
: "${BASE_RECEIPT:?}" "${EXPECTED_BASE_RECEIPT_SHA:?}"
[[ "$(sha256sum "${V4_RECEIPT}" | awk '{print $1}')" == \
   "${EXPECTED_V4_RECEIPT_SHA}" ]]
(cd "${V4_ROOT}" && sha256sum -c --quiet "${V4_RECEIPT}")
[[ "$(sha256sum "${AUDIT_SOURCE_RECEIPT}" | awk '{print $1}')" == \
   "${EXPECTED_AUDIT_SOURCE_RECEIPT_SHA}" ]]
(cd "${AUDIT_SOURCE_ROOT}" && sha256sum -c --quiet "${AUDIT_SOURCE_RECEIPT}")
[[ "$(sha256sum "${SOURCE_TASK_RECEIPT}" | awk '{print $1}')" == \
   "${EXPECTED_SOURCE_TASK_RECEIPT_SHA}" ]]
(cd "${SOURCE_TASK_ROOT}" && sha256sum -c --quiet "${SOURCE_TASK_RECEIPT}")
[[ "$(sha256sum "${BASE_RECEIPT}" | awk '{print $1}')" == \
   "${EXPECTED_BASE_RECEIPT_SHA}" ]]
(cd "${BASE_SOURCE_ROOT}" && sha256sum -c --quiet "${BASE_RECEIPT}")

BASE_SIF=/share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif
PY=/scratch/lg154/conda-envs/habitat/bin/python
HAB_REQUESTS_VENDOR=/scratch/lg154/conda-envs/habitat/lib/python3.9/site-packages/pip/_vendor
SCENE=$(
  /scratch/lg154/conda-envs/memnav/bin/python - \
    "${RUN_ROOT}/sealed_inputs/parent_manifest.json" \
    "${SLURM_ARRAY_TASK_ID}" <<'PY'
import json, sys
payload = json.load(open(sys.argv[1]))
print(payload["scenes"][int(sys.argv[2])])
PY
)
OUT=${RUN_ROOT}/construct_ab/scenes/$(
  printf '%02d_%s' "${SLURM_ARRAY_TASK_ID}" "${SCENE}"
)
pycache=${SLURM_TMPDIR:-/tmp}/h3life_v4_mat_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}
mkdir -p "${pycache}"
exec singularity exec --nv -B /scratch/lg154 -B /scratch/yz11502 \
  "${BASE_SIF}" env PYTHONDONTWRITEBYTECODE=1 \
  PYTHONPYCACHEPREFIX="${pycache}" \
  PYTHONPATH="${V4_ROOT}/MemNavData:${AUDIT_SOURCE_ROOT}/MemNavData:${SOURCE_TASK_ROOT}:${SOURCE_TASK_ROOT}/MemNavData:${BASE_SOURCE_ROOT}:${BASE_SOURCE_ROOT}/MemNavData:${HAB_REQUESTS_VENDOR}" \
  "${PY}" -u \
    "${V4_ROOT}/MemNavData/materialize_hm3d_fullmono_lifelong_natural_ab.py" \
    --parent-root "${PARENT_ROOT}" --protocol "${PROTOCOL}" \
    --audit-run-root "${AUDIT_RUN_ROOT}" \
    --source-construction-root "${SOURCE_CONSTRUCTION_ROOT}" \
    --scene-index "${SLURM_ARRAY_TASK_ID}" --out "${OUT}"
