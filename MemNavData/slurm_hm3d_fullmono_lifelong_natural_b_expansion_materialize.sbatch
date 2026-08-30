#!/usr/bin/env bash
#SBATCH --job-name=h3NatBxMat
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
: "${TASK_ROOT:?}" "${TASK_RECEIPT:?}" "${EXPECTED_TASK_RECEIPT_SHA:?}"
: "${BASE_SOURCE_ROOT:?}" "${BASE_RECEIPT:?}" "${EXPECTED_BASE_RECEIPT_SHA:?}"
: "${SOURCE_TASK_ROOT:?}" "${SOURCE_TASK_RECEIPT:?}" "${EXPECTED_SOURCE_TASK_RECEIPT_SHA:?}"
: "${AUDIT_SOURCE_ROOT:?}" "${AUDIT_SOURCE_RECEIPT:?}" "${EXPECTED_AUDIT_SOURCE_RECEIPT_SHA:?}"
: "${RUN_ROOT:?}" "${PARENT_ROOT:?}" "${PROTOCOL:?}"
: "${AUDIT_RUN_ROOT:?}" "${SOURCE_CONSTRUCTION_ROOT:?}" "${SOURCE_PROTOCOL:?}"
for spec in \
  "${TASK_RECEIPT}:${EXPECTED_TASK_RECEIPT_SHA}:${TASK_ROOT}" \
  "${BASE_RECEIPT}:${EXPECTED_BASE_RECEIPT_SHA}:${BASE_SOURCE_ROOT}" \
  "${SOURCE_TASK_RECEIPT}:${EXPECTED_SOURCE_TASK_RECEIPT_SHA}:${SOURCE_TASK_ROOT}" \
  "${AUDIT_SOURCE_RECEIPT}:${EXPECTED_AUDIT_SOURCE_RECEIPT_SHA}:${AUDIT_SOURCE_ROOT}"; do
  IFS=: read -r receipt expected root <<<"${spec}"
  [[ "$(sha256sum "${receipt}" | awk '{print $1}')" == "${expected}" ]]
  (cd "${root}" && sha256sum -c --quiet "${receipt}")
done

scene=$(/scratch/lg154/conda-envs/memnav/bin/python - \
  "${PARENT_ROOT}/sealed_inputs/parent_manifest.json" \
  "${SLURM_ARRAY_TASK_ID}" <<'PY'
import json,sys
p=json.load(open(sys.argv[1])); print(p['scenes'][int(sys.argv[2])])
PY
)
out=${RUN_ROOT}/construct_ab/scenes/$(printf '%02d_%s' \
  "${SLURM_ARRAY_TASK_ID}" "${scene}")
base_sif=/share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif
py=/scratch/lg154/conda-envs/habitat/bin/python
vendor=/scratch/lg154/conda-envs/habitat/lib/python3.9/site-packages/pip/_vendor
pycache=${SLURM_TMPDIR:-/tmp}/h3_natbx_mat_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}
mkdir -p "${pycache}" /scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs
exec singularity exec --nv -B /scratch/lg154 -B /scratch/yz11502 \
  "${base_sif}" env PYTHONDONTWRITEBYTECODE=1 \
  PYTHONPYCACHEPREFIX="${pycache}" \
  PYTHONPATH="${TASK_ROOT}:${TASK_ROOT}/MemNavData:${AUDIT_SOURCE_ROOT}/MemNavData:${SOURCE_TASK_ROOT}:${SOURCE_TASK_ROOT}/MemNavData:${BASE_SOURCE_ROOT}:${BASE_SOURCE_ROOT}/MemNavData:${vendor}" \
  "${py}" -u \
    "${TASK_ROOT}/MemNavData/materialize_hm3d_fullmono_lifelong_natural_b_expansion.py" \
    --parent-root "${PARENT_ROOT}" --protocol "${PROTOCOL}" \
    --audit-run-root "${AUDIT_RUN_ROOT}" \
    --source-construction-root "${SOURCE_CONSTRUCTION_ROOT}" \
    --source-protocol "${SOURCE_PROTOCOL}" \
    --scene-index "${SLURM_ARRAY_TASK_ID}" --out "${out}"
