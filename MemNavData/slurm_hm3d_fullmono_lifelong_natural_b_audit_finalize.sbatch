#!/usr/bin/env bash
#SBATCH --job-name=h3lifeNatBSeal
#SBATCH --output=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/%x_%j.out
#SBATCH --error=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/%x_%j.err
#SBATCH --time=00:20:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --partition=cpu_short
#SBATCH --account=torch_pr_769_tandon_advanced

set -euo pipefail
: "${AUDIT_ROOT:?}" "${SOURCE_TASK_ROOT:?}" "${BASE_SOURCE_ROOT:?}"
: "${RUN_ROOT:?}" "${AUDIT_RECEIPT:?}" "${EXPECTED_AUDIT_RECEIPT_SHA:?}"
: "${SOURCE_TASK_RECEIPT:?}" "${EXPECTED_SOURCE_TASK_RECEIPT_SHA:?}"
: "${BASE_RECEIPT:?}" "${EXPECTED_BASE_RECEIPT_SHA:?}"
[[ "$(sha256sum "${AUDIT_RECEIPT}" | awk '{print $1}')" == \
   "${EXPECTED_AUDIT_RECEIPT_SHA}" ]]
(cd "${AUDIT_ROOT}" && sha256sum -c --quiet "${AUDIT_RECEIPT}")
[[ "$(sha256sum "${SOURCE_TASK_RECEIPT}" | awk '{print $1}')" == \
   "${EXPECTED_SOURCE_TASK_RECEIPT_SHA}" ]]
(cd "${SOURCE_TASK_ROOT}" && sha256sum -c --quiet "${SOURCE_TASK_RECEIPT}")
[[ "$(sha256sum "${BASE_RECEIPT}" | awk '{print $1}')" == \
   "${EXPECTED_BASE_RECEIPT_SHA}" ]]
(cd "${BASE_SOURCE_ROOT}" && sha256sum -c --quiet "${BASE_RECEIPT}")

BASE_SIF=/share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif
PY=/scratch/lg154/conda-envs/habitat/bin/python
HAB_REQUESTS_VENDOR=/scratch/lg154/conda-envs/habitat/lib/python3.9/site-packages/pip/_vendor
pycache=${SLURM_TMPDIR:-/tmp}/h3life_natb_seal_${SLURM_JOB_ID}
mkdir -p "${pycache}"
singularity exec -B /scratch/lg154 -B /scratch/yz11502 \
  "${BASE_SIF}" env PYTHONDONTWRITEBYTECODE=1 \
  PYTHONPYCACHEPREFIX="${pycache}" \
  PYTHONPATH="${AUDIT_ROOT}/MemNavData:${SOURCE_TASK_ROOT}:${SOURCE_TASK_ROOT}/MemNavData:${BASE_SOURCE_ROOT}:${BASE_SOURCE_ROOT}/MemNavData:${HAB_REQUESTS_VENDOR}" \
  "${PY}" -u \
    "${AUDIT_ROOT}/MemNavData/audit_hm3d_fullmono_lifelong_natural_b.py" \
    aggregate --audit-root "${RUN_ROOT}/natural_b_audit/scenes" \
    --expected-scenes 54 --out "${RUN_ROOT}/natural_b_audit/summary.json"
(cd "${RUN_ROOT}/natural_b_audit" && \
  sha256sum -c summary.json.sha256 >/dev/null)
chmod -R a-w "${RUN_ROOT}/natural_b_audit"
