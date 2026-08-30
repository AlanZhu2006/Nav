#!/usr/bin/env bash
#SBATCH --job-name=h3T3Construct
#SBATCH --output=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/%x_%A_%a.out
#SBATCH --error=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/%x_%A_%a.err
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --partition=h100_tandon,a100_tandon
#SBATCH --gres=gpu:1
#SBATCH --account=torch_pr_769_tandon_advanced

set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
: "${SLURM_ARRAY_TASK_ID:?}" "${WRAPPER_ROOT:?}" "${WRAPPER_RECEIPT:?}"
: "${EXPECTED_WRAPPER_RECEIPT_SHA:?}" "${RUN_ROOT:?}" "${TABLE3_PLAN:?}"
: "${TABLE3_PROTOCOL:?}" "${EXPECTED_TABLE3_PLAN_SHA:?}"
: "${TASK_RECEIPT:?}" "${EXPECTED_TASK_RECEIPT_SHA:?}"
: "${BASE_RECEIPT:?}" "${EXPECTED_BASE_RECEIPT_SHA:?}"
[[ "${SLURM_ARRAY_TASK_ID}" =~ ^([0-9]|[1-9][0-9]|1[01][0-9]|12[0-4])$ ]]
[[ "$(sha256sum "${WRAPPER_RECEIPT}" | awk '{print $1}')" == \
   "${EXPECTED_WRAPPER_RECEIPT_SHA}" ]]
(cd "${WRAPPER_ROOT}" && sha256sum -c --quiet "${WRAPPER_RECEIPT}")
[[ "$(sha256sum "${TABLE3_PLAN}" | awk '{print $1}')" == \
   "${EXPECTED_TABLE3_PLAN_SHA}" ]]
[[ "$(sha256sum "${TASK_RECEIPT}" | awk '{print $1}')" == \
   "${EXPECTED_TASK_RECEIPT_SHA}" ]]
(cd "${TASK_ROOT}" && sha256sum -c --quiet "${TASK_RECEIPT}")
[[ "$(sha256sum "${BASE_RECEIPT}" | awk '{print $1}')" == \
   "${EXPECTED_BASE_RECEIPT_SHA}" ]]
(cd "${BASE_SOURCE_ROOT}" && sha256sum -c --quiet "${BASE_RECEIPT}")
BASE_SIF=/share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif
HAB_PY=/scratch/lg154/conda-envs/habitat/bin/python
HAB_SITE=$(${HAB_PY} -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
export PYTHONPYCACHEPREFIX=${SLURM_TMPDIR:-/tmp}/h3_t3_construct_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}
exec singularity exec --nv -B /scratch/lg154 -B /scratch/yz11502 \
  "${BASE_SIF}" env PYTHONDONTWRITEBYTECODE=1 \
  PYTHONPATH="${WRAPPER_ROOT}:${WRAPPER_ROOT}/MemNavData:${TASK_ROOT}:${TASK_ROOT}/MemNavData:${BASE_SOURCE_ROOT}:${BASE_SOURCE_ROOT}/MemNavData:${HAB_SITE}/pip/_vendor" \
  "${HAB_PY}" -u \
  "${WRAPPER_ROOT}/MemNavData/construct_hm3d_table3_actual_mono_role_pair.py" \
    --run-root "${RUN_ROOT}" --candidate-plan "${TABLE3_PLAN}" \
    --protocol "${TABLE3_PROTOCOL}" --history-index "${SLURM_ARRAY_TASK_ID}"
