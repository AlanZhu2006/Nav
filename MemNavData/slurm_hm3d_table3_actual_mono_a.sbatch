#!/usr/bin/env bash
#SBATCH --job-name=h3T3MonoA
#SBATCH --output=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/%x_%A_%a.out
#SBATCH --error=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/%x_%A_%a.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=72G
#SBATCH --partition=h100_tandon,a100_tandon
#SBATCH --gres=gpu:1
#SBATCH --account=torch_pr_769_tandon_advanced

set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
export PYTHONFAULTHANDLER=1
: "${SLURM_ARRAY_TASK_ID:?}" "${WRAPPER_ROOT:?}" "${WRAPPER_RECEIPT:?}"
: "${EXPECTED_WRAPPER_RECEIPT_SHA:?}" "${TASK_ROOT:?}" "${RUN_ROOT:?}"
: "${TABLE3_PLAN:?}" "${TABLE3_EXECUTION_PROTOCOL:?}"
[[ "${SLURM_ARRAY_TASK_ID}" =~ ^([0-9]|[1-9][0-9]|1[01][0-9]|12[0-4])$ ]]
[[ "$(sha256sum "${WRAPPER_RECEIPT}" | awk '{print $1}')" == \
   "${EXPECTED_WRAPPER_RECEIPT_SHA}" ]]
(cd "${WRAPPER_ROOT}" && sha256sum -c --quiet "${WRAPPER_RECEIPT}")
BASE_SIF=/share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif
exec singularity exec --nv -B /scratch/lg154 -B /scratch/yz11502 \
  "${BASE_SIF}" env MODE=table3_a TASK_ROOT="${TASK_ROOT}" \
  WRAPPER_ROOT="${WRAPPER_ROOT}" SERVER_SOURCE_ROOT="${SERVER_SOURCE_ROOT}" \
  BASE_SOURCE_ROOT="${BASE_SOURCE_ROOT}" RUN_ROOT="${RUN_ROOT}" \
  PARENT_MANIFEST="${TABLE3_PLAN}" PROTOCOL="${TABLE3_EXECUTION_PROTOCOL}" \
  TABLE3_PLAN="${TABLE3_PLAN}" TABLE3_EXECUTION_PROTOCOL="${TABLE3_EXECUTION_PROTOCOL}" \
  SCENE_INDEX="${SLURM_ARRAY_TASK_ID}" \
  RUNTIME_ATTEMPT="formal$(printf '%03d' "${SLURM_ARRAY_TASK_ID}")" \
  HAB_PY=/scratch/lg154/conda-envs/habitat/bin/python \
  MEMNAV_PY=/scratch/lg154/conda-envs/memnav/bin/python \
  TASK_RECEIPT="${TASK_RECEIPT}" EXPECTED_TASK_RECEIPT_SHA="${EXPECTED_TASK_RECEIPT_SHA}" \
  SERVER_SOURCE_RECEIPT="${SERVER_SOURCE_RECEIPT}" \
  EXPECTED_SERVER_SOURCE_RECEIPT_SHA="${EXPECTED_SERVER_SOURCE_RECEIPT_SHA}" \
  BASE_RECEIPT="${BASE_RECEIPT}" EXPECTED_BASE_RECEIPT_SHA="${EXPECTED_BASE_RECEIPT_SHA}" \
  EXPECTED_HAB_REQUESTS_VERSION=2.32.4 EXPECTED_HAB_REQUESTS_INIT_BYTES=5057 \
  EXPECTED_HAB_REQUESTS_INIT_SHA=1e507f1f386bcc6b5f0ff69a614c14875cd65cb67be7f6022f28adef9774573f \
  EXPECTED_HAB_REQUESTS_VERSION_BYTES=435 \
  EXPECTED_HAB_REQUESTS_VERSION_SHA=143abaf3563712f063743a7952aa65319dbcb934d894cfc989bd2c015f8da577 \
  bash "${WRAPPER_ROOT}/MemNavData/run_hm3d_fullmono_server_scene.sh"
