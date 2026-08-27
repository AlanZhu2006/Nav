#!/usr/bin/env bash
#SBATCH --job-name=h3mix_build
#SBATCH --output=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/%x_%A_%a.out
#SBATCH --error=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/%x_%A_%a.err
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --partition=h100_tandon,a100_tandon,h200_public,l40s_public
#SBATCH --gres=gpu:1
#SBATCH --account=torch_pr_769_tandon_advanced

set -euo pipefail
: "${SLURM_ARRAY_TASK_ID:?submit scene array}"
: "${TASK_ROOT:?set task bundle}"
: "${EXPECTED_TASK_RECEIPT_SHA:?set task receipt SHA}"
: "${HM3D_RUNTIME_ROOT:?set parent runtime bundle}"
: "${EXPECTED_HM3D_RUNTIME_RECEIPT_SHA:?set parent runtime receipt SHA}"
: "${EXEC_SOURCE_ROOT:?set Final14 execution bundle}"
: "${EXPECTED_EXEC_SOURCE_RECEIPT_SHA:?set execution receipt SHA}"
: "${RUN_ROOT:?set run root}"
: "${PARENT_RUN_ROOT:?set parent HM3D run}"
: "${PARENT_MANIFEST:?set copied parent manifest}"

receipt=${TASK_ROOT}/SOURCE_BUNDLE.sha256
[[ "$(sha256sum "${receipt}" | awk '{print $1}')" == \
   "${EXPECTED_TASK_RECEIPT_SHA}" ]]
(cd "${TASK_ROOT}" && sha256sum -c SOURCE_BUNDLE.sha256 >/dev/null)
[[ "$(sha256sum "${HM3D_RUNTIME_ROOT}/SOURCE_BUNDLE.sha256" | awk '{print $1}')" == \
   "${EXPECTED_HM3D_RUNTIME_RECEIPT_SHA}" ]]
(cd "${HM3D_RUNTIME_ROOT}" && sha256sum -c SOURCE_BUNDLE.sha256 >/dev/null)
[[ "$(sha256sum "${EXEC_SOURCE_ROOT}/SOURCE_BUNDLE.sha256" | awk '{print $1}')" == \
   "${EXPECTED_EXEC_SOURCE_RECEIPT_SHA}" ]]
(cd "${EXEC_SOURCE_ROOT}" && sha256sum -c SOURCE_BUNDLE.sha256 >/dev/null)
[[ "${SLURM_ARRAY_TASK_ID}" =~ ^[0-8]$ ]]

HAB_PY=/scratch/lg154/conda-envs/habitat/bin/python
BASE_SIF=/share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif
export PYTHONPYCACHEPREFIX=${SLURM_TMPDIR:-/tmp}/h3mix_build_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}
mkdir -p "${PYTHONPYCACHEPREFIX}"
singularity exec --nv -B /scratch/lg154 -B /scratch/yz11502 \
  "${BASE_SIF}" env TASK_ROOT="${TASK_ROOT}" \
  HM3D_RUNTIME_ROOT="${HM3D_RUNTIME_ROOT}" \
  EXEC_SOURCE_ROOT="${EXEC_SOURCE_ROOT}" RUN_ROOT="${RUN_ROOT}" \
  PARENT_RUN_ROOT="${PARENT_RUN_ROOT}" PARENT_MANIFEST="${PARENT_MANIFEST}" \
  PROTOCOL="${TASK_ROOT}/MemNavData/hm3d_mixed_role_protocol_20260818.json" \
  SCENE_RANK="${SLURM_ARRAY_TASK_ID}" HAB_PY="${HAB_PY}" \
  PYTHONPYCACHEPREFIX="${PYTHONPYCACHEPREFIX}" \
  bash "${TASK_ROOT}/MemNavData/run_hm3d_mixed_role_construct_scene.sh"
