#!/usr/bin/env bash
#SBATCH --job-name=h3monoBuild
#SBATCH --output=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/%x_%A_%a.out
#SBATCH --error=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/%x_%A_%a.err
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=56G
#SBATCH --partition=h100_tandon,a100_tandon
#SBATCH --gres=gpu:1
#SBATCH --account=torch_pr_769_tandon_advanced

set -euo pipefail
: "${SLURM_ARRAY_TASK_ID:?scene array index required}"
: "${TASK_ROOT:?}" "${BASE_SOURCE_ROOT:?}" "${RUN_ROOT:?}"
: "${PARENT_MANIFEST:?}" "${PROTOCOL:?}"
[[ "$(sha256sum "${TASK_RECEIPT}" | awk '{print $1}')" == \
   "${EXPECTED_TASK_RECEIPT_SHA}" ]]
(cd "${TASK_ROOT}" && sha256sum -c --quiet "${TASK_RECEIPT}")
[[ "$(sha256sum "${BASE_RECEIPT}" | awk '{print $1}')" == \
   "${EXPECTED_BASE_RECEIPT_SHA}" ]]
(cd "${BASE_SOURCE_ROOT}" && sha256sum -c --quiet "${BASE_RECEIPT}")
BASE_SIF=/share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif
pycache=${SLURM_TMPDIR:-/tmp}/h3mono_build_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}
mkdir -p "${pycache}"
exec singularity exec --nv -B /scratch/lg154 -B /scratch/yz11502 \
  "${BASE_SIF}" env PYTHONDONTWRITEBYTECODE=1 \
  PYTHONPYCACHEPREFIX="${pycache}" \
  PYTHONPATH="${TASK_ROOT}:${TASK_ROOT}/MemNavData:${BASE_SOURCE_ROOT}:${BASE_SOURCE_ROOT}/MemNavData" \
  /scratch/lg154/conda-envs/habitat/bin/python -u \
    "${TASK_ROOT}/MemNavData/construct_hm3d_fullmono_role_pairs.py" \
      --run-root "${RUN_ROOT}" --protocol "${PROTOCOL}" \
      --parent-manifest "${PARENT_MANIFEST}" \
      --scene-index "${SLURM_ARRAY_TASK_ID}"
