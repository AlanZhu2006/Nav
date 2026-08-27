#!/bin/bash
# Paired causal rerun of the cV4 three-leg failure after temporal top-K routing.
# The selected scene is frozen before submission; both native NavDP and the
# geometry-router arm use the same episode and seed.
#SBATCH --job-name=topk3_cV4
#SBATCH --output=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_topk3_cV4_%j.out
#SBATCH --error=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_topk3_cV4_%j.err
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --partition=h200_public
#SBATCH --gres=gpu:h200:1
#SBATCH --account=torch_pr_769_tandon_advanced
#SBATCH --export=ALL

set -euo pipefail
umask 0022

: "${RUN_TAG:?submit with RUN_TAG=<unique tag>}"
: "${EXPECTED_COMMIT:?submit with EXPECTED_COMMIT=<full SHA>}"

ROOT=${ROOT:-/scratch/yz11502/Research/Nav-axis-uturn}
RESULT_BASE=/scratch/yz11502/Research/Nav-axis-uturn-results/targeted_3leg_topk_20260805
RUN_ROOT=${RESULT_BASE}/${RUN_TAG}
MANIFEST=${ROOT}/MemNavData/expanded_3leg_router_eval_20260805.json
RUNNER=${ROOT}/MemNavData/run_expanded_navdp_router_scene.sh
EVALUATOR=${ROOT}/MemNavData/eval_3leg_habitat.py
VALIDATOR=${ROOT}/MemNavData/validate_expanded_3leg_router_eval.py
BASE_SIF=/share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif
HAB_PY=/scratch/lg154/conda-envs/habitat/bin/python
MEMNAV_PY=/scratch/lg154/conda-envs/memnav/bin/python
EXPECTED_MANIFEST_SHA=55096038d0493723d2280fe9abbcb2ea9ea7f2059c50b6eb08d02f560cdb281b
SCENE_INDEX=7
EXPECTED_SCENE=cV4RVeZvu5T

for required in "${BASE_SIF}" "${HAB_PY}" "${MEMNAV_PY}" \
                "${MANIFEST}" "${RUNNER}" "${EVALUATOR}" "${VALIDATOR}"; do
  test -r "${required}" || {
    echo "ABORT: missing dependency ${required}" >&2
    exit 1
  }
done
actual_commit=$(git -C "${ROOT}" rev-parse HEAD)
[[ "${actual_commit}" == "${EXPECTED_COMMIT}" ]] || {
  echo "ABORT: code commit ${actual_commit} != ${EXPECTED_COMMIT}" >&2
  exit 1
}
actual_scene=$("${HAB_PY}" - "${MANIFEST}" "${SCENE_INDEX}" <<'PY'
import json
import sys
manifest = json.load(open(sys.argv[1], encoding="utf-8"))
print(manifest["selection"]["selected_scenes"][int(sys.argv[2])])
PY
)
[[ "${actual_scene}" == "${EXPECTED_SCENE}" ]] || {
  echo "ABORT: frozen scene ${actual_scene} != ${EXPECTED_SCENE}" >&2
  exit 1
}
[[ ! -e "${RUN_ROOT}" ]] || {
  echo "ABORT: output already exists: ${RUN_ROOT}" >&2
  exit 1
}
mkdir -p "${RESULT_BASE}"

echo "host=$(hostname) job=${SLURM_JOB_ID} commit=${actual_commit} scene=${actual_scene}"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

exec singularity exec --nv \
  -B /scratch/lg154 \
  -B /scratch/yz11502 \
  "${BASE_SIF}" \
  env \
    ROOT="${ROOT}" \
    RUN_ROOT="${RUN_ROOT}" \
    MANIFEST="${MANIFEST}" \
    EXPECTED_MANIFEST_SHA="${EXPECTED_MANIFEST_SHA}" \
    EXPECTED_COMMIT="${EXPECTED_COMMIT}" \
    SCENE_INDEX="${SCENE_INDEX}" \
    HAB_PY="${HAB_PY}" \
    MEMNAV_PY="${MEMNAV_PY}" \
    EVALUATOR="${EVALUATOR}" \
    VALIDATOR="${VALIDATOR}" \
    UNIT_TEST_MODULE=MemNavData.test_expanded_3leg_router_eval \
    MAX_STEPS=1200 \
    "${RUNNER}"
