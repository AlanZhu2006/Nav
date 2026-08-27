#!/bin/bash
# One scene per array task; submit index 0 first, then 1-19 after it passes.
# For strict Phase-B P0 pairing, export P0_SHARED_PREFIX=1 together with
# RUN_NAVDP_NATIVE=1,RUN_GEOMETRY_TOP1=0,RUN_GEOMETRY_ROUTER=1,
# RUN_LEARNED_RANK_GEOMETRY=1,DETERMINISTIC_PLAN_SEEDS=1.  Each array task
# then runs geometry -> learned -> native on one node and one pair of servers.
# Example:
#   sbatch --array=0 --export=ALL,RUN_TAG=<tag>,EXPECTED_COMMIT=<sha>,EXPECTED_MANIFEST_SHA=<sha> <this-file>
#   sbatch --array=1-19%2 --dependency=afterok:<smoke-job> --export=... <this-file>
#SBATCH --job-name=navdp_router20
#SBATCH --output=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_navdp_router20_%A_%a.out
#SBATCH --error=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_navdp_router20_%A_%a.err
#SBATCH --time=08:00:00
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

: "${SLURM_ARRAY_TASK_ID:?submit this script as an array}"
: "${RUN_TAG:?submit with one shared RUN_TAG}"
: "${EXPECTED_COMMIT:?submit with EXPECTED_COMMIT=<full SHA>}"
: "${EXPECTED_MANIFEST_SHA:?submit with EXPECTED_MANIFEST_SHA=<SHA256>}"

ROOT=/scratch/yz11502/Research/Nav-axis-uturn
RESULT_BASE=/scratch/yz11502/Research/Nav-axis-uturn-results/expanded_navdp_router_20260805
RUN_ROOT=${RESULT_BASE}/${RUN_TAG}
MANIFEST=${ROOT}/MemNavData/expanded_navdp_router_eval_20260805.json
RUNNER=${ROOT}/MemNavData/run_expanded_navdp_router_scene.sh
BASE_SIF=/share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif
HAB_PY=/scratch/lg154/conda-envs/habitat/bin/python
MEMNAV_PY=/scratch/lg154/conda-envs/memnav/bin/python
RUN_LEARNED_RANK_GEOMETRY=${RUN_LEARNED_RANK_GEOMETRY:-0}
P0_SHARED_PREFIX=${P0_SHARED_PREFIX:-0}
PHASE_B_CKPT=${PHASE_B_CKPT:-}
EXPECTED_PHASE_B_CKPT_SHA=${EXPECTED_PHASE_B_CKPT_SHA:-}

mkdir -p "${RESULT_BASE}" "${RUN_ROOT}"
for required in "${BASE_SIF}" "${HAB_PY}" "${MEMNAV_PY}" \
                "${MANIFEST}" "${RUNNER}"; do
  test -r "${required}" || { echo "ABORT: missing dependency ${required}" >&2; exit 1; }
done
actual_commit=$(git -C "${ROOT}" rev-parse HEAD)
[[ "${actual_commit}" == "${EXPECTED_COMMIT}" ]] || {
  echo "ABORT: code commit ${actual_commit} != ${EXPECTED_COMMIT}" >&2
  exit 1
}

echo "host=$(hostname) job=${SLURM_JOB_ID} task=${SLURM_ARRAY_TASK_ID} commit=${actual_commit}"
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
    SCENE_INDEX="${SLURM_ARRAY_TASK_ID}" \
    HAB_PY="${HAB_PY}" \
    MEMNAV_PY="${MEMNAV_PY}" \
    RUN_LEARNED_RANK_GEOMETRY="${RUN_LEARNED_RANK_GEOMETRY}" \
    P0_SHARED_PREFIX="${P0_SHARED_PREFIX}" \
    PHASE_B_CKPT="${PHASE_B_CKPT}" \
    EXPECTED_PHASE_B_CKPT_SHA="${EXPECTED_PHASE_B_CKPT_SHA}" \
    "${RUNNER}"
