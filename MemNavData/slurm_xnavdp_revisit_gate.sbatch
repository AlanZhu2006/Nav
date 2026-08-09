#!/bin/bash
# One scene per array task for the frozen four-arm revisit controller gate.
# Submit a one-episode smoke under a separate RUN_TAG before the 20-scene run.
#SBATCH --job-name=xnavdp_revisit
#SBATCH --output=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_xnavdp_revisit_%A_%a.out
#SBATCH --error=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_xnavdp_revisit_%A_%a.err
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
: "${RUN_TAG:?export a unique RUN_TAG}"
: "${EXPECTED_COMMIT:?export the exact committed source SHA}"
: "${EXPECTED_MANIFEST_SHA:?export the frozen manifest SHA256}"

ROOT=${ROOT:-/scratch/yz11502/Research/Nav-axis-uturn}
RESULT_BASE=${RESULT_BASE:-/scratch/yz11502/Research/Nav-axis-uturn-results/xnavdp_revisit_gate_20260809}
RUN_ROOT=${RESULT_BASE}/${RUN_TAG}
MANIFEST=${ROOT}/MemNavData/expanded_navdp_router_eval_20260805.json
RUNNER=${ROOT}/MemNavData/run_expanded_navdp_router_scene.sh
BASE_SIF=${BASE_SIF:-/share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif}
HAB_PY=${HAB_PY:-/scratch/lg154/conda-envs/habitat/bin/python}
MEMNAV_PY=${MEMNAV_PY:-/scratch/lg154/conda-envs/memnav/bin/python}
XNAVDP_PY=${XNAVDP_PY:-/scratch/lg154/conda-envs/navdp/bin/python}
XNAVDP_ASSET_ROOT=${XNAVDP_ASSET_ROOT:-/scratch/yz11502/Research/xnavdp_official_878740a2011856d0}
XNAVDP_OFFICIAL_ROOT=${XNAVDP_OFFICIAL_ROOT:-${XNAVDP_ASSET_ROOT}/NavDP}
XNAVDP_CKPT=${XNAVDP_CKPT:-${XNAVDP_ASSET_ROOT}/x-navdp_posttrain.ckpt}
EPISODE_LIMIT=${EPISODE_LIMIT:-0}

for required in "${ROOT}" "${RUNNER}" "${MANIFEST}" "${BASE_SIF}" \
                "${HAB_PY}" "${MEMNAV_PY}" "${XNAVDP_PY}" \
                "${XNAVDP_OFFICIAL_ROOT}" "${XNAVDP_CKPT}"; do
  test -r "${required}" || {
    echo "ABORT: missing dependency ${required}" >&2; exit 1; }
done
[[ "$(git -C "${ROOT}" rev-parse HEAD)" == "${EXPECTED_COMMIT}" ]] || {
  echo "ABORT: benchmark checkout differs from EXPECTED_COMMIT" >&2; exit 1; }

mkdir -p "${RESULT_BASE}" "${RUN_ROOT}"
echo "host=$(hostname) job=${SLURM_JOB_ID} task=${SLURM_ARRAY_TASK_ID} commit=${EXPECTED_COMMIT}"
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
    XNAVDP_PY="${XNAVDP_PY}" \
    XNAVDP_OFFICIAL_ROOT="${XNAVDP_OFFICIAL_ROOT}" \
    XNAVDP_CKPT="${XNAVDP_CKPT}" \
    XNAVDP_REVISIT_GATE=1 \
    RUN_NAVDP_NATIVE=1 \
    RUN_GEOMETRY_TOP1=0 \
    RUN_GEOMETRY_ROUTER=1 \
    RUN_LEARNED_RANK_GEOMETRY=0 \
    P0_SHARED_PREFIX=0 \
    MAX_STEPS=500 \
    EPISODE_LIMIT="${EPISODE_LIMIT}" \
    LEG1_MODE=policy \
    STOP_AFTER_LEG1=0 \
    WRITE_LEG1_TRACE=0 \
    DETERMINISTIC_PLAN_SEEDS=1 \
    NAVDP_GOAL_SWITCH_RESET=carry \
    TRAJECTORY_SELECTOR=server \
    TRAJECTORY_SELECTOR_SCOPE=all \
    ORACLE_CANDIDATE_SEED_COUNT=1 \
    RUN_CONDITIONAL_ORACLES=0 \
    RUN_CONDITIONAL_ORACLE_ANCHOR=0 \
    RUN_CONDITIONAL_ORACLE_POINT=0 \
    TERMINAL_UTURN=off \
    TERMINAL_VISUAL_REFINE=off \
    ARRIVAL_SHADOW=off \
    "${RUNNER}"
