#!/bin/bash
#SBATCH --job-name=mem_graph20
#SBATCH --output=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_mem_graph20_%j.out
#SBATCH --error=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_mem_graph20_%j.err
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

: "${EXPECTED_COMMIT:?submit with EXPECTED_COMMIT}"
: "${RUN_TAG:?submit with RUN_TAG}"
ROOT=${ROOT:-/scratch/yz11502/Research/Nav-memnav-graph}
RESULT_BASE=/scratch/yz11502/Research/Nav-axis-uturn-results/graph_router_ablation_20260806
RUN_ROOT=${RESULT_BASE}/${RUN_TAG}_job_${SLURM_JOB_ID}
REFERENCE_ROOT=/scratch/yz11502/Research/Nav-axis-uturn-results/all_in_one_router_eval_20260805/all1_2leg3leg_6dc745f_20260805/expanded_navdp_router
RUNNER=${ROOT}/MemNavData/run_graph_router_ablation.sh
BASE_SIF=/share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif
HAB_PY=/scratch/lg154/conda-envs/habitat/bin/python
MEMNAV_PY=/scratch/lg154/conda-envs/memnav/bin/python

mkdir -p "${RESULT_BASE}"
for required in "${BASE_SIF}" "${HAB_PY}" "${MEMNAV_PY}" \
                "${REFERENCE_ROOT}" "${RUNNER}"; do
  test -r "${required}" || {
    echo "ABORT: missing dependency ${required}" >&2; exit 1; }
done
[[ "$(git -C "${ROOT}" rev-parse HEAD)" == "${EXPECTED_COMMIT}" ]] || {
  echo "ABORT: remote worktree commit mismatch" >&2; exit 1; }

echo "host=$(hostname) job=${SLURM_JOB_ID} commit=${EXPECTED_COMMIT}"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
exec singularity exec --nv \
  -B /scratch/lg154 \
  -B /scratch/yz11502 \
  "${BASE_SIF}" \
  env \
    ROOT="${ROOT}" \
    RUN_ROOT="${RUN_ROOT}" \
    REFERENCE_ROOT="${REFERENCE_ROOT}" \
    EXPECTED_COMMIT="${EXPECTED_COMMIT}" \
    HAB_PY="${HAB_PY}" \
    MEMNAV_PY="${MEMNAV_PY}" \
    GRAPH_SPACING_M="${GRAPH_SPACING_M:-1.25}" \
    GRAPH_ARRIVAL_M="${GRAPH_ARRIVAL_M:-0.60}" \
    "${RUNNER}"
