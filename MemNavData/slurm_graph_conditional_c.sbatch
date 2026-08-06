#!/bin/bash
#SBATCH --job-name=mem_graph_condC
#SBATCH --output=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_mem_graph_condC_%j.out
#SBATCH --error=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_mem_graph_condC_%j.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --partition=h100_tandon
#SBATCH --gres=gpu:h100:1
#SBATCH --account=torch_pr_769_tandon_advanced
#SBATCH --export=ALL

set -euo pipefail
umask 0022

: "${MODE:?submit with MODE=smoke or full}"
: "${RUN_TAG:?submit with a unique RUN_TAG}"
: "${EXPECTED_COMMIT:?submit with the full EXPECTED_COMMIT}"
[[ "${MODE}" =~ ^(smoke|full)$ ]] || {
  echo "ABORT: MODE must be smoke or full" >&2; exit 1; }
[[ "${RUN_TAG}" =~ ^[A-Za-z0-9_.-]+$ ]] || {
  echo "ABORT: invalid RUN_TAG=${RUN_TAG}" >&2; exit 1; }

ROOT=${ROOT:-/scratch/yz11502/Research/Nav-graph-blind}
RESULT_BASE=/scratch/yz11502/Research/Nav-axis-uturn-results/graph_conditional_c_20260806
RUN_ROOT=${RESULT_BASE}/${RUN_TAG}_job_${SLURM_JOB_ID}
RUNNER=${ROOT}/MemNavData/run_graph_conditional_c.sh
BASE_SIF=/share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif
HAB_PY=/scratch/lg154/conda-envs/habitat/bin/python
MEMNAV_PY=/scratch/lg154/conda-envs/memnav/bin/python

for required in "${BASE_SIF}" "${HAB_PY}" "${MEMNAV_PY}" "${RUNNER}"; do
  test -r "${required}" || {
    echo "ABORT: missing dependency ${required}" >&2; exit 1; }
done
[[ "$(git -C "${ROOT}" rev-parse HEAD)" == "${EXPECTED_COMMIT}" ]] || {
  echo "ABORT: remote worktree commit mismatch" >&2; exit 1; }
[[ ! -e "${RUN_ROOT}" ]] || {
  echo "ABORT: output already exists: ${RUN_ROOT}" >&2; exit 1; }
mkdir -p "${RESULT_BASE}"

echo "host=$(hostname) job=${SLURM_JOB_ID} mode=${MODE} commit=${EXPECTED_COMMIT}"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

exec singularity exec --nv \
  -B /scratch/lg154 \
  -B /scratch/yz11502 \
  "${BASE_SIF}" \
  env \
    ROOT="${ROOT}" \
    RUN_ROOT="${RUN_ROOT}" \
    EXPECTED_COMMIT="${EXPECTED_COMMIT}" \
    HAB_PY="${HAB_PY}" \
    MEMNAV_PY="${MEMNAV_PY}" \
    MODE="${MODE}" \
    "${RUNNER}"
