#!/bin/bash
# Submit scene 0 first; scenes 1-19 depend on its transport success.
#SBATCH --job-name=bearingA40
#SBATCH --output=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_bearingA40_%A_%a.out
#SBATCH --error=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_bearingA40_%A_%a.err
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH --partition=h200_public
#SBATCH --gres=gpu:h200:1
#SBATCH --account=torch_pr_769_tandon_advanced
#SBATCH --export=ALL

set -euo pipefail
umask 0022
: "${SLURM_ARRAY_TASK_ID:?submit as an array}"
: "${RUN_TAG:?set RUN_TAG}"
: "${EXPECTED_COMMIT:?set EXPECTED_COMMIT}"

ROOT=/scratch/yz11502/Research/Nav-axis-uturn
RESULT_BASE=/scratch/yz11502/Research/Nav-axis-uturn-results/novel_a_bearing_gate_20260808
RUN_ROOT=${RESULT_BASE}/${RUN_TAG}
HAB_PY=/scratch/lg154/conda-envs/habitat/bin/python
POLICY_PY=/scratch/lg154/conda-envs/memnav/bin/python
RUNNER=${ROOT}/MemNavData/run_novel_a_bearing_scene.sh
BASE_SIF=/share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif

mkdir -p "${RESULT_BASE}" "${RUN_ROOT}"
actual_commit=$(git -C "${ROOT}" rev-parse HEAD)
[[ "${actual_commit}" == "${EXPECTED_COMMIT}" ]] || {
  echo "ABORT: commit mismatch" >&2; exit 1; }
echo "host=$(hostname) job=${SLURM_JOB_ID} task=${SLURM_ARRAY_TASK_ID} commit=${actual_commit}"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

exec singularity exec --nv \
  -B /scratch/lg154 -B /scratch/yz11502 \
  "${BASE_SIF}" env \
    ROOT="${ROOT}" RUN_ROOT="${RUN_ROOT}" \
    EXPECTED_COMMIT="${EXPECTED_COMMIT}" \
    SCENE_INDEX="${SLURM_ARRAY_TASK_ID}" \
    HAB_PY="${HAB_PY}" POLICY_PY="${POLICY_PY}" \
    "${RUNNER}"
