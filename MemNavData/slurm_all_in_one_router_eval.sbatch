#!/bin/bash
# Single-allocation version requested for simpler H200 scheduling.
# It runs 20-scene 2-leg and 10-scene 3-leg paired evaluations sequentially,
# summarizes each protocol, then runs the four-scene learned-router blind audit.
#SBATCH --job-name=router_all1
#SBATCH --output=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_router_all1_%j.out
#SBATCH --error=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_router_all1_%j.err
#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --partition=h200_public
#SBATCH --gres=gpu:h200:1
#SBATCH --account=torch_pr_769_tandon_advanced
#SBATCH --export=ALL

set -euo pipefail
umask 0022

: "${RUN_TAG:?submit with RUN_TAG=<unique tag>}"
: "${EXPECTED_COMMIT:?submit with EXPECTED_COMMIT=<full SHA>}"

ROOT=/scratch/yz11502/Research/Nav-axis-uturn
RESULT_BASE=/scratch/yz11502/Research/Nav-axis-uturn-results/all_in_one_router_eval_20260805
RUN_ROOT=${RESULT_BASE}/${RUN_TAG}
RUNNER=${ROOT}/MemNavData/run_all_in_one_router_eval.sh
BASE_SIF=/share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif
OVERLAY=/scratch/lg154/Research/datasets/_overlays/mp3d_revisit_v0_pt1.sqf
HAB_PY=/scratch/lg154/conda-envs/habitat/bin/python
MEMNAV_PY=/scratch/lg154/conda-envs/memnav/bin/python
LINGBOT_REPO=/scratch/lg154/Research/Nav/NavDP/baselines/memnav/lingbot-map
EXPECTED_OVERLAY_BYTES=128854888448
EXPECTED_OVERLAY_HEAD_SHA=43d803c2b39735e764f0c6df8f7c40a7035a1c23789e22fcc9831d29928c801b

mkdir -p "${RESULT_BASE}"
for required in "${BASE_SIF}" "${OVERLAY}" "${HAB_PY}" "${MEMNAV_PY}" \
                "${LINGBOT_REPO}/weights/lingbot-map-long.pt" "${RUNNER}"; do
  test -r "${required}" || { echo "ABORT: missing dependency ${required}" >&2; exit 1; }
done
actual_commit=$(git -C "${ROOT}" rev-parse HEAD)
[[ "${actual_commit}" == "${EXPECTED_COMMIT}" ]] || {
  echo "ABORT: code commit ${actual_commit} != ${EXPECTED_COMMIT}" >&2
  exit 1
}
overlay_bytes=$(stat -c '%s' "${OVERLAY}")
overlay_head_sha=$(dd if="${OVERLAY}" bs=1048576 count=1 status=none | \
  sha256sum | awk '{print $1}')
[[ "${overlay_bytes}" == "${EXPECTED_OVERLAY_BYTES}" ]] || {
  echo "ABORT: overlay size mismatch" >&2; exit 1; }
[[ "${overlay_head_sha}" == "${EXPECTED_OVERLAY_HEAD_SHA}" ]] || {
  echo "ABORT: overlay header SHA mismatch" >&2; exit 1; }

echo "host=$(hostname) job=${SLURM_JOB_ID} commit=${actual_commit}"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}

exec singularity exec --nv \
  --overlay "${OVERLAY}:ro" \
  -B /scratch/lg154 \
  -B /scratch/yz11502 \
  "${BASE_SIF}" \
  env \
    ROOT="${ROOT}" \
    RUN_ROOT="${RUN_ROOT}" \
    EXPECTED_COMMIT="${EXPECTED_COMMIT}" \
    HAB_PY="${HAB_PY}" \
    MEMNAV_PY="${MEMNAV_PY}" \
    LINGBOT_REPO="${LINGBOT_REPO}" \
    EPISODE_OVERLAY_ROOT=/mp3d_revisit_v0/vln_n1/traj_data/mp3d_2leg \
    "${RUNNER}"
