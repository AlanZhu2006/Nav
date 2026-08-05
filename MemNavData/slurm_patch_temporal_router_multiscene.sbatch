#!/bin/bash
# Eight-hour ceiling for the scene-balanced router distillation/audit.
# Submit smoke first; full is allowed only after the smoke report completes.
# Example:
#   sbatch --export=ALL,MODE=smoke,EXPECTED_COMMIT=<sha> <this-file>
#   sbatch --export=ALL,MODE=full,EXPECTED_COMMIT=<sha> <this-file>
#SBATCH --job-name=router_scenes
#SBATCH --output=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_router_scenes_%j.out
#SBATCH --error=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_router_scenes_%j.err
#SBATCH --time=08:00:00
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

MODE=${MODE:-full}
case "${MODE}" in
  smoke|full) ;;
  *) echo "ABORT: MODE must be smoke or full" >&2; exit 2 ;;
esac

ROOT=/scratch/yz11502/Research/Nav-axis-uturn
RESULT_BASE=${RESULT_BASE:-/scratch/yz11502/Research/Nav-axis-uturn-results/patch_router_multiscene_20260805}
BASE_SIF=/share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif
OVERLAY=/scratch/lg154/Research/datasets/_overlays/mp3d_revisit_v0_pt1.sqf
MEMNAV_PY=/scratch/lg154/conda-envs/memnav/bin/python
LINGBOT_REPO=/scratch/lg154/Research/Nav/NavDP/baselines/memnav/lingbot-map
SPLIT_MANIFEST=${SPLIT_MANIFEST:-${ROOT}/MemNavData/router_multiscene_split_20260805.json}
RUNNER=${ROOT}/MemNavData/run_patch_temporal_router_multiscene.sh
EXPECTED_OVERLAY_BYTES=128854888448
EXPECTED_OVERLAY_HEAD_SHA=43d803c2b39735e764f0c6df8f7c40a7035a1c23789e22fcc9831d29928c801b
: "${EXPECTED_COMMIT:?submit with EXPECTED_COMMIT=<full git SHA>}"

mkdir -p "${RESULT_BASE}"
for required in "${BASE_SIF}" "${OVERLAY}" "${MEMNAV_PY}" \
                "${LINGBOT_REPO}/weights/lingbot-map-long.pt" \
                "${SPLIT_MANIFEST}" "${RUNNER}"; do
  test -r "${required}" || {
    echo "ABORT: missing dependency ${required}" >&2
    exit 1
  }
done

overlay_bytes=$(stat -c '%s' "${OVERLAY}")
overlay_head_sha=$(dd if="${OVERLAY}" bs=1048576 count=1 status=none | \
  sha256sum | awk '{print $1}')
[[ "${overlay_bytes}" == "${EXPECTED_OVERLAY_BYTES}" ]] || {
  echo "ABORT: overlay size mismatch ${overlay_bytes}" >&2; exit 1; }
[[ "${overlay_head_sha}" == "${EXPECTED_OVERLAY_HEAD_SHA}" ]] || {
  echo "ABORT: overlay header SHA mismatch ${overlay_head_sha}" >&2; exit 1; }

actual_commit=$(git -C "${ROOT}" rev-parse HEAD)
[[ "${actual_commit}" == "${EXPECTED_COMMIT}" ]] || {
  echo "ABORT: code commit ${actual_commit} != ${EXPECTED_COMMIT}" >&2
  exit 1
}

echo "host=$(hostname) job=${SLURM_JOB_ID:-none} mode=${MODE} commit=${actual_commit}"
echo "overlay_bytes=${overlay_bytes} overlay_head_sha=${overlay_head_sha}"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}

exec singularity exec --nv \
  --overlay "${OVERLAY}:ro" \
  -B /scratch/lg154 \
  -B /scratch/yz11502 \
  "${BASE_SIF}" \
  env \
    MODE="${MODE}" \
    ROOT="${ROOT}" \
    RUN_ROOT="${RESULT_BASE}/${MODE}_job_${SLURM_JOB_ID}" \
    EPISODE_ROOT=/mp3d_revisit_v0/vln_n1/traj_data/mp3d_2leg \
    MEMNAV_PY="${MEMNAV_PY}" \
    LINGBOT_REPO="${LINGBOT_REPO}" \
    SPLIT_MANIFEST="${SPLIT_MANIFEST}" \
    EXPECTED_SPLIT_SHA="${EXPECTED_SPLIT_SHA:-}" \
    REFERENCE_ROUTER_MODEL="${REFERENCE_ROUTER_MODEL:-}" \
    EXPECTED_REFERENCE_ROUTER_SHA="${EXPECTED_REFERENCE_ROUTER_SHA:-}" \
    EXPECTED_COMMIT="${EXPECTED_COMMIT}" \
    "${RUNNER}" "${MODE}"
