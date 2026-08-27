#!/bin/bash
# One Slurm task owns one NavDP server and visits scenes sequentially.  The
# 24-hour default avoids a speculative multi-day reservation; submit smoke
# with ``--time=03:00:00`` and size the full job from its measured runtime.
# Safe smoke submission:
#   sbatch --time=03:00:00 --export=ALL,MODE=smoke,RUN_TAG=<tag>,EXPECTED_COMMIT=<sha> <file>
# After smoke's paired audit passes, preserve the dependency in the full job:
#   sbatch --dependency=afterok:<smoke-job> --export=ALL,MODE=full,RUN_TAG=<tag>,EXPECTED_COMMIT=<sha> <file>
#SBATCH --job-name=novel_b_ub
#SBATCH --output=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_novel_b_ub_%j.out
#SBATCH --error=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_novel_b_ub_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --partition=h100_tandon
#SBATCH --gres=gpu:h100:1
#SBATCH --account=torch_pr_769_tandon_advanced
#SBATCH --export=ALL

set -euo pipefail
umask 0022

: "${RUN_TAG:?submit with a unique RUN_TAG}"
: "${EXPECTED_COMMIT:?submit with EXPECTED_COMMIT=<full SHA>}"
[[ "${RUN_TAG}" =~ ^[A-Za-z0-9._-]+$ ]] || {
  echo "ABORT: RUN_TAG contains unsafe characters" >&2
  exit 2
}
MODE=${MODE:-smoke}
case "${MODE}" in
  smoke|full) ;;
  *) echo "ABORT: MODE must be smoke or full" >&2; exit 2 ;;
esac
for command_name in git singularity nvidia-smi stat dd sha256sum; do
  command -v "${command_name}" >/dev/null 2>&1 || {
    echo "ABORT: missing command dependency ${command_name}" >&2
    exit 1
  }
done

ROOT=${ROOT:-/scratch/yz11502/Research/Nav-graph-blind}
RESULT_BASE=${RESULT_BASE:-/scratch/yz11502/Research/Nav-axis-uturn-results/novel_b_upper_bound_20260807}
RUN_ROOT=${RESULT_BASE}/${RUN_TAG}_${MODE}
MANIFEST=${ROOT}/MemNavData/expanded_3leg_router_eval_20260805.json
RUNNER=${ROOT}/MemNavData/run_novel_b_upper_bound.sh
BASE_SIF=${BASE_SIF:-/share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif}
HAB_PY=${HAB_PY:-/scratch/lg154/conda-envs/habitat/bin/python}
NAVDP_PY=${NAVDP_PY:-/scratch/lg154/conda-envs/memnav/bin/python}
NAVDP_CKPT=${NAVDP_CKPT:-/scratch/yz11502/Research/Nav-axis-uturn/.diagnostics/unseen_scene_eval_20260803/checkpoints/navdp_checkpoint.ckpt}
EXPECTED_BASE_SIF_BYTES=10035580928
EXPECTED_BASE_SIF_HEAD_SHA256=990c1377491fbf58c1d7d37bcf8043249826bc7a094eacc4c7a443b9905126c7

mkdir -p "${RESULT_BASE}"
for required in "${BASE_SIF}" "${HAB_PY}" "${NAVDP_PY}" \
                "${MANIFEST}" "${RUNNER}" "${NAVDP_CKPT}"; do
  test -r "${required}" || {
    echo "ABORT: missing dependency ${required}" >&2
    exit 1
  }
done
actual_sif_bytes=$(stat -c '%s' "${BASE_SIF}")
[[ "${actual_sif_bytes}" == "${EXPECTED_BASE_SIF_BYTES}" ]] || {
  echo "ABORT: container image byte count changed" >&2
  exit 1
}
actual_sif_head_sha=$(dd if="${BASE_SIF}" bs=1048576 count=1 status=none | \
  sha256sum)
actual_sif_head_sha=${actual_sif_head_sha%% *}
[[ "${actual_sif_head_sha}" == "${EXPECTED_BASE_SIF_HEAD_SHA256}" ]] || {
  echo "ABORT: container image header SHA256 changed" >&2
  exit 1
}
actual_commit=$(git -C "${ROOT}" rev-parse HEAD)
[[ "${actual_commit}" == "${EXPECTED_COMMIT}" ]] || {
  echo "ABORT: code commit ${actual_commit} != ${EXPECTED_COMMIT}" >&2
  exit 1
}
[[ ! -e "${RUN_ROOT}" ]] || {
  echo "ABORT: output already exists: ${RUN_ROOT}" >&2
  exit 1
}

echo "host=$(hostname) job=${SLURM_JOB_ID} mode=${MODE} commit=${actual_commit}"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

exec singularity exec --nv \
  -B /scratch/lg154 \
  -B /scratch/yz11502 \
  "${BASE_SIF}" \
  env \
    ROOT="${ROOT}" \
    RUN_ROOT="${RUN_ROOT}" \
    MANIFEST="${MANIFEST}" \
    EXPECTED_COMMIT="${EXPECTED_COMMIT}" \
    HAB_PY="${HAB_PY}" \
    NAVDP_PY="${NAVDP_PY}" \
    NAVDP_CKPT="${NAVDP_CKPT}" \
    CONTAINER_IMAGE_PATH="${BASE_SIF}" \
    CONTAINER_IMAGE_BYTES="${actual_sif_bytes}" \
    CONTAINER_IMAGE_HEAD_SHA256="${actual_sif_head_sha}" \
    "${RUNNER}" "${MODE}"
