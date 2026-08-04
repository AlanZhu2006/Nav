#!/bin/bash
# Strict controller ablation on the fixed unseen-scene split.
# Submit MODE=smoke first; submit MODE=full only after the smoke reproduces
# the local Goal-A trajectory under the cluster Habitat runtime.
#SBATCH --job-name=gatecurr_direct
#SBATCH --output=/scratch/yz11502/Research/Nav-axis-uturn-results/gatecurr_direct_control_20260804/slurm_%j.out
#SBATCH --error=/scratch/yz11502/Research/Nav-axis-uturn-results/gatecurr_direct_control_20260804/slurm_%j.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=h200_public
#SBATCH --gres=gpu:h200:1
#SBATCH --account=torch_pr_769_tandon_advanced
#SBATCH --export=ALL

set -euo pipefail
umask 0022

MODE=${MODE:-smoke}
case "${MODE}" in
  smoke|full) ;;
  *) echo "ABORT: MODE must be smoke or full" >&2; exit 2 ;;
esac

ROOT=/scratch/yz11502/Research/Nav-axis-uturn
SOURCE_RUN=${ROOT}/.diagnostics/unseen_scene_eval_20260803
RESULT_BASE=/scratch/yz11502/Research/Nav-axis-uturn-results/gatecurr_direct_control_20260804
BASE_SIF=/share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif
HAB_PY=/scratch/lg154/conda-envs/habitat/bin/python
MEMNAV_PY=/scratch/lg154/conda-envs/memnav/bin/python
LINGBOT_REPO=/scratch/lg154/Research/Nav/NavDP/baselines/memnav/lingbot-map
NAVDP_CKPT=${SOURCE_RUN}/checkpoints/navdp_checkpoint.ckpt
: "${EXPECTED_COMMIT:?submit with EXPECTED_COMMIT=<full git SHA>}"

mkdir -p "${RESULT_BASE}"
for required in "${BASE_SIF}" "${HAB_PY}" "${MEMNAV_PY}" \
                "${LINGBOT_REPO}/weights/lingbot-map-long.pt" \
                "${NAVDP_CKPT}" \
                "${SOURCE_RUN}/checkpoints/gatecurr600.memnav.ckpt" \
                "${SOURCE_RUN}/manifest.json"; do
  test -r "${required}" || { echo "ABORT: missing dependency ${required}" >&2; exit 1; }
done

actual_commit=$(git -C "${ROOT}" rev-parse HEAD)
[[ "${actual_commit}" == "${EXPECTED_COMMIT}" ]] || {
  echo "ABORT: code commit ${actual_commit} != ${EXPECTED_COMMIT}" >&2
  exit 1
}

echo "host=$(hostname) job=${SLURM_JOB_ID:-none} mode=${MODE} commit=${actual_commit}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

exec singularity exec --nv \
  -B /scratch/lg154 \
  -B /scratch/yz11502 \
  "${BASE_SIF}" \
  env \
    ROOT="${ROOT}" \
    SOURCE_RUN="${SOURCE_RUN}" \
    RUN_ROOT="${RESULT_BASE}" \
    RESULT_SET="${MODE}_raw_e32_hpc_${SLURM_JOB_ID:-manual}" \
    HAB_PY="${HAB_PY}" \
    MEMNAV_PY="${MEMNAV_PY}" \
    INTERNNAV_ROOT="${ROOT}/InternNav" \
    LINGBOT_REPO="${LINGBOT_REPO}" \
    NAVDP_CKPT="${NAVDP_CKPT}" \
    MEMNAV_PORT=18903 \
    NAVDP_PORT=18904 \
    MAX_STEPS=500 \
    SEED=20260803 \
    RETRIEVAL=raw \
    EXCLUDE_RECENT=32 \
    "${ROOT}/MemNavData/run_gatecurr_direct_control_ab.sh" "${MODE}"
