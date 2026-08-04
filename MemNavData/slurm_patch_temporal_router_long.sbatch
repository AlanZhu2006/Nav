#!/bin/bash
# Eight-hour ceiling for training-only router distillation and held-out audit.
# Submit with EXPECTED_COMMIT=<full SHA>; results remain diagnostic-only.
#SBATCH --job-name=patch_router
#SBATCH --output=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_patch_router_%j.out
#SBATCH --error=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_patch_router_%j.err
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

ROOT=/scratch/yz11502/Research/Nav-axis-uturn
RESULT_BASE=/scratch/yz11502/Research/Nav-axis-uturn-results/patch_temporal_router_20260805
BASE_SIF=/share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif
MEMNAV_PY=/scratch/lg154/conda-envs/memnav/bin/python
LINGBOT_REPO=/scratch/lg154/Research/Nav/NavDP/baselines/memnav/lingbot-map
: "${EXPECTED_COMMIT:?submit with EXPECTED_COMMIT=<full git SHA>}"

mkdir -p "${RESULT_BASE}"
for required in "${BASE_SIF}" "${MEMNAV_PY}" \
                "${LINGBOT_REPO}/weights/lingbot-map-long.pt" \
                "${ROOT}/MemNavData/run_patch_temporal_router_long.sh"; do
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

echo "host=$(hostname) job=${SLURM_JOB_ID:-none} commit=${actual_commit}"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}

exec singularity exec --nv \
  -B /scratch/lg154 \
  -B /scratch/yz11502 \
  "${BASE_SIF}" \
  env \
    MODE=full \
    ROOT="${ROOT}" \
    RUN_ROOT="${RESULT_BASE}/job_${SLURM_JOB_ID}" \
    MEMNAV_PY="${MEMNAV_PY}" \
    LINGBOT_REPO="${LINGBOT_REPO}" \
    EXPECTED_COMMIT="${EXPECTED_COMMIT}" \
    PATH_MAPS="/home/asus/Research/Nav-axis-uturn=/scratch/yz11502/Research/Nav-axis-uturn /home/asus/Research/Nav=/scratch/yz11502/Research/Nav" \
    "${ROOT}/MemNavData/run_patch_temporal_router_long.sh" full
