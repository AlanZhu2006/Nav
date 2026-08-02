#!/bin/bash
# Submit the causal revisit-gate experiment as:
#   one-batch preflight --afterok--> eight-hour all-leg training.
#
# Run this from the dedicated lg154 deployment worktree. Environment variables
# RUN_NAME and MEMNAV_INIT_CKPT may override the timestamped name and warm start.
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../.." && pwd)
SBATCH_SCRIPT="${SCRIPT_DIR}/train_memnav_mp3d.sbatch"
RUN_NAME=${RUN_NAME:-memnav_gatecurr_warm2600_alllegs_$(date +%Y%m%d_%H%M%S)}
MEMNAV_INIT_CKPT=${MEMNAV_INIT_CKPT:-/scratch/lg154/Research/Nav/InternNav/checkpoints/memnav_mp3d_flowgate/checkpoint-2600/memnav.ckpt}

command -v sbatch >/dev/null || { echo "ABORT: sbatch is unavailable"; exit 1; }
command -v scontrol >/dev/null || { echo "ABORT: scontrol is unavailable"; exit 1; }
[[ -f "${SBATCH_SCRIPT}" ]] || { echo "ABORT: missing ${SBATCH_SCRIPT}"; exit 1; }
[[ -f "${MEMNAV_INIT_CKPT}" ]] || {
  echo "ABORT: warm-start checkpoint missing: ${MEMNAV_INIT_CKPT}"; exit 1;
}
if [[ -n "$(git -C "${REPO_ROOT}" status --porcelain)" ]]; then
  echo "ABORT: worktree has tracked or untracked changes in ${REPO_ROOT}"
  git -C "${REPO_ROOT}" status --short
  exit 1
fi

COMMIT=$(git -C "${REPO_ROOT}" rev-parse HEAD)
BRANCH=$(git -C "${REPO_ROOT}" branch --show-current)
mkdir -p "${REPO_ROOT}/logs/train_memnav"
bash -n "${SBATCH_SCRIPT}"
cd "${REPO_ROOT}"

echo "commit=${COMMIT} branch=${BRANCH}"
echo "repo=${REPO_ROOT}"
echo "run=${RUN_NAME} init_ckpt=${MEMNAV_INIT_CKPT}"
echo "training=all legs, gate teacher 1->0 over 500 optimizer steps, 8 hours"

# One GPU is enough for the real-batch preflight. All data/model/cache knobs are
# identical to the long job; only batch/DDP/logging differ.
PREFLIGHT_JOB=$(
  REPO_ROOT="${REPO_ROOT}" \
  NAME="${RUN_NAME}_preflight" \
  WANDB_NAME="${RUN_NAME}_preflight" \
  MEMNAV_INIT_CKPT="${MEMNAV_INIT_CKPT}" \
  MEMNAV_MAX_LEGS=0 \
  MEMNAV_GATE_TEACHER_START=1.0 \
  MEMNAV_GATE_TEACHER_END=0.0 \
  MEMNAV_GATE_TEACHER_STEPS=500 \
  MEMNAV_PREFLIGHT_ONLY=1 \
  MEMNAV_REPORT_TO=none \
  NPROC=1 BATCH_SIZE=1 NUM_WORKERS=0 \
  sbatch --parsable --job-name="${RUN_NAME}_pre" --time=00:30:00 \
    --gres=gpu:1 --cpus-per-task=8 --mem=80G \
    --output="${REPO_ROOT}/logs/train_memnav/${RUN_NAME}_pre-%j.out" \
    --error="${REPO_ROOT}/logs/train_memnav/${RUN_NAME}_pre-%j.err" \
    --export=ALL "${SBATCH_SCRIPT}"
)
PREFLIGHT_JOB=${PREFLIGHT_JOB%%;*}

TRAIN_JOB=$(
  REPO_ROOT="${REPO_ROOT}" \
  NAME="${RUN_NAME}" \
  WANDB_NAME="${RUN_NAME}" \
  MEMNAV_INIT_CKPT="${MEMNAV_INIT_CKPT}" \
  MEMNAV_MAX_LEGS=0 \
  MEMNAV_GATE_TEACHER_START=1.0 \
  MEMNAV_GATE_TEACHER_END=0.0 \
  MEMNAV_GATE_TEACHER_STEPS=500 \
  MEMNAV_PREFLIGHT_ONLY=0 \
  MEMNAV_REPORT_TO=wandb \
  MEMNAV_SAVE_STEPS=100 \
  NPROC=2 BATCH_SIZE=4 NUM_WORKERS=8 EPOCHS=10 \
  sbatch --parsable --job-name="${RUN_NAME}" --time=08:00:00 \
    --dependency="afterok:${PREFLIGHT_JOB}" \
    --output="${REPO_ROOT}/logs/train_memnav/${RUN_NAME}-%j.out" \
    --error="${REPO_ROOT}/logs/train_memnav/${RUN_NAME}-%j.err" \
    --export=ALL "${SBATCH_SCRIPT}"
)
TRAIN_JOB=${TRAIN_JOB%%;*}

echo "preflight_job=${PREFLIGHT_JOB}"
echo "train_job=${TRAIN_JOB} dependency=afterok:${PREFLIGHT_JOB}"
scontrol show job -o "${PREFLIGHT_JOB}"
scontrol show job -o "${TRAIN_JOB}"
squeue -j "${PREFLIGHT_JOB},${TRAIN_JOB}" \
  -o '%.18i %.28j %.10T %.10M %.9l %.28R'
