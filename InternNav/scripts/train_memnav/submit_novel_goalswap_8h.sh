#!/bin/bash
# Submit the Novel goal-collapse repair as:
#   real two-row preflight --afterok--> eight-hour all-leg training.
#
# Causal changes relative to residualgate1000:
#   1) Goal-A current frames start at the live inference boundary k=40;
#   2) a same-state/same-noise counterfactual denoising margin prevents the
#      decoder from producing the same action for a same-scene wrong goal;
#   3) the optimizer now honors AdamW weight decay + cosine warmup.
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../.." && pwd)
SBATCH_SCRIPT="${SCRIPT_DIR}/train_memnav_mp3d.sbatch"
RUN_NAME=${RUN_NAME:-memnav_novelgs_res1000_early40_w025_$(date +%Y%m%d_%H%M%S)}
MEMNAV_INIT_CKPT=${MEMNAV_INIT_CKPT:-/scratch/yz11502/Research/checkpoints/residualgate1000.memnav.ckpt}

command -v sbatch >/dev/null || { echo "ABORT: sbatch is unavailable"; exit 1; }
command -v scontrol >/dev/null || { echo "ABORT: scontrol is unavailable"; exit 1; }
[[ "${REPO_ROOT}" == */Nav-axis-uturn/InternNav ]] || {
  echo "ABORT: use the dedicated Nav-axis-uturn checkout: ${REPO_ROOT}"; exit 1;
}
[[ -f "${SBATCH_SCRIPT}" ]] || { echo "ABORT: missing ${SBATCH_SCRIPT}"; exit 1; }
[[ -f "${MEMNAV_INIT_CKPT}" ]] || {
  echo "ABORT: warm-start checkpoint missing: ${MEMNAV_INIT_CKPT}"; exit 1;
}
if [[ -n "$(git -C "${REPO_ROOT}" status --porcelain)" ]]; then
  echo "ABORT: worktree is not clean: ${REPO_ROOT}"
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
echo "hypothesis=early Goal-A + counterfactual denoising restores goal conditioning"

COMMON_EXPORTS=(
  REPO_ROOT="${REPO_ROOT}"
  MEMNAV_INIT_CKPT="${MEMNAV_INIT_CKPT}"
  MEMNAV_MAX_LEGS=0
  MEMNAV_GATE_TEACHER_START=0.0
  MEMNAV_GATE_TEACHER_END=0.0
  MEMNAV_GATE_TEACHER_STEPS=0
  MEMNAV_GATE_FUSION=residual
  MEMNAV_RETRIEVAL_TOP1_WEIGHT=0.0
  MEMNAV_RETRIEVAL_TOP1_MARGIN=0.2
  MEMNAV_GOAL_A_MIN_K=40
  MEMNAV_GOAL_SWAP_WEIGHT=0.25
  MEMNAV_GOAL_SWAP_MARGIN=0.05
  MEMNAV_GOAL_SWAP_MIN_ANGLE_DEG=30.0
  LR=5e-5
)

PREFLIGHT_JOB=$(
  env "${COMMON_EXPORTS[@]}" \
  NAME="${RUN_NAME}_preflight" \
  WANDB_NAME="${RUN_NAME}_preflight" \
  MEMNAV_PREFLIGHT_ONLY=1 \
  MEMNAV_TRAIN_MAX_STEPS=1 \
  MEMNAV_REPORT_TO=none \
  NPROC=1 BATCH_SIZE=2 NUM_WORKERS=0 \
  sbatch --parsable --job-name="${RUN_NAME}_pre" --time=00:30:00 \
    --gres=gpu:1 --cpus-per-task=8 --mem=80G \
    --output="${REPO_ROOT}/logs/train_memnav/${RUN_NAME}_pre-%j.out" \
    --error="${REPO_ROOT}/logs/train_memnav/${RUN_NAME}_pre-%j.err" \
    --export=ALL "${SBATCH_SCRIPT}"
)
PREFLIGHT_JOB=${PREFLIGHT_JOB%%;*}

TRAIN_JOB=$(
  env "${COMMON_EXPORTS[@]}" \
  NAME="${RUN_NAME}" \
  WANDB_NAME="${RUN_NAME}" \
  MEMNAV_PREFLIGHT_ONLY=0 \
  MEMNAV_TRAIN_MAX_STEPS=-1 \
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
