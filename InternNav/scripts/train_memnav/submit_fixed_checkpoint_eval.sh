#!/bin/bash
# Submit a real-path preflight followed by the full paired checkpoint evaluator.
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../.." && pwd)
SBATCH_SCRIPT=${SCRIPT_DIR}/eval_memnav_checkpoints.sbatch
BASELINE_CKPT=${BASELINE_CKPT:-/scratch/lg154/Research/Nav/InternNav/checkpoints/memnav_mp3d_flowgate/ckpts/checkpoint-2600/memnav.ckpt}
CANDIDATE_CKPT=${CANDIDATE_CKPT:-/scratch/yz11502/Research/Nav-axis-uturn/InternNav/checkpoints/memnav_gatecurr_warm2600_alllegs_warmfix_20260803_014435/ckpts/checkpoint-600/memnav.ckpt}
EVAL_NAME=${EVAL_NAME:-fixed_ckpt2600_vs_gatecurr600_$(date +%Y%m%d_%H%M%S)}
RESULT_ROOT=${RESULT_ROOT:-/scratch/yz11502/Research/Nav-axis-uturn-results/fixed_checkpoint_eval}

command -v sbatch >/dev/null || { echo "ABORT: sbatch unavailable"; exit 1; }
[[ -f "${SBATCH_SCRIPT}" ]] || { echo "ABORT: missing ${SBATCH_SCRIPT}"; exit 1; }
[[ -f "${BASELINE_CKPT}" ]] || { echo "ABORT: missing ${BASELINE_CKPT}"; exit 1; }
[[ -f "${CANDIDATE_CKPT}" ]] || { echo "ABORT: missing ${CANDIDATE_CKPT}"; exit 1; }
if [[ -n "$(git -C "${REPO_ROOT}" status --porcelain)" ]]; then
  echo "ABORT: evaluator must be submitted from a clean committed worktree"
  git -C "${REPO_ROOT}" status --short
  exit 1
fi
bash -n "${SBATCH_SCRIPT}"

mkdir -p "${REPO_ROOT}/logs/train_memnav" "${RESULT_ROOT}"
COMMIT=$(git -C "${REPO_ROOT}" rev-parse HEAD)
PRE_OUT=${RESULT_ROOT}/${EVAL_NAME}_preflight
FULL_OUT=${RESULT_ROOT}/${EVAL_NAME}

echo "commit=${COMMIT} repo=${REPO_ROOT}"
echo "baseline=${BASELINE_CKPT}"
echo "candidate=${CANDIDATE_CKPT}"
echo "preflight_out=${PRE_OUT} full_out=${FULL_OUT}"

PRE_JOB=$(
  REPO_ROOT="${REPO_ROOT}" BASELINE_CKPT="${BASELINE_CKPT}" CANDIDATE_CKPT="${CANDIDATE_CKPT}" \
  EVAL_OUT="${PRE_OUT}" EVAL_SAMPLES_PER_GROUP=1 EVAL_BATCH_SIZE=3 EVAL_TRIALS=2 \
  sbatch --parsable --job-name="${EVAL_NAME}_pre" --time=00:45:00 \
    --cpus-per-task=8 --mem=96G --gres=gpu:1 \
    --output="${REPO_ROOT}/logs/train_memnav/${EVAL_NAME}_pre-%j.out" \
    --error="${REPO_ROOT}/logs/train_memnav/${EVAL_NAME}_pre-%j.err" \
    --export=ALL "${SBATCH_SCRIPT}"
)
PRE_JOB=${PRE_JOB%%;*}

FULL_JOB=$(
  REPO_ROOT="${REPO_ROOT}" BASELINE_CKPT="${BASELINE_CKPT}" CANDIDATE_CKPT="${CANDIDATE_CKPT}" \
  EVAL_OUT="${FULL_OUT}" EVAL_SAMPLES_PER_GROUP=12 EVAL_BATCH_SIZE=4 EVAL_TRIALS=8 \
  sbatch --parsable --job-name="${EVAL_NAME}" --time=04:00:00 \
    --dependency="afterok:${PRE_JOB}" --cpus-per-task=12 --mem=128G --gres=gpu:1 \
    --output="${REPO_ROOT}/logs/train_memnav/${EVAL_NAME}-%j.out" \
    --error="${REPO_ROOT}/logs/train_memnav/${EVAL_NAME}-%j.err" \
    --export=ALL "${SBATCH_SCRIPT}"
)
FULL_JOB=${FULL_JOB%%;*}

echo "preflight_job=${PRE_JOB}"
echo "full_job=${FULL_JOB} dependency=afterok:${PRE_JOB}"
squeue -j "${PRE_JOB},${FULL_JOB}" -o '%.18i %.34j %.10T %.10M %.9l %.28R'
