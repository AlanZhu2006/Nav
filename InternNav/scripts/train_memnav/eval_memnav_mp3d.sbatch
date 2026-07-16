#!/bin/bash
#SBATCH --job-name=memnav_eval
#SBATCH --output=logs/eval_memnav/offline-%j.out
#SBATCH --error=logs/eval_memnav/offline-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=4:00:00
#SBATCH --account=torch_pr_769_tandon_advanced
#SBATCH --partition=a100_tandon,h100_tandon,h200_tandon
#SBATCH --gres=gpu:1
#SBATCH --export=ALL
# REQUIRED: complete scripts/train_memnav/README.md before every sbatch submission.
set -euo pipefail

REPO_ROOT=${REPO_ROOT:-/scratch/lg154/Research/Nav/InternNav}
IMG=${IMG:-/scratch/lg154/Research/datasets/_overlays/mp3d_revisit_v0_pt1.sqf}
BASE_SIF=${BASE_SIF:-/share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif}
LOG_DIR=${LOG_DIR:-${REPO_ROOT}/logs/eval_memnav}
mkdir -p "${LOG_DIR}"

export MEMNAV_ROOT_DIR=${MEMNAV_ROOT_DIR:-/mp3d_revisit_v0/vln_n1/traj_data}
export MEMNAV_FEATURE_ROOT=${MEMNAV_FEATURE_ROOT:-/scratch/lg154/Research/datasets/mp3d_revisit_v0_feat/vln_n1/traj_data}
export LINGBOT_REPO=${LINGBOT_REPO:-/scratch/lg154/Research/Nav/NavDP/baselines/memnav/lingbot-map}
export LINGBOT_WEIGHTS=${LINGBOT_WEIGHTS:-${LINGBOT_REPO}/weights/lingbot-map-long.pt}
export MEMNAV_DINO_WEIGHTS=${MEMNAV_DINO_WEIGHTS:-${REPO_ROOT}/checkpoints/depth_anything_v2_vits.pth}
export MEMNAV_WINDOW=${MEMNAV_WINDOW:-32}
export MEMNAV_NUM_SCALE=${MEMNAV_NUM_SCALE:-8}
export MEMNAV_MAX_FRAME_NUM=${MEMNAV_MAX_FRAME_NUM:-4096}
export MEMNAV_REQUIRE_GENERATED_POSE_CONVENTION=1

export DATA_SPLIT=${DATA_SPLIT:-val}
export VALIDATION_FRACTION=${VALIDATION_FRACTION:-0.1}
export SPLIT_SEED=${SPLIT_SEED:-0}
export SAMPLING_SEED=${SAMPLING_SEED:-0}
export BATCH_SIZE=${BATCH_SIZE:-4}
export NUM_WORKERS=${NUM_WORKERS:-4}
export MAX_SAMPLES=${MAX_SAMPLES:-0}
export ORACLE_POSITIVE=${ORACLE_POSITIVE:-1}
export FULL_DIFFUSION_GOAL_SHUFFLE=${FULL_DIFFUSION_GOAL_SHUFFLE:-0}
export DIFFUSION_SEED=${DIFFUSION_SEED:-104729}
export SAVE_PER_SAMPLE=${SAVE_PER_SAMPLE:-1}
export NAME=${NAME:-memnav_fixed_val}
export OUTPUT=${OUTPUT:-${LOG_DIR}/${NAME}-${SLURM_JOB_ID}.json}
: "${CHECKPOINT:?Set CHECKPOINT to memnav.ckpt or its checkpoint directory}"
[[ -f "${MEMNAV_DINO_WEIGHTS}" ]] \
  || { echo "Missing DINO weights: ${MEMNAV_DINO_WEIGHTS}" >&2; exit 1; }

apptainer exec --nv --overlay "${IMG}:ro" -B /scratch "${BASE_SIF}" bash -c '
    set -euo pipefail
    source /scratch/lg154/miniconda3/etc/profile.d/conda.sh
    conda activate /scratch/lg154/conda-envs/memnav
    cd '"${REPO_ROOT}"'
    export PYTHONPATH="'"${REPO_ROOT}"':'"${REPO_ROOT}"'/src/diffusion-policy:${PYTHONPATH:-}"
    command -v python
    python -V
    python -c "import torch, transformers, diffusers; print(\"torch\", torch.__version__, \"cuda\", torch.cuda.is_available(), \"transformers\", transformers.__version__, \"diffusers\", diffusers.__version__); assert torch.cuda.is_available()"
    extra=()
    (( MAX_SAMPLES > 0 )) && extra+=(--max-samples "${MAX_SAMPLES}")
    [[ "${ORACLE_POSITIVE}" == 1 ]] && extra+=(--oracle-positive)
    [[ "${FULL_DIFFUSION_GOAL_SHUFFLE}" == 1 ]] && extra+=(--full-diffusion-goal-shuffle)
    [[ "${SAVE_PER_SAMPLE}" == 1 ]] && extra+=(--save-per-sample)
    exec python scripts/eval/eval_memnav_offline.py \
        --checkpoint "${CHECKPOINT}" --output "${OUTPUT}" \
        --root-dir "${MEMNAV_ROOT_DIR}" --feature-root "${MEMNAV_FEATURE_ROOT}" \
        --data-split "${DATA_SPLIT}" --validation-fraction "${VALIDATION_FRACTION}" \
        --split-seed "${SPLIT_SEED}" --sampling-seed "${SAMPLING_SEED}" \
        --diffusion-seed "${DIFFUSION_SEED}" \
        --batch-size "${BATCH_SIZE}" --num-workers "${NUM_WORKERS}" \
        "${extra[@]}"
'
