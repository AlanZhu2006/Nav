#!/bin/bash
#SBATCH --job-name=memnav_eval
#SBATCH --output=/scratch/%u/Research/Nav/InternNav/logs/eval_memnav/offline-%j.out
#SBATCH --error=/scratch/%u/Research/Nav/InternNav/logs/eval_memnav/offline-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=4:00:00
#SBATCH --account=torch_pr_769_tandon_advanced
#SBATCH --partition=h100_tandon,h200_tandon
#SBATCH --gres=gpu:1
#SBATCH --export=ALL
set -euo pipefail

export REPO_ROOT=${REPO_ROOT:-/scratch/${USER}/Research/Nav/InternNav}
export LOG_DIR=${LOG_DIR:-${REPO_ROOT}/logs/eval_memnav}
export IMG=${IMG:-/scratch/lg154/Research/datasets/_overlays/mp3d_revisit_v0_pt1.sqf}
export BASE_SIF=${BASE_SIF:-/share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif}
export CONDA_SH=${CONDA_SH:-/scratch/lg154/miniconda3/etc/profile.d/conda.sh}
export CONDA_ENV=${CONDA_ENV:-/scratch/lg154/conda-envs/memnav}

export MEMNAV_ROOT_DIR=${MEMNAV_ROOT_DIR:-/mp3d_revisit_v0/vln_n1/traj_data}
export MEMNAV_FEATURE_ROOT=${MEMNAV_FEATURE_ROOT:-/scratch/lg154/Research/datasets/mp3d_revisit_v0_feat/vln_n1/traj_data}
export LINGBOT_REPO=${LINGBOT_REPO:-/scratch/lg154/Research/Nav/NavDP/baselines/memnav/lingbot-map}
export LINGBOT_WEIGHTS=${LINGBOT_WEIGHTS:-${LINGBOT_REPO}/weights/lingbot-map-long.pt}
export MEMNAV_WINDOW=${MEMNAV_WINDOW:-32}
export MEMNAV_NUM_SCALE=${MEMNAV_NUM_SCALE:-8}
export MEMNAV_MAX_FRAME_NUM=${MEMNAV_MAX_FRAME_NUM:-4096}
export MEMNAV_STRICT_FEATURE_COVERAGE=${MEMNAV_STRICT_FEATURE_COVERAGE:-1}
export MEMNAV_SAMPLING_MODE=${MEMNAV_SAMPLING_MODE:-random_leg}
export MEMNAV_ADD_GOAL_A=${MEMNAV_ADD_GOAL_A:-1}
export MEMNAV_GLIMPSE_POS=${MEMNAV_GLIMPSE_POS:-14}
export MEMNAV_GLIMPSE_NEG=${MEMNAV_GLIMPSE_NEG:-83}
export MEMNAV_GOAL_SLACK=${MEMNAV_GOAL_SLACK:-4}
export MEMNAV_SCENE_SPLIT=${MEMNAV_SCENE_SPLIT:-all}
export MEMNAV_W_RETRIEVAL=${MEMNAV_W_RETRIEVAL:-1.0}
export MEMNAV_W_AUX_POSE=${MEMNAV_W_AUX_POSE:-0.5}

export NAME=${NAME:-memnav_e1_offline}
export BATCH_SIZE=${BATCH_SIZE:-4}
export NUM_WORKERS=${NUM_WORKERS:-4}
export MAX_SAMPLES=${MAX_SAMPLES:-0}
export EVAL_SEED=${EVAL_SEED:-0}
export DATASET_ROLE=${DATASET_ROLE:-train-diagnostic}
export OUTPUT=${OUTPUT:-${LOG_DIR}/${NAME}-${SLURM_JOB_ID}.json}
: "${CHECKPOINT:?Set CHECKPOINT to a memnav.ckpt path}"

mkdir -p "${LOG_DIR}"

echo "host=$(hostname) start=$(date)"
nvidia-smi -L

apptainer exec --nv \
    --overlay "${IMG}:ro" \
    -B /scratch/lg154 \
    -B "/scratch/${USER}" \
    "${BASE_SIF}" \
    bash -c '
        set -euo pipefail
        source "${CONDA_SH}"
        conda activate "${CONDA_ENV}"
        cd "${REPO_ROOT}"
        export PYTHONPATH="${REPO_ROOT}/src/diffusion-policy:${PYTHONPATH:-}"

        [[ -f "${CHECKPOINT}" ]] || { echo "Missing checkpoint: ${CHECKPOINT}"; exit 1; }
        [[ -d "${MEMNAV_FEATURE_ROOT}" ]] || { echo "Missing features: ${MEMNAV_FEATURE_ROOT}"; exit 1; }
        ls "${MEMNAV_ROOT_DIR}" >/dev/null

        extra_args=()
        if (( MAX_SAMPLES > 0 )); then
            extra_args+=(--max-samples "${MAX_SAMPLES}")
        fi

        echo "git_commit=$(git rev-parse HEAD)"
        echo "checkpoint=${CHECKPOINT} batch=${BATCH_SIZE} max_samples=${MAX_SAMPLES}"
        echo "sampling=${MEMNAV_SAMPLING_MODE} goalA=${MEMNAV_ADD_GOAL_A} glimpse=${MEMNAV_GLIMPSE_POS}/${MEMNAV_GLIMPSE_NEG} slack=${MEMNAV_GOAL_SLACK}"
        echo "scene_split=${MEMNAV_SCENE_SPLIT} role=${DATASET_ROLE} w_retrieval=${MEMNAV_W_RETRIEVAL} w_aux_pose=${MEMNAV_W_AUX_POSE}"
        exec python scripts/eval/eval_memnav_offline.py \
            --checkpoint "${CHECKPOINT}" \
            --root-dir "${MEMNAV_ROOT_DIR}" \
            --feature-root "${MEMNAV_FEATURE_ROOT}" \
            --output "${OUTPUT}" \
            --batch-size "${BATCH_SIZE}" \
            --num-workers "${NUM_WORKERS}" \
            --seed "${EVAL_SEED}" \
            --scene-split "${MEMNAV_SCENE_SPLIT}" \
            --dataset-role "${DATASET_ROLE}" \
            "${extra_args[@]}"
    '
