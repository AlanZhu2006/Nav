#!/bin/bash
#SBATCH --job-name=memnav_mp3d
#SBATCH --output=logs/train_memnav/mp3d-%j.out
#SBATCH --error=logs/train_memnav/mp3d-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=8:00:00
#SBATCH --account=torch_pr_769_tandon_advanced
#SBATCH --partition=a100_tandon,h100_tandon,h200_tandon
#SBATCH --gres=gpu:1
#SBATCH --export=ALL
# REQUIRED: complete scripts/train_memnav/README.md before every sbatch submission.
#
# MemNav training on the PACKED mp3d revisit data (scenes 0-53).
#
# Frames live in a read-only squashfs (pt1.sqf); the LingBot KV caches live in a
# SEPARATE writable tree (precompute_mp3d_pt1.sbatch wrote them there via --out_root).
# So we run training INSIDE the same apptainer overlay used for precompute:
#   apptainer exec --overlay pt1.sqf:ro <base.sif>
# Inside the container the frames appear at /mp3d_revisit_v0/vln_n1/traj_data, which is
# MEMNAV_ROOT_DIR; caches are read from MEMNAV_FEATURE_ROOT on host /scratch. The
# Training is deliberately strict: it aborts if either cache is missing for any
# source-ready episode. This prevents the dataset composition from changing silently
# between restarts or while feature precomputation is still running.
#
# CONSISTENCY (critical): the memory partition geometry MUST match the precompute.
# The caches were built with --kv_cache_sliding_window 32 and --max_frame_num 4096, so:
#   MEMNAV_WINDOW=32   -> MemNav_Dataset.window_size AND LingBotStream.window (== KV
#                          manager sliding window). Using 8 here would mis-partition the
#                          scale/history/window regions vs. how anchor_k/v was captured.
#   MEMNAV_NUM_SCALE=8 -> scale frames [0..7].
#   MEMNAV_MAX_FRAME_NUM=4096 -> covers the actual longest packed episode (3954 frames)
#                          and sizes both aggregator and camera-head 3D-RoPE tables.
# These three flow config(memnav.py) -> dataset + LingBotStream via env vars.
#
# Overridable env vars (defaults shown):
#   NAME=memnav_correctness
#   BATCH_SIZE=4
#   EPOCHS=1
#   NUM_WORKERS=4
#   RESUME_FROM_CHECKPOINT=auto
#   MEMNAV_INIT_CHECKPOINT=  (optional compact memnav.ckpt; resets optimizer/step)
#   MEMNAV_LR=1e-4
#   MEMNAV_SEED=0
#   MEMNAV_MAX_TRAIN_STEPS=-1  (-1 = epoch-based; positive = bounded smoke)
#   MEMNAV_SAVE_STEPS=25
#   MEMNAV_SAMPLING_MODE=random_leg (set decision_curriculum for the opt-in arm)
#   MEMNAV_DECISION_CURRICULUM_PROB=0.5
#   MEMNAV_DECISION_LOOKAHEAD_FRAMES=16
#   MEMNAV_DECISION_MIN_REMAINING_FRAMES=128
#   MEMNAV_DECISION_MIN_ANGLE_DEG=45.0
#   MEMNAV_REPORT_TO=wandb   (wandb | tensorboard | none)
#   MEMNAV_W_AUX_DIRECTION=0.2
#   MEMNAV_W_AUX_RANGE=0.0   (default-off experimental range supervision)
#   MEMNAV_AUX_RANGE_BETA=0.1
#   MEMNAV_AUX_RANGE_GRAD_CAP_RATIO=0.0 (0 = legacy gradients; positive = action-safe cap)
#   MEMNAV_ANCHOR_TF_START=1.0
#   MEMNAV_ANCHOR_TF_END=1.0
#   MEMNAV_ANCHOR_TF_DECAY_STEPS=0
#   MEMNAV_W_POSE_RELIABILITY=0.0
#   MEMNAV_POSE_SCALE_WINDOW=64
#
# Submit:
#   sbatch InternNav/scripts/train_memnav/train_memnav_mp3d.sbatch
#   NAME=memnav_mp3d_smoke BATCH_SIZE=2 EPOCHS=1 NUM_WORKERS=0 MEMNAV_REPORT_TO=none \
#       sbatch --export=ALL InternNav/scripts/train_memnav/train_memnav_mp3d.sbatch
set -euo pipefail

REPO_ROOT=${REPO_ROOT:-/scratch/lg154/Research/Nav/InternNav}
IMG=${IMG:-/scratch/lg154/Research/datasets/_overlays/mp3d_revisit_v0_pt1.sqf}
BASE_SIF=${BASE_SIF:-/share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif}
CONDA_SH=${CONDA_SH:-/scratch/lg154/miniconda3/etc/profile.d/conda.sh}
CONDA_ENV=${CONDA_ENV:-/scratch/lg154/conda-envs/memnav}
mkdir -p "${REPO_ROOT}/logs/train_memnav"

# in-container overlay path (frames) + host feature tree (caches)
export MEMNAV_ROOT_DIR=${MEMNAV_ROOT_DIR:-/mp3d_revisit_v0/vln_n1/traj_data}
export MEMNAV_FEATURE_ROOT=${MEMNAV_FEATURE_ROOT:-/scratch/lg154/Research/datasets/mp3d_revisit_v0_feat/vln_n1/traj_data}
export LINGBOT_REPO=${LINGBOT_REPO:-/scratch/lg154/Research/Nav/NavDP/baselines/memnav/lingbot-map}
export LINGBOT_WEIGHTS=${LINGBOT_WEIGHTS:-${LINGBOT_REPO}/weights/lingbot-map-long.pt}
export MEMNAV_DINO_WEIGHTS=${MEMNAV_DINO_WEIGHTS:-${REPO_ROOT}/checkpoints/depth_anything_v2_vits.pth}
export CONDA_SH CONDA_ENV

# memory-partition geometry — MUST match the precompute
export MEMNAV_WINDOW=32
export MEMNAV_NUM_SCALE=8
export MEMNAV_MAX_FRAME_NUM=4096
export MEMNAV_STRICT_FEATURE_COVERAGE=1
export MEMNAV_REQUIRE_VERSIONED_CACHE=${MEMNAV_REQUIRE_VERSIONED_CACHE:-1}
export MEMNAV_EXPECTED_CACHE_SIGNATURE=${MEMNAV_EXPECTED_CACHE_SIGNATURE:-}
export MEMNAV_EXPECTED_CODE_COMMIT=${MEMNAV_EXPECTED_CODE_COMMIT:-}
export MEMNAV_REQUIRE_GENERATED_POSE_CONVENTION=1
export MEMNAV_DATA_SPLIT=${MEMNAV_DATA_SPLIT:-train}
export MEMNAV_VALIDATION_FRACTION=${MEMNAV_VALIDATION_FRACTION:-0.1}
export MEMNAV_SPLIT_SEED=${MEMNAV_SPLIT_SEED:-0}
export MEMNAV_SAMPLING_MODE=${MEMNAV_SAMPLING_MODE:-random_leg}
export MEMNAV_DECISION_CURRICULUM_PROB=${MEMNAV_DECISION_CURRICULUM_PROB:-0.5}
export MEMNAV_DECISION_LOOKAHEAD_FRAMES=${MEMNAV_DECISION_LOOKAHEAD_FRAMES:-16}
export MEMNAV_DECISION_MIN_REMAINING_FRAMES=${MEMNAV_DECISION_MIN_REMAINING_FRAMES:-128}
export MEMNAV_DECISION_MIN_ANGLE_DEG=${MEMNAV_DECISION_MIN_ANGLE_DEG:-45.0}

# training knobs
export NAME=${NAME:-memnav_correctness}
export BATCH_SIZE=${BATCH_SIZE:-4}
export EPOCHS=${EPOCHS:-1}
export NUM_WORKERS=${NUM_WORKERS:-4}
export RESUME_FROM_CHECKPOINT=${RESUME_FROM_CHECKPOINT:-auto}
export MEMNAV_INIT_CHECKPOINT=${MEMNAV_INIT_CHECKPOINT:-}
export MEMNAV_LR=${MEMNAV_LR:-1e-4}
export MEMNAV_SEED=${MEMNAV_SEED:-0}
export MEMNAV_SAVE_STEPS=${MEMNAV_SAVE_STEPS:-25}
export MEMNAV_LOGGING_STEPS=${MEMNAV_LOGGING_STEPS:-10}
export MEMNAV_EVAL_SAMPLES=${MEMNAV_EVAL_SAMPLES:-16}
export MEMNAV_EVAL_STEPS=${MEMNAV_EVAL_STEPS:-25}
export MEMNAV_EVAL_BATCH_SIZE=${MEMNAV_EVAL_BATCH_SIZE:-4}
export MEMNAV_MAX_TRAIN_STEPS=${MEMNAV_MAX_TRAIN_STEPS:--1}
export MEMNAV_W_AUX_DIRECTION=${MEMNAV_W_AUX_DIRECTION:-0.2}
export MEMNAV_W_AUX_RANGE=${MEMNAV_W_AUX_RANGE:-0.0}
export MEMNAV_AUX_RANGE_BETA=${MEMNAV_AUX_RANGE_BETA:-0.1}
export MEMNAV_AUX_RANGE_GRAD_CAP_RATIO=${MEMNAV_AUX_RANGE_GRAD_CAP_RATIO:-0.0}
export MEMNAV_ANCHOR_TF_START=${MEMNAV_ANCHOR_TF_START:-1.0}
export MEMNAV_ANCHOR_TF_END=${MEMNAV_ANCHOR_TF_END:-1.0}
export MEMNAV_ANCHOR_TF_DECAY_STEPS=${MEMNAV_ANCHOR_TF_DECAY_STEPS:-0}
export MEMNAV_USE_POSE_RELIABILITY_CONDITIONING=${MEMNAV_USE_POSE_RELIABILITY_CONDITIONING:-0}
export MEMNAV_W_POSE_RELIABILITY=${MEMNAV_W_POSE_RELIABILITY:-0.0}
export MEMNAV_POSE_SCALE_WINDOW=${MEMNAV_POSE_SCALE_WINDOW:-64}
export MEMNAV_POSE_RELIABILITY_HIDDEN=${MEMNAV_POSE_RELIABILITY_HIDDEN:-16}
export MEMNAV_POSE_RELIABILITY_INIT=${MEMNAV_POSE_RELIABILITY_INIT:-0.99}
export MEMNAV_POSE_RELIABILITY_LR_MULTIPLIER=${MEMNAV_POSE_RELIABILITY_LR_MULTIPLIER:-5.0}
export MEMNAV_REPORT_TO=${MEMNAV_REPORT_TO:-wandb}
export WANDB_PROJECT=${WANDB_PROJECT:-memnav}
export WANDB_NAME=${WANDB_NAME:-${NAME}}
export WANDB_RUN_ID=${WANDB_RUN_ID:-${NAME}}
export WANDB_RESUME=${WANDB_RESUME:-allow}

echo "host: $(hostname); start: $(date)"
nvidia-smi -L || true

apptainer exec --nv \
    --overlay "${IMG}:ro" \
    -B /scratch \
    "${BASE_SIF}" \
    bash -c '
        set -euo pipefail
        ls "${MEMNAV_ROOT_DIR}" >/dev/null || { echo "ABORT: overlay missing ${MEMNAV_ROOT_DIR}"; exit 1; }
        [[ -d "${MEMNAV_FEATURE_ROOT}" ]] || { echo "ABORT: feature tree missing ${MEMNAV_FEATURE_ROOT}"; exit 1; }
        [[ -f "${LINGBOT_WEIGHTS}" ]]     || { echo "ABORT: weights missing ${LINGBOT_WEIGHTS}"; exit 1; }
        [[ -f "${MEMNAV_DINO_WEIGHTS}" ]] || { echo "ABORT: DINO weights missing ${MEMNAV_DINO_WEIGHTS}"; exit 1; }
        if [[ -n "${MEMNAV_INIT_CHECKPOINT}" ]]; then
          [[ -f "${MEMNAV_INIT_CHECKPOINT}" ]] || {
            echo "ABORT: initialization checkpoint missing ${MEMNAV_INIT_CHECKPOINT}" >&2
            exit 1
          }
          NORMALIZED_RESUME=$(printf "%s" "${RESUME_FROM_CHECKPOINT}" | tr "[:upper:]" "[:lower:]")
          case "${NORMALIZED_RESUME}" in
            ""|none|false|0) ;;
            *)
              echo "ABORT: MEMNAV_INIT_CHECKPOINT cannot be combined with resume=${RESUME_FROM_CHECKPOINT}" >&2
              exit 1
              ;;
          esac
        fi
        if [[ "${MEMNAV_REQUIRE_VERSIONED_CACHE}" == "1" && -z "${MEMNAV_EXPECTED_CACHE_SIGNATURE}" ]]; then
          echo "ABORT: versioned cache requires MEMNAV_EXPECTED_CACHE_SIGNATURE" >&2
          exit 1
        fi
        if [[ "${MEMNAV_REQUIRE_VERSIONED_CACHE}" == "1" && -z "${MEMNAV_EXPECTED_CODE_COMMIT}" ]]; then
          echo "ABORT: versioned cache run requires MEMNAV_EXPECTED_CODE_COMMIT" >&2
          exit 1
        fi

        source "${CONDA_SH}"
        conda activate "${CONDA_ENV}"
        cd '"${REPO_ROOT}"'

        [[ -z "$(git status --porcelain --untracked-files=no)" ]] || {
          echo "ABORT: deployed InternNav has tracked modifications" >&2; exit 1;
        }
        [[ -z "$(git -C "${LINGBOT_REPO}" status --porcelain --untracked-files=no)" ]] || {
          echo "ABORT: LingBot checkout has tracked modifications" >&2; exit 1;
        }
        ACTUAL_CODE_COMMIT=$(git rev-parse HEAD)
        if [[ -n "${MEMNAV_EXPECTED_CODE_COMMIT}" && "${ACTUAL_CODE_COMMIT}" != "${MEMNAV_EXPECTED_CODE_COMMIT}" ]]; then
          echo "ABORT: deployed commit ${ACTUAL_CODE_COMMIT} != expected ${MEMNAV_EXPECTED_CODE_COMMIT}" >&2
          exit 1
        fi
        LINGBOT_COMMIT=$(git -C "${LINGBOT_REPO}" rev-parse HEAD)
        echo "code_commit=${ACTUAL_CODE_COMMIT} lingbot_commit=${LINGBOT_COMMIT}"
        sha256sum "${LINGBOT_WEIGHTS}" "${MEMNAV_DINO_WEIGHTS}"
        if [[ -n "${MEMNAV_INIT_CHECKPOINT}" ]]; then
          sha256sum "${MEMNAV_INIT_CHECKPOINT}"
        fi

        export PYTHONPATH="'"${REPO_ROOT}"':'"${REPO_ROOT}"'/src/diffusion-policy:${PYTHONPATH:-}"

        # per-job scratch (isolated TMP/XDG/wandb so concurrent jobs do not collide)
        JOB_TMP_ROOT="/tmp/${USER}/memnav_${SLURM_JOB_ID:-$$}"
        export TMPDIR="${JOB_TMP_ROOT}/tmp" TMP="${TMPDIR}" TEMP="${TMPDIR}"
        export XDG_CACHE_HOME="${JOB_TMP_ROOT}/xdg_cache"
        export XDG_CONFIG_HOME="${JOB_TMP_ROOT}/xdg_config"
        export XDG_STATE_HOME="${JOB_TMP_ROOT}/xdg_state"
        mkdir -p "${TMPDIR}" "${XDG_CACHE_HOME}" "${XDG_CONFIG_HOME}" "${XDG_STATE_HOME}"

        # big-node malloc / thread hygiene (see train_memnav.sbatch rationale)
        export MALLOC_ARENA_MAX=${MALLOC_ARENA_MAX:-2}
        export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
        export MKL_NUM_THREADS=${MKL_NUM_THREADS:-4}
        export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-4}
        export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
        export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}

        if [[ "${MEMNAV_REPORT_TO}" == "wandb" ]]; then
          export WANDB_DIR="${JOB_TMP_ROOT}/wandb_root"
          export WANDB_CACHE_DIR="${JOB_TMP_ROOT}/wandb_cache"
          export WANDB_CONFIG_DIR="${JOB_TMP_ROOT}/wandb_config"
          export WANDB_DATA_DIR="${JOB_TMP_ROOT}/wandb_data"
          mkdir -p "${WANDB_DIR}" "${WANDB_CACHE_DIR}" "${WANDB_CONFIG_DIR}" "${WANDB_DATA_DIR}"
        fi

        echo "===== TRAIN ENV CHECK ====="
        command -v python; python -V
        python -m pip check
        python -c "import torch, transformers, diffusers; print(\"torch\", torch.__version__, \"cuda\", torch.cuda.is_available(), \"transformers\", transformers.__version__, \"diffusers\", diffusers.__version__); assert torch.cuda.is_available()"
        echo "name=${NAME} bs=${BATCH_SIZE} ep=${EPOCHS} max_steps=${MEMNAV_MAX_TRAIN_STEPS} wk=${NUM_WORKERS} lr=${MEMNAV_LR} seed=${MEMNAV_SEED} resume=${RESUME_FROM_CHECKPOINT} init_checkpoint=${MEMNAV_INIT_CHECKPOINT:-none}"
        echo "root=${MEMNAV_ROOT_DIR}"
        echo "feat=${MEMNAV_FEATURE_ROOT}"
        echo "window=${MEMNAV_WINDOW} num_scale=${MEMNAV_NUM_SCALE} max_frame_num=${MEMNAV_MAX_FRAME_NUM}"
        echo "strict_features=${MEMNAV_STRICT_FEATURE_COVERAGE} require_versioned_cache=${MEMNAV_REQUIRE_VERSIONED_CACHE} expected_cache_signature=${MEMNAV_EXPECTED_CACHE_SIGNATURE} expected_code_commit=${MEMNAV_EXPECTED_CODE_COMMIT} require_pose_marker=${MEMNAV_REQUIRE_GENERATED_POSE_CONVENTION}"
        echo "sampling=${MEMNAV_SAMPLING_MODE} decision_prob=${MEMNAV_DECISION_CURRICULUM_PROB} decision_lookahead=${MEMNAV_DECISION_LOOKAHEAD_FRAMES} decision_min_remaining=${MEMNAV_DECISION_MIN_REMAINING_FRAMES} decision_min_angle_deg=${MEMNAV_DECISION_MIN_ANGLE_DEG}"
        echo "aux_direction_weight=${MEMNAV_W_AUX_DIRECTION} aux_range_weight=${MEMNAV_W_AUX_RANGE} aux_range_beta=${MEMNAV_AUX_RANGE_BETA} aux_range_grad_cap_ratio=${MEMNAV_AUX_RANGE_GRAD_CAP_RATIO} anchor_tf=${MEMNAV_ANCHOR_TF_START}->${MEMNAV_ANCHOR_TF_END}@${MEMNAV_ANCHOR_TF_DECAY_STEPS}"
        echo "pose_scale_window=${MEMNAV_POSE_SCALE_WINDOW} pose_rel_conditioning=${MEMNAV_USE_POSE_RELIABILITY_CONDITIONING} pose_rel_weight=${MEMNAV_W_POSE_RELIABILITY} pose_rel_hidden=${MEMNAV_POSE_RELIABILITY_HIDDEN} pose_rel_init=${MEMNAV_POSE_RELIABILITY_INIT} pose_rel_lr_mult=${MEMNAV_POSE_RELIABILITY_LR_MULTIPLIER}"
        echo "dino_weights=${MEMNAV_DINO_WEIGHTS}"

        TRAIN_ARGS=(
          --name "${NAME}"
          --model-name memnav
          --batch-size "${BATCH_SIZE}"
          --num-workers "${NUM_WORKERS}"
          --epochs "${EPOCHS}"
          --lr "${MEMNAV_LR}"
          --seed "${MEMNAV_SEED}"
          --root-dir "${MEMNAV_ROOT_DIR}"
          --resume-from-checkpoint "${RESUME_FROM_CHECKPOINT}"
        )
        if [[ -n "${MEMNAV_INIT_CHECKPOINT}" ]]; then
          TRAIN_ARGS+=(--ckpt-to-load "${MEMNAV_INIT_CHECKPOINT}")
        fi
        exec python scripts/train/train.py "${TRAIN_ARGS[@]}"
    '

echo "=== done: $(date) ==="
