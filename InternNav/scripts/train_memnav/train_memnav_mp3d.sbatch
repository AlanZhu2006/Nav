#!/bin/bash
#SBATCH --job-name=memnav_mp3d
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --account=torch_pr_769_tandon_advanced
#SBATCH --partition=a100_tandon,h100_tandon,h200_tandon
#SBATCH --gres=gpu:2
#SBATCH --requeue
#SBATCH --export=ALL
#
# MemNav training on the PACKED mp3d revisit data (scenes 0-53).
#
# Frames live in a read-only squashfs (pt1.sqf); the LingBot KV caches live in a
# SEPARATE writable tree (precompute_mp3d_pt1.sbatch wrote them there via --out_root).
# So we run training INSIDE the same apptainer overlay used for precompute:
#   apptainer exec --overlay pt1.sqf:ro <base.sif>
# Inside the container the frames appear at /mp3d_revisit_v0/vln_n1/traj_data, which is
# MEMNAV_ROOT_DIR; caches are read from MEMNAV_FEATURE_ROOT on host /scratch. The
# dataset only enumerates trajectories that HAVE a cache (memnav_dataset_lerobot.py
# skips uncached ones). The pt1 cam caches are COMPLETE (1919 trajectories: 736 two-leg
# + 1183 three-leg <=2048 frames, each with the whole-episode ground_h_est for
# scale_mode='ground'); 25 long (>2048-frame) three-leg episodes were intentionally
# dropped. Both training fixes are live: per-trajectory ground-scale rescale of t_rel
# (scale_mode='ground', policy default) and the rank_loss NaN-overflow fix (9b93877).
#
# CONSISTENCY (critical): the memory partition geometry MUST match the precompute.
# The caches were built with --kv_cache_sliding_window 32 and --max_frame_num 2048, so:
#   MEMNAV_WINDOW=32   -> MemNav_Dataset.window_size AND LingBotStream.window (== KV
#                          manager sliding window). Using 8 here would mis-partition the
#                          scale/history/window regions vs. how anchor_k/v was captured.
#   MEMNAV_NUM_SCALE=8 -> scale frames [0..7].
#   MEMNAV_MAX_FRAME_NUM=2048 -> sizes the 3D-RoPE table for long (>1024-frame) 3leg
#                          episodes; LingBotStream now extends the camera-head rope too.
# These three flow config(memnav.py) -> dataset + LingBotStream via env vars.
#
# Runs 2-GPU DDP (torchrun --nproc_per_node=2) for ~2x throughput. BATCH_SIZE is
# per-GPU, so the effective global batch is 2x BATCH_SIZE. NUM_WORKERS is also per-GPU
# (2 x 8 = 16 = cpus-per-task). MEMNAV_MAX_LEGS=2 trains on two-leg episodes only.
#
# Overridable env vars (defaults shown):
#   NAME=memnav_mp3d
#   BATCH_SIZE=4             (per-GPU; effective global batch = 4 x NPROC = 8)
#   LR=1e-4
#   EPOCHS=10
#   NUM_WORKERS=8            (per-GPU)
#   NPROC=2                  (GPUs / DDP ranks; must be <= --gres=gpu:N)
#   MEMNAV_MAX_LEGS=2        (2 = two-leg episodes only; unset/0 = all legs)
#   MEMNAV_GROUND_SCALE_MAX=6.0  (ground-scale clamp ceiling; corrected per-episode scale
#                            estimates above this are CLAMPED to it. 6.0 leaves ~95.6% of
#                            pt1 eps untouched; the ~4% above clamp to 6.0.)
#   MEMNAV_STREAM_GROUP=1    (per-sample streaming; >1 batches G streams in encode_memory)
#   MEMNAV_REQUIRE_VERSIONED_CACHE=1 (reject mixed/legacy sparse-cache layouts)
#   MEMNAV_GATE_TEACHER_START=1.0  (decoder uses GT revisit gate initially)
#   MEMNAV_GATE_TEACHER_END=0.0    (fully predicted gate after curriculum)
#   MEMNAV_GATE_TEACHER_STEPS=1000 (linear handoff in optimizer steps)
#   MEMNAV_INIT_CKPT=''        (optional weights-only warm start; use a fresh NAME)
#   MEMNAV_GOAL_A_MIN_K=''     (40 enables early Goal-A inference-state coverage)
#   MEMNAV_GOAL_SWAP_WEIGHT=0  (counterfactual Novel denoising; 0 = historical)
#   MEMNAV_GOAL_SWAP_MARGIN=.05
#   MEMNAV_GOAL_SWAP_MIN_ANGLE_DEG=30
#   MEMNAV_PREFLIGHT_ONLY=0    (1 = one real batch forward/backward, then exit)
#   MEMNAV_REPORT_TO=wandb   (wandb | tensorboard | none)
#
# Submit:
#   sbatch InternNav/scripts/train_memnav/train_memnav_mp3d.sbatch
#   NAME=memnav_mp3d_smoke BATCH_SIZE=2 EPOCHS=1 NUM_WORKERS=0 MEMNAV_REPORT_TO=none \
#       sbatch --export=ALL InternNav/scripts/train_memnav/train_memnav_mp3d.sbatch
set -euo pipefail

IMG=/scratch/lg154/Research/datasets/_overlays/mp3d_revisit_v0_pt1.sqf
BASE_SIF=/share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif
# Source/output worktree.  Never silently fall back to the shared main checkout:
# this experiment must execute the committed branch containing the curriculum.
REPO_ROOT=${REPO_ROOT:-${SLURM_SUBMIT_DIR:-/scratch/lg154/Research/Nav-axis-uturn/InternNav}}
[[ -f "${REPO_ROOT}/scripts/train/train.py" ]] || {
  echo "ABORT: REPO_ROOT is not an InternNav checkout: ${REPO_ROOT}"; exit 1;
}
mkdir -p "${REPO_ROOT}/logs/train_memnav"

# in-container overlay path (frames) + host feature tree (caches).  The former
# dense ``mp3d_revisit_v0_feat`` tree was retired; the current schema-v2 caches
# use flow-gated sparse keyframes and live in ``*_feat_flowgate``.
export MEMNAV_ROOT_DIR=/mp3d_revisit_v0/vln_n1/traj_data
export MEMNAV_FEATURE_ROOT=${MEMNAV_FEATURE_ROOT:-/scratch/lg154/Research/datasets/mp3d_revisit_v0_feat_flowgate/vln_n1/traj_data}
export LINGBOT_REPO=/scratch/lg154/Research/Nav/NavDP/baselines/memnav/lingbot-map
export LINGBOT_WEIGHTS=${LINGBOT_REPO}/weights/lingbot-map-long.pt

# memory-partition geometry — MUST match the precompute
export MEMNAV_WINDOW=32
export MEMNAV_NUM_SCALE=8
export MEMNAV_MAX_FRAME_NUM=2048
# This job is specifically for the versioned pt1 sparse cache tree.  Fail closed if
# a path typo or mixed tree would otherwise silently fall back to dense row==frame
# arithmetic (the exact long-horizon interval bug fixed earlier).
export MEMNAV_REQUIRE_VERSIONED_CACHE=${MEMNAV_REQUIRE_VERSIONED_CACHE:-1}

# Batch G trajectory streams on the aggregator's batch dim inside encode_memory
# (window_forward + goal_append_warm). 1 = original one-at-a-time loop; >1 collapses
# the per-sample Python loop into the batch axis for ~Gx fewer transformer forwards,
# at ~Gx resident KV (the batch is processed in chunks of G so peak memory is bounded).
# Default G=1: in practice G>1 slowed our runs (fatter forwards), and the per-sample
# loop keeps the GPU continuously busy (high util). Numerically equivalent to G=4
# (validated: scripts/diag_batched_window.py — cos>0.9999, poses match).
export MEMNAV_STREAM_GROUP=${MEMNAV_STREAM_GROUP:-1}

# episode leg filter: 2 = train on two-leg episodes only (drops three-leg goal-C
# episodes). Read by scripts/train/configs/memnav.py -> MemNav_Dataset(max_legs=...).
export MEMNAV_MAX_LEGS=${MEMNAV_MAX_LEGS:-2}

# ground-scale clamp ceiling (scale_mode='ground'): corrected per-episode scale estimates
# above this are CLAMPED to it (ground_scale_from_h_est clamps, no longer rejects). Default
# 6.0 (was 4.0) leaves ~95.6% of pt1 episodes untouched; the ~4% above clamp to 6.0.
# memnav.py -> IlCfg -> MemNavNet.ground_scale_range -> ground_scale_from_h_est(scale_range=).
export MEMNAV_GROUND_SCALE_MAX=${MEMNAV_GROUND_SCALE_MAX:-6.0}

# checkpoint every N optimizer steps (memnav.py -> save_strategy='steps'). train.py
# auto-resumes from the latest checkpoint in checkpoints/<NAME>/ckpts, so a re-submitted
# job continues where a timed-out one stopped. save_total_limit=8 bounds retained ckpts.
export MEMNAV_SAVE_STEPS=${MEMNAV_SAVE_STEPS:-100}

# Prevent the untrained revisit gate from hiding the already teacher-anchored revisit
# token from action loss at the start of training.  The ratio reaches zero by step 1000,
# so the latter part of training exactly matches inference-time predicted-gate routing.
export MEMNAV_GATE_TEACHER_START=${MEMNAV_GATE_TEACHER_START:-1.0}
export MEMNAV_GATE_TEACHER_END=${MEMNAV_GATE_TEACHER_END:-0.0}
export MEMNAV_GATE_TEACHER_STEPS=${MEMNAV_GATE_TEACHER_STEPS:-1000}
export MEMNAV_GATE_FUSION=${MEMNAV_GATE_FUSION:-complementary}
export MEMNAV_RETRIEVAL_TOP1_WEIGHT=${MEMNAV_RETRIEVAL_TOP1_WEIGHT:-0.0}
export MEMNAV_RETRIEVAL_TOP1_MARGIN=${MEMNAV_RETRIEVAL_TOP1_MARGIN:-0.2}
export MEMNAV_INIT_CKPT=${MEMNAV_INIT_CKPT:-}
export MEMNAV_GOAL_A_MIN_K=${MEMNAV_GOAL_A_MIN_K:-}
export MEMNAV_GOAL_SWAP_WEIGHT=${MEMNAV_GOAL_SWAP_WEIGHT:-0.0}
export MEMNAV_GOAL_SWAP_MARGIN=${MEMNAV_GOAL_SWAP_MARGIN:-0.05}
export MEMNAV_GOAL_SWAP_MIN_ANGLE_DEG=${MEMNAV_GOAL_SWAP_MIN_ANGLE_DEG:-30.0}
export MEMNAV_PREFLIGHT_ONLY=${MEMNAV_PREFLIGHT_ONLY:-0}
# -1 = normal epoch-based training. The guarded preflight sets 1 and executes the
# exact production train.py entrypoint for one optimizer step after gradient checks.
export MEMNAV_TRAIN_MAX_STEPS=${MEMNAV_TRAIN_MAX_STEPS:--1}

# training knobs
export NAME=${NAME:-memnav_mp3d}
export BATCH_SIZE=${BATCH_SIZE:-4}             # per-GPU; effective = BATCH_SIZE x NPROC
export LR=${LR:-1e-4}
export EPOCHS=${EPOCHS:-10}
export NUM_WORKERS=${NUM_WORKERS:-8}
export NPROC=${NPROC:-2}                       # DDP ranks (GPUs); keep <= --gres=gpu:N
export MEMNAV_REPORT_TO=${MEMNAV_REPORT_TO:-wandb}
export WANDB_PROJECT=${WANDB_PROJECT:-memnav}
export WANDB_NAME=${WANDB_NAME:-${NAME}}

echo "host: $(hostname); start: $(date)"
echo "restart_count=${SLURM_RESTART_COUNT:-0}"
nvidia-smi -L || true

# On shared EXCLUSIVE_PROCESS a100_tandon nodes SLURM occasionally deals us a GPU
# that already has another job's context; torch dies at set_device (CUDA err 46).
# The in-container preflight (below) exits 42 in that case; requeue for a fresh
# allocation instead of burning the run. Bounded to avoid an infinite loop on a
# real (non-transient) bug.
MAX_REQUEUES=3

set +e
# Launch through `srun` (single task) — NOT as a bare child of the batch step.
# srun re-establishes the GRES cgroup + CUDA_VISIBLE_DEVICES binding for the step,
# which is what keeps each DDP rank pinned to a GPU we actually own on shared
# a100_tandon nodes. Running torchrun directly (no srun) let a rank grab a busy
# neighbor GPU -> set_device CUDA err 46. This matches the proven-good openpi
# launch (openpi/scripts/sbatch_lg/train_pi0_libero.sbatch: srun ... torch.distributed.run).
srun --export=ALL --ntasks=1 --ntasks-per-node=1 \
     --cpus-per-task="${SLURM_CPUS_PER_TASK}" --kill-on-bad-exit=1 \
  apptainer exec --nv \
    --overlay "${IMG}:ro" \
    -B /scratch/lg154 \
    -B "${REPO_ROOT}" \
    "${BASE_SIF}" \
    bash -c '
        set -euo pipefail
        ls "${MEMNAV_ROOT_DIR}" >/dev/null || { echo "ABORT: overlay missing ${MEMNAV_ROOT_DIR}"; exit 1; }
        [[ -d "${MEMNAV_FEATURE_ROOT}" ]] || { echo "ABORT: feature tree missing ${MEMNAV_FEATURE_ROOT}"; exit 1; }
        [[ -f "${LINGBOT_WEIGHTS}" ]]     || { echo "ABORT: weights missing ${LINGBOT_WEIGHTS}"; exit 1; }
        [[ -z "${MEMNAV_INIT_CKPT}" || -f "${MEMNAV_INIT_CKPT}" ]] || {
          echo "ABORT: warm-start checkpoint missing ${MEMNAV_INIT_CKPT}"; exit 1;
        }

        source /scratch/lg154/miniconda3/etc/profile.d/conda.sh
        conda activate /scratch/lg154/conda-envs/memnav
        cd '"${REPO_ROOT}"'

        export PYTHONPATH="'"${REPO_ROOT}"'/src/diffusion-policy:${PYTHONPATH:-}"

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
        python -c "import torch; print(\"torch\", torch.__version__, \"cuda\", torch.cuda.is_available())"
        echo "name=${NAME} bs/gpu=${BATCH_SIZE} lr=${LR} ep=${EPOCHS} wk/gpu=${NUM_WORKERS} nproc=${NPROC}"
        echo "root=${MEMNAV_ROOT_DIR}"
        echo "feat=${MEMNAV_FEATURE_ROOT}"
        echo "window=${MEMNAV_WINDOW} num_scale=${MEMNAV_NUM_SCALE} max_frame_num=${MEMNAV_MAX_FRAME_NUM} stream_group=${MEMNAV_STREAM_GROUP} max_legs=${MEMNAV_MAX_LEGS} versioned_cache=${MEMNAV_REQUIRE_VERSIONED_CACHE} save_steps=${MEMNAV_SAVE_STEPS}"
        echo "gate_teacher=${MEMNAV_GATE_TEACHER_START}->${MEMNAV_GATE_TEACHER_END} over ${MEMNAV_GATE_TEACHER_STEPS} steps"
        echo "gate_fusion=${MEMNAV_GATE_FUSION} retrieval_top1_weight=${MEMNAV_RETRIEVAL_TOP1_WEIGHT} margin=${MEMNAV_RETRIEVAL_TOP1_MARGIN}"
        echo "goal_a_min_k=${MEMNAV_GOAL_A_MIN_K:-<historical>} goal_swap_weight=${MEMNAV_GOAL_SWAP_WEIGHT} goal_swap_margin=${MEMNAV_GOAL_SWAP_MARGIN} goal_swap_min_angle=${MEMNAV_GOAL_SWAP_MIN_ANGLE_DEG}deg"
        echo "init_ckpt=${MEMNAV_INIT_CKPT:-<fresh>}"
        echo "entrypoint_max_steps=${MEMNAV_TRAIN_MAX_STEPS}"
        nvidia-smi -L || true

        # CPU-side dependency/config/cache/output preflight.  Keep this separate from
        # the CUDA-context probe below: dependency failures are deterministic and must
        # fail the job, whereas transient busy GPUs are eligible for bounded requeue.
        python scripts/train_memnav/dependency_preflight.py

        # Preflight: prove every allocated GPU can create a CUDA context BEFORE we
        # commit to the run. On a busy/exclusive GPU this exits 1 -> we return 42 so
        # the host requeues for a fresh allocation instead of dying mid-train.
        python scripts/train_memnav/gpu_preflight.py || exit 42

        TRAIN_CMD=(
          torchrun --standalone --nnodes=1 --nproc_per_node="${NPROC}"
          scripts/train/train.py
          --name "${NAME}"
          --model-name memnav
          --batch-size "${BATCH_SIZE}"
          --lr "${LR}"
          --num-workers "${NUM_WORKERS}"
          --epochs "${EPOCHS}"
          --root-dir "${MEMNAV_ROOT_DIR}"
        )

        if [[ "${MEMNAV_PREFLIGHT_ONLY}" == "1" ]]; then
          echo "===== ONE-BATCH TRAINING PREFLIGHT ====="
          python scripts/train_memnav/full_training_preflight.py
          [[ "${MEMNAV_TRAIN_MAX_STEPS}" == "1" ]] || {
            echo "ABORT: preflight must run production entrypoint with MEMNAV_TRAIN_MAX_STEPS=1"
            exit 1
          }
          echo "===== PRODUCTION ENTRYPOINT ONE-STEP PREFLIGHT ====="
          "${TRAIN_CMD[@]}"
          echo "===== ENTRYPOINT-PREFLIGHT PASS ====="
          echo "===== PREFLIGHT-ONLY PASS ====="
          exit 0
        fi

        # 2-GPU DDP. --standalone runs a single-node c10d rendezvous on a free port and
        # exports RANK/LOCAL_RANK/WORLD_SIZE/MASTER_ADDR/MASTER_PORT, which train.py reads.
        exec "${TRAIN_CMD[@]}"
    '
rc=$?
set -e

# rc 42 == GPU preflight failed (busy/exclusive GPU in this allocation). Requeue
# for a clean set of GPUs, up to MAX_REQUEUES, then give up so a real bug surfaces.
if [[ "${rc}" -eq 42 ]]; then
    n="${SLURM_RESTART_COUNT:-0}"
    if [[ "${n}" -lt "${MAX_REQUEUES}" ]]; then
        echo "GPU preflight failed on $(hostname); requeue #$((n+1))/${MAX_REQUEUES}"
        scontrol requeue "${SLURM_JOB_ID}"
        exit 0
    fi
    echo "GPU preflight failed after ${MAX_REQUEUES} requeues; giving up (likely not transient)"
    exit 1
fi

echo "=== done: $(date) === (rc=${rc})"
exit "${rc}"
