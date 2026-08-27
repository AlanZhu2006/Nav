#!/bin/bash
#SBATCH --job-name=memnav_fixed_eval
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --account=torch_pr_769_tandon_advanced
#SBATCH --partition=a100_tandon,h100_tandon,h200_tandon
#SBATCH --gres=gpu:1
#SBATCH --export=ALL

# Paired checkpoint evaluator.  It uses the same pt1 overlay, schema-v2 W32
# cache tree, mainline dataloader, production MemNav config, and frozen LingBot
# weights as training.  The policy itself runs in eval mode: no positive-anchor
# teacher forcing and no decoder-gate teacher forcing.
set -euo pipefail

IMG=/scratch/lg154/Research/datasets/_overlays/mp3d_revisit_v0_pt1.sqf
BASE_SIF=/share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif
REPO_ROOT=${REPO_ROOT:?REPO_ROOT must name the dedicated personal InternNav checkout}
BASELINE_CKPT=${BASELINE_CKPT:?BASELINE_CKPT is required}
CANDIDATE_CKPT=${CANDIDATE_CKPT:?CANDIDATE_CKPT is required}
EVAL_OUT=${EVAL_OUT:?EVAL_OUT is required}
EVAL_SAMPLES_PER_GROUP=${EVAL_SAMPLES_PER_GROUP:-12}
EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE:-4}
EVAL_TRIALS=${EVAL_TRIALS:-8}
EVAL_SEED=${EVAL_SEED:-20260803}

[[ -f "${REPO_ROOT}/scripts/train_memnav/fixed_checkpoint_eval.py" ]] || {
  echo "ABORT: evaluator missing under ${REPO_ROOT}"; exit 1;
}
[[ -f "${BASELINE_CKPT}" ]] || { echo "ABORT: baseline missing: ${BASELINE_CKPT}"; exit 1; }
[[ -f "${CANDIDATE_CKPT}" ]] || { echo "ABORT: candidate missing: ${CANDIDATE_CKPT}"; exit 1; }
mkdir -p "${REPO_ROOT}/logs/train_memnav" "${EVAL_OUT}"

export MEMNAV_ROOT_DIR=/mp3d_revisit_v0/vln_n1/traj_data
export MEMNAV_FEATURE_ROOT=/scratch/lg154/Research/datasets/mp3d_revisit_v0_feat_flowgate/vln_n1/traj_data
export LINGBOT_REPO=/scratch/lg154/Research/Nav/NavDP/baselines/memnav/lingbot-map
export LINGBOT_WEIGHTS=${LINGBOT_REPO}/weights/lingbot-map-long.pt
export MEMNAV_WINDOW=32
export MEMNAV_NUM_SCALE=8
export MEMNAV_MAX_FRAME_NUM=2048
export MEMNAV_REQUIRE_VERSIONED_CACHE=1
export MEMNAV_MAX_LEGS=0
export MEMNAV_GROUND_SCALE_MAX=6.0
export MEMNAV_STREAM_GROUP=1
export MEMNAV_REPORT_TO=none

echo "host=$(hostname) start=$(date) commit=$(git -C "${REPO_ROOT}" rev-parse HEAD)"
echo "baseline=${BASELINE_CKPT}"
echo "candidate=${CANDIDATE_CKPT}"
echo "out=${EVAL_OUT} groups=${EVAL_SAMPLES_PER_GROUP} batch=${EVAL_BATCH_SIZE} trials=${EVAL_TRIALS} seed=${EVAL_SEED}"
sha256sum "${BASELINE_CKPT}" "${CANDIDATE_CKPT}"
nvidia-smi -L

srun --export=ALL --ntasks=1 --ntasks-per-node=1 \
  --cpus-per-task="${SLURM_CPUS_PER_TASK}" --kill-on-bad-exit=1 \
  apptainer exec --nv \
    --overlay "${IMG}:ro" \
    -B /scratch/lg154 \
    -B /scratch/yz11502 \
    -B "${REPO_ROOT}" \
    "${BASE_SIF}" \
    bash -c '
      set -euo pipefail
      source /scratch/lg154/miniconda3/etc/profile.d/conda.sh
      conda activate /scratch/lg154/conda-envs/memnav
      cd '"${REPO_ROOT}"'
      export PYTHONPATH='"${REPO_ROOT}"'/src/diffusion-policy:${PYTHONPATH:-}
      export MALLOC_ARENA_MAX=2 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
      export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false
      JOB_TMP_ROOT=/tmp/${USER}/memnav_eval_${SLURM_JOB_ID}
      export TMPDIR=${JOB_TMP_ROOT}/tmp XDG_CACHE_HOME=${JOB_TMP_ROOT}/xdg_cache
      export XDG_CONFIG_HOME=${JOB_TMP_ROOT}/xdg_config XDG_STATE_HOME=${JOB_TMP_ROOT}/xdg_state
      mkdir -p "${TMPDIR}" "${XDG_CACHE_HOME}" "${XDG_CONFIG_HOME}" "${XDG_STATE_HOME}"

      python -V
      python -c "import torch, transformers, diffusers, numpy, pandas, pyarrow, cv2; print(\"dependencies OK\", torch.__version__, torch.cuda.is_available())"
      python scripts/train_memnav/gpu_preflight.py
      python -m py_compile scripts/train_memnav/fixed_checkpoint_eval.py
      python scripts/train_memnav/fixed_checkpoint_eval.py \
        --checkpoint flowgate2600='"${BASELINE_CKPT}"' \
        --checkpoint gatecurr600='"${CANDIDATE_CKPT}"' \
        --out '"${EVAL_OUT}"' \
        --samples-per-group '"${EVAL_SAMPLES_PER_GROUP}"' \
        --batch-size '"${EVAL_BATCH_SIZE}"' \
        --trials '"${EVAL_TRIALS}"' \
        --seed '"${EVAL_SEED}"'
    '

echo "done=$(date)"
