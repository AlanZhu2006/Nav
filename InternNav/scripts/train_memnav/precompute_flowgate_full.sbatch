#!/bin/bash
#SBATCH --job-name=flowgate_full
#SBATCH --output=/scratch/lg154/Research/Nav/InternNav/logs/mp3d_gen/flowfull-%A_%a.out
#SBATCH --error=/scratch/lg154/Research/Nav/InternNav/logs/mp3d_gen/flowfull-%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --account=torch_pr_769_tandon_advanced
#SBATCH --partition=h100_tandon,h200_tandon
#SBATCH --gres=gpu:1
#SBATCH --export=ALL
#
# FULL flow-gated LingBot cache generation for the pt1 episodes, with the
# per-episode threshold policy from the 2026-07-28 sweep (job 14888759): smallest
# threshold whose fitted keyframe count (kf ~= a*N + b) fits a 270-kf budget
# (user directive: total memory length = 8 scale + kf < 280; LingBot quality
# drops somewhere around 260-280):
#   thr 20px: N <= 702      (1305 eps, 557k frames)   flowfeat_tier20.txt
#   thr 25px: 703-877       ( 230 eps, 180k frames)   flowfeat_tier25.txt
#   thr 30px: 878-1075      ( 186 eps, 180k frames)   flowfeat_tier30.txt
#   thr 40px: 1076-1506     ( 151 eps, 190k frames)   flowfeat_tier40.txt
#   thr 50px: 1507-2048     (  47 eps,  80k frames)   flowfeat_tier50.txt
# Tier50 (task 12) is EXTRAPOLATED (sweep topped at 40px) — submit it only after
# the flow50 validation arm on the sweep's longest episodes scores clean.
# All tiers write ONE shared tree (disjoint episode sets; per-episode provenance
# stores its own flow_threshold). Full precompute + --skip_scale (--cam_only is
# invalid in flow mode); depth-head call deduped gate<->ground (verified
# bit-identical vs pre-dedupe sweep output, job 14895902).
# Episodes >2048 frames stay excluded (same as the dense training set).
#
# Array map (14 tasks): 0-5 = tier20 shards 0-5/6; 6-7 = tier25 0-1/2;
#                       8-9 = tier30 0-1/2; 10-11 = tier40 0-1/2; 12 = tier50 0/1;
#                       13 = tier60 rework (9 tier50 eps whose actual kf > 270;
#                            DELETE their 50px caches before submitting).
# Submit:  sbatch --array=0-11 scripts/train_memnav/precompute_flowgate_full.sbatch
# then, once flow50 validates:  sbatch --array=12 ...
# then, after the tier50 kf audit:  sbatch --array=13 ...
# Idempotent: re-submit failed tasks freely (skips episodes with finished caches).
set -euo pipefail

TIER=(20 20 20 20 20 20 25 25 30 30 40 40 50 60)
NSH=( 6  6  6  6  6  6  2  2  2  2  2  2  1  1)
SH=(  0  1  2  3  4  5  0  1  0  1  0  1  0  0)
IDX="${SLURM_ARRAY_TASK_ID:?run as an array job, --array=0-11 (12 = tier50 after validation, 13 = tier60 rework)}"
export THR="${TIER[$IDX]}"
export NUM_SHARDS="${NSH[$IDX]}"
export SHARD="${SH[$IDX]}"
export LIMIT="${LIMIT:-0}"

IMG=/scratch/lg154/Research/datasets/_overlays/mp3d_revisit_v0_pt1.sqf
BASE_SIF=/share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif
export ROOT_DIRS=/mp3d_revisit_v0/vln_n1/traj_data
export OUT_ROOT=/scratch/lg154/Research/datasets/mp3d_revisit_v0_feat_flowgate/vln_n1/traj_data
export REPO=/scratch/lg154/Research/Nav/NavDP/baselines/memnav/lingbot-map
export WEIGHTS=${REPO}/weights/lingbot-map-long.pt
export MANIFEST="/scratch/lg154/Research/Nav/InternNav/scripts/train_memnav/flowfeat_tier${THR}.txt"

echo "host: $(hostname); tier thr=${THR}px shard ${SHARD}/${NUM_SHARDS}; start: $(date)"
mkdir -p "${OUT_ROOT}"

apptainer exec --nv \
    --overlay "${IMG}:ro" \
    -B /scratch/lg154 \
    "${BASE_SIF}" \
    bash -c '
        set -euo pipefail
        ls "${ROOT_DIRS}" >/dev/null || { echo "ABORT: overlay missing ${ROOT_DIRS}"; exit 1; }
        source /scratch/lg154/miniconda3/etc/profile.d/conda.sh
        conda activate /scratch/lg154/conda-envs/memnav
        cd /scratch/lg154/Research/Nav/InternNav
        python scripts/dataset_converters/precompute_lingbot_features.py \
            --root_dirs "${ROOT_DIRS}" \
            --out_root  "${OUT_ROOT}" \
            --lingbot_repo "${REPO}" \
            --weights "${WEIGHTS}" \
            --trajectory_list "${MANIFEST}" \
            --flow_threshold "${THR}" --max_non_keyframe_gap 30 \
            --skip_scale --use_sdpa \
            --kv_cache_sliding_window 32 --num_scale_frames 8 \
            --max_frame_num 2048 \
            --num_shards "${NUM_SHARDS}" --shard "${SHARD}" \
            --limit "${LIMIT}"
    '

echo "=== tier ${THR} shard ${SHARD} done: $(date) ==="
