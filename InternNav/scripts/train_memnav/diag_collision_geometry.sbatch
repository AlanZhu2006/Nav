#!/bin/bash
#SBATCH --job-name=diag_collision_geom
#SBATCH --output=/scratch/lg154/Research/Nav/InternNav/logs/mp3d_gen/diagcol-%j.out
#SBATCH --error=/scratch/lg154/Research/Nav/InternNav/logs/mp3d_gen/diagcol-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --account=torch_pr_769_tandon_advanced
#SBATCH --partition=a100_tandon,h100_tandon,h200_tandon
#SBATCH --gres=gpu:1
#SBATCH --export=ALL
#
# Phase-0 validation for the geometric trajectory selector: LingBot depth x
# ground_scale vs GT sensor depth, as BEV nearest-obstacle distance + collision
# precision/recall on a waypoint query grid. Runs on the ~30 episodes whose raw
# frames survive on host (no overlay needed). See scripts/diag_collision_geo/diag_collision_geometry.py.
#
# Submit: sbatch scripts/train_memnav/diag_collision_geometry.sbatch
set -euo pipefail

echo "host: $(hostname); start: $(date)"
source /scratch/lg154/miniconda3/etc/profile.d/conda.sh
conda activate /scratch/lg154/conda-envs/memnav
cd /scratch/lg154/Research/Nav/InternNav

# LingBotStream's baked-in default weights path is another machine's home dir —
# point it at the scratch checkout (same values as precompute_mp3d_pt*.sbatch)
export LINGBOT_REPO=/scratch/lg154/Research/Nav/NavDP/baselines/memnav/lingbot-map
export LINGBOT_WEIGHTS=${LINGBOT_REPO}/weights/lingbot-map-long.pt

python scripts/diag_collision_geo/diag_collision_geometry.py \
    --episodes "${EPISODES:-16}" \
    --max_frames "${MAX_FRAMES:-200}" \
    --eval_every "${EVAL_EVERY:-8}" \
    --json_out logs/mp3d_gen/diag_collision_geometry_${SLURM_JOB_ID}.json

echo "=== done: $(date) ==="
