#!/bin/bash
#SBATCH --job-name=viz_collision_sel
#SBATCH --output=/scratch/lg154/Research/Nav/InternNav/logs/mp3d_gen/vizcol-%j.out
#SBATCH --error=/scratch/lg154/Research/Nav/InternNav/logs/mp3d_gen/vizcol-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --account=torch_pr_769_tandon_advanced
#SBATCH --partition=a100_tandon,h100_tandon,h200_tandon
#SBATCH --gres=gpu:1
#SBATCH --export=ALL
#
# One-frame forward-pass visualization of the geometric trajectory selector
# (scripts/diag_collision_geo/viz_collision_select.py).
# Override via env: TRAJ, FRAME, GOAL_FRAME.
# Submit: sbatch scripts/train_memnav/viz_collision_select.sbatch
set -euo pipefail

echo "host: $(hostname); start: $(date)"
source /scratch/lg154/miniconda3/etc/profile.d/conda.sh
conda activate /scratch/lg154/conda-envs/memnav
cd /scratch/lg154/Research/Nav/InternNav

export LINGBOT_REPO=/scratch/lg154/Research/Nav/NavDP/baselines/memnav/lingbot-map
export LINGBOT_WEIGHTS=${LINGBOT_REPO}/weights/lingbot-map-long.pt

python scripts/diag_collision_geo/viz_collision_select.py \
    ${TRAJ:+--traj "$TRAJ"} \
    --frame "${FRAME:-160}" \
    ${GOAL_FRAME:+--goal_frame "$GOAL_FRAME"}

echo "=== done: $(date) ==="
