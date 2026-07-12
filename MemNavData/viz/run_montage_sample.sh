#!/bin/bash
# Sample 3 complete scenes; for each, montage 2 episodes of the 2-leg set and 2 of the
# 3-leg set over the scene BEV navmesh occupancy. CPU-only (viz_traj_montage.py loads the
# .navmesh directly, no GL). Meant to run inside the cs621 interactive allocation via:
#   srun --jobid=<allocid> --overlap bash MemNavData/viz/run_montage_sample.sh
set -euo pipefail
source /scratch/lg154/miniconda3/etc/profile.d/conda.sh
conda activate /scratch/lg154/conda-envs/habitat

VIZ=/scratch/lg154/Research/Nav/MemNavData/viz/viz_traj_montage.py
OUT_ROOT=/scratch/lg154/Research/datasets/mp3d_revisit_v0/vln_n1/traj_data
MP3D_ROOT=/scratch/lg154/Research/datasets/mp3d/mp3d
DEST=/scratch/lg154/Research/Nav/MemNavData/viz_out/sample_montage
mkdir -p "${DEST}"

SCENES=(17DRP5sb8fy JeFG25nYj2p WYY7iVyf5p8)
for s in "${SCENES[@]}"; do
  nm=${MP3D_ROOT}/${s}/${s}.navmesh
  for leg in 2leg 3leg; do
    echo "=== ${s} ${leg} ==="
    python "${VIZ}" \
      --scene_dir "${OUT_ROOT}/mp3d_${leg}/${s}" \
      --navmesh   "${nm}" \
      --n 2 \
      --out "${DEST}/${s}_${leg}.png"
  done
done
echo "ALL_DONE -> ${DEST}"
ls -la "${DEST}"
