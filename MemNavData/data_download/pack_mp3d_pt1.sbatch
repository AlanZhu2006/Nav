#!/bin/bash
#SBATCH --job-name=sqfs_mp3d_pt1
#SBATCH --output=/scratch/lg154/Research/Nav/InternNav/logs/mp3d_gen/sqfs_pt1-%j.out
#SBATCH --error=/scratch/lg154/Research/Nav/InternNav/logs/mp3d_gen/sqfs_pt1-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --partition=cs
#SBATCH --account=torch_pr_769_tandon_advanced
#
# Pack the finished mp3d_revisit_v0 scenes (task indices 0-53) into a single squashfs image,
# EXCLUDING the 4 partial scenes (task idx 54-57, crashed mid-write) so pt1.sqf stays disjoint
# from the pt2.sqf we'll build after regenerating scenes 54-89. One .sqf = 1 host inode.
# -noD: RGB/depth are already-compressed jpg/png, so skip data compression -> fast, ~same size.
# Solving the INODE quota (5M cap, per-frame images), not disk space.
set -euo pipefail

MKSQ=/share/apps/apptainer/1.5.2/x86_64/utils/bin/mksquashfs
UNSQ=/share/apps/apptainer/1.5.2/x86_64/utils/bin/unsquashfs
SRC=/scratch/lg154/Research/datasets/mp3d_revisit_v0
OUTDIR=/scratch/lg154/Research/datasets/_overlays
OUT="${OUTDIR}/mp3d_revisit_v0_pt1.sqf"
mkdir -p "${OUTDIR}"
rm -f "${OUT}"

echo "host: $(hostname); start: $(date)"
echo "packing ${SRC} -> ${OUT} (excluding 4 partial scenes)"

# exclusion paths are relative to SRC
"${MKSQ}" "${SRC}" "${OUT}" \
    -keep-as-directory -noappend -noD -no-xattrs -processors 16 \
    -wildcards \
    -e vln_n1/traj_data/mp3d_2leg/ac26ZMwG7aT \
       vln_n1/traj_data/mp3d_2leg/b8cTxDM8gDG \
       vln_n1/traj_data/mp3d_2leg/cV4RVeZvu5T \
       vln_n1/traj_data/mp3d_2leg/dhjEzFoUFzH \
       vln_n1/traj_data/mp3d_3leg/ac26ZMwG7aT \
       vln_n1/traj_data/mp3d_3leg/b8cTxDM8gDG \
       vln_n1/traj_data/mp3d_3leg/cV4RVeZvu5T \
       vln_n1/traj_data/mp3d_3leg/dhjEzFoUFzH

echo "=== pack done: $(date) ==="
ls -la "${OUT}"

echo "=== verify: entries inside the image ==="
NIN=$("${UNSQ}" -l "${OUT}" | wc -l)
echo "unsquashfs -l entry count (incl dirs): ${NIN}"
echo "=== scene dirs inside image ==="
echo "2leg scenes: $("${UNSQ}" -l "${OUT}" | grep -cE '/mp3d_2leg/[^/]+$' || true)"
echo "3leg scenes: $("${UNSQ}" -l "${OUT}" | grep -cE '/mp3d_3leg/[^/]+$' || true)"
echo "DONE_OK"
