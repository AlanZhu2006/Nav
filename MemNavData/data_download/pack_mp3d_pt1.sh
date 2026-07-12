#!/bin/bash
# Pack finished mp3d_revisit_v0 scenes (task idx 0-53) into one squashfs image, excluding the
# 4 partial scenes (idx 54-57, crashed mid-write) so pt1.sqf stays disjoint from the later pt2.
# -noD: jpg/png already compressed -> skip data compression (fast, ~same size). Solving INODE quota.
set -euo pipefail
MKSQ=/share/apps/apptainer/1.5.2/x86_64/utils/bin/mksquashfs
UNSQ=/share/apps/apptainer/1.5.2/x86_64/utils/bin/unsquashfs
SRC=/scratch/lg154/Research/datasets/mp3d_revisit_v0
OUTDIR=/scratch/lg154/Research/datasets/_overlays
OUT="${OUTDIR}/mp3d_revisit_v0_pt1.sqf"
mkdir -p "${OUTDIR}"; rm -f "${OUT}"

echo "host: $(hostname); start: $(date)"
"${MKSQ}" "${SRC}" "${OUT}" \
    -keep-as-directory -noappend -noD -no-xattrs -processors 4 -wildcards \
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
echo "2leg scenes in image: $("${UNSQ}" -l "${OUT}" | grep -cE '/mp3d_2leg/[^/]+$' || true)"
echo "3leg scenes in image: $("${UNSQ}" -l "${OUT}" | grep -cE '/mp3d_3leg/[^/]+$' || true)"
echo "DONE_OK"
