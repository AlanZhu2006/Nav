#!/bin/bash
#SBATCH --job-name=sqfs_interndata_n1
#SBATCH --output=/scratch/lg154/Research/Nav/InternNav/logs/mp3d_gen/sqfs_interndata_n1-%j.out
#SBATCH --error=/scratch/lg154/Research/Nav/InternNav/logs/mp3d_gen/sqfs_interndata_n1-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --partition=cs
#SBATCH --account=torch_pr_769_tandon_advanced
#
# Pack the whole InternData-N1 dataset into ONE squashfs image = 1 host inode.
# Measured: vln_n1 alone is 1,563,385 inodes / 309 GB -> essentially the entire 5M-inode
# quota problem. Everything else is trivial (.git 785, .cache 586, preview 359, vln_ce 76).
# We exclude .git and .cache (dev cruft the loader never reads); keep vln_n1, vln_ce,
# preview, README.md, .gitattributes so the mounted view matches the training data.
# -noD: RGB/depth frames are already-compressed jpg/png -> skip data compression (fast,
# ~same size). This solves the INODE quota, not disk space.
#
# After it prints DONE_OK and you have verified the entry count, delete the bare tree:
#   ls -d InternData-N1/vln_n1/_raw/*/*/ | xargs -P 48 -I{} rm -rf "{}"   # then the rest
set -euo pipefail

MKSQ=/share/apps/apptainer/1.5.2/x86_64/utils/bin/mksquashfs
UNSQ=/share/apps/apptainer/1.5.2/x86_64/utils/bin/unsquashfs
SRC=/scratch/lg154/Research/datasets/InternData-N1
OUTDIR=/scratch/lg154/Research/datasets/_overlays
OUT="${OUTDIR}/InternData-N1.sqf"
mkdir -p "${OUTDIR}"
rm -f "${OUT}"

echo "host: $(hostname); start: $(date)"
echo "packing ${SRC} -> ${OUT} (excluding .git .cache)"

# exclusion paths are relative to SRC; -keep-as-directory => image root is InternData-N1/
"${MKSQ}" "${SRC}" "${OUT}" \
    -keep-as-directory -noappend -noD -no-xattrs -processors 16 \
    -wildcards \
    -e .git .cache

echo "=== pack done: $(date) ==="
ls -la "${OUT}"

echo "=== verify: entry count inside the image ==="
NIN=$("${UNSQ}" -l "${OUT}" | wc -l)
echo "unsquashfs -l entry count (incl dirs): ${NIN}"
echo "=== top-level entries inside image ==="
"${UNSQ}" -l "${OUT}" | grep -E '^squashfs-root/InternData-N1/[^/]+$' || true
echo "DONE_OK"
