#!/bin/bash
#SBATCH --job-name=sqfs_mp3d_pt2
#SBATCH --output=/scratch/lg154/Research/Nav/InternNav/logs/mp3d_gen/sqfs_pt2-%j.out
#SBATCH --error=/scratch/lg154/Research/Nav/InternNav/logs/mp3d_gen/sqfs_pt2-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=08:00:00
#SBATCH --partition=cs
#SBATCH --account=torch_pr_769_tandon_advanced
#
# Pack mp3d_revisit_v0 scenes 54-89 (36 scenes) into pt2.sqf.
#
# WHY THE INDIRECTION: as of ~2026-07-12, mksquashfs can NO LONGER write its output onto
# /scratch (VAST) — the fragment-reader threads fail to reopen the destination:
#   "FATAL ERROR:frag_thrd: can't open destination for reading"
# This reproduces for a 2-file test, with -processors 1, and with -no-fragments, so no flag
# avoids it (it worked on VAST on 2026-07-08; VAST behavior changed since). READING the source
# from VAST is fine — only the OUTPUT can't live on VAST. So we build the image in node-local
# RAM (/dev/shm; cs nodes are diskless with 513 GB RAM) then cp it to /scratch (a plain
# sequential write, which VAST handles fine). --mem=200G because the ~95 GB tmpfs image is
# charged to the job's memory cgroup.
#
# Excludes idx48/50 (YmJkqBEsHnH/YFuZgdQ5vWj) — already deleted from disk, kept defensively.
# -noD: jpg/png already compressed. NON-DESTRUCTIVE: only creates pt2.sqf; prune source separately.
set -euo pipefail

MKSQ=/share/apps/apptainer/1.5.2/x86_64/utils/bin/mksquashfs
UNSQ=/share/apps/apptainer/1.5.2/x86_64/utils/bin/unsquashfs
SRC=/scratch/lg154/Research/datasets/mp3d_revisit_v0
OUTDIR=/scratch/lg154/Research/datasets/_overlays
FINAL="${OUTDIR}/mp3d_revisit_v0_pt2.sqf"
LOCAL="/dev/shm/mp3d_revisit_v0_pt2.${SLURM_JOB_ID:-$$}.sqf"
mkdir -p "${OUTDIR}"
rm -f "${FINAL}" "${LOCAL}"
trap 'rm -f "${LOCAL}"' EXIT

echo "host: $(hostname); start: $(date)"
echo "/dev/shm avail: $(df -B1 --output=avail /dev/shm | tail -1 | awk '{print int($1/1e9)}') GB (need ~100)"
echo "building ${SRC} -> ${LOCAL} (RAM), then copy -> ${FINAL}"

"${MKSQ}" "${SRC}" "${LOCAL}" \
    -keep-as-directory -noappend -noD -no-xattrs -processors 16 \
    -wildcards \
    -e vln_n1/traj_data/mp3d_2leg/YmJkqBEsHnH \
       vln_n1/traj_data/mp3d_3leg/YmJkqBEsHnH \
       vln_n1/traj_data/mp3d_2leg/YFuZgdQ5vWj \
       vln_n1/traj_data/mp3d_3leg/YFuZgdQ5vWj

echo "=== built in RAM: $(date) ==="
ls -la "${LOCAL}"
echo "=== copy RAM -> /scratch ==="
cp "${LOCAL}" "${FINAL}"
rm -f "${LOCAL}"
sync
echo "=== copied: $(date) ==="
ls -la "${FINAL}"

echo "=== verify: scene dirs inside image (expect 36 each leg) ==="
N2=$("${UNSQ}" -l "${FINAL}" | grep -cE '/mp3d_2leg/[^/]+$' || true)
N3=$("${UNSQ}" -l "${FINAL}" | grep -cE '/mp3d_3leg/[^/]+$' || true)
echo "2leg scenes: ${N2}"
echo "3leg scenes: ${N3}"
echo "=== confirm excluded scenes ABSENT ==="
"${UNSQ}" -l "${FINAL}" | grep -E 'YmJkqBEsHnH|YFuZgdQ5vWj' && echo "WARN: excluded scene present!" || echo "OK: idx48/50 absent"
echo "DONE_OK"
