#!/bin/bash
#SBATCH --job-name=prune_interndata_n1
#SBATCH --output=/scratch/lg154/Research/Nav/InternNav/logs/mp3d_gen/prune_interndata_n1-%j.out
#SBATCH --error=/scratch/lg154/Research/Nav/InternNav/logs/mp3d_gen/prune_interndata_n1-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --partition=cs
#SBATCH --account=torch_pr_769_tandon_advanced
#
# Delete the bare InternData-N1 tree AFTER verifying InternData-N1.sqf is a complete, valid
# squashfs image. Scheduled to --begin ~3h from submit (pack finishes in ~2h). The .sqf lives
# in _overlays/, NOT under InternData-N1/, so pruning the source never touches the image.
#
# SAFETY GATE: aborts (exit 1, deletes NOTHING) unless the image opens and lists a plausible
# number of entries. If it aborts, the bare tree is left intact for manual inspection.
set -euo pipefail

UNSQ=/share/apps/apptainer/1.5.2/x86_64/utils/bin/unsquashfs
SRC=/scratch/lg154/Research/datasets/InternData-N1
SQF=/scratch/lg154/Research/datasets/_overlays/InternData-N1.sqf
MIN_ENTRIES=4000000          # mksquashfs reported ~4.14M objects; require most of them present

echo "host: $(hostname); start: $(date)"

# --- gate 1: image exists and is non-trivial ---
[[ -f "${SQF}" ]] || { echo "ABORT: ${SQF} missing"; exit 1; }
SZ=$(stat -c%s "${SQF}")
echo "sqf size: ${SZ} bytes"
[[ "${SZ}" -gt 100000000000 ]] || { echo "ABORT: ${SQF} suspiciously small (<100GB)"; exit 1; }

# --- gate 2: image opens and superblock is valid ---
"${UNSQ}" -s "${SQF}" >/dev/null || { echo "ABORT: unsquashfs -s failed (bad/partial image)"; exit 1; }

# --- gate 3: plausible entry count ---
NIN=$("${UNSQ}" -l "${SQF}" | wc -l)
echo "image lists ${NIN} entries (require >= ${MIN_ENTRIES})"
[[ "${NIN}" -ge "${MIN_ENTRIES}" ]] || { echo "ABORT: only ${NIN} entries in image (< ${MIN_ENTRIES})"; exit 1; }

# --- gate 4: vln_n1 present in image (the actual data) ---
"${UNSQ}" -l "${SQF}" | grep -qE '/InternData-N1/vln_n1/_raw/' \
    || { echo "ABORT: vln_n1/_raw not found in image"; exit 1; }

echo "=== all gates passed: image is complete. Pruning bare tree ${SRC} ==="
cd "${SRC}"
# fan out at the trajectory-dir level (deepest cheap partition) for parallel unlinks on VAST
ls -d vln_n1/_raw/*/*/ 2>/dev/null | xargs -r -P 48 -I{} rm -rf "{}"
# then everything remaining (top-level dirs, README, .git, .cache, .gitattributes)
cd /scratch/lg154/Research/datasets
rm -rf InternData-N1

echo "=== prune done: $(date) ==="
test -e "${SRC}" && echo "WARN: ${SRC} still exists (stale handles? re-run rm)" || echo "GONE"
echo "sqf preserved: $(ls -la "${SQF}")"
echo "DONE_OK"
