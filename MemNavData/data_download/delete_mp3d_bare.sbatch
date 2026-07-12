#!/bin/bash
#SBATCH --job-name=rm_mp3d_bare
#SBATCH --output=/scratch/lg154/Research/Nav/InternNav/logs/mp3d_gen/rm_bare-%j.out
#SBATCH --error=/scratch/lg154/Research/Nav/InternNav/logs/mp3d_gen/rm_bare-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=8G
#SBATCH --time=06:00:00
#SBATCH --partition=cs
#SBATCH --account=torch_pr_769_tandon_advanced
#
# Delete the bare mp3d_revisit_v0 tree AFTER it has been packed+verified into
# _overlays/mp3d_revisit_v0_pt1.sqf. Frees ~2.77M inodes (5M quota was at 99%).
# Parallel unlink across scene dirs (VAST metadata ops are the bottleneck, so fan out).
# SAFETY: refuses to run unless the verified .sqf exists and is >100GB.
set -uo pipefail

TREE=/scratch/lg154/Research/datasets/mp3d_revisit_v0
SQF=/scratch/lg154/Research/datasets/_overlays/mp3d_revisit_v0_pt1.sqf

echo "host: $(hostname); start: $(date)"
# guard: the pack must exist and be large, else abort (never delete unpacked data)
if [[ ! -f "$SQF" ]]; then echo "ABORT: pack image missing: $SQF" >&2; exit 1; fi
sz=$(stat -c%s "$SQF")
if (( sz < 100000000000 )); then echo "ABORT: pack image too small ($sz bytes) — not verified" >&2; exit 1; fi
echo "guard OK: pack image is $((sz/1024/1024/1024)) GB"

echo "inodes before: $(myquota 2>/dev/null | grep scratch || true)"

# fan out rm across scene dirs (both legs), 16-way
for leg in 2 3; do
  d="$TREE/vln_n1/traj_data/mp3d_${leg}leg"
  [[ -d "$d" ]] || continue
  echo "deleting scene dirs under $d ..."
  ls -d "$d"/*/ 2>/dev/null | xargs -P 16 -I{} rm -rf "{}"
done

echo "removing now-empty tree root ..."
rm -rf "$TREE"

echo "=== done: $(date) ==="
ls -la "$TREE" 2>&1 | head -1 || true
echo "inodes after: $(myquota 2>/dev/null | grep scratch || true)"
echo "RM_DONE_OK"
