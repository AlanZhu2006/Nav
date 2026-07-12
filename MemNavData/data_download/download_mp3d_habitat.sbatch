#!/bin/bash
#SBATCH --job-name=mp3d_download
#SBATCH --output=/scratch/lg154/Research/Nav/InternNav/logs/mp3d_download/slurm-%j.out
#SBATCH --error=/scratch/lg154/Research/Nav/InternNav/logs/mp3d_download/slurm-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --account=torch_pr_769_tandon_advanced
#SBATCH --partition=cs
#SBATCH --export=ALL
#
# Download + extract the MP3D Habitat bundle (~15 GB, all 90 scenes as
# <scene>/<scene>.glb + <scene>.navmesh). Contains 17DRP5sb8fy used by
# MemNavData/generate_twoleg.py. CPU-only; the TUM link is ~1.8 MB/s so the
# transfer is ~2.5 h. curl -C - resumes any partial file across retries and
# across re-submits, so re-running this job is safe and idempotent.
#
# Submit:
#   sbatch MemNavData/data_download/download_mp3d_habitat.sbatch

set -euo pipefail

URL="https://kaldir.vc.cit.tum.de/matterport/v1/tasks/mp3d_habitat.zip"
DEST="/scratch/lg154/Research/datasets/mp3d"
OUT="${DEST}/mp3d_habitat.zip"
EXPECTED=16085306031

mkdir -p "${DEST}"
cd "${DEST}"

echo "===== MP3D DOWNLOAD ====="
echo "host:   $(hostname)"
echo "dest:   ${OUT}"
echo "url:    ${URL}"

# --- resumable download with retry loop ---
for attempt in $(seq 1 20); do
  have=$( [ -f "${OUT}" ] && stat -c %s "${OUT}" || echo 0 )
  if [ "${have}" -ge "${EXPECTED}" ]; then
    echo "[download] complete: ${have} bytes"
    break
  fi
  echo "[download] attempt ${attempt}: have ${have} / ${EXPECTED} bytes, resuming..."
  curl -sSL -C - --retry 5 --retry-delay 15 -m 25200 -o "${OUT}" "${URL}" || true
  sleep 10
done

have=$( stat -c %s "${OUT}" )
[ "${have}" -ge "${EXPECTED}" ] || { echo "ERROR: incomplete (${have}/${EXPECTED})" >&2; exit 1; }

# --- verify + extract ---
echo "[verify] testing zip integrity..."
unzip -tq "${OUT}" >/dev/null

echo "[extract] unzipping into ${DEST}/ ..."
unzip -oq "${OUT}" -d "${DEST}"

# --- locate the scene used by the generator and expose it at a stable path ---
GLB=$(find "${DEST}" -name '17DRP5sb8fy.glb' | head -1)
NAV=$(find "${DEST}" -name '17DRP5sb8fy.navmesh' | head -1)
echo "[verify] 17DRP5sb8fy.glb     -> ${GLB:-MISSING}"
echo "[verify] 17DRP5sb8fy.navmesh -> ${NAV:-MISSING}"
[ -n "${GLB}" ] || { echo "ERROR: 17DRP5sb8fy.glb not found after extract" >&2; exit 1; }

# convenience symlinks matching the manifest layout (.../mp3d/17DRP5sb8fy.glb)
ln -sf "${GLB}" "${DEST}/17DRP5sb8fy.glb"
[ -n "${NAV}" ] && ln -sf "${NAV}" "${DEST}/17DRP5sb8fy.navmesh"

echo "===== DONE ====="
echo "scenes extracted: $(find "${DEST}" -name '*.glb' | wc -l)"
echo "total size:       $(du -sh "${DEST}" | cut -f1)"
