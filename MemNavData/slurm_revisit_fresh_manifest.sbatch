#!/usr/bin/env bash
# CPU identity gate between data generation and inference.
#SBATCH --job-name=revfresh_manifest
#SBATCH --output=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/%x_%j.out
#SBATCH --error=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/%x_%j.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=cs
#SBATCH --qos=cpu48
#SBATCH --account=torch_pr_769_tandon_advanced
#SBATCH --export=ALL

set -euo pipefail
umask 0022
: "${SOURCE_ROOT:?set immutable SOURCE_ROOT}"
: "${RUN_ROOT:?set RUN_ROOT}"
: "${SOURCE_RECEIPT:?set SOURCE_RECEIPT}"
: "${EXPECTED_SOURCE_RECEIPT_SHA:?set EXPECTED_SOURCE_RECEIPT_SHA}"
[[ "$(sha256sum "${SOURCE_RECEIPT}" | awk '{print $1}')" == \
    "${EXPECTED_SOURCE_RECEIPT_SHA}" ]] || { echo "ABORT: source receipt" >&2; exit 2; }

HAB_PY=/scratch/lg154/conda-envs/habitat/bin/python
BUILDER=${SOURCE_ROOT}/MemNavData/build_revisit_fresh_manifest.py
PROTOCOL=${SOURCE_ROOT}/MemNavData/revisit_fresh_confirmation_protocol_20260811.json
HISTORICAL=${SOURCE_ROOT}/MemNavData/expanded_navdp_router_eval_20260805.json
GENERATOR=${SOURCE_ROOT}/MemNavData/generate_twoleg.py
ASSET_ROOT=/scratch/lg154/Research/datasets/mp3d/mp3d
MEMNAV_CKPT=/scratch/yz11502/Research/Nav-axis-uturn/.diagnostics/unseen_scene_eval_20260803/checkpoints/gatecurr600.memnav.ckpt
NAVDP_CKPT=/scratch/yz11502/Research/Nav-axis-uturn/.diagnostics/unseen_scene_eval_20260803/checkpoints/navdp_checkpoint.ckpt
LINGBOT_WEIGHTS=/scratch/lg154/Research/Nav/NavDP/baselines/memnav/lingbot-map/weights/lingbot-map-long.pt
OUT=${RUN_ROOT}/data_manifest.json
export PYTHONPYCACHEPREFIX=${SLURM_TMPDIR:-/tmp}/revfresh_manifest_pycache_${SLURM_JOB_ID}
mkdir -p "${PYTHONPYCACHEPREFIX}"
[[ ! -e "${OUT}" && ! -e "${OUT}.sha256" ]] || {
  echo "ABORT: manifest already exists" >&2; exit 2; }
export PYTHONPATH=${SOURCE_ROOT}${PYTHONPATH:+:${PYTHONPATH}}
env PYTHONPYCACHEPREFIX="${PYTHONPYCACHEPREFIX}" \
  "${HAB_PY}" -m py_compile "${BUILDER}"
"${HAB_PY}" -u "${BUILDER}" \
  --protocol "${PROTOCOL}" \
  --historical-manifest "${HISTORICAL}" \
  --generated-root "${RUN_ROOT}/data" \
  --asset-root "${ASSET_ROOT}" \
  --generator "${GENERATOR}" \
  --gatecurr-checkpoint "${MEMNAV_CKPT}" \
  --navdp-checkpoint "${NAVDP_CKPT}" \
  --lingbot-weights "${LINGBOT_WEIGHTS}" \
  --out "${OUT}"
sha256sum "${OUT}" > "${OUT}.sha256"
chmod a-w "${OUT}" "${OUT}.sha256"
echo "COMPLETE ${OUT}"
