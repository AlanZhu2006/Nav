#!/usr/bin/env bash
# Outcome-blind identity gate between held-out-val generation and closed-loop eval.
#SBATCH --job-name=hm3dh10_manifest
#SBATCH --output=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/%x_%j.out
#SBATCH --error=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/%x_%j.err
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=cs
#SBATCH --qos=cpu48
#SBATCH --account=torch_pr_769_tandon_advanced

set -euo pipefail
umask 0022

: "${TASK_ROOT:?set immutable task bundle root}"
: "${BASE_SOURCE_ROOT:?set immutable base source root}"
: "${RUN_ROOT:?set run root}"
: "${DATA_ROOT:?set HM3D held-out val10 data root}"
: "${EXPECTED_TASK_RECEIPT_SHA:?set task receipt SHA}"
: "${EXPECTED_BASE_SOURCE_RECEIPT_SHA:?set base receipt SHA}"

readonly TASK_RECEIPT=${TASK_ROOT}/SOURCE_BUNDLE.sha256
readonly BASE_RECEIPT=${BASE_SOURCE_ROOT}/SOURCE_BUNDLE.sha256
readonly PROTOCOL=${TASK_ROOT}/MemNavData/hm3d_heldout_val10_revisit_protocol_20260816.json
readonly BUILDER=${TASK_ROOT}/MemNavData/build_hm3d_heldout_val10_revisit_manifest.py
readonly ASSET_RECEIPT=${DATA_ROOT}/receipts/hm3d-heldout-val10-extraction.json
readonly OUT=${RUN_ROOT}/data_manifest.json
readonly MEMNAV_PY=/scratch/lg154/conda-envs/memnav/bin/python
readonly MEMNAV_CKPT=/scratch/yz11502/Research/Nav-axis-uturn/.diagnostics/unseen_scene_eval_20260803/checkpoints/gatecurr600.memnav.ckpt
readonly NAVDP_CKPT=/scratch/yz11502/Research/Nav-axis-uturn/.diagnostics/unseen_scene_eval_20260803/checkpoints/navdp_checkpoint.ckpt
readonly LINGBOT_WEIGHTS=/scratch/lg154/Research/Nav/NavDP/baselines/memnav/lingbot-map/weights/lingbot-map-long.pt

fail() { echo "ABORT: $*" >&2; exit 2; }
[[ "$(sha256sum "${TASK_RECEIPT}" | awk '{print $1}')" == \
    "${EXPECTED_TASK_RECEIPT_SHA}" ]] || fail "task receipt changed"
(cd "${TASK_ROOT}" && sha256sum -c SOURCE_BUNDLE.sha256 >/dev/null) || \
  fail "task bundle changed"
[[ "$(sha256sum "${BASE_RECEIPT}" | awk '{print $1}')" == \
    "${EXPECTED_BASE_SOURCE_RECEIPT_SHA}" ]] || fail "base receipt changed"
[[ ! -e "${OUT}" && ! -e "${OUT}.sha256" ]] || \
  fail "manifest output already exists"
for path in "${PROTOCOL}" "${BUILDER}" "${ASSET_RECEIPT}" \
  "${MEMNAV_CKPT}" "${NAVDP_CKPT}" "${LINGBOT_WEIGHTS}"; do
  test -r "${path}" || fail "missing input ${path}"
done

export PYTHONPATH=${TASK_ROOT}:${BASE_SOURCE_ROOT}${PYTHONPATH:+:${PYTHONPATH}}
"${MEMNAV_PY}" -u "${BUILDER}" \
  --protocol "${PROTOCOL}" \
  --generated-root "${RUN_ROOT}/data/hm3d_2leg" \
  --data-root "${DATA_ROOT}" --asset-receipt "${ASSET_RECEIPT}" \
  --base-source-receipt "${BASE_RECEIPT}" \
  --expected-base-source-receipt-sha "${EXPECTED_BASE_SOURCE_RECEIPT_SHA}" \
  --gatecurr-checkpoint "${MEMNAV_CKPT}" \
  --navdp-checkpoint "${NAVDP_CKPT}" \
  --lingbot-weights "${LINGBOT_WEIGHTS}" --out "${OUT}"
sha256sum "${OUT}" > "${OUT}.sha256"
chmod a-w "${OUT}" "${OUT}.sha256"
