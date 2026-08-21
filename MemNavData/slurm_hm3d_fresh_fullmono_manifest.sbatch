#!/usr/bin/env bash
# Seal the pre-navigation parent identity/episode manifest.
#SBATCH --job-name=h3freshManifest
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
: "${TASK_ROOT:?}" "${BASE_SOURCE_ROOT:?}" "${RUN_ROOT:?}" \
  "${DATA_ROOT:?}" "${EXPECTED_TASK_RECEIPT_SHA:?}" \
  "${EXPECTED_BASE_RECEIPT_SHA:?}"
TASK_RECEIPT=${TASK_ROOT}/SOURCE_BUNDLE.sha256
BASE_RECEIPT=${BASE_SOURCE_ROOT}/source_inputs.sha256
PROTOCOL=${TASK_ROOT}/MemNavData/hm3d_fresh_fullmono_mixed_role_protocol_20260820.json
BUILDER=${TASK_ROOT}/MemNavData/build_hm3d_fresh_fullmono_parent_manifest.py
AUDITOR=${TASK_ROOT}/MemNavData/audit_hm3d_fullmono_inputs.py
ASSET_RECEIPT=${DATA_ROOT}/receipts/fresh_fullmono54_assets.json
SELECTION_RECEIPT=${DATA_ROOT}/receipts/fresh_fullmono54_selection.json
OUT=${RUN_ROOT}/sealed_inputs/parent_manifest.json
[[ "$(sha256sum "${TASK_RECEIPT}" | awk '{print $1}')" == \
   "${EXPECTED_TASK_RECEIPT_SHA}" ]]
(cd "${TASK_ROOT}" && sha256sum -c --quiet SOURCE_BUNDLE.sha256)
[[ "$(sha256sum "${BASE_RECEIPT}" | awk '{print $1}')" == \
   "${EXPECTED_BASE_RECEIPT_SHA}" ]]
(cd "${BASE_SOURCE_ROOT}" && sha256sum -c --quiet source_inputs.sha256)
[[ ! -e "${OUT}" && ! -e "${OUT}.sha256" ]]
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="${TASK_ROOT}:${TASK_ROOT}/MemNavData" \
  /scratch/lg154/conda-envs/memnav/bin/python -u "${BUILDER}" \
    --protocol "${PROTOCOL}" --asset-root \
      "${DATA_ROOT}/data/scene_datasets/hm3d/fresh_fullmono54" \
    --asset-receipt "${ASSET_RECEIPT}" \
    --selection-receipt "${SELECTION_RECEIPT}" \
    --generated-root "${RUN_ROOT}/source_generation" --out "${OUT}"
sha256sum "${OUT}" >"${OUT}.sha256"
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="${TASK_ROOT}:${TASK_ROOT}/MemNavData" \
  /scratch/lg154/conda-envs/memnav/bin/python -u "${AUDITOR}" \
    --protocol "${PROTOCOL}" --parent-manifest "${OUT}" \
    --out "${RUN_ROOT}/sealed_inputs/input_audit.json" >/dev/null
sha256sum "${RUN_ROOT}/sealed_inputs/input_audit.json" \
  >"${RUN_ROOT}/sealed_inputs/input_audit.json.sha256"
chmod -R a-w "${RUN_ROOT}/sealed_inputs"
