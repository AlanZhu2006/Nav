#!/usr/bin/env bash
# Resume the frozen 2026-08-20 HM3D pipeline after the parent-manifest
# legacy-carrier eligibility bug.  Source generation and the formal protocol
# remain byte-identical; only the parent builder comes from REPAIR_ROOT.
#SBATCH --job-name=h3freshManifestR1
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
: "${REPAIR_ROOT:?}" "${EXPECTED_REPAIR_RECEIPT_SHA:?}"
: "${TASK_ROOT:?}" "${BASE_SOURCE_ROOT:?}" "${RUN_ROOT:?}" \
  "${DATA_ROOT:?}" "${EXPECTED_TASK_RECEIPT_SHA:?}" \
  "${EXPECTED_BASE_RECEIPT_SHA:?}"

REPAIR_RECEIPT=${REPAIR_ROOT}/REPAIR_BUNDLE.sha256
TASK_RECEIPT=${TASK_ROOT}/SOURCE_BUNDLE.sha256
BASE_RECEIPT=${BASE_SOURCE_ROOT}/source_inputs.sha256
PROTOCOL=${TASK_ROOT}/MemNavData/hm3d_fresh_fullmono_mixed_role_protocol_20260820.json
BUILDER=${REPAIR_ROOT}/MemNavData/build_hm3d_fresh_fullmono_parent_manifest.py
AUDITOR=${TASK_ROOT}/MemNavData/audit_hm3d_fullmono_inputs.py
ASSET_RECEIPT=${DATA_ROOT}/receipts/fresh_fullmono54_assets.json
SELECTION_RECEIPT=${DATA_ROOT}/receipts/fresh_fullmono54_selection.json
OUT=${RUN_ROOT}/sealed_inputs/parent_manifest.json
INPUT_AUDIT=${RUN_ROOT}/sealed_inputs/input_audit.json
EXPECTED_EXISTING_PARENT_SHA=${EXPECTED_EXISTING_PARENT_SHA:-}

fail() { echo "ABORT: $*" >&2; exit 2; }
verify_file_hash() {
  local label=$1 path=$2 expected=$3 actual="" attempt
  for attempt in 1 2 3; do
    actual=$(sha256sum -- "${path}" 2>/dev/null | awk '{print $1}') || actual="READ_ERROR"
    printf 'hash_preflight label=%s attempt=%s expected=%s actual=%s path=%s\n' \
      "${label}" "${attempt}" "${expected}" "${actual}" "${path}" >&2
    [[ "${actual}" == "${expected}" ]] && return 0
    sleep 2
  done
  stat --printf='hash_preflight_stat path=%n bytes=%s mtime=%y\n' -- \
    "${path}" 1>&2 2>/dev/null || true
  fail "${label} hash mismatch"
}
verify_receipt_contents() {
  local label=$1 root=$2 receipt=$3 attempt
  for attempt in 1 2 3; do
    if (cd "${root}" && sha256sum -c --quiet "${receipt}"); then
      printf 'receipt_preflight label=%s attempt=%s status=ok root=%s\n' \
        "${label}" "${attempt}" "${root}" >&2
      return 0
    fi
    printf 'receipt_preflight label=%s attempt=%s status=retry root=%s\n' \
      "${label}" "${attempt}" "${root}" >&2
    sleep 2
  done
  (cd "${root}" && sha256sum -c "${receipt}" 2>&1 | \
    grep -E 'FAILED|No such file|cannot open' | head -40) >&2 || true
  fail "${label} bundle contents mismatch"
}

verify_file_hash repair_receipt "${REPAIR_RECEIPT}" \
  "${EXPECTED_REPAIR_RECEIPT_SHA}"
verify_receipt_contents repair "${REPAIR_ROOT}" REPAIR_BUNDLE.sha256
verify_file_hash task_receipt "${TASK_RECEIPT}" \
  "${EXPECTED_TASK_RECEIPT_SHA}"
verify_receipt_contents task "${TASK_ROOT}" SOURCE_BUNDLE.sha256
verify_file_hash base_receipt "${BASE_RECEIPT}" \
  "${EXPECTED_BASE_RECEIPT_SHA}"
verify_receipt_contents base "${BASE_SOURCE_ROOT}" source_inputs.sha256
if [[ -e "${OUT}" || -e "${OUT}.sha256" ]]; then
  [[ -f "${OUT}" && -f "${OUT}.sha256" &&
     "${#EXPECTED_EXISTING_PARENT_SHA}" -eq 64 ]] || \
    fail "incomplete or unauthorized existing parent manifest"
  verify_file_hash existing_parent "${OUT}" \
    "${EXPECTED_EXISTING_PARENT_SHA}"
  (cd "$(dirname "${OUT}")" && \
    sha256sum -c --quiet "$(basename "${OUT}.sha256")") || \
    fail "existing parent receipt mismatch"
  echo "resume_existing_parent sha256=${EXPECTED_EXISTING_PARENT_SHA}" >&2
else
  PYTHONDONTWRITEBYTECODE=1 \
  PYTHONPATH="${REPAIR_ROOT}:${TASK_ROOT}:${TASK_ROOT}/MemNavData" \
    /scratch/lg154/conda-envs/memnav/bin/python -u "${BUILDER}" \
      --protocol "${PROTOCOL}" \
      --asset-root "${DATA_ROOT}/data/scene_datasets/hm3d/fresh_fullmono54" \
      --asset-receipt "${ASSET_RECEIPT}" \
      --selection-receipt "${SELECTION_RECEIPT}" \
      --generated-root "${RUN_ROOT}/source_generation" --out "${OUT}"
  sha256sum "${OUT}" >"${OUT}.sha256"
fi
[[ ! -e "${INPUT_AUDIT}" && ! -e "${INPUT_AUDIT}.sha256" ]] || \
  fail "input audit output already exists"

PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH="${TASK_ROOT}:${TASK_ROOT}/MemNavData:${BASE_SOURCE_ROOT}:${BASE_SOURCE_ROOT}/MemNavData" \
  /scratch/lg154/conda-envs/memnav/bin/python -u "${AUDITOR}" \
    --protocol "${PROTOCOL}" --parent-manifest "${OUT}" \
    --out "${INPUT_AUDIT}" >/dev/null
sha256sum "${INPUT_AUDIT}" >"${INPUT_AUDIT}.sha256"
chmod -R a-w "${RUN_ROOT}/sealed_inputs"
