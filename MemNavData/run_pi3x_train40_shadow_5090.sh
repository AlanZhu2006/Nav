#!/usr/bin/env bash
# Same immutable train40 shadow as the queued HPC chain, executed on the shared
# RTX 5090 after extracting only the manifest-bound PT1 files.
set -euo pipefail
umask 0022

readonly SOURCE_ROOT=/home/cv/memnav_eval/source_bundles/pi3x_learned_relocalizer_925cd1ceb7ea9771
readonly SOURCE_RECEIPT=${SOURCE_ROOT}/SOURCE_BUNDLE.sha256
readonly EXPECTED_SOURCE_RECEIPT_SHA=925cd1ceb7ea97711d62fb9671f802e4ba16cfa3a33d36607ef9788b1ef4fb4c
readonly INPUT_ROOT=/home/cv/memnav_eval/model_assets/pi3x_train40_inputs_434388205f77815b
readonly DATA_ROOT=${INPUT_ROOT}/data
readonly STATIC_CSV=${INPUT_ROOT}/static_top8_480_lightglue_open_set_rows.csv
readonly EXPECTED_STATIC_CSV_SHA=85f9064bff15ce59106ad2a1aa8e5dc4720ee1b1ad894aac1bcedf8581a1d127
readonly MODEL_ROOT=/home/cv/memnav_eval/model_assets/pi3x_69972d6e1c4492c
readonly MODEL=${MODEL_ROOT}/model.safetensors
readonly EXPECTED_MODEL_SHA=69972d6e1c4492cb4d737a84fe940e357087d81c52f5c9b7c160b49c1f41669a
readonly PI3_ROOT=${SOURCE_ROOT}/third_party/Pi3
readonly ENTRYPOINT=${SOURCE_ROOT}/MemNavData/diag_pi3x_multiview_consistency.py
readonly SUMMARIZER=${SOURCE_ROOT}/MemNavData/summarize_pi3x_multiview_shadow.py
readonly PYTHON=/home/cv/miniconda3/envs/memnav/bin/python
: "${RESULT_ROOT:?set a fresh immutable result root}"

fail() { echo "ABORT: $*" >&2; exit 2; }
sha_is() {
  [[ -f "$1" ]] || fail "missing input $1"
  [[ "$(sha256sum "$1" | awk '{print $1}')" == "$2" ]] || \
    fail "SHA mismatch $1"
}

sha_is "${SOURCE_RECEIPT}" "${EXPECTED_SOURCE_RECEIPT_SHA}"
(cd "${SOURCE_ROOT}" && sha256sum -c "${SOURCE_RECEIPT}") >/dev/null || \
  fail "source bundle changed"
sha_is "${STATIC_CSV}" "${EXPECTED_STATIC_CSV_SHA}"
sha_is "${MODEL}" "${EXPECTED_MODEL_SHA}"
[[ -d "${DATA_ROOT}" && -f "${ENTRYPOINT}" && -f "${SUMMARIZER}" ]] || \
  fail "data or source is absent"
[[ ! -e "${RESULT_ROOT}" ]] || fail "result root already exists"
mkdir -p "${RESULT_ROOT}/smoke" "${RESULT_ROOT}/train40"

export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=${SOURCE_ROOT}:${PI3_ROOT}${PYTHONPATH:+:${PYTHONPATH}}
export PYTHONPYCACHEPREFIX=${RESULT_ROOT}/pycache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

"${PYTHON}" - "${EXPECTED_MODEL_SHA}" <<'PY'
import sys
import torch

if not torch.cuda.is_available():
    raise SystemExit("CUDA unavailable")
name = torch.cuda.get_device_name(0)
if "5090" not in name:
    raise SystemExit(f"unexpected GPU {name}")
print({"preflight": "passed", "gpu": name, "torch": torch.__version__})
PY

COMMON=(
  --rows-csv "${STATIC_CSV}"
  --expected-rows-sha256 "${EXPECTED_STATIC_CSV_SHA}"
  --data-root "${DATA_ROOT}"
  --pi3-root "${PI3_ROOT}"
  --snapshot "${MODEL_ROOT}"
  --expected-model-sha256 "${EXPECTED_MODEL_SHA}"
  --history-mode causal_bridge
  --bridge-frames 8
  --anchor-offsets -8 0 8
  --quiet
)

"${PYTHON}" -u "${ENTRYPOINT}" "${COMMON[@]}" \
  --row-indices 40 0 11 16 --expected-output-rows 4 \
  --output-jsonl "${RESULT_ROOT}/smoke/pi3x_smoke.jsonl"

"${PYTHON}" -u "${ENTRYPOINT}" "${COMMON[@]}" \
  --all-available --expected-output-rows 3840 \
  --output-jsonl "${RESULT_ROOT}/train40/pi3x_full.jsonl"

"${PYTHON}" -u "${SUMMARIZER}" \
  --shadow-jsonl "${RESULT_ROOT}/train40/pi3x_full.jsonl" \
  --rows-csv "${STATIC_CSV}" \
  --expected-rows 3840 \
  --expected-rows-sha256 "${EXPECTED_STATIC_CSV_SHA}" \
  --output-summary "${RESULT_ROOT}/train40/pi3x_train40_summary.json" \
  --output-predictions "${RESULT_ROOT}/train40/pi3x_train40_oof_predictions.csv"

find "${RESULT_ROOT}" -type f ! -path '*/pycache/*' -print0 | sort -z | \
  xargs -0 sha256sum > "${RESULT_ROOT}/OUTPUTS.sha256"
find "${RESULT_ROOT}" -type f ! -path '*/pycache/*' -exec chmod a-w {} +
echo "COMPLETE result=${RESULT_ROOT}"
