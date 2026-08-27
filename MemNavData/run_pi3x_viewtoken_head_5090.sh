#!/usr/bin/env bash
# Nested scene-OOF training of the first learned Pi3X reliability/ranking head.
# It consumes only frozen train40 tokens and cannot change navigation actions.
set -euo pipefail
umask 0022

readonly SOURCE_ROOT=/home/cv/memnav_eval/source_bundles/pi3x_viewtoken_head_7d0fc763d5093295
readonly SOURCE_RECEIPT=${SOURCE_ROOT}/SOURCE_BUNDLE.sha256
readonly EXPECTED_SOURCE_RECEIPT_SHA=7d0fc763d5093295738521d53c6c4898192229b7c0c3ec433a65ce3d220a9f77
readonly INPUT_ROOT=/home/cv/memnav_eval/results/pi3x_learned_relocalizer_20260817/viewtokens_b16_3b9db95668bfe54f_20260817_051500
readonly DESCRIPTORS=${INPUT_ROOT}/pi3x_b16_viewtokens.npz
readonly EXPECTED_DESCRIPTORS_SHA=a9e3a87e11efcf979a35d000d4835d9116f33a6e3bbb75dfc69f0e644957b773
readonly SHADOW=${INPUT_ROOT}/pi3x_b16_shadow.jsonl
readonly EXPECTED_SHADOW_SHA=0a4166f590c18e8a979cec05c8439b6b5da20f3a09db2e8a30b2fd9fc2a56be8
readonly STATIC_CSV=/home/cv/memnav_eval/model_assets/pi3x_train40_inputs_434388205f77815b/static_top8_480_lightglue_open_set_rows.csv
readonly EXPECTED_STATIC_CSV_SHA=85f9064bff15ce59106ad2a1aa8e5dc4720ee1b1ad894aac1bcedf8581a1d127
readonly TRAINER=${SOURCE_ROOT}/MemNavData/train_pi3x_viewtoken_reliability_oof.py
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
sha_is "${DESCRIPTORS}" "${EXPECTED_DESCRIPTORS_SHA}"
sha_is "${SHADOW}" "${EXPECTED_SHADOW_SHA}"
sha_is "${STATIC_CSV}" "${EXPECTED_STATIC_CSV_SHA}"
[[ -f "${TRAINER}" && ! -e "${RESULT_ROOT}" ]] || \
  fail "trainer missing or result root already exists"
mkdir -p "${RESULT_ROOT}/smoke" "${RESULT_ROOT}/full"

export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=${SOURCE_ROOT}${PYTHONPATH:+:${PYTHONPATH}}
export PYTHONPYCACHEPREFIX=${RESULT_ROOT}/pycache

"${PYTHON}" - <<'PY'
import json
from pathlib import Path
import numpy as np
import torch

if not torch.cuda.is_available() or "5090" not in torch.cuda.get_device_name(0):
    raise SystemExit("expected the shared RTX 5090")
root = Path("/home/cv/memnav_eval/results/pi3x_learned_relocalizer_20260817/viewtokens_b16_3b9db95668bfe54f_20260817_051500")
receipt = json.loads((root / "pi3x_b16_viewtokens.npz.receipt.json").read_text())
if receipt["contains_reporting_labels"] or receipt["shape"] != [3840, 20, 2048]:
    raise SystemExit("descriptor receipt violates the frozen feature contract")
with np.load(root / "pi3x_b16_viewtokens.npz") as values:
    if values["row_indices"].tolist() != list(range(3840)):
        raise SystemExit("descriptor row order is not the frozen CSV order")
print({"preflight": "passed", "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__})
PY

"${PYTHON}" -m pytest -q -p no:cacheprovider \
  "${SOURCE_ROOT}/MemNavData/test_train_pi3x_viewtoken_reliability_oof.py" \
  "${SOURCE_ROOT}/MemNavData/test_summarize_pi3x_multiview_shadow.py"

COMMON=(
  --rows-csv "${STATIC_CSV}"
  --shadow-jsonl "${SHADOW}"
  --descriptors-npz "${DESCRIPTORS}"
  --expected-rows 3840
  --expected-rows-sha256 "${EXPECTED_STATIC_CSV_SHA}"
  --minimum-precision 0.90
  --maximum-fpr 0.0275
  --model-dim 64
  --layers 2
  --heads 4
  --batch-sessions 24
  --inference-batch-rows 128
  --learning-rate 3e-4
  --weight-decay 1e-3
  --listwise-weight 0.5
  --support-weight 0.25
  --seed 17
  --device cuda
)

"${PYTHON}" -u "${TRAINER}" "${COMMON[@]}" \
  --outer-splits 2 --inner-splits 2 --epochs 1 \
  --checkpoint-dir "${RESULT_ROOT}/smoke/checkpoints" \
  --output-summary "${RESULT_ROOT}/smoke/summary.json" \
  --output-predictions "${RESULT_ROOT}/smoke/predictions.csv"

"${PYTHON}" -u "${TRAINER}" "${COMMON[@]}" \
  --outer-splits 5 --inner-splits 4 --epochs 30 \
  --checkpoint-dir "${RESULT_ROOT}/full/checkpoints" \
  --output-summary "${RESULT_ROOT}/full/summary.json" \
  --output-predictions "${RESULT_ROOT}/full/predictions.csv"

find "${RESULT_ROOT}" -type f ! -path '*/pycache/*' -print0 | sort -z | \
  xargs -0 sha256sum > "${RESULT_ROOT}/OUTPUTS.sha256"
find "${RESULT_ROOT}" -type f ! -path '*/pycache/*' -exec chmod a-w {} +
echo "COMPLETE result=${RESULT_ROOT}"
