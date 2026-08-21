#!/usr/bin/env bash
# Frozen spatial-evidence learned proof: smoke then 5x4 scene cross-fit OOF.
set -euo pipefail
umask 0022

readonly SOURCE_ROOT=/home/cv/memnav_eval/source_bundles/pi3x_spatial_head_749e46bbf930ab1d
readonly SOURCE_RECEIPT=${SOURCE_ROOT}/SOURCE_BUNDLE.sha256
readonly EXPECTED_SOURCE_RECEIPT_SHA=749e46bbf930ab1db98fd3ea1bd21b2994f5032ad147a3b4b3c685b69eba4917
readonly TOKEN_ROOT=/home/cv/memnav_eval/results/pi3x_learned_relocalizer_20260817/viewtokens_b16_3b9db95668bfe54f_20260817_051500
readonly DESCRIPTORS=${TOKEN_ROOT}/pi3x_b16_viewtokens.npz
readonly EXPECTED_DESCRIPTORS_SHA=a9e3a87e11efcf979a35d000d4835d9116f33a6e3bbb75dfc69f0e644957b773
readonly SPATIAL_ROOT=/home/cv/memnav_eval/results/pi3x_learned_relocalizer_20260817/spatial_export_157bb003103f7e11_20260816T214333Z/train40
readonly SPATIAL=${SPATIAL_ROOT}/pi3x_b16_spatial.npz
readonly EXPECTED_SPATIAL_SHA=23bc0eb6357942248561a7509c3751ed4c8fe90e7f845b47ed3ad9a1bf306342
readonly SPATIAL_RECEIPT=${SPATIAL}.receipt.json
readonly EXPECTED_SPATIAL_RECEIPT_SHA=f5fa7110f6022ce63f7e3736936c5b63b9f3bb5d47b4676eb1f149bf310fbc48
readonly SHADOW=${SPATIAL_ROOT}/pi3x_b16_shadow_repeat.jsonl
readonly EXPECTED_SHADOW_SHA=eef5cbf089d5ed607f39b2b64dc87136238584db1df7e05ce3acaebc4b99f155
readonly STATIC_CSV=/home/cv/memnav_eval/model_assets/pi3x_train40_inputs_434388205f77815b/static_top8_480_lightglue_open_set_rows.csv
readonly EXPECTED_STATIC_CSV_SHA=85f9064bff15ce59106ad2a1aa8e5dc4720ee1b1ad894aac1bcedf8581a1d127
readonly TRAINER=${SOURCE_ROOT}/MemNavData/train_pi3x_spatial_reliability_crossfit_oof.py
readonly PYTHON=/home/cv/miniconda3/envs/memnav/bin/python
: "${RESULT_ROOT:?set a fresh immutable result root}"

fail() { echo "ABORT: $*" >&2; exit 2; }
sha_is() {
  [[ -f "$1" ]] || fail "missing input $1"
  [[ "$(sha256sum "$1" | awk '{print $1}')" == "$2" ]] || fail "SHA mismatch $1"
}

sha_is "${SOURCE_RECEIPT}" "${EXPECTED_SOURCE_RECEIPT_SHA}"
(cd "${SOURCE_ROOT}" && sha256sum -c "${SOURCE_RECEIPT}") >/dev/null || fail "source bundle changed"
sha_is "${DESCRIPTORS}" "${EXPECTED_DESCRIPTORS_SHA}"
sha_is "${SPATIAL}" "${EXPECTED_SPATIAL_SHA}"
sha_is "${SPATIAL_RECEIPT}" "${EXPECTED_SPATIAL_RECEIPT_SHA}"
sha_is "${SHADOW}" "${EXPECTED_SHADOW_SHA}"
sha_is "${STATIC_CSV}" "${EXPECTED_STATIC_CSV_SHA}"
[[ -f "${TRAINER}" && ! -e "${RESULT_ROOT}" ]] || fail "trainer missing or result root exists"
mkdir -p "${RESULT_ROOT}/smoke" "${RESULT_ROOT}/full"

export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=${SOURCE_ROOT}${PYTHONPATH:+:${PYTHONPATH}}
export PYTHONPYCACHEPREFIX=${RESULT_ROOT}/pycache

"${PYTHON}" - <<PY
import json
from pathlib import Path
import numpy as np
import torch

if not torch.cuda.is_available() or "5090" not in torch.cuda.get_device_name(0):
    raise SystemExit("expected shared RTX 5090")
receipt = json.loads(Path("${SPATIAL_RECEIPT}").read_text())
if receipt["contains_reporting_labels"] or receipt["contains_certificate_features"]:
    raise SystemExit("spatial input violates the label-blind feature contract")
with np.load("${DESCRIPTORS}") as global_tokens, np.load("${SPATIAL}") as spatial:
    expected = list(range(3840))
    assert global_tokens["row_indices"].tolist() == expected
    assert spatial["row_indices"].tolist() == expected
    assert np.array_equal(global_tokens["view_roles"], spatial["view_roles"])
    assert np.array_equal(global_tokens["view_relative_age"], spatial["view_relative_age"])
    assert np.array_equal(global_tokens["view_valid"], spatial["view_valid"])
print({"preflight": "passed", "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__})
PY

"${PYTHON}" -m pytest -q -p no:cacheprovider \
  "${SOURCE_ROOT}/MemNavData/test_train_pi3x_viewtoken_reliability_oof.py" \
  "${SOURCE_ROOT}/MemNavData/test_train_pi3x_spatial_reliability_crossfit_oof.py" \
  "${SOURCE_ROOT}/MemNavData/test_summarize_pi3x_multiview_shadow.py"

COMMON=(
  --rows-csv "${STATIC_CSV}"
  --shadow-jsonl "${SHADOW}"
  --descriptors-npz "${DESCRIPTORS}"
  --spatial-npz "${SPATIAL}"
  --expected-rows 3840
  --expected-rows-sha256 "${EXPECTED_STATIC_CSV_SHA}"
  --expected-spatial-sha256 "${EXPECTED_SPATIAL_SHA}"
  --minimum-precision 0.90
  --maximum-fpr 0.0275
  --model-dim 64
  --layers 2
  --heads 4
  --batch-sessions 12
  --inference-batch-rows 64
  --learning-rate 3e-4
  --weight-decay 1e-3
  --support-weight 0.25
  --seed 17
  --device cuda
)

"${PYTHON}" -u "${TRAINER}" "${COMMON[@]}" \
  --outer-splits 2 --inner-splits 2 --consensus 2 --epochs 1 \
  --checkpoint-dir "${RESULT_ROOT}/smoke/checkpoints" \
  --output-summary "${RESULT_ROOT}/smoke/summary.json" \
  --output-predictions "${RESULT_ROOT}/smoke/predictions.csv"

# Consensus=2/4 was frozen before this spatial model ran.  It was the most
# permissive global-token setting satisfying precision, FPR and catastrophe
# gates under the fixed raw-Pi3X-overlap proposal.
"${PYTHON}" -u "${TRAINER}" "${COMMON[@]}" \
  --outer-splits 5 --inner-splits 4 --consensus 2 --epochs 30 \
  --checkpoint-dir "${RESULT_ROOT}/full/checkpoints" \
  --output-summary "${RESULT_ROOT}/full/summary.json" \
  --output-predictions "${RESULT_ROOT}/full/predictions.csv"

find "${RESULT_ROOT}" -type f ! -path '*/pycache/*' ! -name OUTPUTS.sha256 -print0 | sort -z | \
  xargs -0 sha256sum > "${RESULT_ROOT}/OUTPUTS.sha256"
find "${RESULT_ROOT}" -type f ! -path '*/pycache/*' -exec chmod a-w {} +
echo "COMPLETE result=${RESULT_ROOT}"
