#!/usr/bin/env bash
# Export label-blind spatial Pi3X evidence for the next learned consensus head.
set -euo pipefail
umask 0022

readonly SOURCE_ROOT=/home/cv/memnav_eval/source_bundles/pi3x_spatial_export_016574e131f221b8
readonly SOURCE_RECEIPT=${SOURCE_ROOT}/SOURCE_BUNDLE.sha256
readonly EXPECTED_SOURCE_RECEIPT_SHA=016574e131f221b80c272bfdfeea76a0943f08ec5a291febcb856edbc2902311
readonly BASE_PI3_SOURCE=/home/cv/memnav_eval/source_bundles/pi3x_learned_relocalizer_925cd1ceb7ea9771
readonly BASE_PI3_RECEIPT=${BASE_PI3_SOURCE}/SOURCE_BUNDLE.sha256
readonly EXPECTED_BASE_PI3_RECEIPT_SHA=925cd1ceb7ea97711d62fb9671f802e4ba16cfa3a33d36607ef9788b1ef4fb4c
readonly INPUT_ROOT=/home/cv/memnav_eval/model_assets/pi3x_train40_inputs_434388205f77815b
readonly DATA_ROOT=${INPUT_ROOT}/data
readonly STATIC_CSV=${INPUT_ROOT}/static_top8_480_lightglue_open_set_rows.csv
readonly EXPECTED_STATIC_CSV_SHA=85f9064bff15ce59106ad2a1aa8e5dc4720ee1b1ad894aac1bcedf8581a1d127
readonly REQUIRED_PATHS=${INPUT_ROOT}/pi3x_train40_required_paths_b16.txt
readonly EXPECTED_REQUIRED_PATHS_SHA=a1be7bb104152c623d26368a0acb5e6aef3257b889f61d3ee254a6a6b517ebd7
readonly MODEL_ROOT=/home/cv/memnav_eval/model_assets/pi3x_69972d6e1c4492c
readonly MODEL=${MODEL_ROOT}/model.safetensors
readonly EXPECTED_MODEL_SHA=69972d6e1c4492cb4d737a84fe940e357087d81c52f5c9b7c160b49c1f41669a
readonly PI3_ROOT=${BASE_PI3_SOURCE}/third_party/Pi3
readonly ENTRYPOINT=${SOURCE_ROOT}/MemNavData/diag_pi3x_multiview_consistency.py
readonly PYTHON=/home/cv/miniconda3/envs/memnav/bin/python
readonly EXPECTED_REPEAT_SHADOW_SHA=0a4166f590c18e8a979cec05c8439b6b5da20f3a09db2e8a30b2fd9fc2a56be8
: "${RESULT_ROOT:?set a fresh immutable result root}"

fail() { echo "ABORT: $*" >&2; exit 2; }
sha_is() {
  [[ -f "$1" ]] || fail "missing input $1"
  [[ "$(sha256sum "$1" | awk '{print $1}')" == "$2" ]] || fail "SHA mismatch $1"
}

sha_is "${SOURCE_RECEIPT}" "${EXPECTED_SOURCE_RECEIPT_SHA}"
(cd "${SOURCE_ROOT}" && sha256sum -c "${SOURCE_RECEIPT}") >/dev/null || fail "source bundle changed"
sha_is "${BASE_PI3_RECEIPT}" "${EXPECTED_BASE_PI3_RECEIPT_SHA}"
(cd "${BASE_PI3_SOURCE}" && sha256sum -c "${BASE_PI3_RECEIPT}") >/dev/null || fail "base Pi3 source changed"
sha_is "${STATIC_CSV}" "${EXPECTED_STATIC_CSV_SHA}"
sha_is "${REQUIRED_PATHS}" "${EXPECTED_REQUIRED_PATHS_SHA}"
sha_is "${MODEL}" "${EXPECTED_MODEL_SHA}"
[[ -d "${DATA_ROOT}" && -d "${PI3_ROOT}" && -f "${ENTRYPOINT}" ]] || fail "data or source absent"
while IFS= read -r relative; do
  [[ -f "${DATA_ROOT}/${relative}" ]] || fail "missing required path ${relative}"
done < "${REQUIRED_PATHS}"
[[ ! -e "${RESULT_ROOT}" ]] || fail "result root exists"
mkdir -p "${RESULT_ROOT}/smoke" "${RESULT_ROOT}/train40"

export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=${SOURCE_ROOT}:${PI3_ROOT}${PYTHONPATH:+:${PYTHONPATH}}
export PYTHONPYCACHEPREFIX=${RESULT_ROOT}/pycache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

"${PYTHON}" - <<'PY'
import torch
if not torch.cuda.is_available() or "5090" not in torch.cuda.get_device_name(0):
    raise SystemExit("expected shared RTX 5090")
print({"preflight": "passed", "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__})
PY

"${PYTHON}" -m pytest -q -p no:cacheprovider \
  "${SOURCE_ROOT}/MemNavData/test_diag_pi3x_multiview_consistency.py"

COMMON=(
  --rows-csv "${STATIC_CSV}"
  --expected-rows-sha256 "${EXPECTED_STATIC_CSV_SHA}"
  --data-root "${DATA_ROOT}"
  --pi3-root "${PI3_ROOT}"
  --snapshot "${MODEL_ROOT}"
  --expected-model-sha256 "${EXPECTED_MODEL_SHA}"
  --history-mode causal_bridge
  --bridge-frames 16
  --anchor-offsets -8 0 8
  --quiet
)

"${PYTHON}" -u "${ENTRYPOINT}" "${COMMON[@]}" \
  --row-indices 40 0 11 16 91 --expected-output-rows 5 \
  --output-jsonl "${RESULT_ROOT}/smoke/pi3x_smoke.jsonl" \
  --output-spatial-npz "${RESULT_ROOT}/smoke/pi3x_spatial_smoke.npz"

"${PYTHON}" - <<PY
import json
import numpy as np
from pathlib import Path
p = Path("${RESULT_ROOT}/smoke/pi3x_spatial_smoke.npz")
r = json.loads(Path(str(p) + ".receipt.json").read_text())
with np.load(p) as values:
    assert values["row_indices"].tolist() == [40, 0, 11, 16, 91]
    assert values["view_world_points_in_current"].shape[2:] == (9, 16, 3)
    assert np.isfinite(values["view_world_points_in_current"]).all()
    assert np.isfinite(values["view_local_points"]).all()
    assert r["contains_reporting_labels"] is False
    assert r["contains_certificate_features"] is False
print({"spatial_smoke": "passed", "shape": r["world_points_shape"]})
PY

"${PYTHON}" -u "${ENTRYPOINT}" "${COMMON[@]}" \
  --all-available --expected-output-rows 3840 \
  --output-jsonl "${RESULT_ROOT}/train40/pi3x_b16_shadow_repeat.jsonl" \
  --output-spatial-npz "${RESULT_ROOT}/train40/pi3x_b16_spatial.npz"

sha_is "${RESULT_ROOT}/train40/pi3x_b16_shadow_repeat.jsonl" "${EXPECTED_REPEAT_SHADOW_SHA}"
"${PYTHON}" - <<PY
import json
import numpy as np
from pathlib import Path
p = Path("${RESULT_ROOT}/train40/pi3x_b16_spatial.npz")
r = json.loads(Path(str(p) + ".receipt.json").read_text())
with np.load(p) as values:
    assert values["row_indices"].tolist() == list(range(3840))
    assert values["view_world_points_in_current"].shape == (3840, 20, 9, 16, 3)
    assert values["view_valid"].sum() == values["view_counts"].sum()
    for name in ("view_world_points_in_current", "view_local_points", "view_confidence", "view_poses_in_current"):
        assert np.isfinite(values[name]).all(), name
    assert r["contains_reporting_labels"] is False
    assert r["contains_certificate_features"] is False
print({"spatial_full": "passed", "shape": r["world_points_shape"]})
PY

find "${RESULT_ROOT}" -type f ! -path '*/pycache/*' ! -name OUTPUTS.sha256 -print0 | sort -z | \
  xargs -0 sha256sum > "${RESULT_ROOT}/OUTPUTS.sha256"
find "${RESULT_ROOT}" -type f ! -path '*/pycache/*' -exec chmod a-w {} +
echo "COMPLETE result=${RESULT_ROOT}"
