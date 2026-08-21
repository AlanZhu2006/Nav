#!/usr/bin/env bash
# Freeze the four-member spatial-proof ensemble for the next fresh evaluation.
set -euo pipefail
umask 0022

readonly SOURCE_ROOT=/home/cv/memnav_eval/source_bundles/pi3x_spatial_deployment_689c592debb19d7e
readonly SOURCE_RECEIPT=${SOURCE_ROOT}/SOURCE_BUNDLE.sha256
readonly EXPECTED_SOURCE_RECEIPT_SHA=689c592debb19d7e0cced9583dcff0cfcd89d1f2135fa74f833e45bf2b19e24c
readonly TOKEN_ROOT=/home/cv/memnav_eval/results/pi3x_learned_relocalizer_20260817/viewtokens_b16_3b9db95668bfe54f_20260817_051500
readonly DESCRIPTORS=${TOKEN_ROOT}/pi3x_b16_viewtokens.npz
readonly EXPECTED_DESCRIPTORS_SHA=a9e3a87e11efcf979a35d000d4835d9116f33a6e3bbb75dfc69f0e644957b773
readonly SPATIAL_ROOT=/home/cv/memnav_eval/results/pi3x_learned_relocalizer_20260817/spatial_export_157bb003103f7e11_20260816T214333Z/train40
readonly SPATIAL=${SPATIAL_ROOT}/pi3x_b16_spatial.npz
readonly EXPECTED_SPATIAL_SHA=23bc0eb6357942248561a7509c3751ed4c8fe90e7f845b47ed3ad9a1bf306342
readonly SHADOW=${SPATIAL_ROOT}/pi3x_b16_shadow_repeat.jsonl
readonly EXPECTED_SHADOW_SHA=eef5cbf089d5ed607f39b2b64dc87136238584db1df7e05ce3acaebc4b99f155
readonly STATIC_CSV=/home/cv/memnav_eval/model_assets/pi3x_train40_inputs_434388205f77815b/static_top8_480_lightglue_open_set_rows.csv
readonly EXPECTED_STATIC_CSV_SHA=85f9064bff15ce59106ad2a1aa8e5dc4720ee1b1ad894aac1bcedf8581a1d127
readonly TRAINER=${SOURCE_ROOT}/MemNavData/fit_pi3x_spatial_reliability_deployment.py
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
sha_is "${SHADOW}" "${EXPECTED_SHADOW_SHA}"
sha_is "${STATIC_CSV}" "${EXPECTED_STATIC_CSV_SHA}"
[[ -f "${TRAINER}" && ! -e "${RESULT_ROOT}" ]] || fail "trainer missing or result root exists"
mkdir -p "${RESULT_ROOT}"

export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=${SOURCE_ROOT}${PYTHONPATH:+:${PYTHONPATH}}
export PYTHONPYCACHEPREFIX=${RESULT_ROOT}/pycache

"${PYTHON}" -m pytest -q -p no:cacheprovider \
  "${SOURCE_ROOT}/MemNavData/test_train_pi3x_viewtoken_reliability_oof.py" \
  "${SOURCE_ROOT}/MemNavData/test_train_pi3x_spatial_reliability_crossfit_oof.py" \
  "${SOURCE_ROOT}/MemNavData/test_fit_pi3x_spatial_reliability_deployment.py" \
  "${SOURCE_ROOT}/MemNavData/test_pi3x_spatial_proof_runtime.py"

"${PYTHON}" -u "${TRAINER}" \
  --rows-csv "${STATIC_CSV}" \
  --shadow-jsonl "${SHADOW}" \
  --descriptors-npz "${DESCRIPTORS}" \
  --spatial-npz "${SPATIAL}" \
  --expected-rows 3840 \
  --expected-rows-sha256 "${EXPECTED_STATIC_CSV_SHA}" \
  --expected-spatial-sha256 "${EXPECTED_SPATIAL_SHA}" \
  --members 4 \
  --consensus-numerator 2 \
  --consensus-denominator 4 \
  --minimum-precision 0.90 \
  --maximum-fpr 0.0275 \
  --model-dim 64 \
  --layers 2 \
  --heads 4 \
  --epochs 30 \
  --batch-sessions 12 \
  --inference-batch-rows 64 \
  --learning-rate 3e-4 \
  --weight-decay 1e-3 \
  --support-weight 0.25 \
  --seed 17 \
  --device cuda \
  --checkpoint-dir "${RESULT_ROOT}/checkpoints" \
  --output-manifest "${RESULT_ROOT}/deployment_manifest.json"

# Load the exact frozen artifacts through the production-facing class and run
# one eight-candidate session.  This validates hash binding, relative paths,
# overlap selection, proof voting and fail-closed output without reporting a
# performance metric from the training population.
"${PYTHON}" - <<PY
import json
from pathlib import Path
import numpy as np
from MemNavData.pi3x_spatial_proof_runtime import Pi3XSpatialProofEnsemble

runtime = Pi3XSpatialProofEnsemble(
    Path("${RESULT_ROOT}/deployment_manifest.json"), device="cuda"
)
shadow = [json.loads(line) for line in Path("${SHADOW}").read_text().splitlines()[:8]]
with np.load("${DESCRIPTORS}") as global_tokens, np.load("${SPATIAL}") as spatial:
    decision = runtime.decide(
        overlaps=[row["best_view_f1_20cm"] for row in shadow],
        bearings_forward_left=[row["predicted_scale_free_bearing"] for row in shadow],
        descriptors=global_tokens["view_descriptors"][:8],
        roles=global_tokens["view_roles"][:8],
        relative_age=global_tokens["view_relative_age"][:8],
        valid=global_tokens["view_valid"][:8],
        world_points_in_current=spatial["view_world_points_in_current"][:8],
        local_points=spatial["view_local_points"][:8],
        confidence=spatial["view_confidence"][:8],
        poses_in_current=spatial["view_poses_in_current"][:8],
    )
if decision.status == "error":
    raise SystemExit(decision.reason)
Path("${RESULT_ROOT}/runtime_smoke.json").write_text(
    json.dumps(decision.as_dict(), indent=2, sort_keys=True) + "\n"
)
print({"runtime_smoke": decision.as_dict()})
PY

(cd "${RESULT_ROOT}" && \
  find . -type f ! -path '*/pycache/*' ! -name OUTPUTS.sha256 -print0 | \
  sort -z | xargs -0 sha256sum > OUTPUTS.sha256)
(cd "${SOURCE_ROOT}" && "${PY}" \
  MemNavData/verify_portable_checksum_manifest.py \
  "${RESULT_ROOT}/OUTPUTS.sha256" --quiet)
find "${RESULT_ROOT}" -type f ! -path '*/pycache/*' -exec chmod a-w {} +
echo "COMPLETE result=${RESULT_ROOT}"
