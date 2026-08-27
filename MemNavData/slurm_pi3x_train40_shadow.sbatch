#!/usr/bin/env bash
# Label-blind train40 shadow inference for the unified Pi3X relocalizer.
# This job never changes navigation actions and never reads dev/blind data.
#SBATCH --job-name=pi3x_shadow
#SBATCH --output=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/%x_%j.out
#SBATCH --error=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/%x_%j.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=h100_tandon,a100_tandon
#SBATCH --qos=gpu48
#SBATCH --gres=gpu:1
#SBATCH --account=torch_pr_769_tandon_advanced
#SBATCH --export=ALL

set -euo pipefail
umask 0022

: "${SOURCE_ROOT:?set immutable task source root}"
: "${SOURCE_RECEIPT:?set immutable source receipt}"
: "${EXPECTED_SOURCE_RECEIPT_SHA:?set source receipt SHA}"
: "${RUN_ROOT:?set unique result root}"
: "${MODE:?set smoke or full}"

[[ "${MODE}" == smoke || "${MODE}" == full ]] || {
  echo "ABORT: MODE must be smoke or full" >&2; exit 2; }

readonly ENTRYPOINT=${SOURCE_ROOT}/MemNavData/diag_pi3x_multiview_consistency.py
readonly SUMMARIZER=${SOURCE_ROOT}/MemNavData/summarize_pi3x_multiview_shadow.py
readonly PI3_ROOT=${SOURCE_ROOT}/third_party/Pi3
readonly MODEL_ROOT=/scratch/yz11502/Research/model_assets/pi3x_69972d6e1c4492c
readonly MODEL=${MODEL_ROOT}/model.safetensors
readonly STATIC_CSV=/scratch/yz11502/Research/Nav-axis-uturn-results/lingbot_lightglue_relocalization_v2_20260812/bundle_f73016d5ecfebfa7_job_15634113/static_top8_480/lightglue_open_set_rows.csv
readonly BASE_SIF=/share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif
readonly OVERLAY=/scratch/lg154/Research/datasets/_overlays/mp3d_revisit_v0_pt1.sqf
readonly MEMNAV_PY=/scratch/lg154/conda-envs/memnav/bin/python
readonly EPISODE_ROOT=/mp3d_revisit_v0/vln_n1/traj_data/mp3d_3leg
readonly GAP_ROOT=/scratch/yz11502/Research/Nav-axis-uturn-results/nlsr_gapfill_20260807/nlsr_routes_41c1773_20260807_v1
readonly RAW_GAP_SCENE=${GAP_ROOT}/raw/mp3d_3leg/YmJkqBEsHnH
readonly EXPECTED_MODEL_SHA=69972d6e1c4492cb4d737a84fe940e357087d81c52f5c9b7c160b49c1f41669a
readonly EXPECTED_STATIC_CSV_SHA=85f9064bff15ce59106ad2a1aa8e5dc4720ee1b1ad894aac1bcedf8581a1d127

fail() { echo "ABORT: $*" >&2; exit 2; }
sha_is() {
  [[ -f "$1" && ! -L "$1" ]] || fail "missing physical input: $1"
  [[ "$(sha256sum "$1" | awk '{print $1}')" == "$2" ]] || \
    fail "SHA mismatch: $1"
}

sha_is "${SOURCE_RECEIPT}" "${EXPECTED_SOURCE_RECEIPT_SHA}"
(cd "${SOURCE_ROOT}" && sha256sum -c "${SOURCE_RECEIPT}") >/dev/null || \
  fail "source bundle changed"
sha_is "${MODEL}" "${EXPECTED_MODEL_SHA}"
sha_is "${STATIC_CSV}" "${EXPECTED_STATIC_CSV_SHA}"
[[ -f "${BASE_SIF}" && -f "${OVERLAY}" && -d "${RAW_GAP_SCENE}" ]] || \
  fail "container, PT1 overlay, or gap scene is absent"
[[ -f "${PI3_ROOT}/pi3/models/pi3x.py" ]] || fail "Pi3X source is absent"

mkdir -p "${RUN_ROOT}"
readonly OUTPUT=${RUN_ROOT}/pi3x_${MODE}.jsonl
[[ ! -e "${OUTPUT}" && ! -e "${OUTPUT}.receipt.json" ]] || \
  fail "immutable output already exists"

export PYTHONPYCACHEPREFIX=${SLURM_TMPDIR:-/tmp}/pi3x_shadow_${SLURM_JOB_ID}
mkdir -p "${PYTHONPYCACHEPREFIX}"
export PYTHONPATH=${SOURCE_ROOT}:${PI3_ROOT}${PYTHONPATH:+:${PYTHONPATH}}
export APPTAINERENV_PYTHONPATH=${PYTHONPATH}
export SINGULARITYENV_PYTHONPATH=${PYTHONPATH}
export APPTAINERENV_PYTHONPYCACHEPREFIX=${PYTHONPYCACHEPREFIX}
export SINGULARITYENV_PYTHONPYCACHEPREFIX=${PYTHONPYCACHEPREFIX}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CONTAINER=(singularity exec --nv --overlay "${OVERLAY}:ro"
  -B /scratch/lg154 -B /scratch/yz11502
  -B "${RAW_GAP_SCENE}:${EPISODE_ROOT}/YmJkqBEsHnH:ro"
  "${BASE_SIF}")

echo "host=$(hostname) job=${SLURM_JOB_ID} mode=${MODE} gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1)"
"${CONTAINER[@]}" "${MEMNAV_PY}" -m py_compile "${ENTRYPOINT}" "${SUMMARIZER}"
"${CONTAINER[@]}" "${MEMNAV_PY}" - "${PI3_ROOT}" <<'PY'
import importlib.util
import sys
import torch

sys.path.insert(0, sys.argv[1])
from pi3.models.pi3x import Pi3X

required = ("numpy", "PIL", "pyarrow", "scipy", "safetensors")
missing = [name for name in required if importlib.util.find_spec(name) is None]
if missing:
    raise SystemExit(f"missing dependencies: {missing}")
if not torch.cuda.is_available():
    raise SystemExit("CUDA unavailable")
print({"preflight": "passed", "torch": torch.__version__})
PY

COMMON=(
  --rows-csv "${STATIC_CSV}"
  --expected-rows-sha256 "${EXPECTED_STATIC_CSV_SHA}"
  --data-root "${EPISODE_ROOT}"
  --pi3-root "${PI3_ROOT}"
  --snapshot "${MODEL_ROOT}"
  --expected-model-sha256 "${EXPECTED_MODEL_SHA}"
  --output-jsonl "${OUTPUT}"
  --history-mode causal_bridge
  --bridge-frames 8
  --anchor-offsets -8 0 8
  --quiet
)
if [[ "${MODE}" == smoke ]]; then
  "${CONTAINER[@]}" "${MEMNAV_PY}" -u "${ENTRYPOINT}" \
    "${COMMON[@]}" --row-indices 40 0 11 16 --expected-output-rows 4
else
  "${CONTAINER[@]}" "${MEMNAV_PY}" -u "${ENTRYPOINT}" \
    "${COMMON[@]}" --all-available --expected-output-rows 3840
  "${CONTAINER[@]}" "${MEMNAV_PY}" -u "${SUMMARIZER}" \
    --shadow-jsonl "${OUTPUT}" \
    --rows-csv "${STATIC_CSV}" \
    --expected-rows 3840 \
    --expected-rows-sha256 "${EXPECTED_STATIC_CSV_SHA}" \
    --output-summary "${RUN_ROOT}/pi3x_train40_summary.json" \
    --output-predictions "${RUN_ROOT}/pi3x_train40_oof_predictions.csv"
fi

OUTPUT_FILES=("${OUTPUT}" "${OUTPUT}.receipt.json")
if [[ "${MODE}" == full ]]; then
  OUTPUT_FILES+=("${RUN_ROOT}/pi3x_train40_summary.json"
    "${RUN_ROOT}/pi3x_train40_oof_predictions.csv")
fi
sha256sum "${OUTPUT_FILES[@]}" > "${RUN_ROOT}/OUTPUTS.sha256"
chmod a-w "${OUTPUT_FILES[@]}" "${RUN_ROOT}/OUTPUTS.sha256"
echo "COMPLETE mode=${MODE} output=${OUTPUT}"
