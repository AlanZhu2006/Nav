#!/usr/bin/env bash
# Scene-balanced geometry-teacher distillation from the read-only MP3D overlay.
# This is an offline diagnostic.  It never changes the live navigation policy.

set -euo pipefail
umask 0022

MODE=${1:-${MODE:-full}}
case "${MODE}" in
  smoke|full) ;;
  *) echo "ABORT: MODE must be smoke or full" >&2; exit 2 ;;
esac

ROOT=${ROOT:?set ROOT to the committed Nav-axis-uturn checkout}
RUN_ROOT=${RUN_ROOT:?set RUN_ROOT to a new output directory}
EPISODE_ROOT=${EPISODE_ROOT:?set EPISODE_ROOT to the mounted mp3d_2leg directory}
MEMNAV_PY=${MEMNAV_PY:?set MEMNAV_PY to the pinned Python interpreter}
LINGBOT_REPO=${LINGBOT_REPO:?set LINGBOT_REPO to the pinned LingBot checkout}
WEIGHTS=${WEIGHTS:-${LINGBOT_REPO}/weights/lingbot-map-long.pt}
SPLIT_MANIFEST=${SPLIT_MANIFEST:-${ROOT}/MemNavData/router_multiscene_split_20260805.json}

EXPECTED_SPLIT_SHA=97309c183e25cb3dd65472908748d55a94798a636db6157ab6fe120fca05cf7a
EXPECTED_WEIGHT_SHA=832bc82cbae0bc9bbe946ef5ee1f7226abd8c0e183ccf8beddbb3d133576f409
EXPECTED_LINGBOT_COMMIT=7ff6f3ed0913d4d326f8f13bbb429c4ffc0195c2

BASE_DIAGNOSTIC=${ROOT}/MemNavData/diag_distill_geometry_router.py
EXPANDER=${ROOT}/MemNavData/build_router_cross_episode_pairs.py
PATCH_DIAGNOSTIC=${ROOT}/MemNavData/diag_patch_temporal_router.py
UNIT_TESTS=(
  MemNavData.test_reliability_router
  MemNavData.test_patch_temporal_router
  MemNavData.test_router_cross_episode_pairs
  MemNavData.test_router_dataset_selection
)
TASK_FILES=(
  MemNavData/diag_distill_geometry_router.py
  MemNavData/build_router_cross_episode_pairs.py
  MemNavData/patch_temporal_router.py
  MemNavData/diag_patch_temporal_router.py
  MemNavData/run_patch_temporal_router_multiscene.sh
  MemNavData/router_multiscene_split_20260805.json
)

if [[ -e "${RUN_ROOT}" ]]; then
  echo "ABORT: output already exists: ${RUN_ROOT}" >&2
  exit 1
fi
mkdir -p "${RUN_ROOT}"
exec > >(tee "${RUN_ROOT}/run.log") 2>&1

echo "[preflight] mode=${MODE} root=${ROOT} output=${RUN_ROOT}"
for required in "${MEMNAV_PY}" "${LINGBOT_REPO}" "${WEIGHTS}" \
                "${SPLIT_MANIFEST}" "${EPISODE_ROOT}" \
                "${BASE_DIAGNOSTIC}" "${EXPANDER}" \
                "${PATCH_DIAGNOSTIC}"; do
  test -r "${required}" || {
    echo "ABORT: missing dependency ${required}" >&2
    exit 1
  }
done

actual_split_sha=$(sha256sum "${SPLIT_MANIFEST}" | awk '{print $1}')
actual_weight_sha=$(sha256sum "${WEIGHTS}" | awk '{print $1}')
actual_lingbot_commit=$(git -C "${LINGBOT_REPO}" rev-parse HEAD)
[[ "${actual_split_sha}" == "${EXPECTED_SPLIT_SHA}" ]] || {
  echo "ABORT: split SHA mismatch ${actual_split_sha}" >&2; exit 1; }
[[ "${actual_weight_sha}" == "${EXPECTED_WEIGHT_SHA}" ]] || {
  echo "ABORT: weight SHA mismatch ${actual_weight_sha}" >&2; exit 1; }
[[ "${actual_lingbot_commit}" == "${EXPECTED_LINGBOT_COMMIT}" ]] || {
  echo "ABORT: LingBot commit mismatch ${actual_lingbot_commit}" >&2; exit 1; }

actual_commit=$(git -C "${ROOT}" rev-parse HEAD)
if [[ -n "${EXPECTED_COMMIT:-}" && "${actual_commit}" != "${EXPECTED_COMMIT}" ]]; then
  echo "ABORT: code commit ${actual_commit} != ${EXPECTED_COMMIT}" >&2
  exit 1
fi
git -C "${ROOT}" diff --quiet -- "${TASK_FILES[@]}" || {
  echo "ABORT: router task files differ from the checked-out commit" >&2
  exit 1
}

cd "${ROOT}"
"${MEMNAV_PY}" -m py_compile \
  MemNavData/diag_distill_geometry_router.py \
  MemNavData/build_router_cross_episode_pairs.py \
  MemNavData/patch_temporal_router.py \
  MemNavData/diag_patch_temporal_router.py
"${MEMNAV_PY}" -m unittest "${UNIT_TESTS[@]}" -v
"${MEMNAV_PY}" - <<'PY'
import cv2
import numpy
import pandas
import sklearn
import torch
import PIL
assert torch.cuda.is_available(), "CUDA is unavailable"
print("dependencies OK")
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("opencv", cv2.__version__, "numpy", numpy.__version__)
print("pandas", pandas.__version__, "sklearn", sklearn.__version__)
print("Pillow", PIL.__version__)
PY
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

mapfile -t ALL_TRAIN < <("${MEMNAV_PY}" - "${SPLIT_MANIFEST}" <<'PY'
import json, sys
with open(sys.argv[1], encoding="utf-8") as handle:
    split = json.load(handle)
print(*split["train"], sep="\n")
PY
)
mapfile -t ALL_DEVELOPMENT < <("${MEMNAV_PY}" - "${SPLIT_MANIFEST}" <<'PY'
import json, sys
with open(sys.argv[1], encoding="utf-8") as handle:
    split = json.load(handle)
print(*split["development"], sep="\n")
PY
)
mapfile -t FINAL_RESERVED < <("${MEMNAV_PY}" - "${SPLIT_MANIFEST}" <<'PY'
import json, sys
with open(sys.argv[1], encoding="utf-8") as handle:
    split = json.load(handle)
print(*split["final_reserved"], sep="\n")
PY
)

if [[ "${MODE}" == smoke ]]; then
  TRAIN_SCENES=("${ALL_TRAIN[@]:0:4}")
  DEVELOPMENT_SCENES=("${ALL_DEVELOPMENT[@]:0:2}")
  QUERY_STRIDE=128
  MAX_QUERIES=1
  CROSS_CANDIDATE_STRIDE=4
  TOP_K=4
  GRID_SIZE=4
  BATCH_SIZE=24
else
  TRAIN_SCENES=("${ALL_TRAIN[@]}")
  DEVELOPMENT_SCENES=("${ALL_DEVELOPMENT[@]}")
  QUERY_STRIDE=64
  MAX_QUERIES=4
  CROSS_CANDIDATE_STRIDE=1
  TOP_K=32
  GRID_SIZE=8
  BATCH_SIZE=48
fi

"${MEMNAV_PY}" - "${SPLIT_MANIFEST}" "${EPISODE_ROOT}" \
  "${MODE}" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

manifest = Path(sys.argv[1])
episode_root = Path(sys.argv[2])
mode = sys.argv[3]
with open(manifest, encoding="utf-8") as handle:
    split = json.load(handle)
train = split["train"]
development = split["development"]
reserved = split["final_reserved"]
if not (len(train) == 40 and len(development) == 10 and len(reserved) == 4):
    raise RuntimeError("split cardinality changed")
if set(train) & set(development) or set(train + development) & set(reserved):
    raise RuntimeError("scene roles overlap")
forced = set(split["selection_rule"]["forced_train"])
if not forced.issubset(train):
    raise RuntimeError("forced baseline scenes left training split")
salt = split["selection_rule"]["salt"]
remaining = sorted(
    set(train + development) - forced,
    key=lambda scene: hashlib.sha256(
        f"{salt}:{scene}".encode()).hexdigest())
expected_development = set(remaining[:10])
if set(development) != expected_development:
    raise RuntimeError("development split does not match the frozen hash rule")
available = {path.name for path in episode_root.iterdir() if path.is_dir()}
required = set(train + development + reserved)
if not required.issubset(available):
    raise RuntimeError(
        f"overlay lacks scenes: {sorted(required - available)}")
selected = train[:4] + development[:2] if mode == "smoke" else train + development
for scene in selected:
    episodes = sorted(
        path for path in (episode_root / scene).iterdir() if path.is_dir())
    if len(episodes) < 2:
        raise RuntimeError(f"scene {scene} has fewer than two episodes")
print(
    "split audit OK:", len(selected), "selected scenes;",
    len(reserved), "final scenes reserved")
PY

echo "commit=${actual_commit}"
echo "split_sha=${actual_split_sha}"
echo "weight_sha=${actual_weight_sha}"
echo "lingbot_commit=${actual_lingbot_commit}"
printf "train_scenes=%s\n" "${TRAIN_SCENES[*]}"
printf "development_scenes=%s\n" "${DEVELOPMENT_SCENES[*]}"
printf "final_reserved=%s\n" "${FINAL_RESERVED[*]}"

BASE_DIR=${RUN_ROOT}/base_teacher
BASE_ARGS=(
  --episode-root "${EPISODE_ROOT}"
  --lingbot-repo "${LINGBOT_REPO}"
  --weights "${WEIGHTS}"
  --out-dir "${BASE_DIR}"
  --device cuda:0
  --batch-size "${BATCH_SIZE}"
  --candidate-stride 1
  --max-episodes-per-scene 2
  --teacher-only
  --expected-weight-sha "${EXPECTED_WEIGHT_SHA}"
)
for scene in "${TRAIN_SCENES[@]}" "${DEVELOPMENT_SCENES[@]}"; do
  BASE_ARGS+=(--include-scene "${scene}")
done
for scene in "${DEVELOPMENT_SCENES[@]}"; do
  BASE_ARGS+=(--heldout-scene "${scene}")
done

echo "[stage 1/3] exact CLS and base geometry teacher"
"${MEMNAV_PY}" -u "${BASE_DIAGNOSTIC}" "${BASE_ARGS[@]}"

BASE_TEACHER=${BASE_DIR}/teacher_pairs.csv
CLS_CACHE=${BASE_DIR}/exact_dino_cls.npz
base_sha=$(sha256sum "${BASE_TEACHER}" | awk '{print $1}')
EXPANDED=${RUN_ROOT}/expanded_teacher_pairs.csv
EXPANSION_REPORT=${RUN_ROOT}/expansion_report.json
EXPAND_ARGS=(
  --base-teacher-csv "${BASE_TEACHER}"
  --cls-cache "${CLS_CACHE}"
  --out-csv "${EXPANDED}"
  --report "${EXPANSION_REPORT}"
  --expected-base-sha "${base_sha}"
  --query-stride "${QUERY_STRIDE}"
  --query-margin 16
  --candidate-stride "${CROSS_CANDIDATE_STRIDE}"
  --max-queries-per-episode "${MAX_QUERIES}"
  --teacher-top-k "${TOP_K}"
)
for scene in "${TRAIN_SCENES[@]}"; do
  EXPAND_ARGS+=(--train-scene "${scene}")
done
for scene in "${DEVELOPMENT_SCENES[@]}"; do
  EXPAND_ARGS+=(--evaluation-scene "${scene}")
done

echo "[stage 2/3] sparse cross-episode geometry teacher"
"${MEMNAV_PY}" -u "${EXPANDER}" "${EXPAND_ARGS[@]}"

"${MEMNAV_PY}" - "${BASE_TEACHER}" "${EXPANDED}" \
  "${SPLIT_MANIFEST}" "${MODE}" <<'PY'
import json
import sys
import pandas as pd

base = pd.read_csv(sys.argv[1])
expanded = pd.read_csv(sys.argv[2])
with open(sys.argv[3], encoding="utf-8") as handle:
    split = json.load(handle)
if sys.argv[4] == "smoke":
    train = set(split["train"][:4])
    development = set(split["development"][:2])
else:
    train = set(split["train"])
    development = set(split["development"])
reserved = set(split["final_reserved"])
if not base.equals(expanded.iloc[:len(base)].reset_index(drop=True)):
    raise RuntimeError("base teacher rows changed during expansion")
if set(expanded["scene"]) & reserved:
    raise RuntimeError("final-reserved scene leaked into the diagnostic")
added = expanded.iloc[len(base):]
train_added = added[added["kind"].eq("cross_episode_train")]
dev_added = added[added["kind"].eq("cross_episode_evaluation")]
if set(train_added["scene"]) != train:
    raise RuntimeError("training expansion scene mismatch")
if set(dev_added["scene"]) != development:
    raise RuntimeError("development expansion scene mismatch")
if expanded.duplicated(["session_id", "candidate_path"]).any():
    raise RuntimeError("duplicate session/candidate pair")
print(
    "expanded split audit OK:", len(train_added), "train rows;",
    len(dev_added), "development rows")
PY

PATCH_DIR=${RUN_ROOT}/patch_temporal
PATCH_ARGS=(
  --teacher-csv "${EXPANDED}"
  --cls-cache "${CLS_CACHE}"
  --lingbot-repo "${LINGBOT_REPO}"
  --weights "${WEIGHTS}"
  --out-dir "${PATCH_DIR}"
  --device cuda:0
  --batch-size "${BATCH_SIZE}"
  --top-k "${TOP_K}"
  --grid-size "${GRID_SIZE}"
  --expected-weight-sha "${EXPECTED_WEIGHT_SHA}"
  --expected-lingbot-commit "${EXPECTED_LINGBOT_COMMIT}"
)
for scene in "${DEVELOPMENT_SCENES[@]}"; do
  PATCH_ARGS+=(--heldout-scene "${scene}")
done

echo "[stage 3/3] scene-disjoint patch/temporal audit"
"${MEMNAV_PY}" -u "${PATCH_DIAGNOSTIC}" "${PATCH_ARGS[@]}"

echo "[complete] diagnostic only; deployment_approved remains false"
echo "base_report=${BASE_DIR}/report.json"
echo "patch_report=${PATCH_DIR}/report.json"
