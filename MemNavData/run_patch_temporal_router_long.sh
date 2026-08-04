#!/usr/bin/env bash
# Build training-only router queries and evaluate the frozen-DINO patch router.
# This is an offline, scene-disjoint distillation run.  It does not modify the
# live policy and never promotes the resulting head for deployment.

set -euo pipefail
umask 0022

MODE=${1:-${MODE:-full}}
case "${MODE}" in
  smoke|full) ;;
  *) echo "ABORT: MODE must be smoke or full" >&2; exit 2 ;;
esac
INCLUDE_RETURN=${INCLUDE_RETURN:-0}
case "${INCLUDE_RETURN}" in
  0|1) ;;
  *) echo "ABORT: INCLUDE_RETURN must be 0 or 1" >&2; exit 2 ;;
esac
ALLOW_DIRTY_TASK_FILES=${ALLOW_DIRTY_TASK_FILES:-0}
case "${ALLOW_DIRTY_TASK_FILES}" in
  0|1) ;;
  *) echo "ABORT: ALLOW_DIRTY_TASK_FILES must be 0 or 1" >&2; exit 2 ;;
esac

ROOT=${ROOT:-/home/asus/Research/Nav-axis-uturn}
RUN_ROOT=${RUN_ROOT:?set RUN_ROOT to a new output directory}
MEMNAV_PY=${MEMNAV_PY:-/home/asus/miniconda3/envs/memnav/bin/python}
LINGBOT_REPO=${LINGBOT_REPO:?set LINGBOT_REPO to the pinned LingBot checkout}
WEIGHTS=${WEIGHTS:-${LINGBOT_REPO}/weights/lingbot-map-long.pt}
BASE_DIR=${BASE_DIR:-${ROOT}/.diagnostics/router_distillation_20260804/fixed_train4_test5_exact}
BASE_TEACHER=${BASE_TEACHER:-${BASE_DIR}/teacher_pairs.csv}
CLS_CACHE=${CLS_CACHE:-${BASE_DIR}/exact_dino_cls.npz}

EXPECTED_BASE_SHA=7aa916080eeec15ad505ca6b8c2349ac2383a9846ee1bb20ed704c3df350c779
EXPECTED_CLS_SHA=5d920cf32756c26a45a3c854f1e18103cb6980cbba23684815584130db7a8d7b
EXPECTED_WEIGHT_SHA=832bc82cbae0bc9bbe946ef5ee1f7226abd8c0e183ccf8beddbb3d133576f409
EXPECTED_LINGBOT_COMMIT=7ff6f3ed0913d4d326f8f13bbb429c4ffc0195c2

BUILDER=${ROOT}/MemNavData/build_router_cross_episode_pairs.py
DIAGNOSTIC=${ROOT}/MemNavData/diag_patch_temporal_router.py
UNIT_TESTS=(
  MemNavData.test_reliability_router
  MemNavData.test_patch_temporal_router
  MemNavData.test_router_cross_episode_pairs
  MemNavData.test_router_dataset_selection
)
TRAIN_SCENES=(17DRP5sb8fy 1LXtFkjw3qL 1pXnuDYAj8r Uxmj2M2itWa)
HELDOUT_SCENES=(e9zR4mvMWw7 rqfALeAoiTq s8pcmisQ38h yqstnuAEVhm zsNo4HB9uLZ)

if [[ -e "${RUN_ROOT}" ]]; then
  echo "ABORT: output already exists: ${RUN_ROOT}" >&2
  exit 1
fi
mkdir -p "${RUN_ROOT}"
exec > >(tee "${RUN_ROOT}/run.log") 2>&1

echo "[preflight] mode=${MODE} root=${ROOT} output=${RUN_ROOT}"
for required in "${MEMNAV_PY}" "${LINGBOT_REPO}" "${WEIGHTS}" \
                "${BASE_TEACHER}" "${CLS_CACHE}" "${BUILDER}" \
                "${DIAGNOSTIC}"; do
  test -r "${required}" || {
    echo "ABORT: missing dependency ${required}" >&2
    exit 1
  }
done

actual_base_sha=$(sha256sum "${BASE_TEACHER}" | awk '{print $1}')
actual_cls_sha=$(sha256sum "${CLS_CACHE}" | awk '{print $1}')
actual_weight_sha=$(sha256sum "${WEIGHTS}" | awk '{print $1}')
actual_lingbot_commit=$(git -C "${LINGBOT_REPO}" rev-parse HEAD)
[[ "${actual_base_sha}" == "${EXPECTED_BASE_SHA}" ]] || {
  echo "ABORT: teacher SHA mismatch ${actual_base_sha}" >&2; exit 1; }
[[ "${actual_cls_sha}" == "${EXPECTED_CLS_SHA}" ]] || {
  echo "ABORT: CLS SHA mismatch ${actual_cls_sha}" >&2; exit 1; }
[[ "${actual_weight_sha}" == "${EXPECTED_WEIGHT_SHA}" ]] || {
  echo "ABORT: weight SHA mismatch ${actual_weight_sha}" >&2; exit 1; }
[[ "${actual_lingbot_commit}" == "${EXPECTED_LINGBOT_COMMIT}" ]] || {
  echo "ABORT: LingBot commit mismatch ${actual_lingbot_commit}" >&2; exit 1; }

if [[ -n "${EXPECTED_COMMIT:-}" ]]; then
  actual_commit=$(git -C "${ROOT}" rev-parse HEAD)
  [[ "${actual_commit}" == "${EXPECTED_COMMIT}" ]] || {
    echo "ABORT: code commit ${actual_commit} != ${EXPECTED_COMMIT}" >&2
    exit 1
  }
fi
if [[ "${ALLOW_DIRTY_TASK_FILES}" != 1 ]]; then
  git -C "${ROOT}" diff --quiet -- \
    MemNavData/patch_temporal_router.py \
    MemNavData/diag_patch_temporal_router.py \
    MemNavData/build_router_cross_episode_pairs.py \
    MemNavData/run_patch_temporal_router_long.sh || {
      echo "ABORT: router task files differ from the checked-out commit" >&2
      exit 1
    }
else
  echo "WARNING: dirty task files allowed for local smoke only"
fi

cd "${ROOT}"
"${MEMNAV_PY}" -m py_compile \
  MemNavData/patch_temporal_router.py \
  MemNavData/diag_patch_temporal_router.py \
  MemNavData/build_router_cross_episode_pairs.py
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
echo "teacher_sha=${actual_base_sha}"
echo "cls_sha=${actual_cls_sha}"
echo "weight_sha=${actual_weight_sha}"
echo "lingbot_commit=${actual_lingbot_commit}"

EXPANDED=${RUN_ROOT}/expanded_teacher_pairs.csv
EXPANSION_REPORT=${RUN_ROOT}/expansion_report.json
EXPAND_ARGS=(
  --base-teacher-csv "${BASE_TEACHER}"
  --cls-cache "${CLS_CACHE}"
  --out-csv "${EXPANDED}"
  --report "${EXPANSION_REPORT}"
  --expected-base-sha "${EXPECTED_BASE_SHA}"
)
for scene in "${TRAIN_SCENES[@]}"; do
  EXPAND_ARGS+=(--train-scene "${scene}")
done
for scene in "${HELDOUT_SCENES[@]}"; do
  EXPAND_ARGS+=(--evaluation-scene "${scene}")
done
if [[ "${INCLUDE_RETURN}" == 1 ]]; then
  EXPAND_ARGS+=(--include-return)
fi
for mapping in ${PATH_MAPS:-}; do
  EXPAND_ARGS+=(--path-map "${mapping}")
done

if [[ "${MODE}" == smoke ]]; then
  EXPAND_ARGS+=(
    --query-stride 128 --query-margin 16 --candidate-stride 4
    --max-queries-per-episode 1
    --return-query-stride 128 --return-query-margin 16
    --return-candidate-stride 4 --max-return-queries-per-episode 1
  )
  TOP_K=4
  GRID_SIZE=4
  BATCH_SIZE=16
else
  EXPAND_ARGS+=(
    --query-stride 32 --query-margin 16 --candidate-stride 1
    --return-query-stride 32 --return-query-margin 16
    --return-candidate-stride 1
  )
  TOP_K=32
  GRID_SIZE=8
  BATCH_SIZE=24
fi
EXPAND_ARGS+=(--teacher-top-k "${TOP_K}")

echo "[stage 1/2] training-only geometry-teacher expansion"
"${MEMNAV_PY}" -u "${BUILDER}" "${EXPAND_ARGS[@]}"
"${MEMNAV_PY}" - "${BASE_TEACHER}" "${EXPANDED}" \
  "${INCLUDE_RETURN}" <<'PY'
import sys
import pandas as pd

base = pd.read_csv(sys.argv[1])
expanded = pd.read_csv(sys.argv[2])
train_scenes = {
    "17DRP5sb8fy", "1LXtFkjw3qL", "1pXnuDYAj8r", "Uxmj2M2itWa"}
heldout_scenes = {
    "e9zR4mvMWw7", "rqfALeAoiTq", "s8pcmisQ38h", "yqstnuAEVhm",
    "zsNo4HB9uLZ"}
include_return = bool(int(sys.argv[3]))
if len(expanded) <= len(base):
    raise RuntimeError("expansion did not add any training pairs")
if not base.equals(expanded.iloc[:len(base)].reset_index(drop=True)):
    raise RuntimeError("base teacher rows changed during expansion")
added = expanded.iloc[len(base):]
train_added = added[added["kind"].str.endswith("_train")]
evaluation_added = added[added["kind"].str.endswith("_evaluation")]
if set(train_added["scene"]) != train_scenes:
    raise RuntimeError("training expansion has an incorrect scene split")
if set(evaluation_added["scene"]) != heldout_scenes:
    raise RuntimeError("evaluation expansion has an incorrect scene split")
expected_train_kinds = {"cross_episode_train"}
expected_evaluation_kinds = {"cross_episode_evaluation"}
if include_return:
    expected_train_kinds.add("within_episode_return_train")
    expected_evaluation_kinds.add("within_episode_return_evaluation")
if set(train_added["kind"]) != expected_train_kinds:
    raise RuntimeError("unexpected training expansion kind")
if set(evaluation_added["kind"]) != expected_evaluation_kinds:
    raise RuntimeError("unexpected evaluation expansion kind")
if expanded.duplicated(["session_id", "candidate_path"]).any():
    raise RuntimeError("duplicate session/candidate pair after expansion")
if not train_added["teacher_pass"].isin([-1, 0, 1]).all():
    raise RuntimeError("invalid sparse teacher label in training expansion")
if not evaluation_added["teacher_pass"].isin([-1, 0, 1]).all():
    raise RuntimeError("invalid sparse teacher label in evaluation expansion")
print(
    "split audit OK:", len(train_added), "training pairs;",
    len(evaluation_added), "evaluation-only pairs")
PY

DIAG_ARGS=(
  --teacher-csv "${EXPANDED}"
  --cls-cache "${CLS_CACHE}"
  --lingbot-repo "${LINGBOT_REPO}"
  --weights "${WEIGHTS}"
  --out-dir "${RUN_ROOT}/patch_temporal"
  --device cuda:0
  --batch-size "${BATCH_SIZE}"
  --top-k "${TOP_K}"
  --grid-size "${GRID_SIZE}"
  --expected-weight-sha "${EXPECTED_WEIGHT_SHA}"
  --expected-lingbot-commit "${EXPECTED_LINGBOT_COMMIT}"
)
for scene in "${HELDOUT_SCENES[@]}"; do
  DIAG_ARGS+=(--heldout-scene "${scene}")
done
for mapping in ${PATH_MAPS:-}; do
  DIAG_ARGS+=(--path-map "${mapping}")
done

echo "[stage 2/2] scene-disjoint patch/temporal router ablation"
"${MEMNAV_PY}" -u "${DIAGNOSTIC}" "${DIAG_ARGS[@]}"

echo "[complete] diagnostic only; deployment_approved remains false"
echo "report=${RUN_ROOT}/patch_temporal/report.json"
