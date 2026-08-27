#!/usr/bin/env bash
# One-allocation paired evaluation: three-arm 2-leg plus five-arm conditional C.

set -euo pipefail
umask 0022

ROOT=${ROOT:?set ROOT}
RUN_ROOT=${RUN_ROOT:?set RUN_ROOT}
EXPECTED_COMMIT=${EXPECTED_COMMIT:?set EXPECTED_COMMIT}
HAB_PY=${HAB_PY:?set HAB_PY}
MEMNAV_PY=${MEMNAV_PY:?set MEMNAV_PY}
MODE=${MODE:?set MODE=smoke or full}
RUN_TWO_LEG=${RUN_TWO_LEG:-1}
[[ "${MODE}" =~ ^(smoke|full)$ ]] || {
  echo "ABORT: MODE must be smoke or full" >&2; exit 1; }
[[ "${RUN_TWO_LEG}" =~ ^[01]$ ]] || {
  echo "ABORT: RUN_TWO_LEG must be 0 or 1" >&2; exit 1; }

TWO_MANIFEST=${ROOT}/MemNavData/expanded_navdp_router_eval_20260805.json
TWO_MANIFEST_SHA=ba8f72cb504768c801e6c9c386436ccdc66dea07a5e5fac2d7b4248738946a61
THREE_MANIFEST=${ROOT}/MemNavData/expanded_3leg_router_eval_20260805.json
THREE_MANIFEST_SHA=55096038d0493723d2280fe9abbcb2ea9ea7f2059c50b6eb08d02f560cdb281b
SCENE_RUNNER=${ROOT}/MemNavData/run_expanded_navdp_router_scene.sh
TWO_SUMMARIZER=${ROOT}/MemNavData/summarize_expanded_navdp_router_eval.py
CONDITIONAL_EVALUATOR=${ROOT}/MemNavData/eval_conditional_c_habitat.py
CONDITIONAL_SUMMARIZER=${ROOT}/MemNavData/summarize_conditional_c_eval.py
THREE_VALIDATOR=${ROOT}/MemNavData/validate_expanded_3leg_router_eval.py

TASK_FILES=(
  MemNavData/run_three_arm_conditional_eval.sh
  MemNavData/slurm_three_arm_conditional_eval.sbatch
  MemNavData/run_expanded_navdp_router_scene.sh
  MemNavData/summarize_expanded_navdp_router_eval.py
  MemNavData/test_expanded_navdp_router_eval.py
  MemNavData/conditional_c_protocol.py
  MemNavData/test_conditional_c_protocol.py
  MemNavData/eval_conditional_c_habitat.py
  MemNavData/summarize_conditional_c_eval.py
  MemNavData/test_summarize_conditional_c_eval.py
)

actual_commit=$(git -C "${ROOT}" rev-parse HEAD)
[[ "${actual_commit}" == "${EXPECTED_COMMIT}" ]] || {
  echo "ABORT: code commit ${actual_commit} != ${EXPECTED_COMMIT}" >&2
  exit 1
}
git -C "${ROOT}" diff --quiet -- "${TASK_FILES[@]}" || {
  echo "ABORT: task files differ from checked-out commit" >&2; exit 1; }
git -C "${ROOT}" diff --cached --quiet -- "${TASK_FILES[@]}" || {
  echo "ABORT: staged task files differ from checked-out commit" >&2; exit 1; }
for required in "${TWO_MANIFEST}" "${THREE_MANIFEST}" "${SCENE_RUNNER}" \
                "${TWO_SUMMARIZER}" "${CONDITIONAL_EVALUATOR}" \
                "${CONDITIONAL_SUMMARIZER}" "${THREE_VALIDATOR}"; do
  test -r "${required}" || { echo "ABORT: missing ${required}" >&2; exit 1; }
done
[[ ! -e "${RUN_ROOT}" ]] || {
  echo "ABORT: output already exists: ${RUN_ROOT}" >&2; exit 1; }
mkdir -p "${RUN_ROOT}"
exec > >(tee "${RUN_ROOT}/run.log") 2>&1

echo "[paired-eval] mode=${MODE} commit=${actual_commit} job=${SLURM_JOB_ID:-local}"

if [[ "${MODE}" == smoke ]]; then
  TWO_INDICES=(7)
  CONDITIONAL_INDICES=(7)
else
  mapfile -t TWO_INDICES < <(seq 0 19)
  mapfile -t CONDITIONAL_INDICES < <(seq 0 9)
fi

TWO_ROOT=${RUN_ROOT}/two_leg_three_arm
if [[ "${RUN_TWO_LEG}" -eq 1 ]]; then
  echo "[stage 1] two-leg native/top-1/top-K scenes=${#TWO_INDICES[@]}"
  for scene_index in "${TWO_INDICES[@]}"; do
    env \
      ROOT="${ROOT}" \
      RUN_ROOT="${TWO_ROOT}" \
      MANIFEST="${TWO_MANIFEST}" \
      EXPECTED_MANIFEST_SHA="${TWO_MANIFEST_SHA}" \
      EXPECTED_COMMIT="${EXPECTED_COMMIT}" \
      SCENE_INDEX="${scene_index}" \
      HAB_PY="${HAB_PY}" \
      MEMNAV_PY="${MEMNAV_PY}" \
      MAX_STEPS=500 \
      "${SCENE_RUNNER}"
  done
  if [[ "${MODE}" == full ]]; then
    "${HAB_PY}" "${TWO_SUMMARIZER}" \
      --manifest "${TWO_MANIFEST}" --run-root "${TWO_ROOT}" \
      > "${TWO_ROOT}/summary.json"
    cat "${TWO_ROOT}/summary.json"
  fi
else
  echo "[stage 1] skipped: frozen 20-scene two-leg result already exists"
fi

CONDITIONAL_ROOT=${RUN_ROOT}/conditional_c_five_arm
echo "[stage 2] conditional-C five-arm scenes=${#CONDITIONAL_INDICES[@]}"
for scene_index in "${CONDITIONAL_INDICES[@]}"; do
  env \
    ROOT="${ROOT}" \
    RUN_ROOT="${CONDITIONAL_ROOT}" \
    MANIFEST="${THREE_MANIFEST}" \
    EXPECTED_MANIFEST_SHA="${THREE_MANIFEST_SHA}" \
    EXPECTED_COMMIT="${EXPECTED_COMMIT}" \
    SCENE_INDEX="${scene_index}" \
    HAB_PY="${HAB_PY}" \
    MEMNAV_PY="${MEMNAV_PY}" \
    EVALUATOR="${CONDITIONAL_EVALUATOR}" \
    VALIDATOR="${THREE_VALIDATOR}" \
    UNIT_TEST_MODULE=MemNavData.test_expanded_3leg_router_eval \
    RUN_CONDITIONAL_ORACLES=1 \
    MAX_STEPS=500 \
    "${SCENE_RUNNER}"
done
if [[ "${MODE}" == full ]]; then
  "${HAB_PY}" "${CONDITIONAL_SUMMARIZER}" \
    --manifest "${THREE_MANIFEST}" --run-root "${CONDITIONAL_ROOT}" \
    > "${CONDITIONAL_ROOT}/summary.json"
  cat "${CONDITIONAL_ROOT}/summary.json"
fi

echo "[complete] mode=${MODE} root=${RUN_ROOT}"
