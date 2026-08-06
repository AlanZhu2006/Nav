#!/usr/bin/env bash
# Deterministic conditional-C comparison of direct and graph-gap16 control.

set -euo pipefail
umask 0022

ROOT=${ROOT:?set ROOT}
RUN_ROOT=${RUN_ROOT:?set RUN_ROOT}
EXPECTED_COMMIT=${EXPECTED_COMMIT:?set EXPECTED_COMMIT}
HAB_PY=${HAB_PY:?set HAB_PY}
MEMNAV_PY=${MEMNAV_PY:?set MEMNAV_PY}
MODE=${MODE:?set MODE=smoke or full}
[[ "${MODE}" =~ ^(smoke|full)$ ]] || {
  echo "ABORT: MODE must be smoke or full" >&2; exit 1; }

MANIFEST=${ROOT}/MemNavData/expanded_3leg_router_eval_20260805.json
MANIFEST_SHA=55096038d0493723d2280fe9abbcb2ea9ea7f2059c50b6eb08d02f560cdb281b
SCENE_RUNNER=${ROOT}/MemNavData/run_expanded_navdp_router_scene.sh
EVALUATOR=${ROOT}/MemNavData/eval_conditional_c_habitat.py
VALIDATOR=${ROOT}/MemNavData/validate_expanded_3leg_router_eval.py
SUMMARIZER=${ROOT}/MemNavData/summarize_graph_conditional_c_eval.py

TASK_FILES=(
  MemNavData/run_graph_conditional_c.sh
  MemNavData/slurm_graph_conditional_c.sbatch
  MemNavData/run_expanded_navdp_router_scene.sh
  MemNavData/eval_2leg_habitat.py
  MemNavData/eval_conditional_c_habitat.py
  MemNavData/deterministic_eval_protocol.py
  MemNavData/summarize_conditional_c_eval.py
  MemNavData/summarize_graph_conditional_c_eval.py
  MemNavData/test_deterministic_eval_protocol.py
  MemNavData/test_navdp_memory_replay.py
  MemNavData/test_summarize_graph_conditional_c_eval.py
  NavDP/baselines/navdp/deterministic_seed.py
  NavDP/baselines/navdp/navdp_server.py
  NavDP/baselines/memnav/memnav_server.py
  NavDP/baselines/memnav/policy_agent.py
  NavDP/baselines/memnav/reverse_memory_graph.py
)

actual_commit=$(git -C "${ROOT}" rev-parse HEAD)
[[ "${actual_commit}" == "${EXPECTED_COMMIT}" ]] || {
  echo "ABORT: commit ${actual_commit} != ${EXPECTED_COMMIT}" >&2; exit 1; }
git -C "${ROOT}" diff --quiet -- "${TASK_FILES[@]}" || {
  echo "ABORT: conditional graph task differs from commit" >&2; exit 1; }
git -C "${ROOT}" diff --cached --quiet -- "${TASK_FILES[@]}" || {
  echo "ABORT: staged conditional graph task differs from commit" >&2; exit 1; }
for required in "${HAB_PY}" "${MEMNAV_PY}" "${MANIFEST}" \
                "${SCENE_RUNNER}" "${EVALUATOR}" "${VALIDATOR}" \
                "${SUMMARIZER}"; do
  test -r "${required}" || {
    echo "ABORT: missing dependency ${required}" >&2; exit 1; }
done
[[ "$(sha256sum "${MANIFEST}" | awk '{print $1}')" == "${MANIFEST_SHA}" ]] || {
  echo "ABORT: three-leg manifest SHA mismatch" >&2; exit 1; }
[[ ! -e "${RUN_ROOT}" ]] || {
  echo "ABORT: output exists: ${RUN_ROOT}" >&2; exit 1; }
mkdir -p "${RUN_ROOT}"
exec > >(tee "${RUN_ROOT}/run.log") 2>&1

cd "${ROOT}"
"${MEMNAV_PY}" -m py_compile \
  MemNavData/deterministic_eval_protocol.py \
  MemNavData/summarize_graph_conditional_c_eval.py \
  NavDP/baselines/navdp/deterministic_seed.py
"${MEMNAV_PY}" -m unittest \
  MemNavData.test_deterministic_eval_protocol \
  MemNavData.test_navdp_memory_replay \
  MemNavData.test_summarize_graph_conditional_c_eval -v

if [[ "${MODE}" == smoke ]]; then
  SCENE_INDICES=(7)
else
  mapfile -t SCENE_INDICES < <(seq 0 9)
fi

DIRECT_ROOT=${RUN_ROOT}/direct_gap16
GRAPH_ROOT=${RUN_ROOT}/graph_gap16
for scene_index in "${SCENE_INDICES[@]}"; do
  echo "[conditional direct] scene_index=${scene_index}"
  env \
    ROOT="${ROOT}" \
    RUN_ROOT="${DIRECT_ROOT}" \
    MANIFEST="${MANIFEST}" \
    EXPECTED_MANIFEST_SHA="${MANIFEST_SHA}" \
    EXPECTED_COMMIT="${EXPECTED_COMMIT}" \
    SCENE_INDEX="${scene_index}" \
    HAB_PY="${HAB_PY}" \
    MEMNAV_PY="${MEMNAV_PY}" \
    EVALUATOR="${EVALUATOR}" \
    VALIDATOR="${VALIDATOR}" \
    UNIT_TEST_MODULE=MemNavData.test_expanded_3leg_router_eval \
    RUN_NAVDP_NATIVE=1 \
    RUN_GEOMETRY_TOP1=0 \
    RUN_GEOMETRY_ROUTER=1 \
    RUN_CONDITIONAL_ORACLE_ANCHOR=1 \
    RUN_CONDITIONAL_ORACLE_POINT=1 \
    DETERMINISTIC_PLAN_SEEDS=1 \
    RETRIEVAL_CANDIDATE_MIN_GAP=16 \
    GRAPH_SUBGOAL_SPACING_M=0.0 \
    GRAPH_SUBGOAL_ARRIVAL_M=0.60 \
    MAX_STEPS=500 \
    "${SCENE_RUNNER}"

  echo "[conditional graph] scene_index=${scene_index}"
  env \
    ROOT="${ROOT}" \
    RUN_ROOT="${GRAPH_ROOT}" \
    MANIFEST="${MANIFEST}" \
    EXPECTED_MANIFEST_SHA="${MANIFEST_SHA}" \
    EXPECTED_COMMIT="${EXPECTED_COMMIT}" \
    SCENE_INDEX="${scene_index}" \
    HAB_PY="${HAB_PY}" \
    MEMNAV_PY="${MEMNAV_PY}" \
    EVALUATOR="${EVALUATOR}" \
    VALIDATOR="${VALIDATOR}" \
    UNIT_TEST_MODULE=MemNavData.test_expanded_3leg_router_eval \
    RUN_NAVDP_NATIVE=0 \
    RUN_GEOMETRY_TOP1=0 \
    RUN_GEOMETRY_ROUTER=1 \
    RUN_CONDITIONAL_ORACLE_ANCHOR=1 \
    RUN_CONDITIONAL_ORACLE_POINT=0 \
    DETERMINISTIC_PLAN_SEEDS=1 \
    RETRIEVAL_CANDIDATE_MIN_GAP=16 \
    GRAPH_SUBGOAL_SPACING_M=1.25 \
    GRAPH_SUBGOAL_ARRIVAL_M=0.60 \
    MAX_STEPS=500 \
    "${SCENE_RUNNER}"
done

if [[ "${MODE}" == full ]]; then
  "${HAB_PY}" "${SUMMARIZER}" \
    --manifest "${MANIFEST}" \
    --direct-root "${DIRECT_ROOT}" \
    --graph-root "${GRAPH_ROOT}" \
    > "${RUN_ROOT}/summary.json"
  cat "${RUN_ROOT}/summary.json"
fi
echo "[complete] mode=${MODE} root=${RUN_ROOT}"
