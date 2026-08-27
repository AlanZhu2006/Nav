#!/usr/bin/env bash
# Resume the interrupted graph-router ablation without rerunning valid scenes.

set -euo pipefail
umask 0022

ROOT=${ROOT:?set ROOT}
RUN_ROOT=${RUN_ROOT:?set RUN_ROOT}
SOURCE_RUN_ROOT=${SOURCE_RUN_ROOT:?set SOURCE_RUN_ROOT}
EXPECTED_COMMIT=${EXPECTED_COMMIT:?set EXPECTED_COMMIT}
EXPECTED_SOURCE_RESULTS_SHA=${EXPECTED_SOURCE_RESULTS_SHA:?set EXPECTED_SOURCE_RESULTS_SHA}
HAB_PY=${HAB_PY:?set HAB_PY}
MEMNAV_PY=${MEMNAV_PY:?set MEMNAV_PY}
REFERENCE_ROOT=${REFERENCE_ROOT:?set REFERENCE_ROOT}
GRAPH_SPACING_M=${GRAPH_SPACING_M:-1.25}
GRAPH_ARRIVAL_M=${GRAPH_ARRIVAL_M:-0.60}
PREFLIGHT_ONLY=${PREFLIGHT_ONLY:-0}
[[ "${PREFLIGHT_ONLY}" =~ ^[01]$ ]] || {
  echo "ABORT: PREFLIGHT_ONLY must be 0 or 1" >&2; exit 1; }
MANIFEST=${ROOT}/MemNavData/expanded_navdp_router_eval_20260805.json
MANIFEST_SHA=ba8f72cb504768c801e6c9c386436ccdc66dea07a5e5fac2d7b4248738946a61
SCENE_RUNNER=${ROOT}/MemNavData/run_expanded_navdp_router_scene.sh
SUMMARIZER=${ROOT}/MemNavData/summarize_graph_router_ablation.py

TASK_FILES=(
  MemNavData/run_graph_router_completion.sh
  MemNavData/slurm_graph_router_completion.sbatch
  MemNavData/run_expanded_navdp_router_scene.sh
  MemNavData/summarize_graph_router_ablation.py
  MemNavData/expanded_navdp_router_eval_20260805.json
  NavDP/baselines/memnav/reverse_memory_graph.py
  NavDP/baselines/memnav/policy_agent.py
  NavDP/baselines/memnav/memnav_server.py
)

actual_commit=$(git -C "${ROOT}" rev-parse HEAD)
[[ "${actual_commit}" == "${EXPECTED_COMMIT}" ]] || {
  echo "ABORT: commit ${actual_commit} != ${EXPECTED_COMMIT}" >&2; exit 1; }
git -C "${ROOT}" diff --quiet -- "${TASK_FILES[@]}" || {
  echo "ABORT: completion-task files differ from commit" >&2; exit 1; }
git -C "${ROOT}" diff --cached --quiet -- "${TASK_FILES[@]}" || {
  echo "ABORT: staged completion-task files differ from commit" >&2; exit 1; }
for required in "${HAB_PY}" "${MEMNAV_PY}" "${REFERENCE_ROOT}" \
                "${MANIFEST}" "${SCENE_RUNNER}" "${SUMMARIZER}" \
                "${SOURCE_RUN_ROOT}/direct_gap4/scenes" \
                "${SOURCE_RUN_ROOT}/graph_gap16/scenes"; do
  test -r "${required}" || {
    echo "ABORT: missing dependency ${required}" >&2; exit 1; }
done
[[ "$(sha256sum "${MANIFEST}" | awk '{print $1}')" == "${MANIFEST_SHA}" ]] || {
  echo "ABORT: manifest SHA mismatch" >&2; exit 1; }
if [[ -e "${RUN_ROOT}" ]]; then
  echo "ABORT: output exists: ${RUN_ROOT}" >&2; exit 1
fi

scene_name() {
  "${HAB_PY}" - "${MANIFEST}" "$1" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    manifest = json.load(handle)
print(manifest["selection"]["selected_scenes"][int(sys.argv[2])])
PY
}

# The source digest pins every reused metric and summary.  Interrupted scene 15
# is deliberately excluded because it never produced a summary.
source_results_sha=$(
  {
    for file in "${SOURCE_RUN_ROOT}"/direct_gap4/scenes/*/geometry_router/{metric.csv,summary.json}; do
      sha256sum "${file}" | awk '{print $1}'
    done
    for file in "${SOURCE_RUN_ROOT}"/graph_gap16/scenes/{00..14}_*/geometry_router/{metric.csv,summary.json}; do
      sha256sum "${file}" | awk '{print $1}'
    done
  } | sha256sum | awk '{print $1}'
)
[[ "${source_results_sha}" == "${EXPECTED_SOURCE_RESULTS_SHA}" ]] || {
  echo "ABORT: source result digest ${source_results_sha} != ${EXPECTED_SOURCE_RESULTS_SHA}" >&2
  exit 1
}
if [[ "${PREFLIGHT_ONLY}" == "1" ]]; then
  echo "PREFLIGHT_OK commit=${actual_commit} source_digest=${source_results_sha}"
  exit 0
fi

mkdir -p "${RUN_ROOT}/graph_gap16/scenes"
exec > >(tee "${RUN_ROOT}/run.log") 2>&1
echo "[preflight] commit=${actual_commit} source_digest=${source_results_sha}"

# Reuse only the 15 completed, pinned graph-gap16 scenes.  Symlinks preserve
# provenance and avoid silently copying or mutating prior results.
for scene_index in $(seq 0 14); do
  scene=$(scene_name "${scene_index}")
  name=$(printf "%02d_%s" "${scene_index}" "${scene}")
  source_scene=${SOURCE_RUN_ROOT}/graph_gap16/scenes/${name}
  test -r "${source_scene}/geometry_router/summary.json" || {
    echo "ABORT: incomplete source scene ${source_scene}" >&2; exit 1; }
  ln -s "${source_scene}" "${RUN_ROOT}/graph_gap16/scenes/${name}"
done

# Run the most informative composition first, so a later cluster preemption
# cannot again leave graph-gap4 entirely unmeasured.
for scene_index in $(seq 0 19); do
  echo "[graph_gap4] scene=$((scene_index + 1))/20"
  env \
    ROOT="${ROOT}" \
    RUN_ROOT="${RUN_ROOT}/graph_gap4" \
    MANIFEST="${MANIFEST}" \
    EXPECTED_MANIFEST_SHA="${MANIFEST_SHA}" \
    EXPECTED_COMMIT="${EXPECTED_COMMIT}" \
    SCENE_INDEX="${scene_index}" \
    HAB_PY="${HAB_PY}" \
    MEMNAV_PY="${MEMNAV_PY}" \
    RUN_NAVDP_NATIVE=0 \
    RUN_GEOMETRY_TOP1=0 \
    RUN_GEOMETRY_ROUTER=1 \
    RETRIEVAL_CANDIDATE_MIN_GAP=4 \
    GRAPH_SUBGOAL_SPACING_M="${GRAPH_SPACING_M}" \
    GRAPH_SUBGOAL_ARRIVAL_M="${GRAPH_ARRIVAL_M}" \
    "${SCENE_RUNNER}"
done

for scene_index in $(seq 15 19); do
  echo "[graph_gap16 completion] scene=$((scene_index + 1))/20"
  env \
    ROOT="${ROOT}" \
    RUN_ROOT="${RUN_ROOT}/graph_gap16" \
    MANIFEST="${MANIFEST}" \
    EXPECTED_MANIFEST_SHA="${MANIFEST_SHA}" \
    EXPECTED_COMMIT="${EXPECTED_COMMIT}" \
    SCENE_INDEX="${scene_index}" \
    HAB_PY="${HAB_PY}" \
    MEMNAV_PY="${MEMNAV_PY}" \
    RUN_NAVDP_NATIVE=0 \
    RUN_GEOMETRY_TOP1=0 \
    RUN_GEOMETRY_ROUTER=1 \
    RETRIEVAL_CANDIDATE_MIN_GAP=16 \
    GRAPH_SUBGOAL_SPACING_M="${GRAPH_SPACING_M}" \
    GRAPH_SUBGOAL_ARRIVAL_M="${GRAPH_ARRIVAL_M}" \
    "${SCENE_RUNNER}"
done

"${HAB_PY}" "${SUMMARIZER}" \
  --manifest "${MANIFEST}" \
  --reference-root "${REFERENCE_ROOT}" \
  --config "direct_gap4=${SOURCE_RUN_ROOT}/direct_gap4" \
  --config "graph_gap16=${RUN_ROOT}/graph_gap16" \
  --config "graph_gap4=${RUN_ROOT}/graph_gap4" \
  > "${RUN_ROOT}/summary.json"
cat "${RUN_ROOT}/summary.json"
echo "[complete] ${RUN_ROOT}/summary.json"
