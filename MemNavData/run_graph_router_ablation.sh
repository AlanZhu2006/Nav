#!/usr/bin/env bash
# Paired 20-scene ablation of candidate diversity and reverse-memory subgoals.

set -euo pipefail
umask 0022

ROOT=${ROOT:?set ROOT}
RUN_ROOT=${RUN_ROOT:?set RUN_ROOT}
EXPECTED_COMMIT=${EXPECTED_COMMIT:?set EXPECTED_COMMIT}
HAB_PY=${HAB_PY:?set HAB_PY}
MEMNAV_PY=${MEMNAV_PY:?set MEMNAV_PY}
REFERENCE_ROOT=${REFERENCE_ROOT:?set REFERENCE_ROOT}
GRAPH_SPACING_M=${GRAPH_SPACING_M:-1.25}
GRAPH_ARRIVAL_M=${GRAPH_ARRIVAL_M:-0.60}
MANIFEST=${ROOT}/MemNavData/expanded_navdp_router_eval_20260805.json
MANIFEST_SHA=ba8f72cb504768c801e6c9c386436ccdc66dea07a5e5fac2d7b4248738946a61
SCENE_RUNNER=${ROOT}/MemNavData/run_expanded_navdp_router_scene.sh
SUMMARIZER=${ROOT}/MemNavData/summarize_graph_router_ablation.py
SET_TRAINER=${ROOT}/MemNavData/train_neural_set_localizer.py
ROUTER_SPLIT=${ROOT}/MemNavData/router_multiscene_split_20260805.json
ROUTER_SPLIT_SHA=97309c183e25cb3dd65472908748d55a94798a636db6157ab6fe120fca05cf7a
SET_TEACHER=/scratch/yz11502/Research/Nav-axis-uturn-results/patch_router_multiscene_20260805/full_job_15315411/covisibility_teacher_pairs.csv
SET_TEACHER_SHA=927d0e87fc6b53561d5e5546bbcd5d1b94e11907e0aaecd5c810f4fa8911110d
SET_FEATURES=/scratch/yz11502/Research/Nav-axis-uturn-results/patch_router_multiscene_20260805/full_job_15315411/patch_temporal/patch_temporal_features.npz
SET_FEATURES_SHA=a1a08747fedab45908aa9ecc4cbc40ba112ad889a4b8431d5450bacd9dcd621b

TASK_FILES=(
  MemNavData/run_graph_router_ablation.sh
  MemNavData/slurm_graph_router_ablation.sbatch
  MemNavData/summarize_graph_router_ablation.py
  MemNavData/summarize_expanded_navdp_router_eval.py
  MemNavData/test_summarize_graph_router_ablation.py
  MemNavData/train_neural_set_localizer.py
  MemNavData/test_neural_set_localizer.py
  MemNavData/diag_patch_temporal_router.py
  MemNavData/patch_temporal_router.py
  MemNavData/router_multiscene_split_20260805.json
  MemNavData/expanded_navdp_router_eval_20260805.json
  MemNavData/run_expanded_navdp_router_scene.sh
  MemNavData/test_reverse_memory_graph.py
  MemNavData/test_policy_agent_graph.py
  NavDP/baselines/memnav/reverse_memory_graph.py
  NavDP/baselines/memnav/policy_agent.py
  NavDP/baselines/memnav/memnav_server.py
)

actual_commit=$(git -C "${ROOT}" rev-parse HEAD)
[[ "${actual_commit}" == "${EXPECTED_COMMIT}" ]] || {
  echo "ABORT: commit ${actual_commit} != ${EXPECTED_COMMIT}" >&2; exit 1; }
git -C "${ROOT}" diff --quiet -- "${TASK_FILES[@]}" || {
  echo "ABORT: graph ablation files differ from commit" >&2; exit 1; }
git -C "${ROOT}" diff --cached --quiet -- "${TASK_FILES[@]}" || {
  echo "ABORT: staged graph ablation files differ from commit" >&2; exit 1; }
for required in "${HAB_PY}" "${MEMNAV_PY}" "${REFERENCE_ROOT}" \
                "${MANIFEST}" "${SCENE_RUNNER}" "${SUMMARIZER}" \
                "${SET_TRAINER}" "${ROUTER_SPLIT}" \
                "${SET_TEACHER}" "${SET_FEATURES}"; do
  test -r "${required}" || {
    echo "ABORT: missing dependency ${required}" >&2; exit 1; }
done
[[ "$(sha256sum "${MANIFEST}" | awk '{print $1}')" == "${MANIFEST_SHA}" ]] || {
  echo "ABORT: manifest SHA mismatch" >&2; exit 1; }
[[ "$(sha256sum "${ROUTER_SPLIT}" | awk '{print $1}')" == "${ROUTER_SPLIT_SHA}" ]] || {
  echo "ABORT: router split SHA mismatch" >&2; exit 1; }
[[ "$(sha256sum "${SET_TEACHER}" | awk '{print $1}')" == "${SET_TEACHER_SHA}" ]] || {
  echo "ABORT: set-localizer teacher SHA mismatch" >&2; exit 1; }
[[ "$(sha256sum "${SET_FEATURES}" | awk '{print $1}')" == "${SET_FEATURES_SHA}" ]] || {
  echo "ABORT: set-localizer feature SHA mismatch" >&2; exit 1; }
if [[ -e "${RUN_ROOT}" ]]; then
  echo "ABORT: output exists: ${RUN_ROOT}" >&2; exit 1
fi
mkdir -p "${RUN_ROOT}"
exec > >(tee "${RUN_ROOT}/run.log") 2>&1

cd "${ROOT}"
"${MEMNAV_PY}" -m py_compile \
  NavDP/baselines/memnav/reverse_memory_graph.py \
  NavDP/baselines/memnav/policy_agent.py \
  NavDP/baselines/memnav/memnav_server.py \
  MemNavData/summarize_graph_router_ablation.py \
  MemNavData/train_neural_set_localizer.py
"${MEMNAV_PY}" -m unittest \
  MemNavData.test_reverse_memory_graph \
  MemNavData.test_policy_agent_graph \
  MemNavData.test_summarize_graph_router_ablation \
  MemNavData.test_neural_set_localizer -v

echo "[training] K+1 neural set localizer with scene-disjoint calibration"
SET_ARGS=()
while IFS= read -r scene; do
  SET_ARGS+=(--heldout-scene "${scene}")
done < <("${MEMNAV_PY}" - "${ROUTER_SPLIT}" <<'PY'
import json, sys
print(*json.load(open(sys.argv[1]))["development"], sep="\n")
PY
)
"${MEMNAV_PY}" -u "${SET_TRAINER}" \
  --teacher-csv "${SET_TEACHER}" \
  --feature-cache "${SET_FEATURES}" \
  --out-dir "${RUN_ROOT}/neural_set_localizer" \
  --epochs 250 \
  --device cpu \
  "${SET_ARGS[@]}"

# The direct/gap-16 arm already exists and is immutable.  These three paired
# arms isolate candidate coverage, graph control, and their composition.
CONFIGS=(
  "direct_gap4:4:0.0"
  "graph_gap16:16:${GRAPH_SPACING_M}"
  "graph_gap4:4:${GRAPH_SPACING_M}"
)
for spec in "${CONFIGS[@]}"; do
  IFS=: read -r name gap spacing <<<"${spec}"
  config_root=${RUN_ROOT}/${name}
  echo "[configuration] name=${name} gap=${gap} spacing_m=${spacing}"
  for scene_index in $(seq 0 19); do
    echo "[${name}] scene=$((scene_index + 1))/20"
    env \
      ROOT="${ROOT}" \
      RUN_ROOT="${config_root}" \
      MANIFEST="${MANIFEST}" \
      EXPECTED_MANIFEST_SHA="${MANIFEST_SHA}" \
      EXPECTED_COMMIT="${EXPECTED_COMMIT}" \
      SCENE_INDEX="${scene_index}" \
      HAB_PY="${HAB_PY}" \
      MEMNAV_PY="${MEMNAV_PY}" \
      RUN_NAVDP_NATIVE=0 \
      RUN_GEOMETRY_TOP1=0 \
      RUN_GEOMETRY_ROUTER=1 \
      RETRIEVAL_CANDIDATE_MIN_GAP="${gap}" \
      GRAPH_SUBGOAL_SPACING_M="${spacing}" \
      GRAPH_SUBGOAL_ARRIVAL_M="${GRAPH_ARRIVAL_M}" \
      "${SCENE_RUNNER}"
  done
done

"${HAB_PY}" "${SUMMARIZER}" \
  --manifest "${MANIFEST}" \
  --reference-root "${REFERENCE_ROOT}" \
  --config "direct_gap4=${RUN_ROOT}/direct_gap4" \
  --config "graph_gap16=${RUN_ROOT}/graph_gap16" \
  --config "graph_gap4=${RUN_ROOT}/graph_gap4" \
  > "${RUN_ROOT}/summary.json"
cat "${RUN_ROOT}/summary.json"
echo "[complete] ${RUN_ROOT}/summary.json"
