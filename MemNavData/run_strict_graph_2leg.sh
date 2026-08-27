#!/usr/bin/env bash
# Shared-Novel, request-seeded direct-vs-graph evaluation on one frozen manifest.

set -euo pipefail
umask 0022
export GIT_OPTIONAL_LOCKS=0

ROOT=${ROOT:?set ROOT}
RUN_ROOT=${RUN_ROOT:?set RUN_ROOT}
EXPECTED_COMMIT=${EXPECTED_COMMIT:?set EXPECTED_COMMIT}
HAB_PY=${HAB_PY:?set HAB_PY}
MEMNAV_PY=${MEMNAV_PY:?set MEMNAV_PY}
MANIFEST=${MANIFEST:?set MANIFEST}
EXPECTED_MANIFEST_SHA=${EXPECTED_MANIFEST_SHA:?set EXPECTED_MANIFEST_SHA}
MODE=${MODE:?set MODE=smoke or full}
SCENE_WORKERS=${SCENE_WORKERS:-1}
SMOKE_SCENE_INDEX=${SMOKE_SCENE_INDEX:-7}
SMOKE_EPISODE_LIMIT=${SMOKE_EPISODE_LIMIT:-1}
TERMINAL_UTURN=${TERMINAL_UTURN:-off}
TERMINAL_VISUAL_REFINE=${TERMINAL_VISUAL_REFINE:-off}
ARRIVAL_SHADOW=${ARRIVAL_SHADOW:-off}
[[ "${MODE}" =~ ^(smoke|full)$ ]] || {
  echo "ABORT: MODE must be smoke or full" >&2; exit 1; }
[[ "${SCENE_WORKERS}" =~ ^[1-9][0-9]*$ ]] || {
  echo "ABORT: SCENE_WORKERS must be positive" >&2; exit 1; }
[[ "${SMOKE_SCENE_INDEX}" =~ ^[0-9]+$ ]] || {
  echo "ABORT: SMOKE_SCENE_INDEX must be non-negative" >&2; exit 1; }
[[ "${SMOKE_EPISODE_LIMIT}" =~ ^[1-9][0-9]*$ ]] || {
  echo "ABORT: SMOKE_EPISODE_LIMIT must be positive" >&2; exit 1; }
[[ "${TERMINAL_UTURN}" =~ ^(off|oracle|lingbot_yaw|lingbot_local|lingbot)$ ]] || {
  echo "ABORT: invalid TERMINAL_UTURN=${TERMINAL_UTURN}" >&2; exit 1; }
[[ "${TERMINAL_VISUAL_REFINE}" =~ ^(off|verify|refine)$ ]] || {
  echo "ABORT: invalid TERMINAL_VISUAL_REFINE=${TERMINAL_VISUAL_REFINE}" >&2
  exit 1
}
[[ "${ARRIVAL_SHADOW}" =~ ^(off|diagnostic)$ ]] || {
  echo "ABORT: invalid ARRIVAL_SHADOW=${ARRIVAL_SHADOW}" >&2; exit 1; }
if [[ "${TERMINAL_UTURN}" == off && "${TERMINAL_VISUAL_REFINE}" != off ]]; then
  echo "ABORT: terminal visual refinement requires a terminal U-turn" >&2
  exit 1
fi

SCENE_RUNNER=${ROOT}/MemNavData/run_expanded_navdp_router_scene.sh
SUMMARIZER=${ROOT}/MemNavData/summarize_graph_router_ablation.py
TERMINAL_SUMMARIZER=${ROOT}/MemNavData/summarize_terminal_uturn.py
ARRIVAL_SUMMARIZER=${ROOT}/MemNavData/summarize_arrival_shadow.py
SMOKE_VALIDATOR=${ROOT}/MemNavData/validate_strict_graph_smoke.py
TASK_FILES=(
  MemNavData/run_strict_graph_2leg.sh
  MemNavData/slurm_strict_graph_2leg.sbatch
  MemNavData/run_expanded_navdp_router_scene.sh
  MemNavData/eval_2leg_habitat.py
  MemNavData/deterministic_eval_protocol.py
  MemNavData/summarize_expanded_navdp_router_eval.py
  MemNavData/summarize_graph_router_ablation.py
  MemNavData/validate_strict_graph_smoke.py
  MemNavData/test_deterministic_eval_protocol.py
  MemNavData/test_navdp_memory_replay.py
  MemNavData/test_summarize_graph_router_ablation.py
  MemNavData/test_validate_strict_graph_smoke.py
  MemNavData/terminal_uturn.py
  MemNavData/visual_yaw_refinement.py
  MemNavData/summarize_terminal_uturn.py
  MemNavData/test_terminal_uturn.py
  MemNavData/arrival_shadow.py
  MemNavData/test_arrival_shadow.py
  MemNavData/summarize_arrival_shadow.py
  MemNavData/test_summarize_arrival_shadow.py
  NavDP/baselines/navdp/deterministic_seed.py
  NavDP/baselines/navdp/navdp_server.py
  NavDP/baselines/navdp/policy_agent.py
  NavDP/baselines/memnav/memnav_server.py
  NavDP/baselines/memnav/policy_agent.py
  NavDP/baselines/memnav/pose_alignment.py
  NavDP/baselines/memnav/reverse_memory_graph.py
  InternNav/internnav/model/basemodel/memnav/memnav_policy.py
)
manifest_relative=$(realpath --relative-to="${ROOT}" "${MANIFEST}")
[[ "${manifest_relative}" != ../* && "${manifest_relative}" != ".." ]] || {
  echo "ABORT: manifest must live in the child repository" >&2; exit 1; }
TASK_FILES+=("${manifest_relative}")

actual_commit=$(git -C "${ROOT}" rev-parse HEAD)
[[ "${actual_commit}" == "${EXPECTED_COMMIT}" ]] || {
  echo "ABORT: commit ${actual_commit} != ${EXPECTED_COMMIT}" >&2; exit 1; }
git -C "${ROOT}" ls-files --error-unmatch -- "${TASK_FILES[@]}" \
  >/dev/null || {
    echo "ABORT: a strict graph input is not tracked by the commit" >&2
    exit 1
  }
[[ -z "$(git -C "${ROOT}" status --porcelain --untracked-files=all)" ]] || {
  echo "ABORT: strict graph worktree is not completely clean" >&2
  git -C "${ROOT}" status --short >&2
  exit 1
}
git -C "${ROOT}" diff --quiet -- "${TASK_FILES[@]}" || {
  echo "ABORT: strict graph task differs from commit" >&2; exit 1; }
git -C "${ROOT}" diff --cached --quiet -- "${TASK_FILES[@]}" || {
  echo "ABORT: staged strict graph task differs from commit" >&2; exit 1; }
for required in "${HAB_PY}" "${MEMNAV_PY}" "${MANIFEST}" \
                "${SCENE_RUNNER}" "${SUMMARIZER}" "${TERMINAL_SUMMARIZER}" \
                "${ARRIVAL_SUMMARIZER}" "${SMOKE_VALIDATOR}"; do
  test -r "${required}" || {
    echo "ABORT: missing dependency ${required}" >&2; exit 1; }
done
[[ "$(sha256sum "${MANIFEST}" | awk '{print $1}')" == \
    "${EXPECTED_MANIFEST_SHA}" ]] || {
  echo "ABORT: manifest SHA mismatch" >&2; exit 1; }
[[ ! -e "${RUN_ROOT}" ]] || {
  echo "ABORT: output exists: ${RUN_ROOT}" >&2; exit 1; }
mkdir -p "${RUN_ROOT}"
exec > >(tee "${RUN_ROOT}/run.log") 2>&1

cd "${ROOT}"
"${MEMNAV_PY}" -m py_compile \
  MemNavData/deterministic_eval_protocol.py \
  MemNavData/summarize_graph_router_ablation.py \
  MemNavData/summarize_arrival_shadow.py \
  MemNavData/validate_strict_graph_smoke.py \
  NavDP/baselines/navdp/deterministic_seed.py
"${MEMNAV_PY}" -m unittest \
  MemNavData.test_deterministic_eval_protocol \
  MemNavData.test_navdp_memory_replay \
  MemNavData.test_summarize_graph_router_ablation \
  MemNavData.test_terminal_uturn \
  MemNavData.test_arrival_shadow \
  MemNavData.test_summarize_arrival_shadow \
  MemNavData.test_validate_strict_graph_smoke -v

scene_count=$("${HAB_PY}" - "${MANIFEST}" <<'PY'
import json, sys
print(len(json.load(open(sys.argv[1]))["selection"]["selected_scenes"]))
PY
)
[[ "${scene_count}" =~ ^[1-9][0-9]*$ ]] || {
  echo "ABORT: manifest contains no scenes" >&2; exit 1; }
if [[ "${MODE}" == smoke ]]; then
  (( SMOKE_SCENE_INDEX < scene_count )) || {
    echo "ABORT: smoke scene index is outside the manifest" >&2; exit 1; }
  SCENE_INDICES=("${SMOKE_SCENE_INDEX}")
  EPISODE_LIMIT=${SMOKE_EPISODE_LIMIT}
else
  mapfile -t SCENE_INDICES < <(seq 0 $((scene_count - 1)))
  EPISODE_LIMIT=0
fi

IFS=',' read -r -a GPU_IDS <<< "${CUDA_VISIBLE_DEVICES:-0}"
(( SCENE_WORKERS <= ${#GPU_IDS[@]} )) || {
  echo "ABORT: SCENE_WORKERS=${SCENE_WORKERS} exceeds visible GPUs " \
       "${CUDA_VISIBLE_DEVICES:-0}" >&2
  exit 1
}
echo "[protocol] mode=${MODE} workers=${SCENE_WORKERS} " \
     "visible_gpus=${CUDA_VISIBLE_DEVICES:-0} episode_limit=${EPISODE_LIMIT} " \
     "terminal_uturn=${TERMINAL_UTURN} " \
     "terminal_visual_refine=${TERMINAL_VISUAL_REFINE} " \
     "arrival_shadow=${ARRIVAL_SHADOW}"

SOURCE_ROOT=${RUN_ROOT}/shared_novel_direct_gap16
DIRECT_ROOT=${RUN_ROOT}/direct_gap16
GRAPH_ROOT=${RUN_ROOT}/graph_gap16

run_scene() {
  local scene_index=$1
  local target_root=$2
  shift 2
  env \
    ROOT="${ROOT}" \
    RUN_ROOT="${target_root}" \
    MANIFEST="${MANIFEST}" \
    EXPECTED_MANIFEST_SHA="${EXPECTED_MANIFEST_SHA}" \
    EXPECTED_COMMIT="${EXPECTED_COMMIT}" \
    SCENE_INDEX="${scene_index}" \
    HAB_PY="${HAB_PY}" \
    MEMNAV_PY="${MEMNAV_PY}" \
    RUN_NAVDP_NATIVE=0 \
    RUN_GEOMETRY_TOP1=0 \
    RUN_GEOMETRY_ROUTER=1 \
    EPISODE_LIMIT="${EPISODE_LIMIT}" \
    DETERMINISTIC_PLAN_SEEDS=1 \
    RETRIEVAL_CANDIDATE_MIN_GAP=16 \
    GRAPH_SUBGOAL_ARRIVAL_M=0.60 \
    TERMINAL_UTURN="${TERMINAL_UTURN}" \
    TERMINAL_VISUAL_REFINE="${TERMINAL_VISUAL_REFINE}" \
    ARRIVAL_SHADOW="${ARRIVAL_SHADOW}" \
    MAX_STEPS=500 \
    "$@" \
    "${SCENE_RUNNER}"
}

run_scene_pipeline() {
  local scene_index=$1
  local gpu_id=$2
  export CUDA_VISIBLE_DEVICES="${gpu_id}"
  echo "[shared direct-controller Novel] scene_index=${scene_index} gpu=${gpu_id}"
  run_scene "${scene_index}" "${SOURCE_ROOT}" \
    LEG1_MODE=policy \
    STOP_AFTER_LEG1=1 \
    WRITE_LEG1_TRACE=1 \
    GRAPH_SUBGOAL_SPACING_M=0.0

  echo "[direct gap16] scene_index=${scene_index}"
  run_scene "${scene_index}" "${DIRECT_ROOT}" \
    LEG1_MODE=shared_trace \
    SHARED_LEG1_ROOT="${SOURCE_ROOT}" \
    GRAPH_SUBGOAL_SPACING_M=0.0

  echo "[graph gap16] scene_index=${scene_index}"
  run_scene "${scene_index}" "${GRAPH_ROOT}" \
    LEG1_MODE=shared_trace \
    SHARED_LEG1_ROOT="${SOURCE_ROOT}" \
    GRAPH_SUBGOAL_SPACING_M=1.25
}

wait_scene_batch() {
  local failed=0
  local pid
  for pid in "${SCENE_PIDS[@]}"; do
    wait "${pid}" || failed=1
  done
  SCENE_PIDS=()
  (( failed == 0 )) || {
    echo "ABORT: at least one parallel scene pipeline failed" >&2
    exit 1
  }
}

SCENE_PIDS=()
worker_slot=0
for scene_index in "${SCENE_INDICES[@]}"; do
  run_scene_pipeline \
    "${scene_index}" "${GPU_IDS[${worker_slot}]}" &
  SCENE_PIDS+=("$!")
  worker_slot=$((worker_slot + 1))
  if (( worker_slot == SCENE_WORKERS )); then
    wait_scene_batch
    worker_slot=0
  fi
done
(( ${#SCENE_PIDS[@]} == 0 )) || wait_scene_batch

if [[ "${MODE}" == smoke ]]; then
  "${HAB_PY}" "${SMOKE_VALIDATOR}" \
    --source-root "${SOURCE_ROOT}" \
    --direct-root "${DIRECT_ROOT}" \
    --graph-root "${GRAPH_ROOT}" \
    --output "${RUN_ROOT}/smoke_validation.json"
else
  "${HAB_PY}" "${SUMMARIZER}" \
    --manifest "${MANIFEST}" \
    --reference-root "${DIRECT_ROOT}" \
    --config "graph_gap16=${GRAPH_ROOT}" \
    > "${RUN_ROOT}/summary.json"
  cat "${RUN_ROOT}/summary.json"
fi
if [[ "${TERMINAL_UTURN}" != off ]]; then
  "${HAB_PY}" "${TERMINAL_SUMMARIZER}" \
    --root "${GRAPH_ROOT}/scenes" \
    --arm geometry_router \
    --out "${RUN_ROOT}/terminal_summary.json"
fi
if [[ "${ARRIVAL_SHADOW}" == diagnostic ]]; then
  "${HAB_PY}" "${ARRIVAL_SUMMARIZER}" \
    --root "${GRAPH_ROOT}/scenes" \
    --arm geometry_router \
    --out "${RUN_ROOT}/arrival_shadow_summary.json"
fi
echo "[complete] mode=${MODE} root=${RUN_ROOT}"
