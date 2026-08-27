#!/usr/bin/env bash
# One-scene end-to-end smoke for the actual-online Novel->Revisit protocol.

set -euo pipefail
umask 0022

PROJECT_ROOT=${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
SCENE=${SCENE:-dhjEzFoUFzH}
EPISODE=${EPISODE:-episode_0005}
EVAL_SEED=${EVAL_SEED:-5}
SOURCE_SCENE_ROOT=${SOURCE_SCENE_ROOT:-${PROJECT_ROOT}/.diagnostics/shared_online_nnr_smoke_20260813/qw_strict_v4/${SCENE}}
ASSET_ROOT=${ASSET_ROOT:-/home/asus/Research/datasets/mp3d_20scene/assets}
RUN_ROOT=${RUN_ROOT:-${PROJECT_ROOT}/.diagnostics/shared_online_nnr_smoke_20260813/local_${SCENE}_${EPISODE}}
TRACE_ROOT_OVERRIDE=${TRACE_ROOT_OVERRIDE:-}
BENCHMARK_SCENE_ROOT_OVERRIDE=${BENCHMARK_SCENE_ROOT_OVERRIDE:-}
MEMNAV_PORT=${MEMNAV_PORT:-21640}
NAVDP_PORT=${NAVDP_PORT:-21641}
MEMNAV_PY=${MEMNAV_PY:-/home/asus/miniconda3/envs/memnav/bin/python}
HAB_PY=${HAB_PY:-/home/asus/miniconda3/envs/habitat/bin/python}
MEMNAV_CKPT=${MEMNAV_CKPT:-/home/asus/Research/Nav-axis-uturn/.diagnostics/unseen_scene_eval_20260803/checkpoints/gatecurr600.memnav.ckpt}
NAVDP_CKPT=${NAVDP_CKPT:-/home/asus/Research/Nav/NavDP/baselines/navdp/checkpoints/navdp_checkpoint.ckpt}
LINGBOT_REPO=${LINGBOT_REPO:-/home/asus/Research/Nav/NavDP/baselines/memnav/lingbot-map}
LINGBOT_WEIGHTS=${LINGBOT_WEIGHTS:-${LINGBOT_REPO}/weights/lingbot-map-long.pt}
LIGHTGLUE_REPO=${LIGHTGLUE_REPO:-${PROJECT_ROOT}/.diagnostics/dependencies/LightGlue}
DEPENDENCY_ROOT=${DEPENDENCY_ROOT:-${PROJECT_ROOT}/.diagnostics/dependencies/python}
INTERNNAV_ROOT=${INTERNNAV_ROOT:-${PROJECT_ROOT}/InternNav}
MAX_STEPS=${MAX_STEPS:-600}

fail() { echo "ABORT: $*" >&2; exit 2; }

SCENE_FILE=${ASSET_ROOT}/${SCENE}/${SCENE}.glb
SOURCE_EPISODE=${SOURCE_SCENE_ROOT}/${EPISODE}
for required in \
  "${MEMNAV_PY}" "${HAB_PY}" "${MEMNAV_CKPT}" "${NAVDP_CKPT}" \
  "${LINGBOT_WEIGHTS}" "${SCENE_FILE}" \
  "${SOURCE_EPISODE}/meta/gen_meta.json" \
  "${SOURCE_EPISODE}/data/chunk-000/episode_000000.parquet" \
  "${PROJECT_ROOT}/MemNavData/eval_3leg_habitat.py" \
  "${PROJECT_ROOT}/MemNavData/build_shared_online_novel_revisit.py" \
  "${PROJECT_ROOT}/MemNavData/eval_shared_online_novel_revisit.py"; do
  test -r "${required}" || fail "missing input ${required}"
done
for port in "${MEMNAV_PORT}" "${NAVDP_PORT}"; do
  if ss -ltn | awk '{print $4}' | grep -Eq "(^|:)${port}$"; then
    fail "port ${port} is already in use"
  fi
done

mkdir -p "${RUN_ROOT}/logs" "${RUN_ROOT}/buffer"
runtime_root=$(mktemp -d /tmp/shared_online_nnr_smoke.XXXXXX)
mkdir -p "${runtime_root}/memnav" "${runtime_root}/navdp"
memnav_pid=
navdp_pid=
cleanup() {
  for process_id in "${navdp_pid}" "${memnav_pid}"; do
    if [[ -n "${process_id}" ]] && kill -0 "${process_id}" 2>/dev/null; then
      kill "${process_id}" 2>/dev/null || true
      wait "${process_id}" 2>/dev/null || true
    fi
  done
}
trap cleanup EXIT INT TERM

"${HAB_PY}" -m py_compile \
  "${PROJECT_ROOT}/MemNavData/eval_2leg_habitat.py" \
  "${PROJECT_ROOT}/MemNavData/eval_3leg_habitat.py" \
  "${PROJECT_ROOT}/MemNavData/build_shared_online_novel_revisit.py" \
  "${PROJECT_ROOT}/MemNavData/eval_shared_online_novel_revisit.py"

server_pythonpath=${PROJECT_ROOT}:${DEPENDENCY_ROOT}:${LIGHTGLUE_REPO}:${INTERNNAV_ROOT}/src/diffusion-policy
(
  cd "${runtime_root}/memnav"
  exec env PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    PYTHONPATH="${server_pythonpath}" \
    LINGBOT_REPO="${LINGBOT_REPO}" LINGBOT_WEIGHTS="${LINGBOT_WEIGHTS}" \
    MEMNAV_WINDOW=32 MEMNAV_NUM_SCALE=8 MEMNAV_MAX_FRAME_NUM=2048 \
    MEMNAV_GROUND_SCALE_MAX=6.0 MEMNAV_GATE_FUSION=complementary \
    MEMNAV_AUX_POSE_CALIBRATION=empirical MEMNAV_COLLISION_SELECT=1 \
    MEMNAV_REPORT_TO=none \
    "${MEMNAV_PY}" -u \
      "${PROJECT_ROOT}/NavDP/baselines/memnav/memnav_server.py" \
      --port "${MEMNAV_PORT}" --checkpoint "${MEMNAV_CKPT}" \
      --internnav_root "${INTERNNAV_ROOT}" --num_samples 16 \
      --exclude_recent 32 --retrieval raw \
      --retrieval_candidate_top_k 32 --retrieval_candidate_min_gap 16 \
      --graph_subgoal_spacing_m 1.25 --graph_subgoal_arrival_m 0.60 \
      --flow_gate auto --buffer_root "${RUN_ROOT}/buffer" \
      --certified_relocalization --lightglue_repo "${LIGHTGLUE_REPO}" \
      --lightglue_dependency_root "${DEPENDENCY_ROOT}" \
      --lightglue_max_keypoints 2048
) > "${RUN_ROOT}/logs/server_memnav.log" 2>&1 &
memnav_pid=$!

(
  cd "${runtime_root}/navdp"
  exec env NAVDP_DISABLE_VIDEO=1 PYTHONUNBUFFERED=1 \
    PYTHONPATH="${server_pythonpath}" \
    "${MEMNAV_PY}" -u \
      "${PROJECT_ROOT}/NavDP/baselines/navdp/navdp_server.py" \
      --port "${NAVDP_PORT}" --checkpoint "${NAVDP_CKPT}"
) > "${RUN_ROOT}/logs/server_navdp.log" 2>&1 &
navdp_pid=$!

for server_spec in \
  "memnav:${memnav_pid}:${MEMNAV_PORT}:${RUN_ROOT}/logs/server_memnav.log" \
  "navdp:${navdp_pid}:${NAVDP_PORT}:${RUN_ROOT}/logs/server_navdp.log"; do
  IFS=: read -r server_label server_pid server_port server_log <<<"${server_spec}"
  ready=0
  for _attempt in $(seq 1 240); do
    if ! kill -0 "${server_pid}" 2>/dev/null; then
      tail -n 160 "${server_log}" >&2
      fail "${server_label} server exited during startup"
    fi
    if ss -ltn | awk '{print $4}' | grep -Eq "(^|:)${server_port}$"; then
      ready=1
      break
    fi
    sleep 2
  done
  [[ "${ready}" -eq 1 ]] || fail "${server_label} did not bind"
done

hab_site_packages=$("${HAB_PY}" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
hab_pythonpath=${PROJECT_ROOT}:${hab_site_packages}/pip/_vendor
trace_root=${TRACE_ROOT_OVERRIDE:-${RUN_ROOT}/trace}
if [[ -n "${TRACE_ROOT_OVERRIDE}" ]]; then
  test -s "${trace_root}/${EPISODE}_leg1_trace.json" || \
    fail "override A trace is missing"
  test -s "${trace_root}/${EPISODE}_legB_trace.json" || \
    fail "override B trace is missing"
elif [[ ! -s "${trace_root}/${EPISODE}_leg1_trace.json" \
      || ! -s "${trace_root}/${EPISODE}_legB_trace.json" ]]; then
  [[ ! -e "${trace_root}" ]] || fail "partial trace root exists: ${trace_root}"
  mkdir -p "${trace_root}"
  env PYTHONPATH="${hab_pythonpath}" "${HAB_PY}" -u \
    "${PROJECT_ROOT}/MemNavData/eval_3leg_habitat.py" \
    --episode_root "${SOURCE_SCENE_ROOT}" --episode_ids "${EPISODE}" \
    --scene "${SCENE_FILE}" --host 127.0.0.1 \
    --port "${MEMNAV_PORT}" --novel_port "${NAVDP_PORT}" \
    --server_backend hybrid_pose --out "${trace_root}" \
    --success_dist 1.0 --max_steps "${MAX_STEPS}" --exec_horizon 8 \
    --trajectory_selector server --trajectory_selector_scope all \
    --navdp_goal_switch_reset carry --leg1_mode policy --write_leg1_trace \
    --leg1_goal_source own --seed "${EVAL_SEED}" \
    --terminal_uturn off --terminal_visual_refine off \
    --deterministic_plan_seeds --retrieval_override off \
    --double_revisit_c_history initial_leg_only \
    --certified_cdec_rescue off --certified_stagnation_graph off \
    --hybrid_route phase --revisit_controller navdp_mixed \
    --revisit_adapter legacy_metric \
    > "${RUN_ROOT}/logs/eval_record.log" 2>&1
fi

benchmark_scene_root=${BENCHMARK_SCENE_ROOT_OVERRIDE:-${RUN_ROOT}/benchmark/${SCENE}}
if [[ -n "${BENCHMARK_SCENE_ROOT_OVERRIDE}" ]]; then
  test -s "${benchmark_scene_root}/manifest.json" || \
    fail "override benchmark manifest is missing"
elif [[ ! -s "${benchmark_scene_root}/manifest.json" ]]; then
  [[ ! -e "${benchmark_scene_root}" ]] || \
    fail "partial benchmark exists: ${benchmark_scene_root}"
  env PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/MemNavData" \
    "${HAB_PY}" -u \
    "${PROJECT_ROOT}/MemNavData/build_shared_online_novel_revisit.py" \
    --episode-root "${SOURCE_SCENE_ROOT}" --episode-ids "${EPISODE}" \
    --trace-root "${trace_root}" --scene-asset "${SCENE_FILE}" \
    --out "${benchmark_scene_root}" \
    > "${RUN_ROOT}/logs/build_benchmark.log" 2>&1
fi

common=(
  --episode_root "${benchmark_scene_root}" --episode_ids "${EPISODE}"
  --scene "${SCENE_FILE}" --host 127.0.0.1
  --port "${MEMNAV_PORT}" --novel_port "${NAVDP_PORT}"
  --server_backend hybrid_pose --success_dist 1.0 --max_steps "${MAX_STEPS}"
  --exec_horizon 8 --trajectory_selector server --trajectory_selector_scope all
  --navdp_goal_switch_reset before_c --leg1_mode shared_trace
  --shared_leg1_trace_root "${trace_root}" --leg1_goal_source own
  --seed "${EVAL_SEED}" --terminal_uturn off --terminal_visual_refine off
  --deterministic_plan_seeds --retrieval_override off
  --double_revisit_c_history initial_leg_only
  --certified_cdec_rescue off
  --revisit_controller navdp_mixed
)
for arm in native known_direct certified certified_budget certified_graph; do
  arm_root=${RUN_ROOT}/arms/${arm}
  if [[ -s "${arm_root}/summary.json" ]]; then
    continue
  fi
  [[ ! -e "${arm_root}" ]] || fail "partial arm exists: ${arm_root}"
  mkdir -p "${arm_root}"
  if [[ "${arm}" == certified* ]]; then
    route=(--hybrid_route certified_relocalization \
           --revisit_adapter verified_bearing_v1)
  else
    route=(--hybrid_route phase --revisit_adapter legacy_metric)
  fi
  case "${arm}" in
    certified_budget) stagnation_mode=budget_control ;;
    certified_graph) stagnation_mode=rescue ;;
    *) stagnation_mode=off ;;
  esac
  env PYTHONPATH="${hab_pythonpath}" "${HAB_PY}" -u \
    "${PROJECT_ROOT}/MemNavData/eval_shared_online_novel_revisit.py" \
    "${common[@]}" "${route[@]}" --shared_online_nnr_arm "${arm}" \
    --certified_stagnation_graph "${stagnation_mode}" \
    --out "${arm_root}" > "${RUN_ROOT}/logs/eval_${arm}.log" 2>&1
done

sha256sum \
  "${trace_root}/${EPISODE}_leg1_trace.json" \
  "${trace_root}/${EPISODE}_legB_trace.json" \
  "${benchmark_scene_root}/manifest.json" \
  "${RUN_ROOT}/arms/native/summary.json" \
  "${RUN_ROOT}/arms/known_direct/summary.json" \
  "${RUN_ROOT}/arms/certified/summary.json" \
  "${RUN_ROOT}/arms/certified_budget/summary.json" \
  "${RUN_ROOT}/arms/certified_graph/summary.json" \
  > "${RUN_ROOT}/result_inputs.sha256"
echo "DONE ${RUN_ROOT}"
