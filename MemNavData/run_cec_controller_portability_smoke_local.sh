#!/usr/bin/env bash
# Consumed-scene integration smoke for the all-CEC controller comparison.
#
# Every controller receives the same role-free CEC proof stream.  A rejected
# action falls back to the same monocular NavDP ImageGoal controller.  An
# accepted action is projected through the selected controller's audited CEC
# adapter.  Both temporal controllers receive observation-only shadow updates,
# so a later per-action accept/reject transition remains causal.

set -euo pipefail
umask 0022

ROOT=${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
CONTROLLER=${CONTROLLER:-navdp}
EVAL_KIND=${EVAL_KIND:-nnr_revisit}
SCENE=${SCENE:-dhjEzFoUFzH}
EPISODE=${EPISODE:-episode_0005}
MAX_STEPS=${MAX_STEPS:-600}
EVAL_SEED=${EVAL_SEED:-5}

MEMNAV_PORT=${MEMNAV_PORT:-21840}
FALLBACK_PORT=${FALLBACK_PORT:-21841}
UPSTREAM_PORT=${UPSTREAM_PORT:-21842}
PROXY_PORT=${PROXY_PORT:-21843}
HUB_PORT=${HUB_PORT:-21844}

MEMNAV_PY=${MEMNAV_PY:-/home/asus/miniconda3/envs/memnav/bin/python}
HAB_PY=${HAB_PY:-/home/asus/miniconda3/envs/habitat/bin/python}
VINT_PY=${VINT_PY:-${ROOT}/.diagnostics/controller_portability_20260821/envs/vint/bin/python}
VIPLANNER_PY=${VIPLANNER_PY:-${ROOT}/.diagnostics/controller_portability_20260821/envs/viplanner-py310-cu118/bin/python}

MEMNAV_CKPT=${MEMNAV_CKPT:-/home/asus/Research/Nav-axis-uturn/.diagnostics/unseen_scene_eval_20260803/checkpoints/gatecurr600.memnav.ckpt}
NAVDP_CKPT=${NAVDP_CKPT:-/home/asus/Research/Nav/NavDP/baselines/navdp/checkpoints/navdp_checkpoint.ckpt}
VINT_CKPT=${VINT_CKPT:-${ROOT}/.diagnostics/controller_portability_20260821/checkpoints/vint.pth}
IPLANNER_CKPT=${IPLANNER_CKPT:-${ROOT}/.diagnostics/controller_portability_20260821/checkpoints/iplanner.pth}
VIPLANNER_CKPT=${VIPLANNER_CKPT:-${ROOT}/.diagnostics/controller_portability_20260821/checkpoints/viplanner.pt}
MASK2FORMER_CKPT=${MASK2FORMER_CKPT:-${ROOT}/.diagnostics/controller_portability_20260821/checkpoints/mask2former_r50_8xb2-lsj-50e_coco-panoptic_20230118_125535-54df384a.pth}
MASK2FORMER_CONFIG=${MASK2FORMER_CONFIG:-${ROOT}/.diagnostics/controller_portability_20260821/envs/viplanner-py310-cu118/lib/python3.10/site-packages/mmdet/.mim/configs/mask2former/mask2former_r50_8xb2-lsj-50e_coco-panoptic.py}

LINGBOT_REPO=${LINGBOT_REPO:-/home/asus/Research/Nav/NavDP/baselines/memnav/lingbot-map}
LINGBOT_WEIGHTS=${LINGBOT_WEIGHTS:-${LINGBOT_REPO}/weights/lingbot-map-long.pt}
LIGHTGLUE_REPO=${LIGHTGLUE_REPO:-${ROOT}/.diagnostics/dependencies/LightGlue}
DEPENDENCY_ROOT=${DEPENDENCY_ROOT:-${ROOT}/.diagnostics/dependencies/python}
INTERNNAV_ROOT=${INTERNNAV_ROOT:-${ROOT}/InternNav}

ASSET_ROOT=${ASSET_ROOT:-/home/asus/Research/datasets/mp3d_20scene/assets}
SCENE_FILE=${SCENE_FILE:-${ASSET_ROOT}/${SCENE}/${SCENE}.glb}
BENCHMARK_ROOT=${BENCHMARK_ROOT:-${ROOT}/.diagnostics/shared_online_nnr_smoke_20260813/benchmark_qw_native/${SCENE}}
TRACE_ROOT=${TRACE_ROOT:-${ROOT}/.diagnostics/shared_online_nnr_smoke_20260813/qw_native_shared_traces/${SCENE}}
RUN_ROOT=${RUN_ROOT:-${ROOT}/.diagnostics/controller_portability_20260821/local_cec_${CONTROLLER}_${SCENE}_${EPISODE}}

fail() { echo "ABORT: $*" >&2; exit 2; }

case "${CONTROLLER}" in
  navdp|vint|iplanner|viplanner) ;;
  *) fail "CONTROLLER must be navdp, vint, iplanner, or viplanner" ;;
esac
case "${EVAL_KIND}" in
  nnr_revisit|role_pair_mixed|lifelong_5leg|lifelong_nnr) ;;
  *) fail "EVAL_KIND must be nnr_revisit, role_pair_mixed, lifelong_5leg, or lifelong_nnr" ;;
esac
[[ "${MAX_STEPS}" =~ ^[1-9][0-9]*$ ]] || fail "MAX_STEPS must be positive"

required=(
  "${MEMNAV_PY}"
  "${HAB_PY}"
  "${MEMNAV_CKPT}"
  "${NAVDP_CKPT}"
  "${LINGBOT_WEIGHTS}"
  "${SCENE_FILE}"
  "${ROOT}/MemNavData/cec_controller_portability_hub.py"
  "${ROOT}/MemNavData/controller_portability_proxy.py"
)
if [[ "${EVAL_KIND}" == nnr_revisit || "${EVAL_KIND}" == lifelong_nnr ]]; then
  required+=(
    "${BENCHMARK_ROOT}/manifest.json"
    "${BENCHMARK_ROOT}/${EPISODE}/benchmark.json"
    "${TRACE_ROOT}/${EPISODE}_leg1_trace.json"
    "${TRACE_ROOT}/${EPISODE}_legB_trace.json"
    "${ROOT}/MemNavData/eval_shared_online_novel_revisit.py"
  )
elif [[ "${EVAL_KIND}" == role_pair_mixed ]]; then
  required+=(
    "${BENCHMARK_ROOT}/../manifest.json"
    "${BENCHMARK_ROOT}/${EPISODE}/role_pairs.json"
    "${ROOT}/MemNavData/eval_shared_online_role_pairs.py"
  )
else
  required+=(
    "${BENCHMARK_ROOT}/${EPISODE}/meta/gen_meta.json"
    "${BENCHMARK_ROOT}/${EPISODE}/goal_1.jpg"
    "${BENCHMARK_ROOT}/${EPISODE}/goal_2.jpg"
    "${ROOT}/MemNavData/eval_lifelong_5leg_habitat.py"
  )
fi
case "${CONTROLLER}" in
  vint) required+=("${VINT_PY}" "${VINT_CKPT}") ;;
  iplanner) required+=("${IPLANNER_CKPT}") ;;
  viplanner)
    required+=("${VIPLANNER_PY}" "${VIPLANNER_CKPT}"
               "${MASK2FORMER_CKPT}" "${MASK2FORMER_CONFIG}")
    ;;
esac
for item in "${required[@]}"; do
  [[ -r "${item}" ]] || fail "missing input ${item}"
done
[[ ! -e "${RUN_ROOT}" ]] || fail "output already exists: ${RUN_ROOT}"

ports=("${MEMNAV_PORT}" "${FALLBACK_PORT}" "${HUB_PORT}")
if [[ "${CONTROLLER}" != navdp ]]; then
  ports+=("${UPSTREAM_PORT}" "${PROXY_PORT}")
fi
for port in "${ports[@]}"; do
  if ss -ltn | awk '{print $4}' | grep -Eq "(^|:)${port}$"; then
    fail "port ${port} is already in use"
  fi
done

mkdir -p "${RUN_ROOT}/logs" "${RUN_ROOT}/buffer"
runtime_root=$(mktemp -d /tmp/cec_controller_portability.XXXXXX)
mkdir -p "${runtime_root}/memnav" "${runtime_root}/fallback" \
         "${runtime_root}/upstream" "${runtime_root}/proxy" \
         "${runtime_root}/hub"

memnav_pid=
fallback_pid=
upstream_pid=
proxy_pid=
hub_pid=
cleanup() {
  for process_id in "${hub_pid}" "${proxy_pid}" "${upstream_pid}" \
                    "${fallback_pid}" "${memnav_pid}"; do
    if [[ -n "${process_id}" ]] && kill -0 "${process_id}" 2>/dev/null; then
      kill "${process_id}" 2>/dev/null || true
      wait "${process_id}" 2>/dev/null || true
    fi
  done
}
trap cleanup EXIT INT TERM

wait_for_port() {
  local label=$1 pid=$2 port=$3 log=$4 ready=0
  for _attempt in $(seq 1 240); do
    if ! kill -0 "${pid}" 2>/dev/null; then
      tail -n 160 "${log}" >&2 || true
      fail "${label} exited during startup"
    fi
    if ss -ltn | awk '{print $4}' | grep -Eq "(^|:)${port}$"; then
      ready=1
      break
    fi
    sleep 2
  done
  [[ "${ready}" -eq 1 ]] || fail "${label} did not bind port ${port}"
}

"${MEMNAV_PY}" -m py_compile \
  "${ROOT}/MemNavData/controller_portability_contract.py" \
  "${ROOT}/MemNavData/controller_portability_proxy.py" \
  "${ROOT}/MemNavData/cec_controller_portability_hub.py"
"${HAB_PY}" -m py_compile \
  "${ROOT}/MemNavData/eval_2leg_habitat.py" \
  "${ROOT}/MemNavData/eval_3leg_habitat.py" \
  "${ROOT}/MemNavData/eval_lifelong_5leg_habitat.py" \
  "${ROOT}/MemNavData/eval_shared_online_lifelong_nnr.py" \
  "${ROOT}/MemNavData/eval_shared_online_novel_revisit.py" \
  "${ROOT}/MemNavData/eval_shared_online_role_pairs.py"

server_pythonpath=${ROOT}:${DEPENDENCY_ROOT}:${LIGHTGLUE_REPO}:${INTERNNAV_ROOT}/src/diffusion-policy
(
  cd "${runtime_root}/memnav"
  exec env PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    PYTHONPATH="${server_pythonpath}" \
    LINGBOT_REPO="${LINGBOT_REPO}" LINGBOT_WEIGHTS="${LINGBOT_WEIGHTS}" \
    MEMNAV_WINDOW=32 MEMNAV_NUM_SCALE=8 MEMNAV_MAX_FRAME_NUM=2048 \
    MEMNAV_GROUND_SCALE_MAX=6.0 MEMNAV_GATE_FUSION=complementary \
    MEMNAV_AUX_POSE_CALIBRATION=empirical MEMNAV_COLLISION_SELECT=1 \
    MEMNAV_REPORT_TO=none \
    "${MEMNAV_PY}" -u \
      "${ROOT}/NavDP/baselines/memnav/memnav_server.py" \
      --port "${MEMNAV_PORT}" --checkpoint "${MEMNAV_CKPT}" \
      --internnav_root "${INTERNNAV_ROOT}" --num_samples 16 \
      --exclude_recent 32 --retrieval raw \
      --retrieval_candidate_top_k 32 --retrieval_candidate_min_gap 16 \
      --graph_subgoal_spacing_m 0.0 --graph_subgoal_arrival_m 0.60 \
      --flow_gate auto --buffer_root "${RUN_ROOT}/buffer" \
      --certified_relocalization --lightglue_repo "${LIGHTGLUE_REPO}" \
      --lightglue_dependency_root "${DEPENDENCY_ROOT}" \
      --lightglue_max_keypoints 2048
) >"${RUN_ROOT}/logs/server_memnav.log" 2>&1 &
memnav_pid=$!

(
  cd "${runtime_root}/fallback"
  exec env NAVDP_DISABLE_VIDEO=1 PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="${ROOT}" \
    "${MEMNAV_PY}" -u \
      "${ROOT}/NavDP/baselines/navdp/navdp_server.py" \
      --port "${FALLBACK_PORT}" --checkpoint "${NAVDP_CKPT}" \
      --depth_source monocular_sidecar \
      --monocular_depth_url \
        "http://127.0.0.1:${MEMNAV_PORT}/monocular_depth_query"
) >"${RUN_ROOT}/logs/server_fallback_navdp.log" 2>&1 &
fallback_pid=$!

wait_for_port memnav "${memnav_pid}" "${MEMNAV_PORT}" \
  "${RUN_ROOT}/logs/server_memnav.log"
wait_for_port fallback_navdp "${fallback_pid}" "${FALLBACK_PORT}" \
  "${RUN_ROOT}/logs/server_fallback_navdp.log"

controller_url=http://127.0.0.1:${FALLBACK_PORT}
if [[ "${CONTROLLER}" != navdp ]]; then
  case "${CONTROLLER}" in
    vint)
      (
        cd "${ROOT}/NavDP/baselines/vint"
        exec env PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
          "${VINT_PY}" -u vint_server.py --port "${UPSTREAM_PORT}" \
          --robot_config configs/robot_config.yaml \
          --vint_config configs/vint.yaml --vint_checkpoint "${VINT_CKPT}"
      ) >"${RUN_ROOT}/logs/server_vint.log" 2>&1 &
      upstream_pid=$!
      proxy_depth=none
      checkpoint_args=(--checkpoint "vint=${VINT_CKPT}")
      ;;
    iplanner)
      (
        cd "${ROOT}/NavDP/baselines/iplanner"
        exec env PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
          "${MEMNAV_PY}" -u iplanner_server.py --port "${UPSTREAM_PORT}" \
          --config configs/iplanner.yaml --checkpoint "${IPLANNER_CKPT}"
      ) >"${RUN_ROOT}/logs/server_iplanner.log" 2>&1 &
      upstream_pid=$!
      proxy_depth=monocular_sidecar
      checkpoint_args=(--checkpoint "iplanner=${IPLANNER_CKPT}")
      ;;
    viplanner)
      (
        cd "${ROOT}/NavDP/baselines/viplanner"
        exec env PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
          "${VIPLANNER_PY}" -u viplanner_server.py \
          --port "${UPSTREAM_PORT}" --config configs/viplanner.yaml \
          --checkpoint "${VIPLANNER_CKPT}" \
          --m2f_config "${MASK2FORMER_CONFIG}" \
          --m2f_checkpoint "${MASK2FORMER_CKPT}"
      ) >"${RUN_ROOT}/logs/server_viplanner.log" 2>&1 &
      upstream_pid=$!
      proxy_depth=monocular_sidecar
      checkpoint_args=(
        --checkpoint "planner=${VIPLANNER_CKPT}"
        --checkpoint "mask2former=${MASK2FORMER_CKPT}"
      )
      ;;
  esac
  wait_for_port "${CONTROLLER}" "${upstream_pid}" "${UPSTREAM_PORT}" \
    "${RUN_ROOT}/logs/server_${CONTROLLER}.log"
  (
    cd "${runtime_root}/proxy"
    exec env PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
      PYTHONPATH="${ROOT}" "${MEMNAV_PY}" -u \
      "${ROOT}/MemNavData/controller_portability_proxy.py" \
        --controller "${CONTROLLER}" --protocol cec_proof_hybrid \
        --depth-source "${proxy_depth}" --query-population mixed_role \
        --reject-policy shared_native_exact --fallback-controller navdp \
        --repo-root "${ROOT}" \
        --upstream-base "http://127.0.0.1:${UPSTREAM_PORT}" \
        "${checkpoint_args[@]}" --host 127.0.0.1 --port "${PROXY_PORT}"
  ) >"${RUN_ROOT}/logs/server_proxy.log" 2>&1 &
  proxy_pid=$!
  wait_for_port proxy "${proxy_pid}" "${PROXY_PORT}" \
    "${RUN_ROOT}/logs/server_proxy.log"
  controller_url=http://127.0.0.1:${PROXY_PORT}
fi

(
  cd "${runtime_root}/hub"
  exec env PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="${ROOT}" \
    "${MEMNAV_PY}" -u \
      "${ROOT}/MemNavData/cec_controller_portability_hub.py" \
      --host 127.0.0.1 --port "${HUB_PORT}" \
      --controller "${CONTROLLER}" \
      --memnav-url "http://127.0.0.1:${MEMNAV_PORT}" \
      --controller-url "${controller_url}" \
      --fallback-navdp-url "http://127.0.0.1:${FALLBACK_PORT}" \
      --camera-height-m 0.5
) >"${RUN_ROOT}/logs/server_hub.log" 2>&1 &
hub_pid=$!
wait_for_port hub "${hub_pid}" "${HUB_PORT}" \
  "${RUN_ROOT}/logs/server_hub.log"

hab_site_packages=$("${HAB_PY}" -c \
  'import sysconfig; print(sysconfig.get_paths()["purelib"])')
hab_pythonpath=${ROOT}:${hab_site_packages}/pip/_vendor
leg1_mode=shared_trace
if [[ "${EVAL_KIND}" == lifelong_5leg ]]; then
  leg1_mode=policy
fi

common_eval=(
  --episode_root "${BENCHMARK_ROOT}" --episode_ids "${EPISODE}"
  --scene "${SCENE_FILE}" --scene_identity "${SCENE}"
  --host 127.0.0.1 --port "${HUB_PORT}"
  --server_backend cec_portability --navdp_depth_source monocular_sidecar
  --out "${RUN_ROOT}/result" --success_dist 1.0
  --max_steps "${MAX_STEPS}" --exec_horizon 8
  --trajectory_selector server --trajectory_selector_scope all
  --leg1_mode "${leg1_mode}"
  --leg1_goal_source own --seed "${EVAL_SEED}"
  --terminal_uturn off --terminal_visual_refine off
  --deterministic_plan_seeds --retrieval_override off
  --certified_cdec_rescue off --certified_stagnation_graph off
  --hybrid_route phase --revisit_controller navdp_mixed
  --revisit_adapter legacy_metric
)
for remap_name in SHARED_PATH_REMAP_1 SHARED_PATH_REMAP_2; do
  remap_value=${!remap_name:-}
  if [[ -n "${remap_value}" ]]; then
    common_eval+=(--shared_path_remap "${remap_value}")
  fi
done
if [[ "${EVAL_KIND}" == nnr_revisit ]]; then
  evaluator=${ROOT}/MemNavData/eval_shared_online_novel_revisit.py
  eval_extra=(
    --navdp_goal_switch_reset before_c
    --shared_leg1_trace_root "${TRACE_ROOT}"
    --double_revisit_c_history initial_leg_only
    --shared_online_nnr_arm cec_portability
  )
elif [[ "${EVAL_KIND}" == lifelong_nnr ]]; then
  evaluator=${ROOT}/MemNavData/eval_shared_online_lifelong_nnr.py
  eval_extra=(
    --navdp_goal_switch_reset before_c
    --shared_leg1_trace_root "${TRACE_ROOT}"
    --double_revisit_c_history initial_leg_only
    --shared_online_nnr_arm cec_portability
    --lifelong_history_scope "${LIFELONG_HISTORY_SCOPE:-all_prior}"
  )
elif [[ "${EVAL_KIND}" == role_pair_mixed ]]; then
  evaluator=${ROOT}/MemNavData/eval_shared_online_role_pairs.py
  eval_extra=(
    --role_pair_scope consumed_integration
    --role_pair_query_role all
  )
else
  evaluator=${ROOT}/MemNavData/eval_lifelong_5leg_habitat.py
  eval_extra=(
    --navdp_goal_switch_reset carry
    --lifelong_sequence natural_abcbc
    --lifelong_history_scope "${LIFELONG_HISTORY_SCOPE:-all_prior}"
  )
fi
env PYTHONPATH="${hab_pythonpath}" PYTHONDONTWRITEBYTECODE=1 \
  "${HAB_PY}" -u "${evaluator}" "${common_eval[@]}" "${eval_extra[@]}" \
  >"${RUN_ROOT}/logs/evaluator.log" 2>&1

curl --fail --silent "http://127.0.0.1:${HUB_PORT}/healthz" \
  >"${RUN_ROOT}/hub_health.json"
if [[ "${EVAL_KIND}" == nnr_revisit || "${EVAL_KIND}" == lifelong_nnr ]]; then
  receipt_inputs=(
    "${BENCHMARK_ROOT}/manifest.json"
    "${BENCHMARK_ROOT}/${EPISODE}/benchmark.json"
    "${TRACE_ROOT}/${EPISODE}_leg1_trace.json"
    "${TRACE_ROOT}/${EPISODE}_legB_trace.json"
  )
elif [[ "${EVAL_KIND}" == role_pair_mixed ]]; then
  receipt_inputs=(
    "${BENCHMARK_ROOT}/../manifest.json"
    "${BENCHMARK_ROOT}/${EPISODE}/role_pairs.json"
  )
else
  receipt_inputs=(
    "${BENCHMARK_ROOT}/${EPISODE}/meta/gen_meta.json"
    "${BENCHMARK_ROOT}/${EPISODE}/goal_1.jpg"
    "${BENCHMARK_ROOT}/${EPISODE}/goal_2.jpg"
  )
fi
mapfile -t result_files < <(find "${RUN_ROOT}/result" -maxdepth 1 -type f \
  -name '*.json' -print | sort)
sha256sum "${receipt_inputs[@]}" "${result_files[@]}" \
  >"${RUN_ROOT}/result_inputs.sha256"
echo "DONE controller=${CONTROLLER} result=${RUN_ROOT}/result/summary.json"
