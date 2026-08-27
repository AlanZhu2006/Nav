#!/usr/bin/env bash
# Run the frozen seven-episode Revisit actionability shadow on the local GPU.
# The native counterfactual is read-only and never changes factual control.

set -euo pipefail
umask 0022

ROOT=${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
OUT_ROOT=${OUT_ROOT:-${ROOT}/.diagnostics/revisit_action_shadow_20260811}
REFERENCE_ROOT=${REFERENCE_ROOT:-${ROOT}/.diagnostics/revisit_known_phase_ablation_20260811}
DATA_ROOT=${DATA_ROOT:-/home/asus/Research/datasets/mp3d_20scene}
MANIFEST=${MANIFEST:-${ROOT}/MemNavData/expanded_navdp_router_eval_20260805.json}
PROTOCOL=${PROTOCOL:-${ROOT}/MemNavData/REVISIT_ACTIONABILITY_SHADOW_PROTOCOL_20260811.md}
SUMMARIZER=${SUMMARIZER:-${ROOT}/MemNavData/summarize_revisit_action_shadow.py}
MEMNAV_PORT=${MEMNAV_PORT:-21210}
NAVDP_PORT=${NAVDP_PORT:-21211}
MEMNAV_PY=${MEMNAV_PY:-/home/asus/miniconda3/envs/memnav/bin/python}
HAB_PY=${HAB_PY:-/home/asus/miniconda3/envs/habitat/bin/python}
MEMNAV_CKPT=${MEMNAV_CKPT:-/home/asus/Research/Nav-axis-uturn/.diagnostics/unseen_scene_eval_20260803/checkpoints/gatecurr600.memnav.ckpt}
NAVDP_CKPT=${NAVDP_CKPT:-/home/asus/Research/Nav/NavDP/baselines/navdp/checkpoints/navdp_checkpoint.ckpt}
LINGBOT_REPO=${LINGBOT_REPO:-/home/asus/Research/Nav/NavDP/baselines/memnav/lingbot-map}
LINGBOT_WEIGHTS=${LINGBOT_WEIGHTS:-${LINGBOT_REPO}/weights/lingbot-map-long.pt}
INTERNNAV_ROOT=${INTERNNAV_ROOT:-${ROOT}/InternNav}
TARGET_INDICES=${TARGET_INDICES:-0,1,2,3,4,5,6}

TARGET_SCENE_INDEX=(6 4 11 13 15 17 19)
TARGET_SCENE=(pLe4wQe7qrG yqstnuAEVhm uNb9QFRL6hY ac26ZMwG7aT qoiz87JEwZ2 i5noydFURQK gZ6f7yhEvPG)
TARGET_EPISODE=(episode_0001 episode_0001 episode_0000 episode_0000 episode_0000 episode_0000 episode_0000)
TARGET_EPISODE_SEED=(20260804 20260804 20260803 20260803 20260803 20260803 20260803)

for required in "${MEMNAV_PY}" "${HAB_PY}" "${MEMNAV_CKPT}" \
                "${NAVDP_CKPT}" "${LINGBOT_WEIGHTS}" "${MANIFEST}" \
                "${PROTOCOL}" "${SUMMARIZER}" \
                "${ROOT}/MemNavData/eval_2leg_habitat.py" \
                "${ROOT}/MemNavData/revisit_action_shadow.py"; do
  test -r "${required}" || { echo "ABORT: missing ${required}" >&2; exit 1; }
done
for port in "${MEMNAV_PORT}" "${NAVDP_PORT}"; do
  if ss -ltn | awk '{print $4}' | grep -Eq "(^|:)${port}$"; then
    echo "ABORT: port ${port} is already in use" >&2
    exit 1
  fi
done

IFS=',' read -r -a SELECTED_TARGETS <<<"${TARGET_INDICES}"
for target in "${SELECTED_TARGETS[@]}"; do
  [[ "${target}" =~ ^[0-6]$ ]] || {
    echo "ABORT: TARGET_INDICES must contain only 0..6" >&2; exit 1; }
done

mkdir -p "${OUT_ROOT}/logs" "${OUT_ROOT}/buffer" "${OUT_ROOT}/targets"
contract_tmp=$(mktemp)
runtime_root=$(mktemp -d /tmp/revisit_action_shadow.XXXXXX)
MEMNAV_PID=
NAVDP_PID=
cleanup() {
  for pid in "${NAVDP_PID}" "${MEMNAV_PID}"; do
    if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
      kill "${pid}" 2>/dev/null || true
      wait "${pid}" 2>/dev/null || true
    fi
  done
  rm -f -- "${contract_tmp}"
  rm -rf -- "${runtime_root}"
}
trap cleanup EXIT INT TERM

trace_inputs=()
for target in $(seq 0 6); do
  scene_index=${TARGET_SCENE_INDEX[${target}]}
  scene=${TARGET_SCENE[${target}]}
  episode=${TARGET_EPISODE[${target}]}
  trace_inputs+=(
    "${REFERENCE_ROOT}/scenes/$(printf '%02d' "${scene_index}")_${scene}/geometry_router/${episode}_leg1_trace.json"
  )
done
sha256sum \
  "${ROOT}/MemNavData/run_revisit_action_shadow_local.sh" \
  "${PROTOCOL}" \
  "${ROOT}/MemNavData/eval_2leg_habitat.py" \
  "${ROOT}/MemNavData/revisit_action_shadow.py" \
  "${SUMMARIZER}" \
  "${ROOT}/NavDP/baselines/memnav/memnav_server.py" \
  "${ROOT}/NavDP/baselines/memnav/policy_agent.py" \
  "${ROOT}/NavDP/baselines/navdp/navdp_server.py" \
  "${ROOT}/NavDP/baselines/navdp/policy_agent.py" \
  "${MANIFEST}" "${trace_inputs[@]}" > "${contract_tmp}"
if [[ -e "${OUT_ROOT}/source_inputs.sha256" ]]; then
  cmp --silent "${contract_tmp}" "${OUT_ROOT}/source_inputs.sha256" || {
    echo "ABORT: source inputs changed during resume" >&2; exit 1; }
else
  mv "${contract_tmp}" "${OUT_ROOT}/source_inputs.sha256"
  contract_tmp=$(mktemp)
fi

"${MEMNAV_PY}" -m py_compile \
  "${ROOT}/MemNavData/revisit_action_shadow.py" "${SUMMARIZER}" \
  "${ROOT}/NavDP/baselines/memnav/memnav_server.py" \
  "${ROOT}/NavDP/baselines/navdp/navdp_server.py"
"${HAB_PY}" -m py_compile "${ROOT}/MemNavData/eval_2leg_habitat.py"

mkdir -p "${runtime_root}/memnav" "${runtime_root}/navdp"
(
  cd "${runtime_root}/memnav"
  exec env PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    PYTHONPATH="${INTERNNAV_ROOT}/src/diffusion-policy:${PYTHONPATH:-}" \
    LINGBOT_REPO="${LINGBOT_REPO}" LINGBOT_WEIGHTS="${LINGBOT_WEIGHTS}" \
    MEMNAV_WINDOW=32 MEMNAV_NUM_SCALE=8 MEMNAV_MAX_FRAME_NUM=2048 \
    MEMNAV_GROUND_SCALE_MAX=6.0 MEMNAV_GATE_FUSION=complementary \
    MEMNAV_AUX_POSE_CALIBRATION=empirical MEMNAV_COLLISION_SELECT=1 \
    MEMNAV_REPORT_TO=none \
    "${MEMNAV_PY}" -u "${ROOT}/NavDP/baselines/memnav/memnav_server.py" \
      --port "${MEMNAV_PORT}" --checkpoint "${MEMNAV_CKPT}" \
      --internnav_root "${INTERNNAV_ROOT}" --num_samples 16 \
      --exclude_recent 32 --retrieval raw --retrieval_candidate_top_k 32 \
      --retrieval_candidate_min_gap 16 --graph_subgoal_spacing_m 0.0 \
      --graph_subgoal_arrival_m 0.60 --flow_gate auto \
      --buffer_root "${OUT_ROOT}/buffer"
) > "${OUT_ROOT}/logs/server_memnav.log" 2>&1 &
MEMNAV_PID=$!

(
  cd "${runtime_root}/navdp"
  exec env NAVDP_DISABLE_VIDEO=1 PYTHONUNBUFFERED=1 \
    "${MEMNAV_PY}" -u "${ROOT}/NavDP/baselines/navdp/navdp_server.py" \
      --port "${NAVDP_PORT}" --checkpoint "${NAVDP_CKPT}"
) > "${OUT_ROOT}/logs/server_navdp.log" 2>&1 &
NAVDP_PID=$!

for spec in \
    "memnav:${MEMNAV_PID}:${MEMNAV_PORT}:${OUT_ROOT}/logs/server_memnav.log" \
    "navdp:${NAVDP_PID}:${NAVDP_PORT}:${OUT_ROOT}/logs/server_navdp.log"; do
  IFS=: read -r label pid port log <<<"${spec}"
  ready=0
  for _ in $(seq 1 240); do
    if ! kill -0 "${pid}" 2>/dev/null; then
      echo "ABORT: ${label} exited during startup" >&2
      tail -n 120 "${log}" >&2
      exit 1
    fi
    if ss -ltn | awk '{print $4}' | grep -Eq "(^|:)${port}$"; then
      ready=1
      break
    fi
    sleep 2
  done
  [[ "${ready}" -eq 1 ]] || {
    echo "ABORT: ${label} did not bind ${port}" >&2; exit 1; }
done

HAB_SITE_PACKAGES=$("${HAB_PY}" -c \
  'import sysconfig; print(sysconfig.get_paths()["purelib"])')
HAB_PYTHONPATH=${HAB_SITE_PACKAGES}/pip/_vendor${PYTHONPATH:+:${PYTHONPATH}}

for target in "${SELECTED_TARGETS[@]}"; do
  scene_index=${TARGET_SCENE_INDEX[${target}]}
  scene=${TARGET_SCENE[${target}]}
  episode=${TARGET_EPISODE[${target}]}
  episode_seed=${TARGET_EPISODE_SEED[${target}]}
  prefix=$(printf '%02d' "${scene_index}")_${scene}
  trace_root=${REFERENCE_ROOT}/scenes/${prefix}/geometry_router
  target_root=${OUT_ROOT}/targets/${prefix}_${episode}
  if [[ -s "${target_root}/summary.json" && -s "${target_root}/metric.csv" \
        && -s "${target_root}/${episode}_plans.json" ]]; then
    echo "[skip] target=${target} ${scene}/${episode} complete"
    continue
  fi
  if [[ -e "${target_root}" ]]; then
    mv "${target_root}" "${target_root}.partial.$(date +%Y%m%dT%H%M%S)"
  fi
  mkdir -p "${target_root}"
  echo "[shadow] target=${target} ${scene}/${episode}"
  env PYTHONPATH="${HAB_PYTHONPATH}" "${HAB_PY}" -u \
    "${ROOT}/MemNavData/eval_2leg_habitat.py" \
    --episode_root "${DATA_ROOT}/episodes/${scene}" \
    --scene "${DATA_ROOT}/assets/${scene}/${scene}.glb" \
    --host 127.0.0.1 --port "${MEMNAV_PORT}" --novel_port "${NAVDP_PORT}" \
    --out "${target_root}" --server_backend hybrid_pose \
    --leg1_mode shared_trace --shared_leg1_trace_root "${trace_root}" \
    --hybrid_route phase --revisit_controller navdp_mixed \
    --revisit_adapter legacy_metric \
    --revisit_action_shadow native_counterfactual \
    --success_dist 1.0 --max_steps 500 --exec_horizon 8 \
    --trajectory_selector server --navdp_goal_switch_reset carry \
    --leg1_goal_source own --seed "${episode_seed}" --episode_ids "${episode}" \
    --terminal_uturn off --terminal_visual_refine off \
    --deterministic_plan_seeds \
    > "${OUT_ROOT}/logs/eval_${scene}_${episode}.log" 2>&1
  "${HAB_PY}" - "${target_root}/summary.json" <<'PY'
import json, sys
s=json.load(open(sys.argv[1]))
assert s["episodes"] == 1
assert s["revisit_action_shadow"] == "native_counterfactual"
assert s["revisit_action_shadow_plan_count"] > 0
assert s["revisit_action_shadow_available_plan_count"] > 0
assert s["deterministic_plan_seeds"] is True
PY
done

complete=1
for target in $(seq 0 6); do
  scene_index=${TARGET_SCENE_INDEX[${target}]}
  scene=${TARGET_SCENE[${target}]}
  episode=${TARGET_EPISODE[${target}]}
  target_root=${OUT_ROOT}/targets/$(printf '%02d' "${scene_index}")_${scene}_${episode}
  [[ -s "${target_root}/summary.json" ]] || complete=0
done
if [[ "${complete}" -eq 1 && ! -e "${OUT_ROOT}/report.json" ]]; then
  (
    cd "${ROOT}"
    "${HAB_PY}" -u -m MemNavData.summarize_revisit_action_shadow \
      --run-root "${OUT_ROOT}" --reference-root "${REFERENCE_ROOT}" \
      --out "${OUT_ROOT}/report.json"
  ) > "${OUT_ROOT}/logs/summarize.log" 2>&1
  sha256sum "${OUT_ROOT}/report.json" > "${OUT_ROOT}/report.json.sha256"
fi
echo "[complete] selected targets finished; root=${OUT_ROOT} all=${complete}"
