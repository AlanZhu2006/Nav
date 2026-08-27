#!/usr/bin/env bash
# One-episode true-3-leg collection-contract smoke.  Impossible router
# thresholds force every action to remain native NavDP while top-8 evidence is
# recorded for offline temporal modeling.

set -euo pipefail
umask 0022

ROOT=${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
OUT_ROOT=${OUT_ROOT:-${ROOT}/.diagnostics/unknown_goal_natural_stream_smoke_v2_20260811}
SCENE=${SCENE:-17DRP5sb8fy}
EPISODE_IDS=${EPISODE_IDS:-episode_0000}
MEMNAV_PORT=${MEMNAV_PORT:-21130}
NAVDP_PORT=${NAVDP_PORT:-21131}
MEMNAV_PY=${MEMNAV_PY:-/home/asus/miniconda3/envs/memnav/bin/python}
HAB_PY=${HAB_PY:-/home/asus/miniconda3/envs/habitat/bin/python}
MEMNAV_CKPT=${MEMNAV_CKPT:-/home/asus/Research/Nav-axis-uturn/.diagnostics/unseen_scene_eval_20260803/checkpoints/gatecurr600.memnav.ckpt}
NAVDP_CKPT=${NAVDP_CKPT:-/home/asus/Research/Nav/NavDP/baselines/navdp/checkpoints/navdp_checkpoint.ckpt}
LINGBOT_REPO=${LINGBOT_REPO:-/home/asus/Research/Nav/NavDP/baselines/memnav/lingbot-map}
LINGBOT_WEIGHTS=${LINGBOT_WEIGHTS:-${LINGBOT_REPO}/weights/lingbot-map-long.pt}
INTERNNAV_ROOT=${INTERNNAV_ROOT:-${ROOT}/InternNav}
EPISODE_ROOT=${EPISODE_ROOT:-/home/asus/Research/Nav/memnav_viz/validate_gated/mp3d_3leg/${SCENE}}
SCENE_FILE=${SCENE_FILE:-/home/asus/Research/datasets/mp3d/${SCENE}.glb}
MAX_STEPS=${MAX_STEPS:-500}

for required in "${MEMNAV_PY}" "${HAB_PY}" "${MEMNAV_CKPT}" \
    "${NAVDP_CKPT}" "${LINGBOT_WEIGHTS}" "${SCENE_FILE}" \
    "${EPISODE_ROOT}/${EPISODE_IDS}/meta/gen_meta.json"; do
  test -r "${required}" || { echo "ABORT: missing ${required}" >&2; exit 1; }
done
test ! -e "${OUT_ROOT}" || { echo "ABORT: output exists ${OUT_ROOT}" >&2; exit 1; }
for port in "${MEMNAV_PORT}" "${NAVDP_PORT}"; do
  if ss -ltn | awk '{print $4}' | grep -Eq "(^|:)${port}$"; then
    echo "ABORT: port ${port} is in use" >&2; exit 1
  fi
done

mkdir -p "${OUT_ROOT}/logs" "${OUT_ROOT}/buffer" "${OUT_ROOT}/plans"
runtime_root=$(mktemp -d /tmp/unknown_goal_stream.XXXXXX)
MEMNAV_PID=
NAVDP_PID=
cleanup() {
  for pid in "${NAVDP_PID}" "${MEMNAV_PID}"; do
    if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
      kill "${pid}" 2>/dev/null || true
      wait "${pid}" 2>/dev/null || true
    fi
  done
  rm -rf -- "${runtime_root}"
}
trap cleanup EXIT INT TERM

sha256sum \
  "${ROOT}/MemNavData/UNKNOWN_GOAL_NATURAL_STREAM_SHADOW_PROTOCOL_20260811.md" \
  "${ROOT}/MemNavData/run_unknown_goal_natural_stream_smoke.sh" \
  "${ROOT}/MemNavData/summarize_unknown_goal_natural_stream_shadow.py" \
  "${ROOT}/MemNavData/eval_2leg_habitat.py" \
  "${ROOT}/MemNavData/eval_3leg_habitat.py" \
  "${ROOT}/NavDP/baselines/memnav/memnav_server.py" \
  "${ROOT}/NavDP/baselines/memnav/policy_agent.py" \
  "${ROOT}/NavDP/baselines/navdp/navdp_server.py" \
  "${ROOT}/NavDP/baselines/navdp/policy_agent.py" \
  > "${OUT_ROOT}/source_inputs.sha256"

"${HAB_PY}" -m unittest -v \
  MemNavData.test_summarize_unknown_goal_natural_stream_shadow \
  > "${OUT_ROOT}/logs/tests.log" 2>&1

mkdir -p "${runtime_root}/memnav" "${runtime_root}/navdp"
(
  cd "${runtime_root}/memnav"
  exec env \
    PYTHONUNBUFFERED=1 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    PYTHONPATH="${INTERNNAV_ROOT}/src/diffusion-policy:${PYTHONPATH:-}" \
    LINGBOT_REPO="${LINGBOT_REPO}" \
    LINGBOT_WEIGHTS="${LINGBOT_WEIGHTS}" \
    MEMNAV_WINDOW=32 \
    MEMNAV_NUM_SCALE=8 \
    MEMNAV_MAX_FRAME_NUM=2048 \
    MEMNAV_GROUND_SCALE_MAX=6.0 \
    MEMNAV_GATE_FUSION=complementary \
    MEMNAV_AUX_POSE_CALIBRATION=empirical \
    MEMNAV_COLLISION_SELECT=1 \
    MEMNAV_REPORT_TO=none \
    "${MEMNAV_PY}" -u "${ROOT}/NavDP/baselines/memnav/memnav_server.py" \
      --port "${MEMNAV_PORT}" \
      --checkpoint "${MEMNAV_CKPT}" \
      --internnav_root "${INTERNNAV_ROOT}" \
      --num_samples 16 \
      --exclude_recent 32 \
      --retrieval raw \
      --retrieval_candidate_top_k 32 \
      --retrieval_candidate_min_gap 16 \
      --graph_subgoal_spacing_m 0.0 \
      --graph_subgoal_arrival_m 0.60 \
      --flow_gate auto \
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
      echo "ABORT: ${label} server exited" >&2; tail -n 120 "${log}" >&2; exit 1
    fi
    if ss -ltn | awk '{print $4}' | grep -Eq "(^|:)${port}$"; then
      ready=1; break
    fi
    sleep 2
  done
  [[ "${ready}" -eq 1 ]] || {
    echo "ABORT: ${label} server did not bind" >&2; tail -n 120 "${log}" >&2; exit 1; }
done

"${HAB_PY}" -u "${ROOT}/MemNavData/eval_3leg_habitat.py" \
  --episode_root "${EPISODE_ROOT}" \
  --scene "${SCENE_FILE}" \
  --port "${MEMNAV_PORT}" \
  --novel_port "${NAVDP_PORT}" \
  --out "${OUT_ROOT}/plans" \
  --server_backend hybrid_pose \
  --hybrid_route memory_geometry \
  --revisit_adapter legacy_metric \
  --leg1_mode policy \
  --navdp_goal_switch_reset carry \
  --router_visual_floor -1.0 \
  --router_min_matches 1000000000 \
  --router_min_inliers 1000000000 \
  --router_min_inlier_ratio 1.0 \
  --router_confirm_plans 100000 \
  --router_verify_top_k 8 \
  --success_dist 1.0 \
  --max_steps "${MAX_STEPS}" \
  --exec_horizon 8 \
  --trajectory_selector server \
  --trajectory_selector_scope all \
  --seed 20260803 \
  --deterministic_plan_seeds \
  --episode_ids "${EPISODE_IDS}" \
  > "${OUT_ROOT}/logs/evaluator.log" 2>&1

"${HAB_PY}" -u -m MemNavData.summarize_unknown_goal_natural_stream_shadow \
  --run-root "${OUT_ROOT}/plans" \
  --buffer-root "${OUT_ROOT}/buffer" \
  --out "${OUT_ROOT}/report.json" \
  > "${OUT_ROOT}/logs/summarizer.log" 2>&1
sha256sum "${OUT_ROOT}/report.json" > "${OUT_ROOT}/report.json.sha256"
echo "COMPLETE ${OUT_ROOT}"
