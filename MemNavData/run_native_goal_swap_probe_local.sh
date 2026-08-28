#!/usr/bin/env bash
# Local goal-swap probe for mono-native NavDP (post-hoc diagnostic).
#
# Starts the memnav + navdp server pair with the full-mono sidecar wiring
# (mirroring run_hm3d_fullmono_server_scene.sh minus certified/LightGlue,
# which the native probe never calls), then runs
# diag_native_revisit_goal_swap_20260828.py against the pulled fresh-run
# receipts. Plan-level only: no Habitat, no SR claim.

set -euo pipefail
umask 0022

ROOT=${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
PULLED_ROOT=${PULLED_ROOT:-${ROOT}/.diagnostics/hm3d_fresh_fullmono_mixed_role_20260820/pulled_20260828}
OUT_ROOT=${OUT_ROOT:-${ROOT}/.diagnostics/native_goal_swap_probe_20260828}
MEMNAV_PORT=${MEMNAV_PORT:-23140}
NAVDP_PORT=${NAVDP_PORT:-23141}
LIMIT=${LIMIT:-0}

MEMNAV_PY=${MEMNAV_PY:-/home/asus/miniconda3/envs/memnav/bin/python}
MEMNAV_CKPT=${MEMNAV_CKPT:-/home/asus/Research/Nav-axis-uturn/.diagnostics/unseen_scene_eval_20260803/checkpoints/gatecurr600.memnav.ckpt}
NAVDP_CKPT=${NAVDP_CKPT:-/home/asus/Research/Nav/NavDP/baselines/navdp/checkpoints/navdp_checkpoint.ckpt}
LINGBOT_REPO=${LINGBOT_REPO:-/home/asus/Research/lingbot-map}
LINGBOT_WEIGHTS=${LINGBOT_WEIGHTS:-${LINGBOT_REPO}/weights/lingbot-map-long.pt}
DEPENDENCY_ROOT=${DEPENDENCY_ROOT:-${ROOT}/.diagnostics/dependencies/python}
INTERNNAV_ROOT=${INTERNNAV_ROOT:-${ROOT}/InternNav}

fail() { echo "ABORT: $*" >&2; exit 2; }
[[ ! -e "${OUT_ROOT}/goal_swap_probe.json" ]] || \
  fail "output already exists: ${OUT_ROOT}/goal_swap_probe.json"
for path in "${PULLED_ROOT}/evaluation_natural_direction" \
  "${PULLED_ROOT}/online_a" "${MEMNAV_PY}" "${MEMNAV_CKPT}" \
  "${NAVDP_CKPT}" "${LINGBOT_WEIGHTS}" "${DEPENDENCY_ROOT}/kornia"; do
  test -r "${path}" || fail "missing input: ${path}"
done
for port in "${MEMNAV_PORT}" "${NAVDP_PORT}"; do
  ! ss -ltn | awk '{print $4}' | grep -Eq "(^|:)${port}$" || \
    fail "port ${port} is already in use"
done

mkdir -p "${OUT_ROOT}/logs" "${OUT_ROOT}/buffer"
runtime_root=$(mktemp -d /tmp/native_goal_swap_probe.XXXXXX)
MEMNAV_PID=
NAVDP_PID=
cleanup() {
  for process_id in "${NAVDP_PID}" "${MEMNAV_PID}"; do
    if [[ -n "${process_id}" ]] && kill -0 "${process_id}" 2>/dev/null; then
      kill "${process_id}" 2>/dev/null || true
      wait "${process_id}" 2>/dev/null || true
    fi
  done
  rm -rf -- "${runtime_root}"
}
trap cleanup EXIT INT TERM

server_pythonpath=${ROOT}:${DEPENDENCY_ROOT}:${INTERNNAV_ROOT}/src/diffusion-policy${PYTHONPATH:+:${PYTHONPATH}}

mkdir -p "${runtime_root}/memnav" "${runtime_root}/navdp"
(
  cd "${runtime_root}/memnav"
  exec env PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    PYTHONPATH="${server_pythonpath}" LINGBOT_REPO="${LINGBOT_REPO}" \
    LINGBOT_WEIGHTS="${LINGBOT_WEIGHTS}" MEMNAV_WINDOW=32 \
    MEMNAV_NUM_SCALE=8 MEMNAV_MAX_FRAME_NUM=2048 \
    MEMNAV_GROUND_SCALE_MAX=6.0 MEMNAV_GATE_FUSION=complementary \
    MEMNAV_AUX_POSE_CALIBRATION=empirical MEMNAV_COLLISION_SELECT=1 \
    MEMNAV_REPORT_TO=none "${MEMNAV_PY}" -u \
    "${ROOT}/NavDP/baselines/memnav/memnav_server.py" \
      --port "${MEMNAV_PORT}" --checkpoint "${MEMNAV_CKPT}" \
      --internnav_root "${INTERNNAV_ROOT}" --num_samples 16 \
      --exclude_recent 32 --retrieval raw --retrieval_candidate_top_k 32 \
      --retrieval_candidate_min_gap 16 --graph_subgoal_spacing_m 0.0 \
      --graph_subgoal_arrival_m 0.60 --flow_gate auto \
      --buffer_root "${OUT_ROOT}/buffer"
) >"${OUT_ROOT}/logs/server_memnav.log" 2>&1 &
MEMNAV_PID=$!
(
  cd "${runtime_root}/navdp"
  exec env NAVDP_DISABLE_VIDEO=1 PYTHONUNBUFFERED=1 \
    PYTHONPATH="${server_pythonpath}" "${MEMNAV_PY}" -u \
    "${ROOT}/NavDP/baselines/navdp/navdp_server.py" \
      --port "${NAVDP_PORT}" --checkpoint "${NAVDP_CKPT}" \
      --depth_source metric_request --allow_depth_source_override \
      --monocular_depth_url "http://127.0.0.1:${MEMNAV_PORT}/monocular_depth_query" \
      --require_monocular_depth_transaction
) >"${OUT_ROOT}/logs/server_navdp.log" 2>&1 &
NAVDP_PID=$!

for spec in \
  "memnav:${MEMNAV_PID}:${MEMNAV_PORT}:${OUT_ROOT}/logs/server_memnav.log" \
  "navdp:${NAVDP_PID}:${NAVDP_PORT}:${OUT_ROOT}/logs/server_navdp.log"; do
  IFS=: read -r label process_id port log_path <<<"${spec}"
  ready=0
  for _ in $(seq 1 240); do
    kill -0 "${process_id}" 2>/dev/null || {
      tail -n 160 "${log_path}" >&2
      fail "${label} server exited during startup"
    }
    if ss -ltn | awk '{print $4}' | grep -Eq "(^|:)${port}$"; then
      ready=1
      break
    fi
    sleep 2
  done
  [[ "${ready}" -eq 1 ]] || fail "${label} server did not bind port ${port}"
done

limit_args=()
if [[ "${LIMIT}" != "0" ]]; then
  limit_args=(--limit "${LIMIT}")
fi
env PYTHONPATH="${ROOT}" "${MEMNAV_PY}" -u \
  "${ROOT}/MemNavData/diag_native_revisit_goal_swap_20260828.py" \
  --pulled-root "${PULLED_ROOT}" --out-dir "${OUT_ROOT}" \
  --memnav_port "${MEMNAV_PORT}" --navdp_port "${NAVDP_PORT}" \
  "${limit_args[@]}" 2>&1 | tee "${OUT_ROOT}/logs/probe.log"
