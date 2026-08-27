#!/usr/bin/env bash
# Consumed-scene integration gate for autonomous GOAT ImageGoal lifecycle.

set -euo pipefail
umask 0022

ROOT=${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
OUT_ROOT=${OUT_ROOT:-${ROOT}/.diagnostics/goat_autonomous_multigoal_local_20260818}
GOAT_CODE=${GOAT_CODE:-${ROOT}/.diagnostics/dependencies/goat-bench}
DATA_ROOT=${DATA_ROOT:-${ROOT}/.diagnostics/datasets/goat-smoke-hm3d-20260814}
GOAT_POLICY_CKPT=${GOAT_POLICY_CKPT:-${ROOT}/.diagnostics/datasets/goat-assets/checkpoints/sense_act_nn_monolithic/ckpt_best.pth}
GOAT_PY=${GOAT_PY:-${ROOT}/.diagnostics/envs/goat-bench-policy-local-20260815/bin/python}
MEMNAV_PY=${MEMNAV_PY:-/home/asus/miniconda3/envs/memnav/bin/python}
MEMNAV_CKPT=${MEMNAV_CKPT:-/home/asus/Research/Nav-axis-uturn/.diagnostics/unseen_scene_eval_20260803/checkpoints/gatecurr600.memnav.ckpt}
NAVDP_CKPT=${NAVDP_CKPT:-/home/asus/Research/Nav/NavDP/baselines/navdp/checkpoints/navdp_checkpoint.ckpt}
LINGBOT_REPO=${LINGBOT_REPO:-/home/asus/Research/Nav/NavDP/baselines/memnav/lingbot-map}
LINGBOT_WEIGHTS=${LINGBOT_WEIGHTS:-${LINGBOT_REPO}/weights/lingbot-map-long.pt}
LIGHTGLUE_REPO=${LIGHTGLUE_REPO:-${ROOT}/.diagnostics/dependencies/LightGlue}
DEPENDENCY_ROOT=${DEPENDENCY_ROOT:-${ROOT}/.diagnostics/dependencies/python}
INTERNNAV_ROOT=${INTERNNAV_ROOT:-${ROOT}/InternNav}
OPENAI_CLIP_ROOT=${OPENAI_CLIP_ROOT:-${ROOT}/.diagnostics/dependencies/openai-clip}
MEMNAV_PORT=${MEMNAV_PORT:-21640}
NAVDP_PORT=${NAVDP_PORT:-21641}
EPISODE=${EPISODE:-5cdEh9F2hJL:1}
MAX_STEPS=${MAX_STEPS:-220}
BASE_SEED=${BASE_SEED:-100}
UNSUPPORTED_IMAGE_CONTROLLER=${UNSUPPORTED_IMAGE_CONTROLLER:-official}
NAVDP_STOP_THRESHOLD=${NAVDP_STOP_THRESHOLD:--0.5}

fail() { echo "ABORT: $*" >&2; exit 2; }
[[ "${MAX_STEPS}" =~ ^[1-9][0-9]*$ ]] || fail "MAX_STEPS must be positive"
[[ "${UNSUPPORTED_IMAGE_CONTROLLER}" == official || \
   "${UNSUPPORTED_IMAGE_CONTROLLER}" == navdp ]] || \
  fail "UNSUPPORTED_IMAGE_CONTROLLER must be official or navdp"
[[ ! -e "${OUT_ROOT}" ]] || fail "output already exists: ${OUT_ROOT}"
for required in \
  "${GOAT_PY}" "${MEMNAV_PY}" "${GOAT_POLICY_CKPT}" \
  "${MEMNAV_CKPT}" "${NAVDP_CKPT}" "${LINGBOT_WEIGHTS}" \
  "${LIGHTGLUE_REPO}/lightglue" "${DEPENDENCY_ROOT}/kornia" \
  "${OPENAI_CLIP_ROOT}/clip/__init__.py" \
  "${DATA_ROOT}/data/scene_datasets/hm3d/val/00853-5cdEh9F2hJL/5cdEh9F2hJL.basis.glb" \
  "${ROOT}/MemNavData/goat_autonomous_multigoal_pilot.py" \
  "${ROOT}/MemNavData/goat_autonomous_stop.py" \
  "${ROOT}/MemNavData/goat_navdp_camera_adapter.py"; do
  [[ -r "${required}" ]] || fail "missing input: ${required}"
done
for port in "${MEMNAV_PORT}" "${NAVDP_PORT}"; do
  ! ss -ltn | awk '{print $4}' | grep -Eq "(^|:)${port}$" || \
    fail "port ${port} is already in use"
done

mkdir -p "${OUT_ROOT}/logs" "${OUT_ROOT}/buffer"
runtime_root=$(mktemp -d /tmp/goat_autonomous_multigoal.XXXXXX)
MEMNAV_PID=
NAVDP_PID=
MONITOR_PID=
cleanup() {
  for process_id in "${MONITOR_PID}" "${NAVDP_PID}" "${MEMNAV_PID}"; do
    if [[ -n "${process_id}" ]] && kill -0 "${process_id}" 2>/dev/null; then
      kill "${process_id}" 2>/dev/null || true
      wait "${process_id}" 2>/dev/null || true
    fi
  done
  find "${runtime_root}" -depth -delete 2>/dev/null || true
}
trap cleanup EXIT INT TERM

server_pythonpath=${ROOT}:${DEPENDENCY_ROOT}:${LIGHTGLUE_REPO}:${INTERNNAV_ROOT}/src/diffusion-policy${PYTHONPATH:+:${PYTHONPATH}}
goat_pythonpath=${ROOT}:${OPENAI_CLIP_ROOT}${PYTHONPATH:+:${PYTHONPATH}}

"${MEMNAV_PY}" -m pytest -q -p no:cacheprovider \
  "${ROOT}/MemNavData/test_goat_autonomous_stop.py" \
  "${ROOT}/MemNavData/test_goat_autonomous_multigoal_pilot.py" \
  "${ROOT}/MemNavData/test_goat_navdp_camera_adapter.py" \
  "${ROOT}/MemNavData/test_goat_navdp_runtime_pilot.py" \
  "${ROOT}/MemNavData/test_goat_terminal_alignment.py" \
  >"${OUT_ROOT}/logs/preflight_tests.log" 2>&1
"${GOAT_PY}" -m py_compile \
  "${ROOT}/MemNavData/goat_autonomous_stop.py" \
  "${ROOT}/MemNavData/goat_navdp_camera_adapter.py" \
  "${ROOT}/MemNavData/goat_autonomous_multigoal_pilot.py"

mkdir -p "${runtime_root}/memnav" "${runtime_root}/navdp"
(
  cd "${runtime_root}/memnav"
  exec env PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    PYTHONPATH="${server_pythonpath}" LINGBOT_REPO="${LINGBOT_REPO}" \
    LINGBOT_WEIGHTS="${LINGBOT_WEIGHTS}" MEMNAV_WINDOW=32 \
    MEMNAV_NUM_SCALE=8 MEMNAV_MAX_FRAME_NUM=6000 \
    MEMNAV_GROUND_SCALE_MAX=6.0 MEMNAV_GATE_FUSION=complementary \
    MEMNAV_AUX_POSE_CALIBRATION=empirical MEMNAV_COLLISION_SELECT=1 \
    MEMNAV_REPORT_TO=none "${MEMNAV_PY}" -u \
    "${ROOT}/NavDP/baselines/memnav/memnav_server.py" \
      --port "${MEMNAV_PORT}" --checkpoint "${MEMNAV_CKPT}" \
      --internnav_root "${INTERNNAV_ROOT}" --num_samples 16 \
      --exclude_recent 32 --retrieval raw --retrieval_candidate_top_k 32 \
      --retrieval_candidate_min_gap 16 --graph_subgoal_spacing_m 0.0 \
      --graph_subgoal_arrival_m 0.60 --flow_gate auto \
      --buffer_root "${OUT_ROOT}/buffer" --certified_relocalization \
      --lightglue_repo "${LIGHTGLUE_REPO}" \
      --lightglue_dependency_root "${DEPENDENCY_ROOT}" \
      --lightglue_max_keypoints 2048
) >"${OUT_ROOT}/logs/server_memnav.log" 2>&1 &
MEMNAV_PID=$!

(
  cd "${runtime_root}/navdp"
  exec env NAVDP_DISABLE_VIDEO=1 PYTHONUNBUFFERED=1 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    PYTHONPATH="${server_pythonpath}" "${MEMNAV_PY}" -u \
    "${ROOT}/NavDP/baselines/navdp/navdp_server.py" \
      --port "${NAVDP_PORT}" --checkpoint "${NAVDP_CKPT}"
) >"${OUT_ROOT}/logs/server_navdp.log" 2>&1 &
NAVDP_PID=$!

for spec in \
  "memnav:${MEMNAV_PID}:${MEMNAV_PORT}:${OUT_ROOT}/logs/server_memnav.log" \
  "navdp:${NAVDP_PID}:${NAVDP_PORT}:${OUT_ROOT}/logs/server_navdp.log"; do
  IFS=: read -r label process_id port log_path <<<"${spec}"
  ready=0
  for _ in $(seq 1 300); do
    kill -0 "${process_id}" 2>/dev/null || {
      tail -n 200 "${log_path}" >&2
      fail "${label} server exited during startup"
    }
    if ss -ltn | awk '{print $4}' | grep -Eq "(^|:)${port}$"; then
      ready=1
      break
    fi
    sleep 2
  done
  [[ "${ready}" -eq 1 ]] || fail "${label} server startup timed out"
done

(
  echo "timestamp_utc,memory_used_mib,utilization_gpu_percent"
  while true; do
    timestamp=$(date -u +%Y-%m-%dT%H:%M:%SZ)
    nvidia-smi --query-gpu=memory.used,utilization.gpu \
      --format=csv,noheader,nounits | head -1 | sed "s/^/${timestamp},/"
    sleep 10
  done
) >"${OUT_ROOT}/logs/gpu.csv" 2>/dev/null &
MONITOR_PID=$!

# The local GOAT environment uses an older Torch allocator.  The two model
# services received their allocator option explicitly above; do not leak it to
# the official GOAT process.
unset PYTORCH_CUDA_ALLOC_CONF
(
  cd "${GOAT_CODE}"
  exec env PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 \
    MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet PYTHONPATH="${goat_pythonpath}" \
    "${GOAT_PY}" -u "${ROOT}/MemNavData/goat_autonomous_multigoal_pilot.py" \
      --goat-code "${GOAT_CODE}" --data-root "${DATA_ROOT}" \
      --checkpoint "${GOAT_POLICY_CKPT}" --output-dir "${OUT_ROOT}/result" \
      --episode "${EPISODE}" --memnav-url "http://127.0.0.1:${MEMNAV_PORT}" \
      --navdp-url "http://127.0.0.1:${NAVDP_PORT}" \
      --policy-device cuda --base-seed "${BASE_SEED}" \
      --max-steps "${MAX_STEPS}" \
      --navdp-stop-threshold "${NAVDP_STOP_THRESHOLD}" \
      --unsupported-image-controller "${UNSUPPORTED_IMAGE_CONTROLLER}"
) >"${OUT_ROOT}/logs/runner.log" 2>&1

result=${OUT_ROOT}/result/goat_autonomous_multigoal_pilot.json
[[ -s "${result}" ]] || fail "pilot result is absent"
"${GOAT_PY}" - "${result}" "${UNSUPPORTED_IMAGE_CONTROLLER}" \
  "${NAVDP_STOP_THRESHOLD}" <<'PY'
import json
import math
import sys
payload = json.load(open(sys.argv[1]))
assert payload["complete"] is True
assert payload["ground_truth_used_by_decision"] is False
assert payload["unsupported_image_controller"] == sys.argv[2]
assert math.isclose(
    float(payload["navdp_upstream_critic_threshold"]), float(sys.argv[3]),
    rel_tol=0.0, abs_tol=1e-12)
assert payload["imagegoal_official_stop_head_used"] is (sys.argv[2] == "official")
assert payload["episodes"]
print(json.dumps({
    "result": sys.argv[1],
    "episodes": len(payload["episodes"]),
    "termination": payload["episodes"][0]["termination_reason"],
    "image_successes": payload["episodes"][0]["image_subtask_successes"],
    "image_subtasks": payload["episodes"][0]["image_subtask_count"],
    "terminal_searches": payload["episodes"][0]["terminal_search_count"],
}, sort_keys=True))
PY
