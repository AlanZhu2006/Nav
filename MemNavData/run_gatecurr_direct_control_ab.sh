#!/usr/bin/env bash
set -euo pipefail

# Paired controller ablation on the fixed 5-scene/10-episode unseen split.
# Goal A is controlled by frozen official NavDP while every observation is
# streamed into MemNav.  Goal B is controlled directly by gatecurr600's
# diffusion decoder.  This keeps Novel protection and the memory stream fixed
# while removing the LingBot-metric-pose -> NavDP point-goal controller.

MODE=${1:-full}
case "${MODE}" in
  smoke|full) ;;
  *) echo "usage: $0 [smoke|full]" >&2; exit 2 ;;
esac

ROOT=${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
SOURCE_RUN=${SOURCE_RUN:-${ROOT}/.diagnostics/unseen_scene_eval_20260803}
RUN_ROOT=${RUN_ROOT:-${ROOT}/.diagnostics/gatecurr_direct_control_20260804}
RESULT_SET=${RESULT_SET:-${MODE}_raw_e32}
MEMNAV_PORT=${MEMNAV_PORT:-18903}
NAVDP_PORT=${NAVDP_PORT:-18904}
MAX_STEPS=${MAX_STEPS:-500}
SEED=${SEED:-20260803}
RETRIEVAL=${RETRIEVAL:-raw}
EXCLUDE_RECENT=${EXCLUDE_RECENT:-32}

HAB_PY=${HAB_PY:-/home/asus/miniconda3/envs/habitat/bin/python}
MEMNAV_PY=${MEMNAV_PY:-/home/asus/miniconda3/envs/memnav/bin/python}
EVALUATOR=${ROOT}/MemNavData/eval_2leg_habitat.py
VALIDATOR=${ROOT}/MemNavData/validate_unseen_eval.py
MEMNAV_SERVER=${ROOT}/NavDP/baselines/memnav/memnav_server.py
NAVDP_SERVER=${ROOT}/NavDP/baselines/navdp/navdp_server.py
INTERNNAV_ROOT=${INTERNNAV_ROOT:-${ROOT}/InternNav}
LINGBOT_REPO=${LINGBOT_REPO:-/home/asus/Research/Nav/NavDP/baselines/memnav/lingbot-map}
LINGBOT_WEIGHTS=${LINGBOT_REPO}/weights/lingbot-map-long.pt
NAVDP_CKPT=${NAVDP_CKPT:-/home/asus/Research/Nav/NavDP/baselines/navdp/checkpoints/navdp_checkpoint.ckpt}
MEMNAV_CKPT=${MEMNAV_CKPT:-${SOURCE_RUN}/checkpoints/gatecurr600.memnav.ckpt}
MANIFEST=${SOURCE_RUN}/manifest.json
EXPECTED_MEMNAV_SHA=9b7a5811ff0aea212503f58b45258ba4f66b06420f87c350946aead39db6fdb7
EXPECTED_NAVDP_SHA=3bb3ad4ab241e857bb57a4021cc6aab76d5263e81fbf80298d579053ef011947
EXPECTED_LINGBOT_SHA=832bc82cbae0bc9bbe946ef5ee1f7226abd8c0e183ccf8beddbb3d133576f409

RESULT_ROOT=${RUN_ROOT}/results/${RESULT_SET}/gatecurr600_direct
LOG_ROOT=${RUN_ROOT}/logs/${RESULT_SET}
BUFFER_ROOT=${RUN_ROOT}/buffer/${RESULT_SET}
mkdir -p "${RESULT_ROOT}" "${LOG_ROOT}" "${BUFFER_ROOT}"

for required in "${HAB_PY}" "${MEMNAV_PY}" "${EVALUATOR}" "${VALIDATOR}" \
                "${MEMNAV_SERVER}" "${NAVDP_SERVER}" "${MANIFEST}" \
                "${LINGBOT_WEIGHTS}" "${NAVDP_CKPT}" "${MEMNAV_CKPT}"; do
  test -e "${required}" || { echo "ABORT: missing dependency ${required}" >&2; exit 1; }
done

actual_memnav_sha=$(sha256sum "${MEMNAV_CKPT}" | awk '{print $1}')
actual_navdp_sha=$(sha256sum "${NAVDP_CKPT}" | awk '{print $1}')
actual_lingbot_sha=$(sha256sum "${LINGBOT_WEIGHTS}" | awk '{print $1}')
[[ "${actual_memnav_sha}" == "${EXPECTED_MEMNAV_SHA}" ]] || {
  echo "ABORT: gatecurr600 SHA256 mismatch: ${actual_memnav_sha}" >&2; exit 1;
}
[[ "${actual_navdp_sha}" == "${EXPECTED_NAVDP_SHA}" ]] || {
  echo "ABORT: NavDP SHA256 mismatch: ${actual_navdp_sha}" >&2; exit 1;
}
[[ "${actual_lingbot_sha}" == "${EXPECTED_LINGBOT_SHA}" ]] || {
  echo "ABORT: LingBot SHA256 mismatch: ${actual_lingbot_sha}" >&2; exit 1;
}

"${HAB_PY}" -m py_compile "${EVALUATOR}"
"${MEMNAV_PY}" -m py_compile "${MEMNAV_SERVER}" "${NAVDP_SERVER}"
"${HAB_PY}" -c \
  'import habitat_sim,numpy,pandas,pyarrow,PIL,requests,scipy,quaternion; print("Habitat dependencies OK", habitat_sim.__version__)'
"${MEMNAV_PY}" -c \
  'import torch,torchvision,transformers,diffusers,cv2,flask,imageio; assert torch.cuda.is_available(); print("Policy dependencies OK", torch.__version__)'
"${HAB_PY}" "${VALIDATOR}" \
  --manifest "${MANIFEST}" --run-root "${SOURCE_RUN}" --phase episodes \
  > "${LOG_ROOT}/validation_inputs.json"

mapfile -t SCENES < <("${HAB_PY}" -c \
  'import json,sys; print(*json.load(open(sys.argv[1]))["selection"]["selected_scenes"], sep="\n")' \
  "${MANIFEST}")
if [[ "${MODE}" == smoke ]]; then
  SCENES=("${SCENES[0]}")
  EPISODE_ARGS=(--episode_ids episode_0000)
else
  EPISODE_ARGS=(--episodes 2)
fi

for scene in "${SCENES[@]}"; do
  test -f "${SOURCE_RUN}/assets/${scene}.glb" || {
    echo "ABORT: missing scene ${scene}" >&2; exit 1;
  }
  test -d "${SOURCE_RUN}/episodes/${scene}" || {
    echo "ABORT: missing episodes for ${scene}" >&2; exit 1;
  }
done

if find "${RESULT_ROOT}" -type f -name summary.json -print -quit | grep -q .; then
  echo "ABORT: completed output already exists under ${RESULT_ROOT}" >&2
  exit 1
fi
for port in "${MEMNAV_PORT}" "${NAVDP_PORT}"; do
  if ss -ltn | awk '{print $4}' | grep -Eq "(^|:)${port}$"; then
    echo "ABORT: port ${port} is already in use" >&2
    exit 1
  fi
done

rm -rf "${BUFFER_ROOT}"
mkdir -p "${BUFFER_ROOT}"
MEMNAV_PID=
NAVDP_PID=
cleanup() {
  for pid in "${NAVDP_PID}" "${MEMNAV_PID}"; do
    if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
      kill "${pid}" 2>/dev/null || true
      wait "${pid}" 2>/dev/null || true
    fi
  done
}
trap cleanup EXIT INT TERM

MEMNAV_LOG=${LOG_ROOT}/server_memnav.log
NAVDP_LOG=${LOG_ROOT}/server_navdp.log
(
  cd "$(dirname "${MEMNAV_SERVER}")"
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
    "${MEMNAV_PY}" -u "${MEMNAV_SERVER}" \
      --port "${MEMNAV_PORT}" \
      --checkpoint "${MEMNAV_CKPT}" \
      --internnav_root "${INTERNNAV_ROOT}" \
      --num_samples 16 \
      --exclude_recent "${EXCLUDE_RECENT}" \
      --retrieval "${RETRIEVAL}" \
      --flow_gate auto \
      --buffer_root "${BUFFER_ROOT}"
) >"${MEMNAV_LOG}" 2>&1 &
MEMNAV_PID=$!

(
  cd "$(dirname "${NAVDP_SERVER}")"
  exec env NAVDP_DISABLE_VIDEO=1 PYTHONUNBUFFERED=1 \
    "${MEMNAV_PY}" -u "${NAVDP_SERVER}" \
      --port "${NAVDP_PORT}" --checkpoint "${NAVDP_CKPT}"
) >"${NAVDP_LOG}" 2>&1 &
NAVDP_PID=$!

for spec in "memnav:${MEMNAV_PID}:${MEMNAV_PORT}:${MEMNAV_LOG}" \
            "navdp:${NAVDP_PID}:${NAVDP_PORT}:${NAVDP_LOG}"; do
  IFS=: read -r label pid port log <<<"${spec}"
  ready=0
  for _ in $(seq 1 180); do
    if ! kill -0 "${pid}" 2>/dev/null; then
      echo "ABORT: ${label} server exited during startup" >&2
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
    echo "ABORT: ${label} server did not bind port ${port}" >&2; exit 1;
  }
done

echo "[audit] gatecurr600=${actual_memnav_sha} navdp=${actual_navdp_sha} lingbot=${actual_lingbot_sha} retrieval=${RETRIEVAL} exclude_recent=${EXCLUDE_RECENT}"
for scene in "${SCENES[@]}"; do
  out=${RESULT_ROOT}/${scene}
  mkdir -p "${out}"
  echo "[eval] gatecurr600-direct ${scene} mode=${MODE}"
  (
    cd "${ROOT}"
    "${HAB_PY}" -u "${EVALUATOR}" \
      --episode_root "${SOURCE_RUN}/episodes/${scene}" \
      --scene "${SOURCE_RUN}/assets/${scene}.glb" \
      --host 127.0.0.1 \
      --port "${MEMNAV_PORT}" \
      --novel_port "${NAVDP_PORT}" \
      --out "${out}" \
      --server_backend hybrid_oracle \
      --leg1_mode policy \
      --success_dist 1.0 \
      --max_steps "${MAX_STEPS}" \
      --exec_horizon 8 \
      --trajectory_selector server \
      --leg1_goal_source own \
      --seed "${SEED}" \
      --terminal_uturn off \
      --terminal_visual_refine off \
      "${EPISODE_ARGS[@]}"
  ) >"${LOG_ROOT}/eval_${scene}.log" 2>&1
done

echo "[done] results=${RESULT_ROOT}"
