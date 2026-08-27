#!/usr/bin/env bash
set -euo pipefail

# Controlled start->A closed-loop comparison.  The generated scene-disjoint
# episodes and controller are shared; only the policy server is changed.
# GATE_OVERRIDE=0 isolates MemNav's Novel path.  With no override this evaluates
# the checkpoint's deployed soft gate; GATE_SKIP_BELOW only skips goal-pose
# computation and does NOT hard-clamp that gate.  MODE=smoke runs episode_0000
# in the first scene, while MODE=full runs both episodes in all five scenes.

MODE=${1:-full}
case "${MODE}" in
  smoke|full) ;;
  *) echo "usage: $0 [smoke|full]" >&2; exit 2 ;;
esac

ROOT=/home/asus/Research/Nav-axis-uturn
SOURCE_RUN=${ROOT}/.diagnostics/unseen_scene_eval_20260803
RUN_ROOT=${RUN_ROOT:-${ROOT}/.diagnostics/novel_navdp_ab_20260803}
RESULT_SET=${RESULT_SET:-${MODE}}
PORT=${PORT:-18896}
LABELS=${LABELS:-"navdp flowgate2600 gatecurr600 residualgate1000"}
CUSTOM_LABEL=${CUSTOM_LABEL:-}
CUSTOM_CKPT=${CUSTOM_CKPT:-}
MAX_STEPS=${MAX_STEPS:-500}
SEED=${SEED:-20260803}
GATE_OVERRIDE=${GATE_OVERRIDE:-}
GATE_SKIP_BELOW=${GATE_SKIP_BELOW:-0.0}
COLLISION_SELECT=${COLLISION_SELECT:-1}
COLLISION_HORIZON_WAYPOINTS=${COLLISION_HORIZON_WAYPOINTS:-0}
TRAJECTORY_SELECTOR=${TRAJECTORY_SELECTOR:-server}
LEG1_GOAL_SOURCE=${LEG1_GOAL_SOURCE:-own}

HAB_PY=/home/asus/miniconda3/envs/habitat/bin/python
MEMNAV_PY=/home/asus/miniconda3/envs/memnav/bin/python
EVALUATOR=${ROOT}/MemNavData/eval_2leg_habitat.py
MEMNAV_SERVER=${ROOT}/NavDP/baselines/memnav/memnav_server.py
NAVDP_SERVER=${ROOT}/NavDP/baselines/navdp/navdp_server.py
INTERNNAV_ROOT=${ROOT}/InternNav
LINGBOT_REPO=/home/asus/Research/Nav/NavDP/baselines/memnav/lingbot-map
LINGBOT_WEIGHTS=${LINGBOT_REPO}/weights/lingbot-map-long.pt
NAVDP_CKPT=/home/asus/Research/Nav/NavDP/baselines/navdp/checkpoints/navdp_checkpoint.ckpt

declare -A CKPT
CKPT[flowgate2600]=${SOURCE_RUN}/checkpoints/flowgate2600.memnav.ckpt
CKPT[gatecurr600]=${SOURCE_RUN}/checkpoints/gatecurr600.memnav.ckpt
CKPT[residualgate1000]=${RUN_ROOT}/checkpoints/residualgate1000.memnav.ckpt
if [[ -n "${CUSTOM_LABEL}" || -n "${CUSTOM_CKPT}" ]]; then
  if [[ -z "${CUSTOM_LABEL}" || -z "${CUSTOM_CKPT}" ]]; then
    echo "ABORT: CUSTOM_LABEL and CUSTOM_CKPT must be set together" >&2
    exit 2
  fi
  CKPT[${CUSTOM_LABEL}]=${CUSTOM_CKPT}
fi

mkdir -p "${RUN_ROOT}"/{logs,results,buffer}
for required in "${EVALUATOR}" "${MEMNAV_SERVER}" "${NAVDP_SERVER}" \
                "${LINGBOT_WEIGHTS}" "${NAVDP_CKPT}"; do
  test -f "${required}"
done
for label in ${LABELS}; do
  if [[ "${label}" != navdp ]]; then
    test -f "${CKPT[${label}]}"
  fi
done
"${HAB_PY}" -m py_compile "${EVALUATOR}"
"${MEMNAV_PY}" -m py_compile "${NAVDP_SERVER}" "${MEMNAV_SERVER}"
"${HAB_PY}" -c 'import habitat_sim,numpy,pandas,pyarrow,PIL,requests,scipy,quaternion; print("Habitat dependencies OK", habitat_sim.__version__)'
"${MEMNAV_PY}" -c 'import torch,torchvision,transformers,diffusers,cv2,flask,imageio; assert torch.cuda.is_available(); print("Policy dependencies OK", torch.__version__)'

mapfile -t SCENES < <("${HAB_PY}" -c \
  'import json,sys; print(*json.load(open(sys.argv[1]))["selection"]["selected_scenes"], sep="\n")' \
  "${SOURCE_RUN}/manifest.json")
if [[ "${MODE}" == smoke ]]; then
  SCENES=("${SCENES[0]}")
  EPISODE_ARGS=(--episode_ids episode_0000)
else
  EPISODE_ARGS=(--episodes 2)
fi

SERVER_PID=
cleanup_server() {
  if [[ -n "${SERVER_PID}" ]] && kill -0 "${SERVER_PID}" 2>/dev/null; then
    kill "${SERVER_PID}" 2>/dev/null || true
    wait "${SERVER_PID}" 2>/dev/null || true
  fi
  SERVER_PID=
  for _ in $(seq 1 30); do
    if ! ss -ltn | awk '{print $4}' | grep -Eq "(^|:)${PORT}$"; then
      break
    fi
    sleep 1
  done
}
trap cleanup_server EXIT INT TERM

if ss -ltn | awk '{print $4}' | grep -Eq "(^|:)${PORT}$"; then
  echo "ABORT: port ${PORT} is already in use" >&2
  exit 1
fi

for label in ${LABELS}; do
  result_root=${RUN_ROOT}/results/${RESULT_SET}/${label}
  server_log=${RUN_ROOT}/logs/${RESULT_SET}_server_${label}.log
  if find "${result_root}" -type f -name summary.json -print -quit 2>/dev/null | grep -q .; then
    echo "ABORT: completed output already exists under ${result_root}" >&2
    exit 1
  fi
  rm -rf "${RUN_ROOT}/buffer/${RESULT_SET}_${label}"
  mkdir -p "${result_root}" "$(dirname "${server_log}")"

  if [[ "${label}" == navdp ]]; then
    (
      cd "$(dirname "${NAVDP_SERVER}")"
      exec env NAVDP_DISABLE_VIDEO=1 PYTHONUNBUFFERED=1 \
        "${MEMNAV_PY}" -u "${NAVDP_SERVER}" \
          --port "${PORT}" --checkpoint "${NAVDP_CKPT}"
    ) >"${server_log}" 2>&1 &
    backend=navdp
  else
    fusion=complementary
    if [[ "${label}" == residualgate1000 ]]; then
      fusion=residual
    fi
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
        MEMNAV_GATE_FUSION="${fusion}" \
        MEMNAV_AUX_POSE_CALIBRATION=empirical \
        MEMNAV_COLLISION_SELECT="${COLLISION_SELECT}" \
        MEMNAV_COLLISION_HORIZON_WAYPOINTS="${COLLISION_HORIZON_WAYPOINTS}" \
        MEMNAV_REPORT_TO=none \
        "${MEMNAV_PY}" -u "${MEMNAV_SERVER}" \
          --port "${PORT}" \
          --checkpoint "${CKPT[${label}]}" \
          --internnav_root "${INTERNNAV_ROOT}" \
          --num_samples 16 \
          --exclude_recent 83 \
          --retrieval head \
          --gate_skip_below "${GATE_SKIP_BELOW}" \
          --flow_gate auto \
          --buffer_root "${RUN_ROOT}/buffer/${RESULT_SET}_${label}"
    ) >"${server_log}" 2>&1 &
    backend=memnav
  fi
  SERVER_PID=$!

  ready=0
  for _ in $(seq 1 180); do
    if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
      echo "ABORT: ${label} server exited during startup" >&2
      tail -n 100 "${server_log}" >&2
      exit 1
    fi
    if ss -ltn | awk '{print $4}' | grep -Eq "(^|:)${PORT}$"; then
      ready=1
      break
    fi
    sleep 2
  done
  if [[ "${ready}" -ne 1 ]]; then
    echo "ABORT: ${label} server did not bind port ${PORT}" >&2
    exit 1
  fi

  for scene in "${SCENES[@]}"; do
    out=${result_root}/${scene}
    mkdir -p "${out}"
    echo "[novel-eval] ${label} ${scene} mode=${MODE} max_steps=${MAX_STEPS}"
    gate_args=()
    if [[ -n "${GATE_OVERRIDE}" && "${backend}" == memnav ]]; then
      gate_args=(--gate_override "${GATE_OVERRIDE}")
    fi
    (
      cd "${ROOT}"
      "${HAB_PY}" -u "${EVALUATOR}" \
        --episode_root "${SOURCE_RUN}/episodes/${scene}" \
        --scene "${SOURCE_RUN}/assets/${scene}.glb" \
        --host 127.0.0.1 \
        --port "${PORT}" \
        --out "${out}" \
        --server_backend "${backend}" \
        --leg1_mode policy \
        --stop_after_leg1 \
        --success_dist 1.0 \
        --max_steps "${MAX_STEPS}" \
        --exec_horizon 8 \
        --trajectory_selector "${TRAJECTORY_SELECTOR}" \
        --leg1_goal_source "${LEG1_GOAL_SOURCE}" \
        --seed "${SEED}" \
        --terminal_uturn off \
        --terminal_visual_refine off \
        "${gate_args[@]}" \
        "${EPISODE_ARGS[@]}"
    ) >"${RUN_ROOT}/logs/${RESULT_SET}_eval_${label}_${scene}.log" 2>&1
  done
  cleanup_server
done

echo "[done] Goal-A ${RESULT_SET}: labels=${LABELS} gate_override=${GATE_OVERRIDE:-none}"
