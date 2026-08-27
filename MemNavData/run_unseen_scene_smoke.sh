#!/usr/bin/env bash
set -euo pipefail

MODE=${1:-all}
case "${MODE}" in
  generate|eval|all) ;;
  *) echo "usage: $0 [generate|eval|all]" >&2; exit 2 ;;
esac

ROOT=/home/asus/Research/Nav-axis-uturn
RUN_ROOT=${RUN_ROOT:-${ROOT}/.diagnostics/unseen_scene_eval_20260803}
MANIFEST=${RUN_ROOT}/manifest.json
HAB_PY=/home/asus/miniconda3/envs/habitat/bin/python
MEMNAV_PY=/home/asus/miniconda3/envs/memnav/bin/python
GENERATOR=${ROOT}/MemNavData/generate_twoleg.py
EVALUATOR=${ROOT}/MemNavData/eval_2leg_habitat.py
VALIDATOR=${ROOT}/MemNavData/validate_unseen_eval.py
SERVER=${ROOT}/NavDP/baselines/memnav/memnav_server.py
INTERNNAV_ROOT=${ROOT}/InternNav
LINGBOT_REPO=/home/asus/Research/Nav/NavDP/baselines/memnav/lingbot-map
LINGBOT_WEIGHTS=${LINGBOT_REPO}/weights/lingbot-map-long.pt
PORT=${PORT:-18893}

mkdir -p "${RUN_ROOT}"/{episodes,results,logs,buffer}

mapfile -t SCENES < <("${HAB_PY}" -c \
  'import json,sys; print(*json.load(open(sys.argv[1]))["selection"]["selected_scenes"], sep="\n")' \
  "${MANIFEST}")
EPISODES_PER_SCENE=$("${HAB_PY}" -c \
  'import json,sys; print(json.load(open(sys.argv[1]))["episode_generation"]["episodes_per_scene"])' \
  "${MANIFEST}")
GEN_BASE_SEED=$("${HAB_PY}" -c \
  'import json,sys; print(json.load(open(sys.argv[1]))["episode_generation"]["base_seed"])' \
  "${MANIFEST}")
GEN_SEED_STRIDE=$("${HAB_PY}" -c \
  'import json,sys; print(json.load(open(sys.argv[1]))["episode_generation"]["seed_stride"])' \
  "${MANIFEST}")
EVAL_SEED=$("${HAB_PY}" -c \
  'import json,sys; print(json.load(open(sys.argv[1]))["evaluation"]["base_seed"])' \
  "${MANIFEST}")

echo "[preflight] root=${ROOT} run_root=${RUN_ROOT} mode=${MODE}"
test -f "${MANIFEST}"
test -f "${GENERATOR}"
test -f "${EVALUATOR}"
test -f "${VALIDATOR}"
test -f "${SERVER}"
test -f "${LINGBOT_WEIGHTS}"
"${HAB_PY}" -c 'import habitat_sim,numpy,pandas,pyarrow,PIL,requests,scipy,quaternion; print("Habitat dependencies OK", habitat_sim.__version__)'
"${MEMNAV_PY}" -c 'import torch,transformers,diffusers,flask,cv2,numpy; assert torch.cuda.is_available(); print("MemNav dependencies OK", torch.__version__)'
"${HAB_PY}" "${VALIDATOR}" --manifest "${MANIFEST}" --run-root "${RUN_ROOT}" --phase assets \
  > "${RUN_ROOT}/logs/validation_assets.json"

if [[ "${MODE}" == generate || "${MODE}" == all ]]; then
  for index in "${!SCENES[@]}"; do
    scene=${SCENES[${index}]}
    scene_root=${RUN_ROOT}/episodes/${scene}
    seed=$((GEN_BASE_SEED + index * GEN_SEED_STRIDE))
    if [[ -d "${scene_root}" ]]; then
      existing=$(find "${scene_root}" -mindepth 1 -maxdepth 1 -type d -name 'episode_*' | wc -l)
    else
      existing=0
    fi
    if [[ "${existing}" -eq "${EPISODES_PER_SCENE}" ]]; then
      echo "[generate] ${scene}: already has ${existing} episodes"
      continue
    fi
    if [[ "${existing}" -ne 0 ]]; then
      echo "ABORT: ${scene_root} has partial output (${existing} episodes)" >&2
      exit 1
    fi
    mkdir -p "${scene_root}"
    echo "[generate] ${scene}: n=${EPISODES_PER_SCENE} seed=${seed}"
    "${HAB_PY}" "${GENERATOR}" \
      --scene "${RUN_ROOT}/assets/${scene}.glb" \
      --out "${scene_root}" \
      --n "${EPISODES_PER_SCENE}" \
      --n_legs 2 \
      --seed "${seed}" \
      --window 32 \
      --num_scale 8 \
      > "${RUN_ROOT}/logs/generate_${scene}.log" 2>&1
  done
  "${HAB_PY}" "${VALIDATOR}" --manifest "${MANIFEST}" --run-root "${RUN_ROOT}" --phase episodes \
    > "${RUN_ROOT}/logs/validation_episodes.json"
fi

if [[ "${MODE}" == eval || "${MODE}" == all ]]; then
  "${HAB_PY}" "${VALIDATOR}" --manifest "${MANIFEST}" --run-root "${RUN_ROOT}" --phase ready \
    > "${RUN_ROOT}/logs/validation_ready.json"
  if ss -ltn | awk '{print $4}' | grep -Eq "(^|:)${PORT}$"; then
    echo "ABORT: port ${PORT} is already in use" >&2
    exit 1
  fi

  SERVER_PID=
  cleanup_server() {
    if [[ -n "${SERVER_PID}" ]] && kill -0 "${SERVER_PID}" 2>/dev/null; then
      kill "${SERVER_PID}" 2>/dev/null || true
      wait "${SERVER_PID}" 2>/dev/null || true
    fi
    SERVER_PID=
  }
  trap cleanup_server EXIT INT TERM

  for label in flowgate2600 gatecurr600; do
    checkpoint=${RUN_ROOT}/checkpoints/${label}.memnav.ckpt
    server_log=${RUN_ROOT}/logs/server_${label}.log
    rm -rf "${RUN_ROOT}/buffer/${label}"
    echo "[server] starting ${label} on :${PORT}"
    env \
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
      "${MEMNAV_PY}" "${SERVER}" \
        --port "${PORT}" \
        --checkpoint "${checkpoint}" \
        --internnav_root "${INTERNNAV_ROOT}" \
        --num_samples 16 \
        --exclude_recent 83 \
        --retrieval head \
        --flow_gate auto \
        --buffer_root "${RUN_ROOT}/buffer/${label}" \
        > "${server_log}" 2>&1 &
    SERVER_PID=$!

    ready=0
    for _ in $(seq 1 120); do
      if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
        echo "ABORT: ${label} server exited during startup" >&2
        tail -n 100 "${server_log}" >&2
        exit 1
      fi
      if curl -fsS --max-time 5 -X POST "http://127.0.0.1:${PORT}/navigator_reset" \
          -H 'Content-Type: application/json' -d '{}' | grep -q 'memnav'; then
        ready=1
        break
      fi
      sleep 5
    done
    if [[ "${ready}" -ne 1 ]]; then
      echo "ABORT: ${label} server did not become ready" >&2
      tail -n 100 "${server_log}" >&2
      exit 1
    fi

    for scene in "${SCENES[@]}"; do
      out=${RUN_ROOT}/results/${label}/${scene}
      mkdir -p "${out}"
      echo "[eval] ${label} ${scene}"
      (
        cd "${ROOT}/MemNavData"
        "${HAB_PY}" "${EVALUATOR}" \
          --episode_root "${RUN_ROOT}/episodes/${scene}" \
          --scene "${RUN_ROOT}/assets/${scene}.glb" \
          --host 127.0.0.1 \
          --port "${PORT}" \
          --out "${out}" \
          --leg1_mode replay \
          --success_dist 1.0 \
          --max_steps 1200 \
          --exec_horizon 8 \
          --seed "${EVAL_SEED}" \
          --episodes "${EPISODES_PER_SCENE}" \
          --terminal_uturn off \
          --terminal_visual_refine off
      ) > "${RUN_ROOT}/logs/eval_${label}_${scene}.log" 2>&1
    done
    cleanup_server
  done
fi

echo "[done] ${MODE} completed"
