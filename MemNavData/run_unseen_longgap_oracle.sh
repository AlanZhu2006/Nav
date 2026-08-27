#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/asus/Research/Nav-axis-uturn
RUN_ROOT=${RUN_ROOT:-${ROOT}/.diagnostics/unseen_scene_eval_20260803}
MANIFEST=${RUN_ROOT}/manifest.json
HAB_PY=/home/asus/miniconda3/envs/habitat/bin/python
MEMNAV_PY=/home/asus/miniconda3/envs/memnav/bin/python
EVALUATOR=${ROOT}/MemNavData/eval_2leg_habitat.py
VALIDATOR=${ROOT}/MemNavData/validate_unseen_eval.py
SERVER=${ROOT}/NavDP/baselines/memnav/memnav_server.py
INTERNNAV_ROOT=${ROOT}/InternNav
LINGBOT_REPO=/home/asus/Research/Nav/NavDP/baselines/memnav/lingbot-map
LINGBOT_WEIGHTS=${LINGBOT_REPO}/weights/lingbot-map-long.pt
CHECKPOINT=${RUN_ROOT}/checkpoints/gatecurr600.memnav.ckpt
PORT=${PORT:-18894}

CASES=(
  "e9zR4mvMWw7:episode_0000:20260803"
  "rqfALeAoiTq:episode_0000:20260803"
  "zsNo4HB9uLZ:episode_0001:20260804"
)

echo "[preflight] gatecurr600 long-gap oracle-covis"
test -f "${MANIFEST}"
test -f "${EVALUATOR}"
test -f "${VALIDATOR}"
test -f "${SERVER}"
test -f "${LINGBOT_WEIGHTS}"
test -f "${CHECKPOINT}"
"${HAB_PY}" "${VALIDATOR}" \
  --manifest "${MANIFEST}" \
  --run-root "${RUN_ROOT}" \
  --phase ready \
  > "${RUN_ROOT}/logs/validation_longgap_oracle_ready.json"
"${HAB_PY}" -c \
  'import habitat_sim,numpy,pandas,pyarrow,PIL,requests,scipy,quaternion; print("Habitat dependencies OK", habitat_sim.__version__)'
"${MEMNAV_PY}" -c \
  'import torch,transformers,diffusers,flask,cv2,numpy; assert torch.cuda.is_available(); print("MemNav dependencies OK", torch.__version__)'

if ss -ltn | awk '{print $4}' | grep -Eq "(^|:)${PORT}$"; then
  echo "ABORT: port ${PORT} is already in use" >&2
  exit 1
fi

for item in "${CASES[@]}"; do
  IFS=: read -r scene episode seed <<< "${item}"
  output=${RUN_ROOT}/results/gatecurr600_oracle/${scene}
  if [[ -e "${output}/metric.csv" ]]; then
    echo "ABORT: refusing to overwrite ${output}/metric.csv" >&2
    exit 1
  fi
done

SERVER_PID=
cleanup_server() {
  if [[ -n "${SERVER_PID}" ]] && kill -0 "${SERVER_PID}" 2>/dev/null; then
    kill "${SERVER_PID}" 2>/dev/null || true
    wait "${SERVER_PID}" 2>/dev/null || true
  fi
  SERVER_PID=
}
trap cleanup_server EXIT INT TERM

SERVER_LOG=${RUN_ROOT}/logs/server_gatecurr600_oracle.log
rm -rf "${RUN_ROOT}/buffer/gatecurr600_oracle"
echo "[server] starting gatecurr600 on :${PORT}"
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
    --checkpoint "${CHECKPOINT}" \
    --internnav_root "${INTERNNAV_ROOT}" \
    --num_samples 16 \
    --exclude_recent 83 \
    --retrieval head \
    --flow_gate auto \
    --buffer_root "${RUN_ROOT}/buffer/gatecurr600_oracle" \
    > "${SERVER_LOG}" 2>&1 &
SERVER_PID=$!

ready=0
for _ in $(seq 1 120); do
  if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
    echo "ABORT: server exited during startup" >&2
    tail -n 100 "${SERVER_LOG}" >&2
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
  echo "ABORT: server did not become ready" >&2
  tail -n 100 "${SERVER_LOG}" >&2
  exit 1
fi

for item in "${CASES[@]}"; do
  IFS=: read -r scene episode seed <<< "${item}"
  output=${RUN_ROOT}/results/gatecurr600_oracle/${scene}
  mkdir -p "${output}"
  echo "[oracle] ${scene}/${episode} seed=${seed}"
  (
    cd "${ROOT}/MemNavData"
    "${HAB_PY}" "${EVALUATOR}" \
      --episode_root "${RUN_ROOT}/episodes/${scene}" \
      --episode_ids "${episode}" \
      --episodes 1 \
      --scene "${RUN_ROOT}/assets/${scene}.glb" \
      --host 127.0.0.1 \
      --port "${PORT}" \
      --out "${output}" \
      --leg1_mode replay \
      --success_dist 1.0 \
      --max_steps 1200 \
      --exec_horizon 8 \
      --seed "${seed}" \
      --retrieval_override gt_covis \
      --terminal_uturn off \
      --terminal_visual_refine off
  ) > "${RUN_ROOT}/logs/eval_gatecurr600_oracle_${scene}_${episode}.log" 2>&1
done

echo "[done] gatecurr600 long-gap oracle-covis"
