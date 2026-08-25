#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
MEMNAV_PY="${MEMNAV_PY:-/home/asus/miniconda3/envs/memnav/bin/python}"
MEMNAV_PORT="${MEMNAV_PORT:-18888}"
NAVDP_PORT="${NAVDP_PORT:-8888}"
CEC_HUB_PORT="${CEC_HUB_PORT:-18889}"
CEC_CAMERA_HEIGHT_M="${CEC_CAMERA_HEIGHT_M:?set measured camera optical-center height in metres}"
CEC_GOAL_CANDIDATE_DIR="${CEC_GOAL_CANDIDATE_DIR:-$ROOT/.diagnostics/realworld_cec_stack/goal_candidates}"

cd "$ROOT"
exec env PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" \
  "$MEMNAV_PY" -u "$ROOT/MemNavData/realworld_cec_hub.py" \
    --host 127.0.0.1 --port "$CEC_HUB_PORT" \
    --memnav-url "http://127.0.0.1:$MEMNAV_PORT" \
    --navdp-url "http://127.0.0.1:$NAVDP_PORT" \
    --camera-height-m "$CEC_CAMERA_HEIGHT_M" \
    --goal-candidate-dir "$CEC_GOAL_CANDIDATE_DIR"
