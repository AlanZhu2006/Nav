#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
SESSION="${CEC_TMUX_SESSION:-cec-realworld}"
MEMNAV_PORT="${MEMNAV_PORT:-18888}"
NAVDP_PORT="${NAVDP_PORT:-8888}"
CEC_HUB_PORT="${CEC_HUB_PORT:-18889}"
OUT_ROOT="${CEC_OUT_ROOT:-$ROOT/.diagnostics/realworld_cec_stack}"
CEC_CAMERA_HEIGHT_M="${CEC_CAMERA_HEIGHT_M:?set measured camera optical-center height in metres}"
export CEC_CAMERA_HEIGHT_M

command -v tmux >/dev/null || { echo "tmux is required" >&2; exit 1; }
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux session already exists: $SESSION" >&2
  exit 1
fi
for port in "$MEMNAV_PORT" "$NAVDP_PORT" "$CEC_HUB_PORT"; do
  if ss -ltn | awk '{print $4}' | grep -Eq "(^|:)$port$"; then
    echo "port already in use: $port" >&2
    exit 1
  fi
done
mkdir -p "$OUT_ROOT/logs" "$OUT_ROOT/buffer"

tmux new-session -d -s "$SESSION" -n memnav \
  "exec '$ROOT/MemNavData/run_realworld_memnav_server.sh' >'$OUT_ROOT/logs/memnav.log' 2>&1"
tmux new-window -t "$SESSION" -n navdp \
  "exec '$ROOT/MemNavData/run_realworld_navdp_server.sh' >'$OUT_ROOT/logs/navdp.log' 2>&1"
tmux new-window -t "$SESSION" -n hub \
  "exec '$ROOT/MemNavData/run_realworld_cec_hub.sh' >'$OUT_ROOT/logs/hub.log' 2>&1"

ready=false
for _ in $(seq 1 240); do
  if curl -fsS --max-time 1 "http://127.0.0.1:$CEC_HUB_PORT/healthz" \
      | grep -q 'causal_monocular_rgb_v1' \
      && ss -ltn | awk '{print $4}' | grep -Eq "(^|:)$MEMNAV_PORT$" \
      && ss -ltn | awk '{print $4}' | grep -Eq "(^|:)$NAVDP_PORT$"; then
    ready=true
    break
  fi
  if ! tmux has-session -t "$SESSION" 2>/dev/null; then
    break
  fi
  sleep 1
done
if [[ "$ready" != true ]]; then
  echo "CEC policy stack failed to become ready" >&2
  for log in "$OUT_ROOT"/logs/*.log; do
    echo "===== $log" >&2
    tail -n 100 "$log" >&2 || true
  done
  tmux kill-session -t "$SESSION" 2>/dev/null || true
  exit 1
fi

echo "CEC real-world policy stack ready"
echo "  sensor: causal monocular RGB (client depth is local safety only)"
echo "  camera optical-center height: ${CEC_CAMERA_HEIGHT_M} m"
echo "  hub:    http://127.0.0.1:$CEC_HUB_PORT"
echo "  memnav: http://127.0.0.1:$MEMNAV_PORT"
echo "  navdp:  http://127.0.0.1:$NAVDP_PORT"
echo "  logs:   $OUT_ROOT/logs"
echo "  tmux:   tmux attach -t $SESSION"
