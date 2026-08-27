#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
MEMNAV_PY="${MEMNAV_PY:-/home/asus/miniconda3/envs/memnav/bin/python}"
NAVDP_PORT="${NAVDP_PORT:-8888}"
MEMNAV_PORT="${MEMNAV_PORT:-18888}"
NAVDP_CKPT="${NAVDP_CKPT:-/home/asus/Research/Nav/NavDP/baselines/navdp/checkpoints/navdp_checkpoint.ckpt}"
DEPENDENCY_ROOT="${DEPENDENCY_ROOT:-$ROOT/.diagnostics/dependencies/python}"
INTERNNAV_ROOT="${INTERNNAV_ROOT:-$ROOT/InternNav}"

for path in "$MEMNAV_PY" "$NAVDP_CKPT" "$DEPENDENCY_ROOT" "$INTERNNAV_ROOT"; do
  [[ -e "$path" ]] || { echo "Missing NavDP input: $path" >&2; exit 1; }
done
server_pythonpath="$ROOT:$ROOT/NavDP/baselines/navdp:$DEPENDENCY_ROOT:$INTERNNAV_ROOT/src/diffusion-policy${PYTHONPATH:+:$PYTHONPATH}"
cd "$ROOT/.diagnostics/realworld_cec_stack"
exec env NAVDP_DISABLE_VIDEO=1 PYTHONUNBUFFERED=1 \
  PYTHONPATH="$server_pythonpath" \
  "$MEMNAV_PY" -u "$ROOT/NavDP/baselines/navdp/navdp_server.py" \
    --port "$NAVDP_PORT" --checkpoint "$NAVDP_CKPT" \
    --depth_source monocular_sidecar \
    --monocular_depth_url \
      "http://127.0.0.1:${MEMNAV_PORT}/monocular_depth_query"
