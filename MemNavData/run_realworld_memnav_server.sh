#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
MEMNAV_PY="${MEMNAV_PY:-/home/asus/miniconda3/envs/memnav/bin/python}"
MEMNAV_PORT="${MEMNAV_PORT:-18888}"
MEMNAV_CKPT="${MEMNAV_CKPT:-/home/asus/Research/Nav-axis-uturn/.diagnostics/unseen_scene_eval_20260803/checkpoints/gatecurr600.memnav.ckpt}"
INTERNNAV_ROOT="${INTERNNAV_ROOT:-$ROOT/InternNav}"
LINGBOT_REPO="${LINGBOT_REPO:-/home/asus/Research/Nav/NavDP/baselines/memnav/lingbot-map}"
LINGBOT_WEIGHTS="${LINGBOT_WEIGHTS:-$LINGBOT_REPO/weights/lingbot-map-long.pt}"
LIGHTGLUE_REPO="${LIGHTGLUE_REPO:-$ROOT/.diagnostics/dependencies/LightGlue}"
DEPENDENCY_ROOT="${DEPENDENCY_ROOT:-$ROOT/.diagnostics/dependencies/python}"
BUFFER_ROOT="${CEC_BUFFER_ROOT:-$ROOT/.diagnostics/realworld_cec_stack/buffer}"

for path in "$MEMNAV_PY" "$MEMNAV_CKPT" "$INTERNNAV_ROOT" \
  "$LINGBOT_WEIGHTS" "$LIGHTGLUE_REPO" "$DEPENDENCY_ROOT"; do
  [[ -e "$path" ]] || { echo "Missing CEC input: $path" >&2; exit 1; }
done
mkdir -p "$BUFFER_ROOT"
extra_args=()
if [[ "${CEC_EAGER_DEPTH_CACHE:-0}" == "1" ]]; then
  extra_args+=(--certified_eager_depth_cache)
fi
server_pythonpath="$ROOT:$DEPENDENCY_ROOT:$LIGHTGLUE_REPO:$INTERNNAV_ROOT/src/diffusion-policy${PYTHONPATH:+:$PYTHONPATH}"
cd "$ROOT/.diagnostics/realworld_cec_stack"
exec env PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  PYTHONPATH="$server_pythonpath" \
  LINGBOT_REPO="$LINGBOT_REPO" LINGBOT_WEIGHTS="$LINGBOT_WEIGHTS" \
  MEMNAV_WINDOW=32 MEMNAV_NUM_SCALE=8 MEMNAV_MAX_FRAME_NUM=2048 \
  MEMNAV_GROUND_SCALE_MAX=6.0 MEMNAV_GATE_FUSION=complementary \
  MEMNAV_AUX_POSE_CALIBRATION=empirical MEMNAV_COLLISION_SELECT=1 \
  MEMNAV_REPORT_TO=none \
  "$MEMNAV_PY" -u "$ROOT/NavDP/baselines/memnav/memnav_server.py" \
    --host 127.0.0.1 --port "$MEMNAV_PORT" --checkpoint "$MEMNAV_CKPT" \
    --internnav_root "$INTERNNAV_ROOT" --num_samples 16 \
    --exclude_recent 32 --retrieval raw \
    --retrieval_candidate_top_k 32 --retrieval_candidate_min_gap 16 \
    --graph_subgoal_spacing_m 0.0 --graph_subgoal_arrival_m 0.60 \
    --flow_gate auto --buffer_root "$BUFFER_ROOT" \
    --certified_relocalization \
    --lightglue_repo "$LIGHTGLUE_REPO" \
    --lightglue_dependency_root "$DEPENDENCY_ROOT" \
    --lightglue_max_keypoints 2048 \
    "${extra_args[@]}"
