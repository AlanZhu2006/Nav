#!/usr/bin/env bash
# One real LingBot/CEC RGB replay gate; no Habitat, NavDP, actions, or SR.
set -euo pipefail
umask 0022

ROOT=${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
MIRROR_ROOT=${MIRROR_ROOT:-${ROOT}/.diagnostics/long_range_path_field_20260902/local_mirror}
MANIFEST=${MANIFEST:-${MIRROR_ROOT}/scratch/yz11502/Research/Nav-axis-uturn-results/hm3d_table3_causal_survey_expansion_20260831/expansion_20260830T165729Z_a8dd232e/merged_query_population/role_pairs/manifest.json}
HISTORY_INDEX=${HISTORY_INDEX:-0}
OUT_ROOT=${OUT_ROOT:-${ROOT}/.diagnostics/long_range_path_field_20260902/local_rgb_gate_index${HISTORY_INDEX}_attempt1}
MEMNAV_PORT=${MEMNAV_PORT:-22670}
MEMNAV_PY=${MEMNAV_PY:-/home/asus/miniconda3/envs/memnav/bin/python}
MEMNAV_CKPT=${MEMNAV_CKPT:-/home/asus/Research/Nav-axis-uturn/.diagnostics/unseen_scene_eval_20260803/checkpoints/gatecurr600.memnav.ckpt}
LINGBOT_REPO=${LINGBOT_REPO:-/home/asus/Research/Nav/NavDP/baselines/memnav/lingbot-map}
LINGBOT_WEIGHTS=${LINGBOT_WEIGHTS:-${LINGBOT_REPO}/weights/lingbot-map-long.pt}
LIGHTGLUE_REPO=${LIGHTGLUE_REPO:-${ROOT}/.diagnostics/dependencies/LightGlue}
DEPENDENCY_ROOT=${DEPENDENCY_ROOT:-${ROOT}/.diagnostics/dependencies/python}
INTERNNAV_ROOT=${INTERNNAV_ROOT:-${ROOT}/InternNav}
STRIDE=${STRIDE:-8}
MAX_READOUTS=${MAX_READOUTS:-80}

fail() { echo "ABORT: $*" >&2; exit 2; }
[[ "${HISTORY_INDEX}" =~ ^(0|16|32)$ ]] || fail "HISTORY_INDEX must be 0,16,32"
[[ "${STRIDE}" =~ ^[1-9][0-9]*$ ]] || fail "STRIDE must be positive"
[[ "${MAX_READOUTS}" =~ ^[1-9][0-9]*$ ]] || fail "MAX_READOUTS must be positive"
[[ ! -e "${OUT_ROOT}" ]] || fail "output exists: ${OUT_ROOT}"
for path in "${MANIFEST}" "${MEMNAV_PY}" "${MEMNAV_CKPT}" \
  "${LINGBOT_WEIGHTS}" "${LIGHTGLUE_REPO}/lightglue" \
  "${DEPENDENCY_ROOT}/kornia" "${INTERNNAV_ROOT}"; do
  [[ -e "${path}" ]] || fail "missing input: ${path}"
done
if ss -ltn | awk '{print $4}' | grep -Eq "(^|:)${MEMNAV_PORT}$"; then
  fail "port ${MEMNAV_PORT} is already in use"
fi

mkdir -p "${OUT_ROOT}/logs" "${OUT_ROOT}/buffer"
runtime_root=$(mktemp -d /tmp/epf_rgb_replay.XXXXXX)
MEMNAV_PID=
cleanup() {
  if [[ -n "${MEMNAV_PID}" ]] && kill -0 "${MEMNAV_PID}" 2>/dev/null; then
    kill "${MEMNAV_PID}" 2>/dev/null || true
    wait "${MEMNAV_PID}" 2>/dev/null || true
  fi
  rm -rf -- "${runtime_root}"
}
trap cleanup EXIT INT TERM

server_path=${ROOT}:${ROOT}/MemNavData:${ROOT}/NavDP/baselines/memnav:${DEPENDENCY_ROOT}:${LIGHTGLUE_REPO}:${INTERNNAV_ROOT}/src/diffusion-policy

PYTHONPATH="${ROOT}:${ROOT}/MemNavData" "${MEMNAV_PY}" -m py_compile \
  "${ROOT}/MemNavData/episodic_path_field.py" \
  "${ROOT}/MemNavData/run_hm3d_episodic_path_field_gate.py" \
  "${ROOT}/MemNavData/run_local_episodic_path_field_rgb_replay_gate.py" \
  "${ROOT}/NavDP/baselines/memnav/policy_agent.py" \
  "${ROOT}/NavDP/baselines/memnav/memnav_server.py"

(
  cd "${runtime_root}"
  exec env PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    PYTHONPATH="${server_path}" LINGBOT_REPO="${LINGBOT_REPO}" \
    LINGBOT_WEIGHTS="${LINGBOT_WEIGHTS}" MEMNAV_WINDOW=32 \
    MEMNAV_NUM_SCALE=8 MEMNAV_MAX_FRAME_NUM=2048 \
    MEMNAV_GROUND_SCALE_MAX=6.0 MEMNAV_GATE_FUSION=complementary \
    MEMNAV_AUX_POSE_CALIBRATION=empirical MEMNAV_COLLISION_SELECT=1 \
    MEMNAV_REPORT_TO=none "${MEMNAV_PY}" -u \
    "${ROOT}/NavDP/baselines/memnav/memnav_server.py" \
      --host 127.0.0.1 --port "${MEMNAV_PORT}" \
      --checkpoint "${MEMNAV_CKPT}" --internnav_root "${INTERNNAV_ROOT}" \
      --num_samples 16 --exclude_recent 32 --retrieval raw \
      --retrieval_candidate_top_k 32 --retrieval_candidate_min_gap 16 \
      --graph_subgoal_spacing_m 0.0 --graph_subgoal_arrival_m 0.60 \
      --flow_gate auto --buffer_root "${OUT_ROOT}/buffer" \
      --certified_relocalization --lightglue_repo "${LIGHTGLUE_REPO}" \
      --lightglue_dependency_root "${DEPENDENCY_ROOT}" \
      --lightglue_max_keypoints 2048
) >"${OUT_ROOT}/logs/server_memnav.log" 2>&1 &
MEMNAV_PID=$!

ready=0
for _ in $(seq 1 240); do
  if ! kill -0 "${MEMNAV_PID}" 2>/dev/null; then
    tail -n 160 "${OUT_ROOT}/logs/server_memnav.log" >&2
    fail "MemNav server exited during startup"
  fi
  if "${MEMNAV_PY}" - "${MEMNAV_PORT}" <<'PY' >/dev/null 2>&1
import socket,sys
with socket.create_connection(("127.0.0.1", int(sys.argv[1])), timeout=0.5):
    pass
PY
  then
    ready=1
    break
  fi
  sleep 2
done
[[ "${ready}" -eq 1 ]] || fail "MemNav server startup timeout"

PYTHONPATH="${ROOT}:${ROOT}/MemNavData" "${MEMNAV_PY}" -u \
  "${ROOT}/MemNavData/run_local_episodic_path_field_rgb_replay_gate.py" \
    --mirror-root "${MIRROR_ROOT}" --manifest "${MANIFEST}" \
    --history-index "${HISTORY_INDEX}" --port "${MEMNAV_PORT}" \
    --stride "${STRIDE}" --max-readouts "${MAX_READOUTS}" \
    --out "${OUT_ROOT}/result" \
    >"${OUT_ROOT}/logs/client.log" 2>&1

cat "${OUT_ROOT}/logs/client.log"
echo "LOCAL_RGB_REPLAY_GATE_COMPLETE output=${OUT_ROOT}/result/gate_result.json"
