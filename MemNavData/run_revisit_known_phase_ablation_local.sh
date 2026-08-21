#!/usr/bin/env bash
# Start the two frozen policy servers and run the consumed-pool Revisit phase
# ablation.  This launcher owns its ports and cleans up only its own processes.

set -euo pipefail
umask 0022

ROOT=${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
OUT_ROOT=${OUT_ROOT:-${ROOT}/.diagnostics/revisit_known_phase_ablation_20260811}
MEMNAV_PORT=${MEMNAV_PORT:-21110}
NAVDP_PORT=${NAVDP_PORT:-21111}
MEMNAV_PY=${MEMNAV_PY:-/home/asus/miniconda3/envs/memnav/bin/python}
HAB_PY=${HAB_PY:-/home/asus/miniconda3/envs/habitat/bin/python}
MEMNAV_CKPT=${MEMNAV_CKPT:-/home/asus/Research/Nav-axis-uturn/.diagnostics/unseen_scene_eval_20260803/checkpoints/gatecurr600.memnav.ckpt}
NAVDP_CKPT=${NAVDP_CKPT:-/home/asus/Research/Nav/NavDP/baselines/navdp/checkpoints/navdp_checkpoint.ckpt}
PHASE_B_CKPT=${PHASE_B_CKPT:-${ROOT}/.diagnostics/phase_b_model_20260808/lingbot_native_phase_b.pt}
LINGBOT_REPO=${LINGBOT_REPO:-/home/asus/Research/Nav/NavDP/baselines/memnav/lingbot-map}
LINGBOT_WEIGHTS=${LINGBOT_WEIGHTS:-${LINGBOT_REPO}/weights/lingbot-map-long.pt}
INTERNNAV_ROOT=${INTERNNAV_ROOT:-${ROOT}/InternNav}
MANIFEST=${MANIFEST:-${ROOT}/MemNavData/expanded_navdp_router_eval_20260805.json}
BASE_RUNNER=${BASE_RUNNER:-${ROOT}/MemNavData/run_local_phase_b_p0_20scene.sh}
SUMMARIZER=${SUMMARIZER:-${ROOT}/MemNavData/summarize_revisit_phase_ablation.py}
PROTOCOL=${PROTOCOL:-${ROOT}/MemNavData/REVISIT_KNOWN_PHASE_ABLATION_PROTOCOL_20260811.md}
FRONT_SUPPORT_SUMMARIZER=${FRONT_SUPPORT_SUMMARIZER:-${ROOT}/MemNavData/summarize_revisit_front_support.py}
RUN_KNOWN_REVISIT_FRONT_SUPPORT=${RUN_KNOWN_REVISIT_FRONT_SUPPORT:-0}
SCENE_INDICES=${SCENE_INDICES:-}
EXPECTED_PHASE_B_SHA=1232a426458cedf36869304116a2dd5c779bbcdaca587f76abd5ed3572164f2c

for required in "${MEMNAV_PY}" "${HAB_PY}" "${MEMNAV_CKPT}" \
                "${NAVDP_CKPT}" "${PHASE_B_CKPT}" "${LINGBOT_WEIGHTS}" \
                "${MANIFEST}" "${BASE_RUNNER}" "${SUMMARIZER}" \
                "${PROTOCOL}"; do
  test -r "${required}" || {
    echo "ABORT: missing input ${required}" >&2
    exit 1
  }
done
[[ "${RUN_KNOWN_REVISIT_FRONT_SUPPORT}" =~ ^[01]$ ]] || {
  echo "ABORT: RUN_KNOWN_REVISIT_FRONT_SUPPORT must be 0 or 1" >&2
  exit 1
}
if [[ "${RUN_KNOWN_REVISIT_FRONT_SUPPORT}" -eq 1 ]]; then
  test -r "${FRONT_SUPPORT_SUMMARIZER}" || {
    echo "ABORT: missing input ${FRONT_SUPPORT_SUMMARIZER}" >&2
    exit 1
  }
  test -r "${ROOT}/MemNavData/run_revisit_front_support_local.sh" || {
    echo "ABORT: missing front-support launcher" >&2
    exit 1
  }
fi
[[ "$(sha256sum "${PHASE_B_CKPT}" | awk '{print $1}')" == \
    "${EXPECTED_PHASE_B_SHA}" ]] || {
  echo "ABORT: Phase-B checkpoint identity changed" >&2
  exit 1
}
for port in "${MEMNAV_PORT}" "${NAVDP_PORT}"; do
  if ss -ltn | awk '{print $4}' | grep -Eq "(^|:)${port}$"; then
    echo "ABORT: port ${port} is already in use" >&2
    exit 1
  fi
done

mkdir -p "${OUT_ROOT}/logs" "${OUT_ROOT}/buffer"
contract_tmp=$(mktemp)
runtime_root=$(mktemp -d /tmp/revisit_known_phase.XXXXXX)
MEMNAV_PID=
NAVDP_PID=
cleanup() {
  for pid in "${NAVDP_PID}" "${MEMNAV_PID}"; do
    if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
      kill "${pid}" 2>/dev/null || true
      wait "${pid}" 2>/dev/null || true
    fi
  done
  rm -f -- "${contract_tmp}"
  rm -rf -- "${runtime_root}"
}
trap cleanup EXIT INT TERM

extra_source_inputs=()
if [[ "${RUN_KNOWN_REVISIT_FRONT_SUPPORT}" -eq 1 ]]; then
  extra_source_inputs+=(
    "${FRONT_SUPPORT_SUMMARIZER}"
    "${ROOT}/MemNavData/run_revisit_front_support_local.sh"
  )
fi
sha256sum \
  "${PROTOCOL}" \
  "${BASE_RUNNER}" \
  "${SUMMARIZER}" \
  "${ROOT}/MemNavData/eval_2leg_habitat.py" \
  "${ROOT}/MemNavData/revisit_bearing_adapter.py" \
  "${extra_source_inputs[@]}" \
  "${ROOT}/NavDP/baselines/memnav/memnav_server.py" \
  "${ROOT}/NavDP/baselines/memnav/policy_agent.py" \
  "${ROOT}/NavDP/baselines/navdp/navdp_server.py" \
  "${ROOT}/NavDP/baselines/navdp/policy_agent.py" \
  "${MANIFEST}" > "${contract_tmp}"
if [[ -e "${OUT_ROOT}/ablation_source_inputs.sha256" ]]; then
  cmp --silent "${contract_tmp}" "${OUT_ROOT}/ablation_source_inputs.sha256" || {
    echo "ABORT: source inputs changed during resume" >&2
    exit 1
  }
else
  mv "${contract_tmp}" "${OUT_ROOT}/ablation_source_inputs.sha256"
  contract_tmp=$(mktemp)
fi

"${HAB_PY}" -m py_compile \
  "${ROOT}/MemNavData/eval_2leg_habitat.py" \
  "${SUMMARIZER}"
"${MEMNAV_PY}" -m py_compile \
  "${ROOT}/NavDP/baselines/memnav/memnav_server.py" \
  "${ROOT}/NavDP/baselines/memnav/policy_agent.py" \
  "${ROOT}/NavDP/baselines/navdp/navdp_server.py" \
  "${ROOT}/NavDP/baselines/navdp/policy_agent.py"

mkdir -p "${runtime_root}/memnav" "${runtime_root}/navdp"
(
  cd "${runtime_root}/memnav"
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
    "${MEMNAV_PY}" -u "${ROOT}/NavDP/baselines/memnav/memnav_server.py" \
      --port "${MEMNAV_PORT}" \
      --checkpoint "${MEMNAV_CKPT}" \
      --internnav_root "${INTERNNAV_ROOT}" \
      --num_samples 16 \
      --exclude_recent 32 \
      --retrieval raw \
      --retrieval_candidate_top_k 32 \
      --retrieval_candidate_min_gap 16 \
      --graph_subgoal_spacing_m 0.0 \
      --graph_subgoal_arrival_m 0.60 \
      --flow_gate auto \
      --phase_b_checkpoint "${PHASE_B_CKPT}" \
      --phase_b_allow_unapproved \
      --buffer_root "${OUT_ROOT}/buffer"
) > "${OUT_ROOT}/logs/server_memnav.log" 2>&1 &
MEMNAV_PID=$!

(
  cd "${runtime_root}/navdp"
  exec env NAVDP_DISABLE_VIDEO=1 PYTHONUNBUFFERED=1 \
    "${MEMNAV_PY}" -u "${ROOT}/NavDP/baselines/navdp/navdp_server.py" \
      --port "${NAVDP_PORT}" \
      --checkpoint "${NAVDP_CKPT}"
) > "${OUT_ROOT}/logs/server_navdp.log" 2>&1 &
NAVDP_PID=$!

for spec in \
    "memnav:${MEMNAV_PID}:${MEMNAV_PORT}:${OUT_ROOT}/logs/server_memnav.log" \
    "navdp:${NAVDP_PID}:${NAVDP_PORT}:${OUT_ROOT}/logs/server_navdp.log"; do
  IFS=: read -r label pid port log <<<"${spec}"
  ready=0
  for _ in $(seq 1 240); do
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
    echo "ABORT: ${label} server did not bind port ${port}" >&2
    tail -n 120 "${log}" >&2
    exit 1
  }
done

ROOT="${ROOT}" \
OUT_ROOT="${OUT_ROOT}" \
MANIFEST="${MANIFEST}" \
MEMNAV_PORT="${MEMNAV_PORT}" \
NAVDP_PORT="${NAVDP_PORT}" \
RUN_GEOMETRY=1 \
RUN_LEARNED=0 \
RUN_NATIVE=0 \
RUN_KNOWN_REVISIT_DIRECT=1 \
RUN_KNOWN_REVISIT_FRONT_SUPPORT="${RUN_KNOWN_REVISIT_FRONT_SUPPORT}" \
SCENE_INDICES="${SCENE_INDICES}" \
MAX_STEPS=500 \
BASE_SEED=20260803 \
EXPECTED_PHASE_B_SHA="${EXPECTED_PHASE_B_SHA}" \
  bash "${BASE_RUNNER}"

if [[ ! -e "${OUT_ROOT}/report.json" \
      && "${RUN_KNOWN_REVISIT_FRONT_SUPPORT}" -eq 0 ]]; then
  (
    cd "${ROOT}"
    "${HAB_PY}" -u -m MemNavData.summarize_revisit_phase_ablation \
      --manifest "${MANIFEST}" \
      --run-root "${OUT_ROOT}" \
      --out "${OUT_ROOT}/report.json"
  ) > "${OUT_ROOT}/logs/summarize.log" 2>&1
fi
if [[ ! -e "${OUT_ROOT}/report.json" \
      && "${RUN_KNOWN_REVISIT_FRONT_SUPPORT}" -eq 1 ]]; then
  (
    cd "${ROOT}"
    "${HAB_PY}" -u -m MemNavData.summarize_revisit_front_support \
      --manifest "${MANIFEST}" \
      --run-root "${OUT_ROOT}" \
      --scene-indices "${SCENE_INDICES}" \
      --out "${OUT_ROOT}/report.json"
  ) > "${OUT_ROOT}/logs/summarize.log" 2>&1
fi
if [[ -e "${OUT_ROOT}/report.json" ]]; then
  sha256sum "${OUT_ROOT}/report.json" > "${OUT_ROOT}/report.json.sha256"
fi
echo "[complete] ${OUT_ROOT}"
