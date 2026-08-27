#!/usr/bin/env bash
# One formal scene: two paired long-horizon-readout arms share one loaded
# server pair and one shared causal monocular Goal-A trace per episode.
set -euo pipefail
umask 0022

ROOT=${ROOT:?set immutable source ROOT}
RUN_ROOT=${RUN_ROOT:?set RUN_ROOT}
SCENE_INDEX=${SCENE_INDEX:?set SCENE_INDEX}
HAB_PY=${HAB_PY:?set HAB_PY}
MEMNAV_PY=${MEMNAV_PY:?set MEMNAV_PY}
SOURCE_RECEIPT=${SOURCE_RECEIPT:?set SOURCE_RECEIPT}
EXPECTED_SOURCE_RECEIPT_SHA=${EXPECTED_SOURCE_RECEIPT_SHA:?set receipt SHA}
MANIFEST=${MANIFEST:-/scratch/yz11502/Research/Nav-axis-uturn-results/revisit_fresh_confirmation_20260811/fresh160_v3_attempt600_20260811T2000/data_manifest.json}
PROTOCOL=${PROTOCOL:-${ROOT}/MemNavData/mdtec_monocular_cec_composition_protocol_20260819.json}
EXPECTED_MANIFEST_SHA=8013fa2a768d84638a9f9ecc50df46dda67ebb79250d14ad0a8087ac52fd33e5
EXPECTED_PROTOCOL_SHA=5554f78a31ae843c4b87bdbdb0fea39521cf17804b0f13727bfec63592bb58a0
BASE_ROOT=${BASE_ROOT:-/scratch/yz11502/Research/Nav-axis-uturn}
INTERNNAV_ROOT=${INTERNNAV_ROOT:-${BASE_ROOT}/InternNav}
MEMNAV_CKPT=${MEMNAV_CKPT:-${BASE_ROOT}/.diagnostics/unseen_scene_eval_20260803/checkpoints/gatecurr600.memnav.ckpt}
NAVDP_CKPT=${NAVDP_CKPT:-${BASE_ROOT}/.diagnostics/unseen_scene_eval_20260803/checkpoints/navdp_checkpoint.ckpt}
LINGBOT_REPO=${LINGBOT_REPO:-/scratch/lg154/Research/Nav/NavDP/baselines/memnav/lingbot-map}
LINGBOT_WEIGHTS=${LINGBOT_WEIGHTS:-${LINGBOT_REPO}/weights/lingbot-map-long.pt}
LIGHTGLUE_REPO=${LIGHTGLUE_REPO:-${ROOT}/third_party/LightGlue}
DEPENDENCY_ROOT=${DEPENDENCY_ROOT:-${ROOT}/third_party/python}
TORCH_HOME=${TORCH_HOME:-${ROOT}/torch_home}
EXPECTED_MEMNAV_SHA=9b7a5811ff0aea212503f58b45258ba4f66b06420f87c350946aead39db6fdb7
EXPECTED_NAVDP_SHA=3bb3ad4ab241e857bb57a4021cc6aab76d5263e81fbf80298d579053ef011947
EXPECTED_LINGBOT_SHA=832bc82cbae0bc9bbe946ef5ee1f7226abd8c0e183ccf8beddbb3d133576f409
SMOKE=${SMOKE:-0}
MAX_STEPS=${MAX_STEPS:-500}

[[ "${SCENE_INDEX}" =~ ^([0-9]|1[0-9])$ ]] || { echo "bad scene index" >&2; exit 1; }
[[ "${SMOKE}" =~ ^[01]$ ]] || { echo "bad SMOKE" >&2; exit 1; }
if [[ "${SMOKE}" -eq 0 ]]; then
  [[ "${MAX_STEPS}" -eq 500 ]] || { echo "formal max steps changed" >&2; exit 1; }
fi
for item in "${HAB_PY}" "${MEMNAV_PY}" "${SOURCE_RECEIPT}" "${MANIFEST}" \
            "${PROTOCOL}" "${MEMNAV_CKPT}" "${NAVDP_CKPT}" "${LINGBOT_WEIGHTS}"; do
  [[ -r "${item}" ]] || { echo "missing ${item}" >&2; exit 1; }
done
[[ "$(sha256sum "${SOURCE_RECEIPT}" | awk '{print $1}')" == "${EXPECTED_SOURCE_RECEIPT_SHA}" ]] || { echo "source receipt SHA changed" >&2; exit 1; }
(cd "${ROOT}" && sha256sum -c --quiet "${SOURCE_RECEIPT}")
[[ "$(sha256sum "${MANIFEST}" | awk '{print $1}')" == "${EXPECTED_MANIFEST_SHA}" ]] || { echo "Fresh160 manifest SHA changed" >&2; exit 1; }
[[ "$(sha256sum "${PROTOCOL}" | awk '{print $1}')" == "${EXPECTED_PROTOCOL_SHA}" ]] || { echo "composition protocol SHA changed" >&2; exit 1; }
[[ "$(sha256sum "${MEMNAV_CKPT}" | awk '{print $1}')" == "${EXPECTED_MEMNAV_SHA}" ]] || exit 1
[[ "$(sha256sum "${NAVDP_CKPT}" | awk '{print $1}')" == "${EXPECTED_NAVDP_SHA}" ]] || exit 1
[[ "$(sha256sum "${LINGBOT_WEIGHTS}" | awk '{print $1}')" == "${EXPECTED_LINGBOT_SHA}" ]] || exit 1

HAB_SITE=$(${HAB_PY} -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
HAB_PYTHONPATH=${HAB_SITE}/pip/_vendor

scene=$(${HAB_PY} - "${MANIFEST}" "${SCENE_INDEX}" <<'PY'
import json,sys
print(json.load(open(sys.argv[1]))["scenes"][int(sys.argv[2])])
PY
)
scene_root=${RUN_ROOT}/scenes/$(printf '%02d' "${SCENE_INDEX}")_${scene}
[[ ! -e "${scene_root}" ]] || { echo "output exists: ${scene_root}" >&2; exit 1; }
mkdir -p "${scene_root}/logs"
exec > >(tee "${scene_root}/run.log") 2>&1

(cd "${ROOT}" && PYTHONDONTWRITEBYTECODE=1 "${MEMNAV_PY}" -m unittest \
  MemNavData.test_mdtec_monocular_cec_composition \
  MemNavData.test_mdtec_raw_depth_gate_d \
  MemNavData.test_monocular_depth_runtime)

port_key=$(( (${SLURM_JOB_ID:-1000} + SCENE_INDEX * 43) % 14000 ))
MEMNAV_PORT=${MEMNAV_PORT:-$((25000 + port_key))}
NAVDP_PORT=${NAVDP_PORT:-$((MEMNAV_PORT + 1))}
for port in "${MEMNAV_PORT}" "${NAVDP_PORT}"; do
  if ss -ltn | awk '{print $4}' | grep -Eq "(^|:)${port}$"; then
    echo "port ${port} in use" >&2; exit 1
  fi
done
runtime_root=${SLURM_TMPDIR:-/tmp}/mdtec_cec_composition_${SLURM_JOB_ID:-local}_${SCENE_INDEX}
mkdir -p "${runtime_root}/memnav" "${runtime_root}/navdp"
MEMNAV_PID= NAVDP_PID=
cleanup() {
  for pid in "${NAVDP_PID}" "${MEMNAV_PID}"; do
    if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
      kill "${pid}" 2>/dev/null || true; wait "${pid}" 2>/dev/null || true
    fi
  done
}
trap cleanup EXIT INT TERM
(
  cd "${runtime_root}/memnav"
  exec env PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    PYTHONPATH="${ROOT}:${DEPENDENCY_ROOT}:${LIGHTGLUE_REPO}:${INTERNNAV_ROOT}/src/diffusion-policy" \
    TORCH_HOME="${TORCH_HOME}" \
    LINGBOT_REPO="${LINGBOT_REPO}" LINGBOT_WEIGHTS="${LINGBOT_WEIGHTS}" \
    MEMNAV_WINDOW=32 MEMNAV_NUM_SCALE=8 MEMNAV_MAX_FRAME_NUM=2048 \
    MEMNAV_GROUND_SCALE_MAX=6.0 MEMNAV_GATE_FUSION=complementary \
    MEMNAV_AUX_POSE_CALIBRATION=empirical MEMNAV_COLLISION_SELECT=1 \
    MEMNAV_REPORT_TO=none "${MEMNAV_PY}" -u \
    "${ROOT}/NavDP/baselines/memnav/memnav_server.py" \
      --port "${MEMNAV_PORT}" --checkpoint "${MEMNAV_CKPT}" \
      --internnav_root "${INTERNNAV_ROOT}" --num_samples 16 \
      --exclude_recent 32 --retrieval raw --retrieval_candidate_top_k 32 \
      --retrieval_candidate_min_gap 16 --graph_subgoal_spacing_m 0.0 \
      --graph_subgoal_arrival_m 0.60 --flow_gate auto \
      --certified_relocalization --lightglue_repo "${LIGHTGLUE_REPO}" \
      --buffer_root "${scene_root}/buffer"
) >"${scene_root}/logs/server_memnav.log" 2>&1 &
MEMNAV_PID=$!
(
  cd "${runtime_root}/navdp"
  exec env NAVDP_DISABLE_VIDEO=1 PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="${ROOT}" \
    "${MEMNAV_PY}" -u "${ROOT}/NavDP/baselines/navdp/navdp_server.py" \
      --port "${NAVDP_PORT}" --checkpoint "${NAVDP_CKPT}" \
      --depth_source metric_request --allow_depth_source_override \
      --monocular_depth_url "http://127.0.0.1:${MEMNAV_PORT}/monocular_depth_query"
) >"${scene_root}/logs/server_navdp.log" 2>&1 &
NAVDP_PID=$!
for spec in "memnav:${MEMNAV_PID}:${MEMNAV_PORT}" "navdp:${NAVDP_PID}:${NAVDP_PORT}"; do
  IFS=: read -r label pid port <<<"${spec}"; ready=0
  for _ in $(seq 1 240); do
    if ! kill -0 "${pid}" 2>/dev/null; then
      tail -n 100 "${scene_root}/logs/server_${label}.log"; exit 1
    fi
    if ss -ltn | awk '{print $4}' | grep -Eq "(^|:)${port}$"; then ready=1; break; fi
    sleep 2
  done
  [[ "${ready}" -eq 1 ]] || { echo "${label} startup timeout" >&2; exit 1; }
done
gpu_uuid=$(nvidia-smi --query-gpu=uuid --format=csv,noheader | head -1)
mem_start=$(awk '{print $22}' "/proc/${MEMNAV_PID}/stat")
nav_start=$(awk '{print $22}' "/proc/${NAVDP_PID}/stat")
cat >"${scene_root}/server_receipt.json" <<EOF
{
  "same_process_all_arms": true,
  "memnav_pid": ${MEMNAV_PID},
  "memnav_process_start_ticks": ${mem_start},
  "memnav_port": ${MEMNAV_PORT},
  "navdp_pid": ${NAVDP_PID},
  "navdp_process_start_ticks": ${nav_start},
  "navdp_port": ${NAVDP_PORT},
  "gpu_uuid": "${gpu_uuid}",
  "slurm_job_id": "${SLURM_JOB_ID:-local}",
  "scene_index": ${SCENE_INDEX},
  "scene": "${scene}"
}
EOF

env MDTEC_CEC_COMPOSITION_PROTOCOL="${PROTOCOL}" \
  MDTEC_CEC_COMPOSITION_MANIFEST="${MANIFEST}" \
  MDTEC_CEC_COMPOSITION_SCENE_INDEX="${SCENE_INDEX}" \
  MDTEC_CEC_COMPOSITION_SCENE_ROOT="${scene_root}" \
  MDTEC_CEC_COMPOSITION_HAB_PY="${HAB_PY}" \
  MDTEC_CEC_COMPOSITION_EVAL_2LEG_PY="${ROOT}/MemNavData/eval_2leg_habitat.py" \
  MDTEC_CEC_COMPOSITION_HOST=127.0.0.1 \
  MDTEC_CEC_COMPOSITION_MEMNAV_PORT="${MEMNAV_PORT}" \
  MDTEC_CEC_COMPOSITION_NAVDP_PORT="${NAVDP_PORT}" \
  MDTEC_CEC_COMPOSITION_SMOKE="${SMOKE}" \
  MDTEC_CEC_COMPOSITION_MAX_STEPS="${MAX_STEPS}" \
  PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="${ROOT}:${HAB_PYTHONPATH}" \
  "${HAB_PY}" -u "${ROOT}/MemNavData/eval_mdtec_monocular_cec_composition_habitat.py" \
    >"${scene_root}/logs/evaluator.log" 2>&1

kill -0 "${MEMNAV_PID}" && kill -0 "${NAVDP_PID}"
[[ "$(awk '{print $22}' "/proc/${MEMNAV_PID}/stat")" == "${mem_start}" ]] || exit 1
[[ "$(awk '{print $22}' "/proc/${NAVDP_PID}/stat")" == "${nav_start}" ]] || exit 1
rows=$(( $(wc -l <"${scene_root}/depth_arms.csv") - 1 ))
[[ "${rows}" -eq 4 ]] || { echo "expected four rows (2 episodes x 2 arms), got ${rows}" >&2; exit 1; }
cp "${SOURCE_RECEIPT}" "${scene_root}/source_inputs.sha256"
echo "[complete] scene=${scene} rows=${rows}"
