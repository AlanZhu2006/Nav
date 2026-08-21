#!/usr/bin/env bash
# One formal scene: three paired depth arms share one loaded server pair.
set -euo pipefail
umask 0022

ROOT=${ROOT:?set immutable source ROOT}
RUN_ROOT=${RUN_ROOT:?set RUN_ROOT}
SCENE_INDEX=${SCENE_INDEX:?set SCENE_INDEX}
HAB_PY=${HAB_PY:?set HAB_PY}
MEMNAV_PY=${MEMNAV_PY:?set MEMNAV_PY}
SOURCE_RECEIPT=${SOURCE_RECEIPT:?set SOURCE_RECEIPT}
EXPECTED_SOURCE_RECEIPT_SHA=${EXPECTED_SOURCE_RECEIPT_SHA:?set receipt SHA}
MANIFEST=${MANIFEST:-${ROOT}/MemNavData/expanded_navdp_router_eval_20260805.json}
PROTOCOL=${PROTOCOL:-${ROOT}/MemNavData/mdtec_raw_depth_gate_d_protocol_20260819.json}
ANALYSIS=${ANALYSIS:-${ROOT}/MemNavData/mdtec_raw_depth_gate_d_analysis_20260819.json}
INPUTS=${INPUTS:-${ROOT}/MemNavData/novel_a_bearing_inputs_20260808.json}
EXPECTED_MANIFEST_SHA=ba8f72cb504768c801e6c9c386436ccdc66dea07a5e5fac2d7b4248738946a61
EXPECTED_PROTOCOL_SHA=8deb4e13bd169bb39c7696a777656c3e4912f985f48ee364111b4a9a76cf9413
EXPECTED_ANALYSIS_SHA=636b79d1eff9db42a4c3141438c3c469020363e60ab366171b0fdfa2aa639380
EXPECTED_INPUTS_SHA=401d43723a37465fa00778fd21b27eecbe46cf114abb074a3582b524451ce901
BASE_ROOT=${BASE_ROOT:-/scratch/yz11502/Research/Nav-axis-uturn}
INTERNNAV_ROOT=${INTERNNAV_ROOT:-${BASE_ROOT}/InternNav}
MEMNAV_CKPT=${MEMNAV_CKPT:-${BASE_ROOT}/.diagnostics/unseen_scene_eval_20260803/checkpoints/gatecurr600.memnav.ckpt}
NAVDP_CKPT=${NAVDP_CKPT:-${BASE_ROOT}/.diagnostics/unseen_scene_eval_20260803/checkpoints/navdp_checkpoint.ckpt}
LINGBOT_REPO=${LINGBOT_REPO:-/scratch/lg154/Research/Nav/NavDP/baselines/memnav/lingbot-map}
LINGBOT_WEIGHTS=${LINGBOT_WEIGHTS:-${LINGBOT_REPO}/weights/lingbot-map-long.pt}
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
            "${PROTOCOL}" "${ANALYSIS}" "${INPUTS}" "${MEMNAV_CKPT}" \
            "${NAVDP_CKPT}" "${LINGBOT_WEIGHTS}"; do
  [[ -r "${item}" ]] || { echo "missing ${item}" >&2; exit 1; }
done
[[ "$(sha256sum "${SOURCE_RECEIPT}" | awk '{print $1}')" == "${EXPECTED_SOURCE_RECEIPT_SHA}" ]] || { echo "source receipt SHA changed" >&2; exit 1; }
(cd "${ROOT}" && sha256sum -c --quiet "${SOURCE_RECEIPT}")
[[ "$(sha256sum "${MANIFEST}" | awk '{print $1}')" == "${EXPECTED_MANIFEST_SHA}" ]] || exit 1
[[ "$(sha256sum "${PROTOCOL}" | awk '{print $1}')" == "${EXPECTED_PROTOCOL_SHA}" ]] || exit 1
[[ "$(sha256sum "${ANALYSIS}" | awk '{print $1}')" == "${EXPECTED_ANALYSIS_SHA}" ]] || exit 1
[[ "$(sha256sum "${INPUTS}" | awk '{print $1}')" == "${EXPECTED_INPUTS_SHA}" ]] || exit 1
[[ "$(sha256sum "${MEMNAV_CKPT}" | awk '{print $1}')" == "${EXPECTED_MEMNAV_SHA}" ]] || exit 1
[[ "$(sha256sum "${NAVDP_CKPT}" | awk '{print $1}')" == "${EXPECTED_NAVDP_SHA}" ]] || exit 1
[[ "$(sha256sum "${LINGBOT_WEIGHTS}" | awk '{print $1}')" == "${EXPECTED_LINGBOT_SHA}" ]] || exit 1

HAB_SITE=$(${HAB_PY} -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
HAB_PYTHONPATH=${HAB_SITE}/pip/_vendor
scene=$(${HAB_PY} - "${MANIFEST}" "${SCENE_INDEX}" <<'PY'
import json,sys
print(json.load(open(sys.argv[1]))["selection"]["selected_scenes"][int(sys.argv[2])])
PY
)
episode_root=$(${HAB_PY} - "${MANIFEST}" "${scene}" <<'PY'
import json,sys
m=json.load(open(sys.argv[1])); s=sys.argv[2]
key="legacy_anchor_episode_root" if s in m["selection"]["anchor_scenes"] else "expanded_episode_root"
print(m["paths"][key])
PY
)
asset_root=$(${HAB_PY} - "${MANIFEST}" <<'PY'
import json,sys
print(json.load(open(sys.argv[1]))["paths"]["asset_root"])
PY
)
mapfile -t episodes < <(${HAB_PY} - "${MANIFEST}" "${scene}" <<'PY'
import json,sys
print(*(x["episode"] for x in json.load(open(sys.argv[1]))["episodes"][sys.argv[2]]),sep="\n")
PY
)
episode_csv=$(IFS=,; echo "${episodes[*]}")
scene_root=${RUN_ROOT}/scenes/$(printf '%02d' "${SCENE_INDEX}")_${scene}
[[ ! -e "${scene_root}" ]] || { echo "output exists: ${scene_root}" >&2; exit 1; }
mkdir -p "${scene_root}/logs" "${scene_root}/buffer"
exec > >(tee "${scene_root}/run.log") 2>&1

(cd "${ROOT}" && PYTHONDONTWRITEBYTECODE=1 "${MEMNAV_PY}" -m unittest \
  MemNavData.test_mdtec_raw_depth_gate_d \
  MemNavData.test_monocular_depth_runtime \
  MemNavData.test_navdp_memory_replay)

port_key=$(( (${SLURM_JOB_ID:-1000} + SCENE_INDEX * 43) % 14000 ))
MEMNAV_PORT=${MEMNAV_PORT:-$((21000 + port_key))}
NAVDP_PORT=${NAVDP_PORT:-$((MEMNAV_PORT + 1))}
for port in "${MEMNAV_PORT}" "${NAVDP_PORT}"; do
  if ss -ltn | awk '{print $4}' | grep -Eq "(^|:)${port}$"; then
    echo "port ${port} in use" >&2; exit 1
  fi
done
runtime_root=${SLURM_TMPDIR:-/tmp}/mdtec_gate_d_${SLURM_JOB_ID:-local}_${SCENE_INDEX}
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
    PYTHONPATH="${ROOT}:${INTERNNAV_ROOT}/src/diffusion-policy" \
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

smoke_env=()
[[ "${SMOKE}" -eq 1 ]] && smoke_env+=(MDTEC_GATE_D_SMOKE=1)
env MDTEC_GATE_D_PROTOCOL="${PROTOCOL}" MDTEC_GATE_D_MANIFEST="${MANIFEST}" \
  MDTEC_GATE_D_INPUTS="${INPUTS}" MDTEC_GATE_D_SCENE_INDEX="${SCENE_INDEX}" \
  "${smoke_env[@]}" PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="${HAB_PYTHONPATH}" \
  "${HAB_PY}" -u "${ROOT}/MemNavData/eval_mdtec_raw_depth_gate_d_habitat.py" \
    --episode_root "${episode_root}/${scene}" \
    --scene "${asset_root}/${scene}/${scene}.glb" \
    --host 127.0.0.1 --port "${MEMNAV_PORT}" --novel_port "${NAVDP_PORT}" \
    --out "${scene_root}" --server_backend hybrid_pose --leg1_mode policy \
    --stop_after_leg1 --success_dist 1.0 --max_steps "${MAX_STEPS}" \
    --exec_horizon 8 --trajectory_selector server --seed 20260803 \
    --terminal_uturn off --terminal_visual_refine off \
    --episode_ids "${episode_csv}" --deterministic_plan_seeds \
    --navdp_depth_source metric_request \
    >"${scene_root}/logs/evaluator.log" 2>&1

kill -0 "${MEMNAV_PID}" && kill -0 "${NAVDP_PID}"
[[ "$(awk '{print $22}' "/proc/${MEMNAV_PID}/stat")" == "${mem_start}" ]] || exit 1
[[ "$(awk '{print $22}' "/proc/${NAVDP_PID}/stat")" == "${nav_start}" ]] || exit 1
rows=$(( $(wc -l <"${scene_root}/depth_arms.csv") - 1 ))
[[ "${rows}" -eq 6 ]] || { echo "expected six rows, got ${rows}" >&2; exit 1; }
cp "${SOURCE_RECEIPT}" "${scene_root}/source_inputs.sha256"
echo "[complete] scene=${scene} rows=${rows}"
