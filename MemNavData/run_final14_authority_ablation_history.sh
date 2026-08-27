#!/usr/bin/env bash
# One Final14 history, two paired authority arms, one persistent server pair.
set -euo pipefail
umask 0022

REPAIR_ROOT=${REPAIR_ROOT:?}
REPAIR_RECEIPT=${REPAIR_RECEIPT:?}
EXPECTED_REPAIR_RECEIPT_SHA=${EXPECTED_REPAIR_RECEIPT_SHA:?}
BASE_SOURCE_ROOT=${BASE_SOURCE_ROOT:?}
BASE_SOURCE_RECEIPT=${BASE_SOURCE_RECEIPT:?}
EXPECTED_BASE_SOURCE_RECEIPT_SHA=${EXPECTED_BASE_SOURCE_RECEIPT_SHA:?}
RUN_ROOT=${RUN_ROOT:?}
BENCH_ROOT=${BENCH_ROOT:?}
HISTORY_INDEX=${HISTORY_INDEX:?}
HAB_PY=${HAB_PY:?}
MEMNAV_PY=${MEMNAV_PY:?}
PROTOCOL=${PROTOCOL:-${REPAIR_ROOT}/MemNavData/final14_cec_authority_ablation_protocol_20260828.json}
EXPECTED_MANIFEST_SHA=7468703a9efbb10e801ffdd226911f696a30fa9432ef9ab486d3134f6e40fe6a
EXPECTED_PROTOCOL_SHA=d49a2cfb03c869398d1daef56107a2ab0a8679e3174ffd55ba37f4efcf8089f1
BASE_ROOT=${BASE_ROOT:-/scratch/yz11502/Research/Nav-axis-uturn}
INTERNNAV_ROOT=${INTERNNAV_ROOT:-${BASE_ROOT}/InternNav}
MEMNAV_CKPT=${MEMNAV_CKPT:-${BASE_ROOT}/.diagnostics/unseen_scene_eval_20260803/checkpoints/gatecurr600.memnav.ckpt}
NAVDP_CKPT=${NAVDP_CKPT:-${BASE_ROOT}/.diagnostics/unseen_scene_eval_20260803/checkpoints/navdp_checkpoint.ckpt}
LINGBOT_REPO=${LINGBOT_REPO:-/scratch/lg154/Research/Nav/NavDP/baselines/memnav/lingbot-map}
LINGBOT_WEIGHTS=${LINGBOT_WEIGHTS:-${LINGBOT_REPO}/weights/lingbot-map-long.pt}
LIGHTGLUE_REPO=${LIGHTGLUE_REPO:-${BASE_SOURCE_ROOT}/third_party/LightGlue}
DEPENDENCY_ROOT=${DEPENDENCY_ROOT:-${BASE_SOURCE_ROOT}/third_party/python}
TORCH_HOME=${TORCH_HOME:-${BASE_SOURCE_ROOT}/torch_home}
EXPECTED_MEMNAV_SHA=9b7a5811ff0aea212503f58b45258ba4f66b06420f87c350946aead39db6fdb7
EXPECTED_NAVDP_SHA=3bb3ad4ab241e857bb57a4021cc6aab76d5263e81fbf80298d579053ef011947
EXPECTED_LINGBOT_SHA=832bc82cbae0bc9bbe946ef5ee1f7226abd8c0e183ccf8beddbb3d133576f409
SMOKE=${SMOKE:-0}
MAX_STEPS=${MAX_STEPS:-600}

[[ "${HISTORY_INDEX}" =~ ^([0-9]|1[0-9]|20)$ ]]
[[ "${SMOKE}" =~ ^[01]$ ]]
if [[ "${SMOKE}" == 0 ]]; then [[ "${MAX_STEPS}" == 600 ]]; fi
[[ "$(sha256sum "${REPAIR_RECEIPT}" | awk '{print $1}')" == \
   "${EXPECTED_REPAIR_RECEIPT_SHA}" ]]
(cd "${REPAIR_ROOT}" && sha256sum -c --quiet "${REPAIR_RECEIPT}")
[[ "$(sha256sum "${BASE_SOURCE_RECEIPT}" | awk '{print $1}')" == \
   "${EXPECTED_BASE_SOURCE_RECEIPT_SHA}" ]]
(cd "${BASE_SOURCE_ROOT}" && sha256sum -c --quiet "${BASE_SOURCE_RECEIPT}")
[[ "$(sha256sum "${BENCH_ROOT}/manifest.json" | awk '{print $1}')" == \
   "${EXPECTED_MANIFEST_SHA}" ]]
[[ "$(sha256sum "${PROTOCOL}" | awk '{print $1}')" == \
   "${EXPECTED_PROTOCOL_SHA}" ]]
[[ "$(sha256sum "${MEMNAV_CKPT}" | awk '{print $1}')" == "${EXPECTED_MEMNAV_SHA}" ]]
[[ "$(sha256sum "${NAVDP_CKPT}" | awk '{print $1}')" == "${EXPECTED_NAVDP_SHA}" ]]
[[ "$(sha256sum "${LINGBOT_WEIGHTS}" | awk '{print $1}')" == "${EXPECTED_LINGBOT_SHA}" ]]

history_label=$(${HAB_PY} - "${BENCH_ROOT}/manifest.json" \
  "${HISTORY_INDEX}" <<'PY'
import json,sys
x=json.load(open(sys.argv[1]))["episodes"][int(sys.argv[2])]
print(f'{int(sys.argv[2]):03d}_{x["scene"]}_{x["episode"]}')
PY
)
task_root=${RUN_ROOT}/tasks/${history_label}
[[ ! -e "${task_root}" ]]
mkdir -p "${task_root}/logs" "${task_root}/buffer"
exec > >(tee "${task_root}/run.log") 2>&1

PYTHONPATH_VALUE=${REPAIR_ROOT}:${REPAIR_ROOT}/MemNavData:${REPAIR_ROOT}/NavDP/baselines/memnav:${BASE_SOURCE_ROOT}:${BASE_SOURCE_ROOT}/MemNavData:${BASE_SOURCE_ROOT}/NavDP/baselines/memnav:${DEPENDENCY_ROOT}:${LIGHTGLUE_REPO}:${INTERNNAV_ROOT}/src/diffusion-policy
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="${PYTHONPATH_VALUE}" \
  "${MEMNAV_PY}" -m unittest -q \
    MemNavData.test_policy_agent_graph
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="${PYTHONPATH_VALUE}" \
  "${MEMNAV_PY}" - <<'PY'
from MemNavData.certified_relocalization_runtime import runtime_contract
from MemNavData.final14_authority_ablation import AUTHORITY_POLICY
contract = runtime_contract()
assert contract["default_authority_policy"] == "strict_certificate"
assert contract["diagnostic_authority_policies"] == ["pnp_pose_available"]
assert set(AUTHORITY_POLICY.values()) == {
    "strict_certificate", "pnp_pose_available"}
PY

port_key=$(( (${SLURM_JOB_ID:-1000} + HISTORY_INDEX * 43) % 14000 ))
MEMNAV_PORT=${MEMNAV_PORT:-$((25000 + port_key))}
NAVDP_PORT=${NAVDP_PORT:-$((MEMNAV_PORT + 1))}
for port in "${MEMNAV_PORT}" "${NAVDP_PORT}"; do
  ! ss -ltn | awk '{print $4}' | grep -Eq "(^|:)${port}$"
done
runtime_root=${SLURM_TMPDIR:-/tmp}/final14_authority_${SLURM_JOB_ID:-local}_${HISTORY_INDEX}
mkdir -p "${runtime_root}/memnav" "${runtime_root}/navdp"
MEMNAV_PID= NAVDP_PID=
cleanup() {
  for pid in "${NAVDP_PID}" "${MEMNAV_PID}"; do
    if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
      kill "${pid}" 2>/dev/null || true
      wait "${pid}" 2>/dev/null || true
    fi
  done
}
trap cleanup EXIT INT TERM
(
  cd "${runtime_root}/memnav"
  exec env PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    PYTHONPATH="${PYTHONPATH_VALUE}" TORCH_HOME="${TORCH_HOME}" \
    LINGBOT_REPO="${LINGBOT_REPO}" LINGBOT_WEIGHTS="${LINGBOT_WEIGHTS}" \
    MEMNAV_WINDOW=32 MEMNAV_NUM_SCALE=8 MEMNAV_MAX_FRAME_NUM=2048 \
    MEMNAV_GROUND_SCALE_MAX=6.0 MEMNAV_GATE_FUSION=complementary \
    MEMNAV_AUX_POSE_CALIBRATION=empirical MEMNAV_COLLISION_SELECT=1 \
    MEMNAV_REPORT_TO=none "${MEMNAV_PY}" -u \
    "${REPAIR_ROOT}/NavDP/baselines/memnav/memnav_server.py" \
      --port "${MEMNAV_PORT}" --checkpoint "${MEMNAV_CKPT}" \
      --internnav_root "${INTERNNAV_ROOT}" --num_samples 16 \
      --exclude_recent 32 --retrieval raw --retrieval_candidate_top_k 32 \
      --retrieval_candidate_min_gap 16 --graph_subgoal_spacing_m 0.0 \
      --graph_subgoal_arrival_m 0.60 --flow_gate auto \
      --certified_relocalization --lightglue_repo "${LIGHTGLUE_REPO}" \
      --buffer_root "${task_root}/buffer"
) >"${task_root}/logs/server_memnav.log" 2>&1 &
MEMNAV_PID=$!
(
  cd "${runtime_root}/navdp"
  exec env NAVDP_DISABLE_VIDEO=1 PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="${PYTHONPATH_VALUE}" \
    "${MEMNAV_PY}" -u \
    "${BASE_SOURCE_ROOT}/NavDP/baselines/navdp/navdp_server.py" \
      --port "${NAVDP_PORT}" --checkpoint "${NAVDP_CKPT}" \
      --depth_source metric_request --allow_depth_source_override \
      --monocular_depth_url "http://127.0.0.1:${MEMNAV_PORT}/monocular_depth_query"
) >"${task_root}/logs/server_navdp.log" 2>&1 &
NAVDP_PID=$!
for spec in "memnav:${MEMNAV_PID}:${MEMNAV_PORT}" \
            "navdp:${NAVDP_PID}:${NAVDP_PORT}"; do
  IFS=: read -r label pid port <<<"${spec}"
  ready=0
  for _ in $(seq 1 240); do
    kill -0 "${pid}" 2>/dev/null || {
      tail -n 120 "${task_root}/logs/server_${label}.log"; exit 2; }
    if ss -ltn | awk '{print $4}' | grep -Eq "(^|:)${port}$"; then
      ready=1; break
    fi
    sleep 2
  done
  [[ "${ready}" == 1 ]]
done

# A listening socket can precede a usable model.  This reset-only probe is
# repeated by the evaluator and consumes no evaluated observation.
curl --fail --silent --show-error --max-time 30 \
  -H 'Content-Type: application/json' \
  -d '{"camera_height":0.5,"seed":0,"episode_len":600}' \
  "http://127.0.0.1:${MEMNAV_PORT}/navigator_reset" \
  >"${task_root}/logs/memnav_semantic_health.json"

gpu_uuid=$(nvidia-smi --query-gpu=uuid --format=csv,noheader | head -1)
mem_start=$(awk '{print $22}' "/proc/${MEMNAV_PID}/stat")
nav_start=$(awk '{print $22}' "/proc/${NAVDP_PID}/stat")
cat >"${task_root}/server_receipt.json" <<EOF
{
  "same_process_all_arms": true,
  "semantic_health_reset_before_evaluation": true,
  "memnav_pid": ${MEMNAV_PID},
  "memnav_process_start_ticks": ${mem_start},
  "memnav_port": ${MEMNAV_PORT},
  "navdp_pid": ${NAVDP_PID},
  "navdp_process_start_ticks": ${nav_start},
  "navdp_port": ${NAVDP_PORT},
  "gpu_uuid": "${gpu_uuid}",
  "slurm_job_id": "${SLURM_JOB_ID:-local}",
  "history_index": ${HISTORY_INDEX},
  "smoke": ${SMOKE}
}
EOF

HAB_SITE=$(${HAB_PY} -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
runner=("${REPAIR_ROOT}/MemNavData/run_final14_authority_ablation_episode.py"
  --source-root "${REPAIR_ROOT}" --run-root "${RUN_ROOT}"
  --bench-root "${BENCH_ROOT}"
  --expected-manifest-sha256 "${EXPECTED_MANIFEST_SHA}"
  --history-index "${HISTORY_INDEX}" --hab-python "${HAB_PY}"
  --memnav-port "${MEMNAV_PORT}" --navdp-port "${NAVDP_PORT}"
  --max-steps "${MAX_STEPS}")
if [[ "${SMOKE}" == 1 ]]; then runner+=(--smoke); fi
PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH="${REPAIR_ROOT}:${REPAIR_ROOT}/MemNavData:${BASE_SOURCE_ROOT}:${BASE_SOURCE_ROOT}/MemNavData:${HAB_SITE}/pip/_vendor" \
  "${HAB_PY}" -u "${runner[@]}" \
  >"${task_root}/logs/evaluator.log" 2>&1

kill -0 "${MEMNAV_PID}" && kill -0 "${NAVDP_PID}"
[[ "$(awk '{print $22}' "/proc/${MEMNAV_PID}/stat")" == "${mem_start}" ]]
[[ "$(awk '{print $22}' "/proc/${NAVDP_PID}/stat")" == "${nav_start}" ]]
completion=${RUN_ROOT}/evaluation/natural_direction/${history_label}/completion.json
[[ -r "${completion}" ]]
cp "${REPAIR_RECEIPT}" "${task_root}/repair_SOURCE_BUNDLE.sha256"
cp "${BASE_SOURCE_RECEIPT}" "${task_root}/base_source_inputs.sha256"
echo "[complete] history=${HISTORY_INDEX} label=${history_label}"
