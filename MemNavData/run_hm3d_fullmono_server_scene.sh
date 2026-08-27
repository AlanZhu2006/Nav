#!/usr/bin/env bash
# Start one verified LingBot/MemNav + NavDP pair for collection or query eval.
set -euo pipefail
umask 0022

MODE=${MODE:?set collect, smoke, eval, or lifelong_b}
TASK_ROOT=${TASK_ROOT:?set immutable task root}
SERVER_SOURCE_ROOT=${SERVER_SOURCE_ROOT:-${TASK_ROOT}}
BASE_SOURCE_ROOT=${BASE_SOURCE_ROOT:?set verified Final14 mono source root}
RUN_ROOT=${RUN_ROOT:?set isolated run root}
PARENT_MANIFEST=${PARENT_MANIFEST:?set sealed parent HM3D manifest}
PROTOCOL=${PROTOCOL:?set frozen full-mono protocol}
SCENE_INDEX=${SCENE_INDEX:-0}
HAB_PY=${HAB_PY:?set Habitat Python}
MEMNAV_PY=${MEMNAV_PY:?set MemNav Python}
TASK_RECEIPT=${TASK_RECEIPT:?set task bundle receipt}
EXPECTED_TASK_RECEIPT_SHA=${EXPECTED_TASK_RECEIPT_SHA:?set task receipt sha}
BASE_RECEIPT=${BASE_RECEIPT:?set verified base receipt}
EXPECTED_BASE_RECEIPT_SHA=${EXPECTED_BASE_RECEIPT_SHA:?set base receipt sha}
RUNTIME_ATTEMPT=${RUNTIME_ATTEMPT:-}
RESUME_INCOMPLETE=${RESUME_INCOMPLETE:-0}
FORMAL_INDICES_OVERRIDE=${FORMAL_INDICES_OVERRIDE:-}

[[ "${MODE}" == collect || "${MODE}" == smoke || "${MODE}" == eval \
   || "${MODE}" == lifelong_b ]] || {
  echo "invalid MODE=${MODE}" >&2; exit 2; }
[[ "${SCENE_INDEX}" =~ ^[0-9]+$ ]] || { echo "bad scene index" >&2; exit 2; }
[[ "${RESUME_INCOMPLETE}" =~ ^[01]$ ]] || {
  echo "RESUME_INCOMPLETE must be 0 or 1" >&2; exit 2; }
if [[ -n "${RUNTIME_ATTEMPT}" ]]; then
  [[ "${RUNTIME_ATTEMPT}" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]] || {
    echo "invalid runtime attempt" >&2; exit 2; }
fi
if [[ "${MODE}" == lifelong_b ]]; then
  scene_count=$(${MEMNAV_PY} - "${PARENT_MANIFEST}" <<'PY'
import json,sys
print(len(json.load(open(sys.argv[1]))["scenes"]))
PY
  )
else
  scene_count=$(${MEMNAV_PY} - "${PROTOCOL}" <<'PY'
import json,sys
print(len(json.load(open(sys.argv[1]))["dataset"]["scenes"]))
PY
  )
fi
(( SCENE_INDEX >= 0 && SCENE_INDEX < scene_count )) || {
  echo "scene index ${SCENE_INDEX} outside 0..$((scene_count-1))" >&2; exit 2; }
[[ "$(sha256sum "${TASK_RECEIPT}" | awk '{print $1}')" == \
   "${EXPECTED_TASK_RECEIPT_SHA}" ]]
(cd "${TASK_ROOT}" && sha256sum -c --quiet "${TASK_RECEIPT}")
if [[ "${SERVER_SOURCE_ROOT}" != "${TASK_ROOT}" ]]; then
  : "${SERVER_SOURCE_RECEIPT:?set immutable server source receipt}"
  : "${EXPECTED_SERVER_SOURCE_RECEIPT_SHA:?set server source receipt SHA}"
  [[ "$(sha256sum "${SERVER_SOURCE_RECEIPT}" | awk '{print $1}')" == \
     "${EXPECTED_SERVER_SOURCE_RECEIPT_SHA}" ]]
  (cd "${SERVER_SOURCE_ROOT}" && \
    sha256sum -c --quiet "${SERVER_SOURCE_RECEIPT}")
fi
[[ "$(sha256sum "${BASE_RECEIPT}" | awk '{print $1}')" == \
   "${EXPECTED_BASE_RECEIPT_SHA}" ]]
(cd "${BASE_SOURCE_ROOT}" && sha256sum -c --quiet "${BASE_RECEIPT}")

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
for path in "${PARENT_MANIFEST}" "${PROTOCOL}" "${MEMNAV_CKPT}" \
            "${NAVDP_CKPT}" "${LINGBOT_WEIGHTS}"; do
  [[ -r "${path}" ]] || { echo "missing ${path}" >&2; exit 2; }
done
[[ "$(sha256sum "${MEMNAV_CKPT}" | awk '{print $1}')" == "${EXPECTED_MEMNAV_SHA}" ]]
[[ "$(sha256sum "${NAVDP_CKPT}" | awk '{print $1}')" == "${EXPECTED_NAVDP_SHA}" ]]
[[ "$(sha256sum "${LINGBOT_WEIGHTS}" | awk '{print $1}')" == "${EXPECTED_LINGBOT_SHA}" ]]

task_label=${MODE}_${SCENE_INDEX}
if [[ -n "${RUNTIME_ATTEMPT}" ]]; then
  task_label=${task_label}_${RUNTIME_ATTEMPT}
fi
task_run=${RUN_ROOT}/runtime/${task_label}
[[ ! -e "${task_run}" ]] || { echo "runtime output exists ${task_run}" >&2; exit 2; }
mkdir -p "${task_run}/logs" "${task_run}/buffer"
exec > >(tee "${task_run}/run.log") 2>&1

# The selected prefix is sealed only after all pre-query construction tasks.
# Formal array elements outside that prefix complete without loading models.
FORMAL_INDICES=
if [[ "${MODE}" == eval || "${MODE}" == lifelong_b ]]; then
  BENCH_ROOT=${BENCH_ROOT:?set sealed natural-direction benchmark}
  manifest=${BENCH_ROOT}/manifest.json
  FORMAL_INDICES=$(${MEMNAV_PY} - "${manifest}" "${SCENE_INDEX}" <<'PY'
import json,sys
m=json.load(open(sys.argv[1])); rank=int(sys.argv[2])
print(" ".join(str(i) for i,row in enumerate(m["episodes"])
               if int(row["final14_scene_rank"]) == rank))
PY
)
  if [[ -n "${FORMAL_INDICES_OVERRIDE}" ]]; then
    seen_override=" "
    for index in ${FORMAL_INDICES_OVERRIDE}; do
      [[ "${index}" =~ ^[0-9]+$ ]] || {
        echo "invalid formal history override: ${index}" >&2; exit 2; }
      [[ " ${FORMAL_INDICES} " == *" ${index} "* ]] || {
        echo "formal history override escaped scene: ${index}" >&2; exit 2; }
      [[ "${seen_override}" != *" ${index} "* ]] || {
        echo "duplicate formal history override: ${index}" >&2; exit 2; }
      seen_override+="${index} "
    done
    FORMAL_INDICES=${FORMAL_INDICES_OVERRIDE}
  fi
  if [[ -z "${FORMAL_INDICES}" ]]; then
    printf '{"status":"complete","scene_index":%s,"histories":0,"models_loaded":false}\n' \
      "${SCENE_INDEX}" >"${task_run}/empty_scene_receipt.json"
    echo "[complete] mode=${MODE} scene_index=${SCENE_INDEX} histories=0"
    exit 0
  fi
fi

HAB_SITE=$(${HAB_PY} -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
HAB_PYTHONPATH=${HAB_SITE}/pip/_vendor
PYTHONPATH_VALUE=${TASK_ROOT}:${TASK_ROOT}/MemNavData:${SERVER_SOURCE_ROOT}:${SERVER_SOURCE_ROOT}/MemNavData:${BASE_SOURCE_ROOT}:${BASE_SOURCE_ROOT}/MemNavData:${DEPENDENCY_ROOT}:${LIGHTGLUE_REPO}:${INTERNNAV_ROOT}/src/diffusion-policy:${HAB_PYTHONPATH}
REQUESTS_INIT=${HAB_PYTHONPATH}/requests/__init__.py
REQUESTS_VERSION=${HAB_PYTHONPATH}/requests/__version__.py
[[ -r "${REQUESTS_INIT}" && -r "${REQUESTS_VERSION}" ]] || {
  echo "missing Habitat vendored requests dependency" >&2; exit 2; }
if [[ -n "${EXPECTED_HAB_REQUESTS_VERSION:-}" ]]; then
  : "${EXPECTED_HAB_REQUESTS_INIT_BYTES:?}" \
    "${EXPECTED_HAB_REQUESTS_INIT_SHA:?}" \
    "${EXPECTED_HAB_REQUESTS_VERSION_BYTES:?}" \
    "${EXPECTED_HAB_REQUESTS_VERSION_SHA:?}"
  [[ "$(stat -c '%s' "${REQUESTS_INIT}")" == \
     "${EXPECTED_HAB_REQUESTS_INIT_BYTES}" ]]
  [[ "$(sha256sum "${REQUESTS_INIT}" | awk '{print $1}')" == \
     "${EXPECTED_HAB_REQUESTS_INIT_SHA}" ]]
  [[ "$(stat -c '%s' "${REQUESTS_VERSION}")" == \
     "${EXPECTED_HAB_REQUESTS_VERSION_BYTES}" ]]
  [[ "$(sha256sum "${REQUESTS_VERSION}" | awk '{print $1}')" == \
     "${EXPECTED_HAB_REQUESTS_VERSION_SHA}" ]]
  env PYTHONPATH="${HAB_PYTHONPATH}" "${HAB_PY}" -c \
    'import requests,sys; assert requests.__version__ == sys.argv[1]; assert "/pip/_vendor/requests/" in requests.__file__' \
    "${EXPECTED_HAB_REQUESTS_VERSION}"
else
  env PYTHONPATH="${HAB_PYTHONPATH}" "${HAB_PY}" -c \
    'import requests; assert "/pip/_vendor/requests/" in requests.__file__'
fi

port_key=$(( (${SLURM_JOB_ID:-1000} + SCENE_INDEX * 47) % 14000 ))
MEMNAV_PORT=${MEMNAV_PORT:-$((25000 + port_key))}
NAVDP_PORT=${NAVDP_PORT:-$((MEMNAV_PORT + 1))}
for port in "${MEMNAV_PORT}" "${NAVDP_PORT}"; do
  ! ss -ltn | awk '{print $4}' | grep -Eq "(^|:)${port}$" || {
    echo "port ${port} in use" >&2; exit 2; }
done
runtime_tmp=${SLURM_TMPDIR:-/tmp}/h3fullmono_${SLURM_JOB_ID:-local}_${task_label}
mkdir -p "${runtime_tmp}/memnav" "${runtime_tmp}/navdp"
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
  cd "${runtime_tmp}/memnav"
  exec env PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    PYTHONPATH="${PYTHONPATH_VALUE}" TORCH_HOME="${TORCH_HOME}" \
    LINGBOT_REPO="${LINGBOT_REPO}" LINGBOT_WEIGHTS="${LINGBOT_WEIGHTS}" \
    MEMNAV_WINDOW=32 MEMNAV_NUM_SCALE=8 MEMNAV_MAX_FRAME_NUM=2048 \
    MEMNAV_GROUND_SCALE_MAX=6.0 MEMNAV_GATE_FUSION=complementary \
    MEMNAV_AUX_POSE_CALIBRATION=empirical MEMNAV_COLLISION_SELECT=1 \
    MEMNAV_REPORT_TO=none "${MEMNAV_PY}" -u \
    "${SERVER_SOURCE_ROOT}/NavDP/baselines/memnav/memnav_server.py" \
      --port "${MEMNAV_PORT}" --checkpoint "${MEMNAV_CKPT}" \
      --internnav_root "${INTERNNAV_ROOT}" --num_samples 16 \
      --exclude_recent 32 --retrieval raw --retrieval_candidate_top_k 32 \
      --retrieval_candidate_min_gap 16 --graph_subgoal_spacing_m 0.0 \
      --graph_subgoal_arrival_m 0.60 --flow_gate auto \
      --certified_relocalization --lightglue_repo "${LIGHTGLUE_REPO}" \
      --buffer_root "${task_run}/buffer"
) >"${task_run}/logs/server_memnav.log" 2>&1 &
MEMNAV_PID=$!
(
  cd "${runtime_tmp}/navdp"
  exec env NAVDP_DISABLE_VIDEO=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH="${PYTHONPATH_VALUE}" "${MEMNAV_PY}" -u \
    "${SERVER_SOURCE_ROOT}/NavDP/baselines/navdp/navdp_server.py" \
      --port "${NAVDP_PORT}" --checkpoint "${NAVDP_CKPT}" \
      --depth_source metric_request --allow_depth_source_override \
      --monocular_depth_url "http://127.0.0.1:${MEMNAV_PORT}/monocular_depth_query" \
      --require_monocular_depth_transaction
) >"${task_run}/logs/server_navdp.log" 2>&1 &
NAVDP_PID=$!
for spec in "memnav:${MEMNAV_PID}:${MEMNAV_PORT}" "navdp:${NAVDP_PID}:${NAVDP_PORT}"; do
  IFS=: read -r label pid port <<<"${spec}"; ready=0
  for _ in $(seq 1 240); do
    kill -0 "${pid}" 2>/dev/null || {
      tail -n 120 "${task_run}/logs/server_${label}.log"; exit 2; }
    if ss -ltn | awk '{print $4}' | grep -Eq "(^|:)${port}$"; then
      ready=1; break
    fi
    sleep 2
  done
  [[ "${ready}" -eq 1 ]] || { echo "${label} startup timeout" >&2; exit 2; }
done

echo "mode=${MODE} scene_index=${SCENE_INDEX} host=$(hostname)"
nvidia-smi --query-gpu=name,uuid,memory.total --format=csv,noheader
if [[ "${MODE}" == collect ]]; then
  collector=("${TASK_ROOT}/MemNavData/collect_hm3d_fullmono_goal_a.py"
      --source-root "${BASE_SOURCE_ROOT}" --run-root "${RUN_ROOT}" \
      --evaluator-source-root "${TASK_ROOT}" \
      --protocol "${PROTOCOL}" --parent-manifest "${PARENT_MANIFEST}" \
      --scene-index "${SCENE_INDEX}" --hab-python "${HAB_PY}" \
      --memnav-port "${MEMNAV_PORT}" --navdp-port "${NAVDP_PORT}")
  if [[ "${RESUME_INCOMPLETE}" == 1 ]]; then
    collector+=(--resume-incomplete --repair-tag "${RUNTIME_ATTEMPT}")
  fi
  PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="${PYTHONPATH_VALUE}" \
    "${HAB_PY}" -u "${collector[@]}" \
      >"${task_run}/logs/collector.log" 2>&1
elif [[ "${MODE}" == lifelong_b ]]; then
  BENCH_ROOT=${BENCH_ROOT:?set sealed lifelong A/B role-pair benchmark}
  manifest=${BENCH_ROOT}/manifest.json
  expected_manifest_sha=$(sha256sum "${manifest}" | awk '{print $1}')
  MAX_STEPS=${MAX_STEPS:-600}
  [[ "${MAX_STEPS}" -eq 600 ]] || { echo "formal B max steps changed" >&2; exit 2; }
  for history_index in ${FORMAL_INDICES}; do
    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="${PYTHONPATH_VALUE}" \
      "${HAB_PY}" -u \
      "${TASK_ROOT}/MemNavData/collect_hm3d_fullmono_lifelong_b.py" \
        --source-root "${SERVER_SOURCE_ROOT}" --run-root "${RUN_ROOT}" \
        --protocol "${PROTOCOL}" --bench-root "${BENCH_ROOT}" \
        --expected-manifest-sha256 "${expected_manifest_sha}" \
        --history-index "${history_index}" --hab-python "${HAB_PY}" \
        --memnav-port "${MEMNAV_PORT}" --navdp-port "${NAVDP_PORT}" \
        --max-steps "${MAX_STEPS}" \
        >"${task_run}/logs/factual_b_${history_index}.log" 2>&1
  done
else
  BENCH_ROOT=${BENCH_ROOT:?set sealed natural-direction benchmark}
  manifest=${BENCH_ROOT}/manifest.json
  expected_manifest_sha=$(sha256sum "${manifest}" | awk '{print $1}')
  if [[ "${MODE}" == smoke ]]; then
    indices=0
    MAX_STEPS=${MAX_STEPS:-80}
  else
    MAX_STEPS=${MAX_STEPS:-600}
    [[ "${MAX_STEPS}" -eq 600 ]] || { echo "formal max steps changed" >&2; exit 2; }
    indices=${FORMAL_INDICES}
  fi
  if [[ -z "${indices}" ]]; then
    printf '{"status":"complete","scene_index":%s,"histories":0}\n' \
      "${SCENE_INDEX}" >"${task_run}/empty_scene_receipt.json"
  else
    for history_index in ${indices}; do
      runner=("${TASK_ROOT}/MemNavData/run_hm3d_fullmono_query_history.py"
        --source-root "${BASE_SOURCE_ROOT}" --run-root "${RUN_ROOT}"
        --evaluator-source-root "${TASK_ROOT}"
        --bench-root "${BENCH_ROOT}"
        --expected-manifest-sha256 "${expected_manifest_sha}"
        --history-index "${history_index}" --hab-python "${HAB_PY}"
        --memnav-port "${MEMNAV_PORT}" --navdp-port "${NAVDP_PORT}"
        --max-steps "${MAX_STEPS}")
      if [[ "${MODE}" == smoke ]]; then runner+=(--smoke); fi
      PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="${PYTHONPATH_VALUE}" \
        "${HAB_PY}" -u "${runner[@]}" \
        >"${task_run}/logs/query_${history_index}.log" 2>&1
    done
  fi
fi

kill -0 "${MEMNAV_PID}"
kill -0 "${NAVDP_PID}"
echo "[complete] mode=${MODE} scene_index=${SCENE_INDEX}"
