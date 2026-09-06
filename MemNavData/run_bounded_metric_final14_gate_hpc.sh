#!/usr/bin/env bash
# One-node, one-history, two-arm Final14 integration gate.
set -euo pipefail
umask 0022

TASK_ROOT=${TASK_ROOT:?set immutable bounded-metric overlay}
BASE_SOURCE_ROOT=${BASE_SOURCE_ROOT:?set recent receipt-bound source bundle}
FALLBACK_SOURCE_ROOT=${FALLBACK_SOURCE_ROOT:?set verified full Final14 source}
RUN_ROOT=${RUN_ROOT:?set new run root}
BENCH_ROOT=${BENCH_ROOT:?set sealed Final14 natural benchmark}
HISTORY_INDEX=${HISTORY_INDEX:?set one consumed history index}
HAB_PY=${HAB_PY:?set Habitat Python}
MEMNAV_PY=${MEMNAV_PY:?set MemNav Python}
TASK_RECEIPT=${TASK_RECEIPT:-${TASK_ROOT}/SOURCE_BUNDLE.sha256}
BASE_RECEIPT=${BASE_RECEIPT:-${BASE_SOURCE_ROOT}/SOURCE_BUNDLE.sha256}
FALLBACK_RECEIPT=${FALLBACK_RECEIPT:-${FALLBACK_SOURCE_ROOT}/source_inputs.sha256}
EXPECTED_MANIFEST_SHA=${EXPECTED_MANIFEST_SHA:-7468703a9efbb10e801ffdd226911f696a30fa9432ef9ab486d3134f6e40fe6a}
MAX_STEPS=${MAX_STEPS:-16}

BASE_ROOT=/scratch/yz11502/Research/Nav-axis-uturn
INTERNNAV_ROOT=${FALLBACK_SOURCE_ROOT}/InternNav
MEMNAV_CKPT=${BASE_ROOT}/.diagnostics/unseen_scene_eval_20260803/checkpoints/gatecurr600.memnav.ckpt
NAVDP_CKPT=${BASE_ROOT}/.diagnostics/unseen_scene_eval_20260803/checkpoints/navdp_checkpoint.ckpt
LINGBOT_REPO=/scratch/lg154/Research/Nav/NavDP/baselines/memnav/lingbot-map
LINGBOT_WEIGHTS=${LINGBOT_REPO}/weights/lingbot-map-long.pt
LIGHTGLUE_REPO=${FALLBACK_SOURCE_ROOT}/third_party/LightGlue
DEPENDENCY_ROOT=${FALLBACK_SOURCE_ROOT}/third_party/python
TORCH_HOME=${FALLBACK_SOURCE_ROOT}/torch_home

[[ "${HISTORY_INDEX}" =~ ^[0-9]+$ ]] || { echo "bad history index" >&2; exit 2; }
[[ "${MAX_STEPS}" =~ ^[1-9][0-9]*$ ]] || { echo "bad MAX_STEPS" >&2; exit 2; }
for path in \
  "${TASK_RECEIPT}" "${BASE_RECEIPT}" "${FALLBACK_RECEIPT}" \
  "${BENCH_ROOT}/manifest.json" "${MEMNAV_CKPT}" "${NAVDP_CKPT}" \
  "${LINGBOT_WEIGHTS}" "${LIGHTGLUE_REPO}" "${DEPENDENCY_ROOT}" \
  "${TASK_ROOT}/NavDP/baselines/memnav/memnav_server.py" \
  "${TASK_ROOT}/NavDP/baselines/memnav/policy_agent.py" \
  "${TASK_ROOT}/MemNavData/eval_shared_online_role_pairs.py" \
  "${TASK_ROOT}/MemNavData/eval_2leg_habitat.py" \
  "${TASK_ROOT}/MemNavData/run_bounded_metric_final14_gate.py"; do
  [[ -e "${path}" ]] || { echo "missing ${path}" >&2; exit 2; }
done
(cd "${TASK_ROOT}" && sha256sum -c --quiet "${TASK_RECEIPT}")
(cd "${BASE_SOURCE_ROOT}" && sha256sum -c --quiet "${BASE_RECEIPT}")
(cd "${FALLBACK_SOURCE_ROOT}" && sha256sum -c --quiet "${FALLBACK_RECEIPT}")
[[ "$(sha256sum "${BENCH_ROOT}/manifest.json" | awk '{print $1}')" == \
  "${EXPECTED_MANIFEST_SHA}" ]] || { echo "Final14 manifest changed" >&2; exit 2; }

label=$(printf '%03d' "${HISTORY_INDEX}")
task_run=${RUN_ROOT}/tasks/${label}
[[ ! -e "${task_run}" ]] || { echo "task output exists ${task_run}" >&2; exit 2; }
mkdir -p "${task_run}/logs" "${task_run}/buffer"
exec > >(tee "${task_run}/run.log") 2>&1

hab_site=$(${HAB_PY} -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
hab_requests=${hab_site}/pip/_vendor
common_roots=${TASK_ROOT}:${TASK_ROOT}/MemNavData:${BASE_SOURCE_ROOT}:${BASE_SOURCE_ROOT}/MemNavData:${FALLBACK_SOURCE_ROOT}:${FALLBACK_SOURCE_ROOT}/MemNavData
memnav_path=${TASK_ROOT}/NavDP/baselines/memnav:${BASE_SOURCE_ROOT}/NavDP/baselines/memnav:${FALLBACK_SOURCE_ROOT}/NavDP/baselines/memnav:${common_roots}:${DEPENDENCY_ROOT}:${LIGHTGLUE_REPO}:${INTERNNAV_ROOT}/src/diffusion-policy
navdp_path=${BASE_SOURCE_ROOT}/NavDP/baselines/navdp:${FALLBACK_SOURCE_ROOT}/NavDP/baselines/navdp:${common_roots}
evaluator_path=${common_roots}:${hab_requests}

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="${common_roots}" \
  "${MEMNAV_PY}" -m pytest -q -p no:cacheprovider \
    "${TASK_ROOT}/MemNavData/test_revisit_bearing_adapter.py" \
    "${TASK_ROOT}/MemNavData/test_first40_local_pose_scale.py"

source "${TASK_ROOT}/MemNavData/slurm_port_pair.sh"
claim_slurm_tcp_port_pair boundedcec 12000 6000
runtime_tmp=${SLURM_TMPDIR:-/tmp}/bounded_cec_${SLURM_JOB_ID}_${HISTORY_INDEX}
mkdir -p "${runtime_tmp}/memnav" "${runtime_tmp}/navdp"
MEMNAV_PID=
NAVDP_PID=
cleanup() {
  for pid in "${NAVDP_PID}" "${MEMNAV_PID}"; do
    if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
      kill "${pid}" 2>/dev/null || true
      wait "${pid}" 2>/dev/null || true
    fi
  done
  release_slurm_tcp_port_pair
}
trap cleanup EXIT INT TERM

(
  cd "${runtime_tmp}/memnav"
  exec env PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    PYTHONPATH="${memnav_path}" TORCH_HOME="${TORCH_HOME}" \
    LINGBOT_REPO="${LINGBOT_REPO}" LINGBOT_WEIGHTS="${LINGBOT_WEIGHTS}" \
    MEMNAV_WINDOW=32 MEMNAV_NUM_SCALE=8 MEMNAV_MAX_FRAME_NUM=2048 \
    MEMNAV_GROUND_SCALE_MAX=6.0 MEMNAV_GATE_FUSION=complementary \
    MEMNAV_AUX_POSE_CALIBRATION=empirical MEMNAV_COLLISION_SELECT=1 \
    MEMNAV_REPORT_TO=none "${MEMNAV_PY}" -u \
      "${TASK_ROOT}/NavDP/baselines/memnav/memnav_server.py" \
      --port "${MEMNAV_PORT}" --checkpoint "${MEMNAV_CKPT}" \
      --internnav_root "${INTERNNAV_ROOT}" --num_samples 16 \
      --exclude_recent 32 --retrieval raw \
      --retrieval_candidate_top_k 32 --retrieval_candidate_min_gap 16 \
      --graph_subgoal_spacing_m 0.0 --graph_subgoal_arrival_m 0.60 \
      --flow_gate auto --buffer_root "${task_run}/buffer" \
      --certified_relocalization --lightglue_repo "${LIGHTGLUE_REPO}" \
      --lightglue_dependency_root "${DEPENDENCY_ROOT}" \
      --lightglue_max_keypoints 2048 --synchronize_cuda_http_handoff
) >"${task_run}/logs/server_memnav.log" 2>&1 &
MEMNAV_PID=$!
(
  cd "${runtime_tmp}/navdp"
  exec env NAVDP_DISABLE_VIDEO=1 PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="${navdp_path}" \
    "${MEMNAV_PY}" -u \
      "${FALLBACK_SOURCE_ROOT}/NavDP/baselines/navdp/navdp_server.py" \
      --port "${NAVDP_PORT}" --checkpoint "${NAVDP_CKPT}" \
      --depth_source metric_request --allow_depth_source_override \
      --monocular_depth_url \
        "http://127.0.0.1:${MEMNAV_PORT}/monocular_depth_query" \
      --require_monocular_depth_transaction
) >"${task_run}/logs/server_navdp.log" 2>&1 &
NAVDP_PID=$!

for spec in \
  "memnav:${MEMNAV_PID}:${MEMNAV_PORT}" \
  "navdp:${NAVDP_PID}:${NAVDP_PORT}"; do
  IFS=: read -r name pid port <<<"${spec}"
  ready=0
  for _ in $(seq 1 240); do
    if ! kill -0 "${pid}" 2>/dev/null; then
      tail -n 160 "${task_run}/logs/server_${name}.log"; exit 2
    fi
    if ss -ltn | awk '{print $4}' | grep -Eq "(^|:)${port}$"; then
      ready=1; break
    fi
    sleep 2
  done
  [[ "${ready}" -eq 1 ]] || { echo "${name} startup timeout" >&2; exit 2; }
done

cat >"${task_run}/server_receipt.json" <<EOF
{
  "same_server_pair": true,
  "memnav_pid": ${MEMNAV_PID},
  "navdp_pid": ${NAVDP_PID},
  "memnav_port": ${MEMNAV_PORT},
  "navdp_port": ${NAVDP_PORT},
  "slurm_job_id": "${SLURM_JOB_ID}",
  "history_index": ${HISTORY_INDEX},
  "gpu": "$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)",
  "gpu_uuid": "$(nvidia-smi --query-gpu=uuid --format=csv,noheader | head -1)"
}
EOF

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="${evaluator_path}" \
  "${HAB_PY}" -u \
    "${TASK_ROOT}/MemNavData/run_bounded_metric_final14_gate.py" \
    --source-root "${TASK_ROOT}" --run-root "${RUN_ROOT}" \
    --bench-root "${BENCH_ROOT}" \
    --expected-manifest-sha256 "${EXPECTED_MANIFEST_SHA}" \
    --history-index "${HISTORY_INDEX}" --hab-python "${HAB_PY}" \
    --memnav-port "${MEMNAV_PORT}" --navdp-port "${NAVDP_PORT}" \
    --max-steps "${MAX_STEPS}" \
    >"${task_run}/logs/evaluator.log" 2>&1

completion=$(find "${RUN_ROOT}/gate" -mindepth 2 -maxdepth 2 \
  -type f -name completion.json | head -1)
[[ -n "${completion}" && -r "${completion}" ]] || {
  echo "completion missing" >&2; exit 2; }
echo "[complete] ${completion}"
