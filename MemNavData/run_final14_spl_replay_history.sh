#!/usr/bin/env bash
set -euo pipefail
: "${REPLAY_ROOT:?}" "${RUN_ROOT:?}" "${HISTORY_INDEX:?}"
source "${REPLAY_ROOT}/MemNavData/final14_spl_replay_env.sh"
[[ "${HISTORY_INDEX}" =~ ^([0-9]|1[0-9]|20)$ ]]
(cd "${REPLAY_ROOT}" && sha256sum -c --quiet source_inputs.sha256)
[[ "$(sha256sum "${FROZEN_BASE}/source_inputs.sha256" | awk '{print $1}')" == "${BASE_RECEIPT_SHA}" ]]
(cd "${FROZEN_BASE}" && sha256sum -c --quiet source_inputs.sha256)
HAB_SITE=$("${HAB_PY}" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
export PYTHONPATH="${REPLAY_ROOT}:${FROZEN_BASE}:${FROZEN_BASE}/MemNavData:${HAB_SITE}/pip/_vendor"
task_root=${RUN_ROOT}/tasks/${HISTORY_INDEX}
[[ ! -e "${task_root}" ]] || { echo "output already exists: ${task_root}" >&2; exit 1; }
mkdir -p "${task_root}/logs"
exec > >(tee "${task_root}/run.log") 2>&1
runtime_root=$(mktemp -d "${SLURM_TMPDIR:-/tmp}/f14spl_${SLURM_JOB_ID}_${HISTORY_INDEX}_XXXXXX")
mkdir -p "${runtime_root}/memnav" "${runtime_root}/navdp" "${runtime_root}/buffer"
source "${REPLAY_ROOT}/MemNavData/slurm_port_pair.sh"
claim_slurm_tcp_port_pair final14_spl_replay
MEMNAV_PID= NAVDP_PID=
cleanup() {
  local result=$?
  trap - EXIT
  for pid in "${MEMNAV_PID}" "${NAVDP_PID}"; do
    if [[ -n "${pid}" ]]; then kill "${pid}" 2>/dev/null || true; fi
  done
  for pid in "${MEMNAV_PID}" "${NAVDP_PID}"; do
    if [[ -n "${pid}" ]]; then wait "${pid}" 2>/dev/null || true; fi
  done
  if tar -C "${runtime_root}" -czf "${task_root}/runtime.tar.gz.partial" memnav navdp buffer; then
    mv "${task_root}/runtime.tar.gz.partial" "${task_root}/runtime.tar.gz"
    (cd "${task_root}" && sha256sum runtime.tar.gz > runtime.tar.gz.sha256)
  else
    echo "runtime archive incomplete; partial retained" >&2
    result=1
  fi
  release_slurm_tcp_port_pair
  exit "${result}"
}
trap cleanup EXIT
trap 'exit 124' USR1 TERM INT
(
  cd "${runtime_root}/memnav"
  exec env PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    PYTHONPATH="${FROZEN_BASE}:${DEPENDENCY_ROOT}:${LIGHTGLUE_REPO}:${INTERNNAV_ROOT}/src/diffusion-policy" \
    MEMNAV_WINDOW=32 MEMNAV_NUM_SCALE=8 MEMNAV_MAX_FRAME_NUM=2048 \
    MEMNAV_GROUND_SCALE_MAX=6.0 MEMNAV_GATE_FUSION=complementary \
    MEMNAV_AUX_POSE_CALIBRATION=empirical MEMNAV_COLLISION_SELECT=1 MEMNAV_REPORT_TO=none \
    "${MEMNAV_PY}" -u "${FROZEN_BASE}/NavDP/baselines/memnav/memnav_server.py" \
      --port "${MEMNAV_PORT}" --checkpoint "${MEMNAV_CKPT}" --internnav_root "${INTERNNAV_ROOT}" \
      --num_samples 16 --exclude_recent 32 --retrieval raw --retrieval_candidate_top_k 32 \
      --retrieval_candidate_min_gap 16 --graph_subgoal_spacing_m 0.0 --graph_subgoal_arrival_m 0.60 \
      --flow_gate auto --certified_relocalization --lightglue_repo "${LIGHTGLUE_REPO}" \
      --buffer_root "${runtime_root}/buffer"
) >"${task_root}/logs/server_memnav.log" 2>&1 &
MEMNAV_PID=$!
(
  cd "${runtime_root}/navdp"
  exec env NAVDP_DISABLE_VIDEO=1 PYTHONUNBUFFERED=1 PYTHONPATH="${FROZEN_BASE}" \
    "${MEMNAV_PY}" -u "${FROZEN_BASE}/NavDP/baselines/navdp/navdp_server.py" \
      --port "${NAVDP_PORT}" --checkpoint "${NAVDP_CKPT}" --depth_source metric_request \
      --allow_depth_source_override \
      --monocular_depth_url "http://127.0.0.1:${MEMNAV_PORT}/monocular_depth_query"
) >"${task_root}/logs/server_navdp.log" 2>&1 &
NAVDP_PID=$!
for spec in "memnav:${MEMNAV_PID}:${MEMNAV_PORT}" "navdp:${NAVDP_PID}:${NAVDP_PORT}"; do
  IFS=: read -r label pid port <<<"${spec}"
  ready=0
  for _ in $(seq 1 240); do
    if ! kill -0 "${pid}" 2>/dev/null; then tail -n 60 "${task_root}/logs/server_${label}.log"; exit 1; fi
    if "${HAB_PY}" -c 'import socket,sys; s=socket.create_connection(("127.0.0.1",int(sys.argv[1])),1); s.close()' "${port}" 2>/dev/null; then ready=1; break; fi
    sleep 2
  done
  [[ "${ready}" == 1 ]] || { echo "${label} startup timeout" >&2; exit 1; }
done
export MEMNAV_PID NAVDP_PID
"${HAB_PY}" - "${task_root}/server_receipt.json" <<'PY'
import json, os, subprocess, sys
from pathlib import Path
receipt = {"same_process_all_arms": True, "history_index": int(os.environ["HISTORY_INDEX"]),
           "job_id": os.environ["SLURM_JOB_ID"],
           "gpu": subprocess.check_output(["nvidia-smi", "--query-gpu=uuid,name", "--format=csv,noheader"], text=True).strip()}
for key in ("MEMNAV_PID", "NAVDP_PID"):
    pid = int(os.environ[key])
    receipt[key] = {"pid": pid, "start_ticks": Path(f"/proc/{pid}/stat").read_text().split()[21]}
Path(sys.argv[1]).write_text(json.dumps(receipt, indent=2) + "\n")
PY
"${HAB_PY}" -u "${REPLAY_ROOT}/MemNavData/final14_spl_replay.py" run \
  --source-root "${FROZEN_BASE}" --run-root "${RUN_ROOT}" --bench-root "${BENCH_ROOT}" \
  --expected-manifest-sha256 "${EXPECTED_MANIFEST_SHA}" --history-index "${HISTORY_INDEX}" \
  --hab-python "${HAB_PY}" --memnav-port "${MEMNAV_PORT}" --navdp-port "${NAVDP_PORT}" \
  --max-steps 600 >"${task_root}/logs/evaluator.log" 2>&1
"${HAB_PY}" - "${task_root}/server_receipt.json" <<'PY'
import json, sys
from pathlib import Path
x=json.loads(Path(sys.argv[1]).read_text())
for key in ("MEMNAV_PID", "NAVDP_PID"):
    assert Path(f"/proc/{x[key]['pid']}/stat").read_text().split()[21] == x[key]["start_ticks"]
PY
"${MEMNAV_PY}" -u "${REPLAY_ROOT}/MemNavData/final14_spl_replay.py" audit \
  --run-root "${RUN_ROOT}" --bench-root "${BENCH_ROOT}" --history-index "${HISTORY_INDEX}" \
  --out "${task_root}/independent_verification.json" >"${task_root}/logs/verification.log" 2>&1
echo "[complete] history=${HISTORY_INDEX}; ten rollouts independently verified"
