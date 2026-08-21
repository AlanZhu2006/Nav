#!/usr/bin/env bash
# One consumed Novel episode per task; all four causal-control arms are paired.
#SBATCH --job-name=novel_causal
#SBATCH --output=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/%x_%A_%a.out
#SBATCH --error=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/%x_%A_%a.err
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --partition=h100_tandon,h200_public,a100_tandon
#SBATCH --qos=gpu48
#SBATCH --gres=gpu:1
#SBATCH --account=torch_pr_769_tandon_advanced
#SBATCH --export=ALL

set -euo pipefail
umask 0022
: "${SLURM_ARRAY_TASK_ID:?submit an episode array}"
: "${SOURCE_ROOT:?set immutable source root}"
: "${SOURCE_RECEIPT:?set source receipt}"
: "${EXPECTED_SOURCE_RECEIPT_SHA:?set receipt SHA}"
: "${BASE_SOURCE_ROOT:?set base dependency source root}"
: "${BASE_SOURCE_RECEIPT_SHA:?set base receipt SHA}"
: "${DEPENDENCY_RECEIPT:?set model dependency receipt}"
: "${EXPECTED_DEPENDENCY_RECEIPT_SHA:?set dependency receipt SHA}"
: "${RUN_ROOT:?set Novel-control run root}"
: "${BENCH_ROOT:?set sealed natural-direction benchmark root}"
: "${CONTROL_MANIFEST:?set frozen control manifest}"
: "${EXPECTED_CONTROL_MANIFEST_SHA:?set control manifest SHA}"

HAB_PY=/scratch/lg154/conda-envs/habitat/bin/python
MEMNAV_PY=/scratch/lg154/conda-envs/memnav/bin/python
BASE_SIF=/share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif
MEMNAV_SERVER=${SOURCE_ROOT}/NavDP/baselines/memnav/memnav_server.py
NAVDP_SERVER=${SOURCE_ROOT}/NavDP/baselines/navdp/navdp_server.py
SERVER_LAUNCHER=${SOURCE_ROOT}/MemNavData/retrying_server_launcher.py
RUNNER=${SOURCE_ROOT}/MemNavData/run_novel_memory_direction_control_episode.sh
INTERNNAV_ROOT=${BASE_SOURCE_ROOT}/InternNav
LINGBOT_REPO=/scratch/lg154/Research/Nav/NavDP/baselines/memnav/lingbot-map
LIGHTGLUE_REPO=${BASE_SOURCE_ROOT}/third_party/LightGlue
DEPENDENCY_ROOT=${BASE_SOURCE_ROOT}/third_party/python
TORCH_HOME=${BASE_SOURCE_ROOT}/torch_home
BASE_RECEIPT=${BASE_SOURCE_ROOT}/SOURCE_BUNDLE.sha256
fail() { echo "ABORT: $*" >&2; exit 2; }

[[ "${SLURM_ARRAY_TASK_ID}" =~ ^[0-9]+$ ]] || fail "bad array index"
[[ "$(sha256sum "${SOURCE_RECEIPT}" | awk '{print $1}')" == \
   "${EXPECTED_SOURCE_RECEIPT_SHA}" ]] || fail "source receipt changed"
(cd "${SOURCE_ROOT}" && sha256sum -c "${SOURCE_RECEIPT}") >/dev/null || \
  fail "source bundle changed"
[[ "$(sha256sum "${BASE_RECEIPT}" | awk '{print $1}')" == \
   "${BASE_SOURCE_RECEIPT_SHA}" ]] || fail "base source receipt changed"
[[ "$(sha256sum "${DEPENDENCY_RECEIPT}" | awk '{print $1}')" == \
   "${EXPECTED_DEPENDENCY_RECEIPT_SHA}" ]] || fail "dependency receipt changed"
[[ "$(sha256sum "${CONTROL_MANIFEST}" | awk '{print $1}')" == \
   "${EXPECTED_CONTROL_MANIFEST_SHA}" ]] || fail "control manifest changed"
[[ -f "${BASE_SIF}" && -x "${HAB_PY}" && -x "${MEMNAV_PY}" ]] || \
  fail "runtime dependency missing"

accepted_count=$("${MEMNAV_PY}" - "${CONTROL_MANIFEST}" \
  "${BENCH_ROOT}/manifest.json" <<'PY'
import hashlib,json,sys
from pathlib import Path
def sha(path):
    digest=hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda:handle.read(8<<20),b""):
            digest.update(chunk)
    return digest.hexdigest()
control=json.load(open(sys.argv[1]))
if control.get("evaluation_stage") != "consumed_development_mechanism_only":
    raise SystemExit("wrong control stage")
if control.get("confirmation_claim_allowed") is not False:
    raise SystemExit("control manifest claims confirmation")
if sha(sys.argv[2]) != control["benchmark_manifest_sha256"]:
    raise SystemExit("benchmark manifest changed")
print(len(control["episodes"]))
PY
)
if (( SLURM_ARRAY_TASK_ID >= accepted_count )); then
  mkdir -p "${RUN_ROOT}/evaluation/skipped"
  printf 'population=%s\narray_index=%s\n' \
    "${accepted_count}" "${SLURM_ARRAY_TASK_ID}" \
    > "${RUN_ROOT}/evaluation/skipped/$(printf '%03d' "${SLURM_ARRAY_TASK_ID}").txt"
  echo "SKIP index=${SLURM_ARRAY_TASK_ID} population=${accepted_count}"
  exit 0
fi

readarray -t dependencies < <("${MEMNAV_PY}" - "${DEPENDENCY_RECEIPT}" <<'PY'
import json,os,sys
receipt=json.load(open(sys.argv[1]))
for name in ("gatecurr600","navdp_checkpoint","lingbot_map_long"):
    row=receipt["dependencies"][name]
    if os.stat(row["path"]).st_size != int(row["bytes"]):
        raise SystemExit(f"dependency size changed: {name}")
    print(row["path"])
PY
)
[[ "${#dependencies[@]}" -eq 3 ]] || fail "dependency receipt invalid"
MEMNAV_CKPT=${dependencies[0]}; NAVDP_CKPT=${dependencies[1]}
LINGBOT_WEIGHTS=${dependencies[2]}
actual_lingbot_commit=$(git -c safe.directory="${LINGBOT_REPO}" \
  -C "${LINGBOT_REPO}" rev-parse HEAD)
[[ "${actual_lingbot_commit}" == 7ff6f3ed0913d4d326f8f13bbb429c4ffc0195c2 ]] || \
  fail "LingBot source commit changed"

echo "host=$(hostname) job=${SLURM_JOB_ID} task=${SLURM_ARRAY_TASK_ID} episode_index=${SLURM_ARRAY_TASK_ID}"
nvidia-smi --query-gpu=name,uuid,driver_version,memory.total \
  --format=csv,noheader

export PYTHONPYCACHEPREFIX=${SLURM_TMPDIR:-/tmp}/novel_causal_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}
export PYTHONPATH=${SOURCE_ROOT}:${SOURCE_ROOT}/MemNavData:${DEPENDENCY_ROOT}:${LIGHTGLUE_REPO}:${BASE_SOURCE_ROOT}/NavDP/baselines/navdp:${BASE_SOURCE_ROOT}/NavDP/baselines/memnav:${INTERNNAV_ROOT}/src/diffusion-policy${PYTHONPATH:+:${PYTHONPATH}}
export TORCH_HOME
mkdir -p "${PYTHONPYCACHEPREFIX}" "${RUN_ROOT}/logs" "${RUN_ROOT}/buffer"
"${MEMNAV_PY}" -m py_compile "${MEMNAV_SERVER}" "${NAVDP_SERVER}" \
  "${SOURCE_ROOT}/MemNavData/eval_novel_memory_direction_control.py" \
  "${SOURCE_ROOT}/MemNavData/novel_memory_direction_control.py" \
  "${SERVER_LAUNCHER}"

port_key=$(( (SLURM_JOB_ID + SLURM_ARRAY_TASK_ID * 59) % 15000 ))
MEMNAV_BASE_PORT=$((20000 + port_key * 2))
NAVDP_BASE_PORT=$((MEMNAV_BASE_PORT + 1))
runtime_root=${SLURM_TMPDIR:-/tmp}/novel_causal_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}
mkdir -p "${runtime_root}/memnav" "${runtime_root}/navdp"
label=$(printf '%03d' "${SLURM_ARRAY_TASK_ID}")
MEMNAV_PORT_FILE=${runtime_root}/memnav.port
NAVDP_PORT_FILE=${runtime_root}/navdp.port
MEMNAV_LAUNCHER_LOG=${RUN_ROOT}/logs/server_memnav_${label}.log
NAVDP_LAUNCHER_LOG=${RUN_ROOT}/logs/server_navdp_${label}.log
MEMNAV_PID=; NAVDP_PID=
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
  exec env PYTHONUNBUFFERED=1 PYTHONPYCACHEPREFIX="${PYTHONPYCACHEPREFIX}" \
    PYTHONPATH="${PYTHONPATH}" TORCH_HOME="${TORCH_HOME}" \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    LINGBOT_REPO="${LINGBOT_REPO}" LINGBOT_WEIGHTS="${LINGBOT_WEIGHTS}" \
    MEMNAV_WINDOW=32 MEMNAV_NUM_SCALE=8 MEMNAV_MAX_FRAME_NUM=2048 \
    MEMNAV_GROUND_SCALE_MAX=6.0 MEMNAV_GATE_FUSION=complementary \
    MEMNAV_AUX_POSE_CALIBRATION=empirical MEMNAV_COLLISION_SELECT=1 \
    MEMNAV_REPORT_TO=none \
    "${MEMNAV_PY}" -u "${SERVER_LAUNCHER}" \
      --base-port "${MEMNAV_BASE_PORT}" --range-start 20000 --range-end 49999 \
      --stride 1994 --max-attempts 12 --ready-timeout 720 \
      --port-file "${MEMNAV_PORT_FILE}" \
      --receipt-file "${RUN_ROOT}/logs/server_memnav_${label}.launcher.json" \
      --log-prefix "${MEMNAV_LAUNCHER_LOG}.child" \
      --cwd "${runtime_root}/memnav" -- \
      "${MEMNAV_PY}" -u "${MEMNAV_SERVER}" --port '{port}' \
        --checkpoint "${MEMNAV_CKPT}" --internnav_root "${INTERNNAV_ROOT}" \
        --num_samples 16 --exclude_recent 32 --retrieval raw \
        --retrieval_candidate_top_k 32 --retrieval_candidate_min_gap 16 \
        --graph_subgoal_spacing_m 0.0 --graph_subgoal_arrival_m 0.60 \
        --flow_gate auto --buffer_root "${RUN_ROOT}/buffer/${label}" \
        --certified_relocalization --lightglue_repo "${LIGHTGLUE_REPO}" \
        --lightglue_dependency_root "${DEPENDENCY_ROOT}" \
        --lightglue_max_keypoints 2048
) > "${MEMNAV_LAUNCHER_LOG}" 2>&1 &
MEMNAV_PID=$!
(
  exec env NAVDP_DISABLE_VIDEO=1 PYTHONUNBUFFERED=1 \
    PYTHONPYCACHEPREFIX="${PYTHONPYCACHEPREFIX}" PYTHONPATH="${PYTHONPATH}" \
    "${MEMNAV_PY}" -u "${SERVER_LAUNCHER}" \
      --base-port "${NAVDP_BASE_PORT}" --range-start 20000 --range-end 49999 \
      --stride 1994 --max-attempts 12 --ready-timeout 720 \
      --port-file "${NAVDP_PORT_FILE}" \
      --receipt-file "${RUN_ROOT}/logs/server_navdp_${label}.launcher.json" \
      --log-prefix "${NAVDP_LAUNCHER_LOG}.child" \
      --cwd "${runtime_root}/navdp" -- \
      "${MEMNAV_PY}" -u "${NAVDP_SERVER}" --port '{port}' \
        --checkpoint "${NAVDP_CKPT}"
) > "${NAVDP_LAUNCHER_LOG}" 2>&1 &
NAVDP_PID=$!

ready=0
for _ in $(seq 1 420); do
  kill -0 "${MEMNAV_PID}" 2>/dev/null || {
    tail -n 200 "${MEMNAV_LAUNCHER_LOG}" >&2
    fail "MemNav launcher died"
  }
  kill -0 "${NAVDP_PID}" 2>/dev/null || {
    tail -n 200 "${NAVDP_LAUNCHER_LOG}" >&2
    fail "NavDP launcher died"
  }
  if [[ -s "${MEMNAV_PORT_FILE}" && -s "${NAVDP_PORT_FILE}" ]]; then
    ready=1
    break
  fi
  sleep 2
done
[[ "${ready}" -eq 1 ]] || {
  tail -n 200 "${MEMNAV_LAUNCHER_LOG}" "${NAVDP_LAUNCHER_LOG}" >&2
  fail "server launchers timed out"
}
MEMNAV_PORT=$(<"${MEMNAV_PORT_FILE}")
NAVDP_PORT=$(<"${NAVDP_PORT_FILE}")
[[ "${MEMNAV_PORT}" =~ ^[0-9]+$ && "${NAVDP_PORT}" =~ ^[0-9]+$ ]] || \
  fail "invalid launcher port"
(( MEMNAV_PORT >= 20000 && MEMNAV_PORT <= 49999 )) || fail "bad MemNav port"
(( NAVDP_PORT >= 20000 && NAVDP_PORT <= 49999 )) || fail "bad NavDP port"
[[ "${MEMNAV_PORT}" != "${NAVDP_PORT}" ]] || fail "launcher ports collide"

singularity exec --nv -B /scratch/lg154 -B /scratch/yz11502 \
  "${BASE_SIF}" env \
  SOURCE_ROOT="${SOURCE_ROOT}" RUN_ROOT="${RUN_ROOT}" \
  BENCH_ROOT="${BENCH_ROOT}" CONTROL_MANIFEST="${CONTROL_MANIFEST}" \
  EXPECTED_CONTROL_MANIFEST_SHA="${EXPECTED_CONTROL_MANIFEST_SHA}" \
  EPISODE_INDEX="${SLURM_ARRAY_TASK_ID}" \
  HAB_PY="${HAB_PY}" MEMNAV_PORT="${MEMNAV_PORT}" NAVDP_PORT="${NAVDP_PORT}" \
  MAX_STEPS=600 PYTHONPYCACHEPREFIX="${PYTHONPYCACHEPREFIX}" \
  PYTHONPATH="${PYTHONPATH}" bash "${RUNNER}"
