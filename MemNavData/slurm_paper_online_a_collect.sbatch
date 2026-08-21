#!/usr/bin/env bash
# Frozen MP3D paper source collection: one scene / two Goal-A episodes per task.
#SBATCH --job-name=paper_A
#SBATCH --output=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/%x_%A_%a.out
#SBATCH --error=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/%x_%A_%a.err
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --partition=l40s_public,h100_tandon,h200_public,a100_tandon
#SBATCH --gres=gpu:1
#SBATCH --account=torch_pr_769_tandon_advanced
#SBATCH --export=ALL

set -euo pipefail
umask 0022
: "${SLURM_ARRAY_TASK_ID:?submit array 0-15}"
: "${SOURCE_ROOT:?set immutable SOURCE_ROOT}"
: "${SOURCE_RECEIPT:?set SOURCE_RECEIPT}"
: "${EXPECTED_SOURCE_RECEIPT_SHA:?set receipt SHA}"
: "${RUN_ROOT:?set RUN_ROOT}"
: "${MANIFEST:?set frozen MANIFEST}"
: "${EXPECTED_MANIFEST_SHA:?set manifest SHA}"
: "${BASE_SOURCE_ROOT:?set base dependency source root}"
: "${BASE_SOURCE_RECEIPT_SHA:?set base receipt SHA}"
: "${DEPENDENCY_RECEIPT:?set model dependency receipt}"
: "${EXPECTED_DEPENDENCY_RECEIPT_SHA:?set dependency receipt SHA}"
FINAL14_POPULATION_MODE=${FINAL14_POPULATION_MODE:-0}

BASE_SIF=/share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif
SOURCE_OVERLAY=/scratch/lg154/Research/datasets/_overlays/mp3d_revisit_v0_pt1.sqf
HAB_PY=/scratch/lg154/conda-envs/habitat/bin/python
MEMNAV_PY=/scratch/lg154/conda-envs/memnav/bin/python
RUNNER=${SOURCE_ROOT}/MemNavData/run_paper_online_a_scene.sh
SERVER=${SOURCE_ROOT}/NavDP/baselines/navdp/navdp_server.py
SERVER_LAUNCHER=${SOURCE_ROOT}/MemNavData/retrying_server_launcher.py
INTERNNAV_ROOT=${BASE_SOURCE_ROOT}/InternNav
LIGHTGLUE_REPO=${BASE_SOURCE_ROOT}/third_party/LightGlue
DEPENDENCY_ROOT=${BASE_SOURCE_ROOT}/third_party/python
TORCH_HOME=${BASE_SOURCE_ROOT}/torch_home
BASE_RECEIPT=${BASE_SOURCE_ROOT}/SOURCE_BUNDLE.sha256

fail() { echo "ABORT: $*" >&2; exit 2; }
[[ "${SLURM_ARRAY_TASK_ID}" =~ ^([0-9]|1[0-5])$ ]] || fail "bad array index"
[[ "${FINAL14_POPULATION_MODE}" == 0 || "${FINAL14_POPULATION_MODE}" == 1 ]] || \
  fail "FINAL14_POPULATION_MODE must be 0 or 1"
[[ -r "${BASE_SIF}" && -r "${SOURCE_OVERLAY}" \
   && -x "${HAB_PY}" && -x "${MEMNAV_PY}" ]] || \
  fail "runtime dependency missing"
[[ "$(sha256sum "${SOURCE_RECEIPT}" | awk '{print $1}')" == \
    "${EXPECTED_SOURCE_RECEIPT_SHA}" ]] || fail "source receipt changed"
(cd "${SOURCE_ROOT}" && sha256sum -c "${SOURCE_RECEIPT}") >/dev/null || \
  fail "source bundle changed"
[[ "$(sha256sum "${MANIFEST}" | awk '{print $1}')" == \
    "${EXPECTED_MANIFEST_SHA}" ]] || fail "manifest changed"
[[ "$(sha256sum "${BASE_RECEIPT}" | awk '{print $1}')" == \
    "${BASE_SOURCE_RECEIPT_SHA}" ]] || fail "base source receipt changed"
[[ "$(sha256sum "${DEPENDENCY_RECEIPT}" | awk '{print $1}')" == \
    "${EXPECTED_DEPENDENCY_RECEIPT_SHA}" ]] || fail "dependency receipt changed"

readarray -t dependencies < <("${MEMNAV_PY}" - "${DEPENDENCY_RECEIPT}" <<'PY'
import json,os,sys
receipt=json.load(open(sys.argv[1]))
row=receipt["dependencies"]["navdp_checkpoint"]
if os.stat(row["path"]).st_size != int(row["bytes"]):
    raise SystemExit("NavDP checkpoint size changed")
print(row["path"])
PY
)
[[ "${#dependencies[@]}" -eq 1 ]] || fail "dependency receipt invalid"
NAVDP_CHECKPOINT=${dependencies[0]}

mkdir -p "${RUN_ROOT}/logs" \
  /scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs
export PYTHONPYCACHEPREFIX=${SLURM_TMPDIR:-/tmp}/paper_a_pycache_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}
export PYTHONPATH=${SOURCE_ROOT}:${SOURCE_ROOT}/MemNavData:${DEPENDENCY_ROOT}:${LIGHTGLUE_REPO}:${BASE_SOURCE_ROOT}/NavDP/baselines/navdp:${INTERNNAV_ROOT}/src/diffusion-policy${PYTHONPATH:+:${PYTHONPATH}}
export TORCH_HOME
mkdir -p "${PYTHONPYCACHEPREFIX}"
"${MEMNAV_PY}" -m py_compile "${SERVER}" "${SERVER_LAUNCHER}"
"${MEMNAV_PY}" -c \
  'from depth_anything.depth_anything_v2.dpt import DepthAnythingV2; print("depth_anything import ok")'
base_port=$((21000 + (SLURM_JOB_ID + SLURM_ARRAY_TASK_ID * 31) % 22000))
runtime_root=${SLURM_TMPDIR:-/tmp}/paper_a_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}
mkdir -p "${runtime_root}"
port_file=${runtime_root}/navdp.port
launcher_receipt=${RUN_ROOT}/logs/navdp_${SLURM_ARRAY_TASK_ID}.launcher.json
launcher_log=${RUN_ROOT}/logs/navdp_${SLURM_ARRAY_TASK_ID}.log

SERVER_PID=
cleanup() {
  if [[ -n "${SERVER_PID}" ]] && kill -0 "${SERVER_PID}" 2>/dev/null; then
    kill "${SERVER_PID}" 2>/dev/null || true
    wait "${SERVER_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

(
  exec env NAVDP_DISABLE_VIDEO=1 PYTHONUNBUFFERED=1 \
    PYTHONPYCACHEPREFIX="${PYTHONPYCACHEPREFIX}" \
    PYTHONPATH="${PYTHONPATH}" TORCH_HOME="${TORCH_HOME}" \
    "${MEMNAV_PY}" -u "${SERVER_LAUNCHER}" \
      --base-port "${base_port}" --range-start 21000 --range-end 42999 \
      --stride 997 --max-attempts 12 --ready-timeout 480 \
      --port-file "${port_file}" --receipt-file "${launcher_receipt}" \
      --log-prefix "${launcher_log}.child" --cwd "${runtime_root}" -- \
      "${MEMNAV_PY}" -u "${SERVER}" --port '{port}' \
        --checkpoint "${NAVDP_CHECKPOINT}"
) > "${launcher_log}" 2>&1 &
SERVER_PID=$!
ready=0
for _ in $(seq 1 300); do
  kill -0 "${SERVER_PID}" 2>/dev/null || {
    tail -n 200 "${launcher_log}" >&2
    fail "NavDP launcher exited"
  }
  if [[ -s "${port_file}" && -s "${launcher_receipt}" ]]; then
    ready=1; break
  fi
  sleep 2
done
[[ "${ready}" -eq 1 ]] || {
  tail -n 200 "${launcher_log}" >&2
  fail "NavDP launcher startup timeout"
}
port=$(<"${port_file}")
[[ "${port}" =~ ^[0-9]+$ ]] && (( port >= 21000 && port <= 42999 )) || \
  fail "invalid NavDP launcher port"

echo "host=$(hostname) job=${SLURM_JOB_ID} task=${SLURM_ARRAY_TASK_ID}"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
singularity exec --nv --overlay "${SOURCE_OVERLAY}:ro" \
  -B /scratch/lg154 -B /scratch/yz11502 \
  "${BASE_SIF}" env \
    ROOT="${SOURCE_ROOT}" RUN_ROOT="${RUN_ROOT}" MANIFEST="${MANIFEST}" \
    EXPECTED_MANIFEST_SHA="${EXPECTED_MANIFEST_SHA}" \
    SOURCE_RECEIPT="${SOURCE_RECEIPT}" \
    EXPECTED_SOURCE_RECEIPT_SHA="${EXPECTED_SOURCE_RECEIPT_SHA}" \
    SCENE_INDEX="${SLURM_ARRAY_TASK_ID}" HAB_PY="${HAB_PY}" \
    NAVDP_PORT="${port}" NAVDP_CHECKPOINT="${NAVDP_CHECKPOINT}" \
    FINAL14_POPULATION_MODE="${FINAL14_POPULATION_MODE}" \
    PYTHONPYCACHEPREFIX="${PYTHONPYCACHEPREFIX}" \
    PYTHONPATH="${PYTHONPATH}" TORCH_HOME="${TORCH_HOME}" \
    bash "${RUNNER}"
