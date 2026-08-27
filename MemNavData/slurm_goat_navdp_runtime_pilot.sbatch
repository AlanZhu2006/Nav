#!/usr/bin/env bash
# Ten-scene first-ImageGoal runtime gate.  This is explicitly not a GOAT score.
#SBATCH --job-name=goat_navdp10
#SBATCH --output=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/%x_%j.out
#SBATCH --error=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/%x_%j.err
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --partition=h100_tandon,a100_tandon,h200_public,l40s_public
#SBATCH --qos=gpu48
#SBATCH --gres=gpu:1
#SBATCH --account=torch_pr_769_tandon_advanced

set -euo pipefail
umask 0022

: "${SOURCE_ROOT:?set immutable pilot source root}"
: "${SOURCE_RECEIPT:?set pilot source receipt}"
: "${EXPECTED_SOURCE_RECEIPT_SHA:?set pilot receipt SHA-256}"
: "${BASE_SOURCE_ROOT:?set pinned NavDP source root}"
: "${EXPECTED_BASE_SOURCE_RECEIPT_SHA:?set pinned NavDP source receipt SHA-256}"
: "${MANIFEST:?set frozen pilot manifest}"
: "${EXPECTED_MANIFEST_SHA:?set frozen pilot manifest SHA-256}"
: "${RUN_ROOT:?set unique runtime-pilot result root}"

readonly DATA_ROOT="${GOAT_HM3D_ROOT:-/scratch/yz11502/Research/datasets/goat_bench_20260814}"
readonly ENV_PREFIX="${GOAT_ENV_PREFIX:-/scratch/yz11502/conda_envs/goat-bench-habitat023-20260814}"
readonly GOAT_CODE="${GOAT_CODE_ROOT:-${DATA_ROOT}/code/goat-bench}"
readonly BASE_SIF="${GOAT_BASE_SIF:-/share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif}"
readonly ENV_RECEIPT="${DATA_ROOT}/receipts/goat-environment-v023.json"
readonly EXTRACT_RECEIPT="${DATA_ROOT}/receipts/goat-val-unseen-extraction.json"
readonly MEMNAV_PY="${MEMNAV_PY:-/scratch/lg154/conda-envs/memnav/bin/python}"
readonly NAVDP_CKPT="${NAVDP_CKPT:-/scratch/yz11502/Research/Nav-axis-uturn/.diagnostics/unseen_scene_eval_20260803/checkpoints/navdp_checkpoint.ckpt}"
readonly EXPECTED_NAVDP_CKPT_SHA="3bb3ad4ab241e857bb57a4021cc6aab76d5263e81fbf80298d579053ef011947"
readonly NAVDP_SERVER="${BASE_SOURCE_ROOT}/NavDP/baselines/navdp/navdp_server.py"
readonly RUNNER="${SOURCE_ROOT}/MemNavData/goat_navdp_runtime_pilot.py"
readonly ADAPTER="${SOURCE_ROOT}/MemNavData/goat_navdp_discrete_adapter.py"
readonly CONTRACT="${SOURCE_ROOT}/MemNavData/goat_contract_smoke.py"

fail() { echo "ABORT: $*" >&2; exit 2; }
for command_name in singularity sha256sum nvidia-smi ss; do
  command -v "${command_name}" >/dev/null || fail "${command_name} unavailable"
done
[[ "$(sha256sum "${SOURCE_RECEIPT}" | awk '{print $1}')" == \
   "${EXPECTED_SOURCE_RECEIPT_SHA}" ]] || fail "pilot source receipt changed"
(cd "${SOURCE_ROOT}" && sha256sum -c "${SOURCE_RECEIPT}") >/dev/null || \
  fail "pilot source bundle changed"
[[ "$(sha256sum "${BASE_SOURCE_ROOT}/SOURCE_BUNDLE.sha256" | awk '{print $1}')" == \
   "${EXPECTED_BASE_SOURCE_RECEIPT_SHA}" ]] || fail "base NavDP source receipt changed"
(cd "${BASE_SOURCE_ROOT}" && sha256sum -c SOURCE_BUNDLE.sha256) >/dev/null || \
  fail "base NavDP source bundle changed"
[[ "$(sha256sum "${MANIFEST}" | awk '{print $1}')" == \
   "${EXPECTED_MANIFEST_SHA}" ]] || fail "pilot manifest changed"
[[ "$(sha256sum "${NAVDP_CKPT}" | awk '{print $1}')" == \
   "${EXPECTED_NAVDP_CKPT_SHA}" ]] || fail "NavDP checkpoint changed"
[[ -x "${ENV_PREFIX}/bin/python" && -x "${MEMNAV_PY}" ]] || \
  fail "required Python environment is missing"
[[ -f "${ENV_RECEIPT}" && -f "${EXTRACT_RECEIPT}" && \
   -f "${DATA_ROOT}/data/scene_datasets/hm3d/val/SEALED" ]] || \
  fail "GOAT environment or scene extraction is incomplete"
[[ -f "${NAVDP_SERVER}" && -f "${RUNNER}" && -f "${ADAPTER}" && \
   -f "${CONTRACT}" && -d "${GOAT_CODE}/.git" && -r "${BASE_SIF}" ]] || \
  fail "frozen runtime input is missing"
[[ "$(git -C "${GOAT_CODE}" rev-parse HEAD)" == \
   74c41d19d4a4c3608d1575b512087b5a529aee0e ]] || \
  fail "GOAT source commit changed"
[[ -d "${RUN_ROOT}" && -f "${RUN_ROOT}/submission.json" ]] || \
  fail "precommitted run root/submission receipt is missing"
[[ "$(find "${RUN_ROOT}" -mindepth 1 -maxdepth 1 -printf '%f\n' | sort)" == \
   "submission.json" ]] || fail "run root contains pre-run artifacts"

"${ENV_PREFIX}/bin/python" - "${ENV_RECEIPT}" "${EXTRACT_RECEIPT}" <<'PY'
import json
import sys
for path in sys.argv[1:]:
    payload = json.load(open(path))
    if not payload.get("complete"):
        raise SystemExit("incomplete prerequisite receipt: {}".format(path))
PY

mkdir -p "${RUN_ROOT}/logs"
cp "${SOURCE_RECEIPT}" "${RUN_ROOT}/source_bundle.sha256"
cp "${MANIFEST}" "${RUN_ROOT}/pilot_manifest.json"
chmod a-w "${RUN_ROOT}/source_bundle.sha256" "${RUN_ROOT}/pilot_manifest.json"

export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
export PYTHONPYCACHEPREFIX="${SLURM_TMPDIR:-/tmp}/goat_navdp10_${SLURM_JOB_ID}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-12}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-12}"
export MAGNUM_LOG=quiet
export HABITAT_SIM_LOG=quiet
mkdir -p "${PYTHONPYCACHEPREFIX}"

# Python 3.7 cannot safely byte-compile a read-only bundle; AST parsing catches
# syntax incompatibility without creating __pycache__ next to source files.
"${ENV_PREFIX}/bin/python" - "${RUNNER}" "${ADAPTER}" "${CONTRACT}" <<'PY'
import ast
import pathlib
import sys
for raw_path in sys.argv[1:]:
    path = pathlib.Path(raw_path)
    ast.parse(path.read_text(), filename=str(path))
PY

port_key=$(( SLURM_JOB_ID % 18000 ))
NAVDP_PORT=$((20000 + port_key))
ss -ltn | awk '{print $4}' | grep -Eq "(^|:)${NAVDP_PORT}$" && \
  fail "NavDP port ${NAVDP_PORT} is already in use"
readonly NAVDP_URL="http://127.0.0.1:${NAVDP_PORT}"
runtime_root="${SLURM_TMPDIR:-/tmp}/goat_navdp10_${SLURM_JOB_ID}"
mkdir -p "${runtime_root}/navdp"

NAVDP_PID=
MONITOR_PID=
cleanup() {
  for pid in "${MONITOR_PID}" "${NAVDP_PID}"; do
    if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
      kill "${pid}" 2>/dev/null || true
      wait "${pid}" 2>/dev/null || true
    fi
  done
}
trap cleanup EXIT INT TERM

(
  cd "${runtime_root}/navdp"
  exec env NAVDP_DISABLE_VIDEO=1 PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPYCACHEPREFIX="${PYTHONPYCACHEPREFIX}" \
    PYTHONPATH="${BASE_SOURCE_ROOT}:${BASE_SOURCE_ROOT}/NavDP/baselines/navdp:${BASE_SOURCE_ROOT}/InternNav/src/diffusion-policy" \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "${MEMNAV_PY}" -u "${NAVDP_SERVER}" \
      --port "${NAVDP_PORT}" --checkpoint "${NAVDP_CKPT}"
) > "${RUN_ROOT}/logs/navdp_server.log" 2>&1 &
NAVDP_PID=$!

ready=0
for _ in $(seq 1 240); do
  kill -0 "${NAVDP_PID}" 2>/dev/null || {
    tail -n 160 "${RUN_ROOT}/logs/navdp_server.log" >&2
    fail "NavDP server exited before binding"
  }
  if ss -ltn | awk '{print $4}' | grep -Eq "(^|:)${NAVDP_PORT}$"; then
    ready=1
    break
  fi
  sleep 2
done
[[ "${ready}" -eq 1 ]] || fail "NavDP server startup timed out"

(
  echo "timestamp_utc,gpu_index,gpu_name,memory_used_mib,utilization_gpu_percent"
  while true; do
    timestamp=$(date -u +%Y-%m-%dT%H:%M:%SZ)
    nvidia-smi --id="${CUDA_VISIBLE_DEVICES%%,*}" \
      --query-gpu=index,name,memory.used,utilization.gpu \
      --format=csv,noheader,nounits | \
      sed "s/^/${timestamp},/"
    sleep 5
  done
) > "${RUN_ROOT}/gpu_samples.csv" 2> "${RUN_ROOT}/logs/gpu_monitor.err" &
MONITOR_PID=$!

{
  echo "host=$(hostname)"
  echo "job_id=${SLURM_JOB_ID}"
  echo "navdp_port=${NAVDP_PORT}"
  echo "source_receipt_sha256=${EXPECTED_SOURCE_RECEIPT_SHA}"
  echo "base_source_receipt_sha256=${EXPECTED_BASE_SOURCE_RECEIPT_SHA}"
  echo "manifest_sha256=${EXPECTED_MANIFEST_SHA}"
  echo "navdp_checkpoint_sha256=${EXPECTED_NAVDP_CKPT_SHA}"
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
} > "${RUN_ROOT}/runtime_environment.txt"

cd "${GOAT_CODE}"
singularity exec --nv -B /scratch/yz11502 "${BASE_SIF}" env \
  PYTHONNOUSERSITE=1 \
  PYTHONDONTWRITEBYTECODE=1 \
  PYTHONPYCACHEPREFIX="${PYTHONPYCACHEPREFIX}" \
  PYTHONPATH="${SOURCE_ROOT}" \
  MAGNUM_LOG="${MAGNUM_LOG}" \
  HABITAT_SIM_LOG="${HABITAT_SIM_LOG}" \
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
  "${ENV_PREFIX}/bin/python" -u "${RUNNER}" \
    --goat-code "${GOAT_CODE}" \
    --data-root "${DATA_ROOT}" \
    --manifest "${MANIFEST}" \
    --output-dir "${RUN_ROOT}" \
    --navdp-url "${NAVDP_URL}" \
  > "${RUN_ROOT}/logs/runtime_pilot.log" 2>&1

test -s "${RUN_ROOT}/goat_navdp_runtime_pilot.json"
"${ENV_PREFIX}/bin/python" - "${RUN_ROOT}/goat_navdp_runtime_pilot.json" <<'PY'
import json
import sys
payload = json.load(open(sys.argv[1]))
assert payload["complete"] is True
assert payload["is_goat_navigation_score"] is False
assert payload["paper_metric_use_prohibited"] is True
assert payload["scene_count"] == 10
assert payload["episode_count"] == 10
assert payload["all_records_complete"] is True
assert payload["all_goal_render_contracts_passed"] is True
PY
sha256sum "${RUN_ROOT}/goat_navdp_runtime_pilot.json" \
  > "${RUN_ROOT}/goat_navdp_runtime_pilot.json.sha256"
: > "${RUN_ROOT}/SEALED"
