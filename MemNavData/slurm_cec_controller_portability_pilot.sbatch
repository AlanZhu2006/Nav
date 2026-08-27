#!/usr/bin/env bash
#SBATCH --job-name=cecCtlPilot
#SBATCH --output=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/%x_%A_%a.out
#SBATCH --error=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/%x_%A_%a.err
#SBATCH --time=01:00:00
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
: "${SLURM_ARRAY_TASK_ID:?submit as a frozen history array}"
: "${SOURCE_ROOT:?set immutable source root}"
: "${SOURCE_RECEIPT:?set source receipt}"
: "${EXPECTED_SOURCE_RECEIPT_SHA:?set source receipt SHA}"
: "${BASE_SOURCE_ROOT:?set immutable dependency source root}"
: "${BASE_SOURCE_RECEIPT_SHA:?set base source receipt SHA}"
: "${BENCH_ROOT:?set frozen role-pair benchmark root}"
: "${EXPECTED_MANIFEST_SHA:?set benchmark SHA}"
: "${RUN_ROOT:?set unique pilot result root}"
: "${PORTABILITY_ENV_ROOT:?set immutable environment root}"
: "${PORTABILITY_CHECKPOINT_ROOT:?set checkpoint root}"
: "${DEPENDENCY_RECEIPT:?set dependency receipt}"
: "${EXPECTED_DEPENDENCY_RECEIPT_SHA:?set dependency receipt SHA}"
: "${SOURCE_OVERLAY:?set MP3D source overlay}"
: "${EXPECTED_SOURCE_OVERLAY_BYTES:?set overlay size}"

MAX_STEPS=${MAX_STEPS:-80}
BASE_SIF=${BASE_SIF:-/share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif}
fail() { echo "ABORT: $*" >&2; exit 2; }
verify_file_hash() {
  local label=$1 path=$2 expected=$3 actual="" attempt
  for attempt in 1 2 3; do
    actual=$(sha256sum -- "${path}" 2>/dev/null | awk '{print $1}') || actual="READ_ERROR"
    printf 'hash_preflight label=%s attempt=%s expected=%s actual=%s path=%s\n' \
      "${label}" "${attempt}" "${expected}" "${actual}" "${path}" >&2
    [[ "${actual}" == "${expected}" ]] && return 0
    sleep 2
  done
  stat --printf='hash_preflight_stat path=%n bytes=%s mtime=%y\n' -- \
    "${path}" >&2 2>/dev/null || true
  fail "${label} hash mismatch"
}

verify_file_hash source_receipt "${SOURCE_RECEIPT}" \
  "${EXPECTED_SOURCE_RECEIPT_SHA}"
(cd "${SOURCE_ROOT}" && sha256sum -c --quiet "${SOURCE_RECEIPT}") || \
  fail "source bundle changed"
verify_file_hash base_source_receipt \
  "${BASE_SOURCE_ROOT}/source_inputs.sha256" "${BASE_SOURCE_RECEIPT_SHA}"
verify_file_hash benchmark_manifest "${BENCH_ROOT}/manifest.json" \
  "${EXPECTED_MANIFEST_SHA}"
verify_file_hash dependency_receipt "${DEPENDENCY_RECEIPT}" \
  "${EXPECTED_DEPENDENCY_RECEIPT_SHA}"
[[ -r "${PORTABILITY_ENV_ROOT}/environment_receipt.json" ]] || \
  fail "portability environment missing"
(cd "${PORTABILITY_CHECKPOINT_ROOT}" && \
  sha256sum -c --quiet CHECKPOINTS.sha256) || fail "checkpoint receipt failed"
[[ -r "${SOURCE_OVERLAY}" && \
   "$(stat -c %s "${SOURCE_OVERLAY}")" -eq \
   "${EXPECTED_SOURCE_OVERLAY_BYTES}" ]] || fail "source overlay changed"
[[ -r "${BASE_SIF}" ]] || fail "base image missing"

mkdir -p "${RUN_ROOT}" /scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs
port_slot=$(( (SLURM_JOB_ID + SLURM_ARRAY_TASK_ID * 59) % 4000 ))
base_port=$((20000 + port_slot * 5))
export PYTHONPYCACHEPREFIX=${SLURM_TMPDIR:-/tmp}/cec_ctl_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}
mkdir -p "${PYTHONPYCACHEPREFIX}"

echo "host=$(hostname) job=${SLURM_JOB_ID} task=${SLURM_ARRAY_TASK_ID} max_steps=${MAX_STEPS}"
nvidia-smi --query-gpu=name,uuid,driver_version,memory.total \
  --format=csv,noheader

singularity exec --nv --overlay "${SOURCE_OVERLAY}:ro" \
  -B /scratch/lg154 -B /scratch/yz11502 \
  "${BASE_SIF}" env \
    SOURCE_ROOT="${SOURCE_ROOT}" BASE_SOURCE_ROOT="${BASE_SOURCE_ROOT}" \
    BENCH_ROOT="${BENCH_ROOT}" EXPECTED_MANIFEST_SHA="${EXPECTED_MANIFEST_SHA}" \
    HISTORY_INDEX="${SLURM_ARRAY_TASK_ID}" RUN_ROOT="${RUN_ROOT}" \
    PORTABILITY_ENV_ROOT="${PORTABILITY_ENV_ROOT}" \
    PORTABILITY_CHECKPOINT_ROOT="${PORTABILITY_CHECKPOINT_ROOT}" \
    DEPENDENCY_RECEIPT="${DEPENDENCY_RECEIPT}" \
    EXPECTED_DEPENDENCY_RECEIPT_SHA="${EXPECTED_DEPENDENCY_RECEIPT_SHA}" \
    MAX_STEPS="${MAX_STEPS}" BASE_PORT="${base_port}" \
    PYTHONPYCACHEPREFIX="${PYTHONPYCACHEPREFIX}" \
    bash "${SOURCE_ROOT}/MemNavData/run_cec_controller_portability_hpc_history.sh"
