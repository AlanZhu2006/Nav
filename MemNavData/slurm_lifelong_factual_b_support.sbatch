#!/usr/bin/env bash
#SBATCH --job-name=lifeBseal
#SBATCH --output=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/%x_%j.out
#SBATCH --error=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/%x_%j.err
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=h100_tandon,h200_public,a100_tandon
#SBATCH --qos=gpu48
#SBATCH --gres=gpu:1
#SBATCH --account=torch_pr_769_tandon_advanced
#SBATCH --export=ALL

set -euo pipefail
umask 0022
: "${SOURCE_ROOT:?set immutable task source root}"
: "${SOURCE_RECEIPT:?set source receipt}"
: "${EXPECTED_SOURCE_RECEIPT_SHA:?set source receipt SHA}"
: "${NNR_ROOT:?set sealed NNR run root}"
: "${RUN_ROOT:?set unique lifelong run root}"

HAB_PY=/scratch/lg154/conda-envs/habitat/bin/python
BASE_SIF=/share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif
BENCHMARK_ROOT=${NNR_ROOT}/prepared/benchmark
OUT=${RUN_ROOT}/factual_b_support
fail() { echo "ABORT: $*" >&2; exit 2; }

[[ "$(sha256sum "${SOURCE_RECEIPT}" | awk '{print $1}')" == \
   "${EXPECTED_SOURCE_RECEIPT_SHA}" ]] || fail "source receipt changed"
(cd "${SOURCE_ROOT}" && sha256sum -c --quiet "${SOURCE_RECEIPT}") || \
  fail "source bundle changed"
(cd "${BENCHMARK_ROOT}" && sha256sum -c --quiet manifest.json.sha256) || \
  fail "NNR benchmark manifest changed"
[[ -f "${BENCHMARK_ROOT}/SEALED" && ! -e "${OUT}" ]] || \
  fail "NNR seal missing or support output already exists"

export PYTHONPYCACHEPREFIX=${SLURM_TMPDIR:-/tmp}/life_b_support_${SLURM_JOB_ID}
mkdir -p "${PYTHONPYCACHEPREFIX}" "${RUN_ROOT}" \
  /scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs
echo "host=$(hostname) job=${SLURM_JOB_ID}"
nvidia-smi --query-gpu=name,uuid,memory.total --format=csv,noheader

singularity exec --nv -B /scratch/lg154 -B /scratch/yz11502 \
  -B /scratch/qw2440 "${BASE_SIF}" env \
  PYTHONPATH="${SOURCE_ROOT}:${SOURCE_ROOT}/MemNavData" \
  PYTHONPYCACHEPREFIX="${PYTHONPYCACHEPREFIX}" \
  "${HAB_PY}" -u \
    "${SOURCE_ROOT}/MemNavData/audit_lifelong_factual_b_support.py" \
    --benchmark-root "${BENCHMARK_ROOT}" --out "${OUT}"

(cd "${OUT}" && sha256sum -c --quiet population.json.sha256) || \
  fail "support population receipt failed"
chmod -R a-w "${OUT}"
echo "COMPLETE support=$(awk '{print $1}' "${OUT}/population.json.sha256")"
