#!/usr/bin/env bash
# Merge all 22 pre-outcome construction decisions into one sealed benchmark.
#SBATCH --job-name=nnr_finalize
#SBATCH --output=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/%x_%j.out
#SBATCH --error=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/%x_%j.err
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=cpu_short
#SBATCH --account=torch_pr_769_tandon_advanced
#SBATCH --export=ALL

set -euo pipefail
umask 0022
: "${SOURCE_ROOT:?set source root}"
: "${SOURCE_RECEIPT:?set source receipt}"
: "${EXPECTED_SOURCE_RECEIPT_SHA:?set receipt SHA}"
: "${RUN_ROOT:?set run root}"
PY=/scratch/lg154/conda-envs/memnav/bin/python
TRACE_ROOT=${RUN_ROOT}/prepared/traces
OUT=${RUN_ROOT}/prepared/benchmark
export PYTHONPYCACHEPREFIX=${SLURM_TMPDIR:-/tmp}/nnr_finalize_pycache_${SLURM_JOB_ID}
mkdir -p "${PYTHONPYCACHEPREFIX}"
fail() { echo "ABORT: $*" >&2; exit 2; }
[[ "$(sha256sum "${SOURCE_RECEIPT}" | awk '{print $1}')" == \
   "${EXPECTED_SOURCE_RECEIPT_SHA}" ]] || fail "source receipt changed"
(cd "${SOURCE_ROOT}" && sha256sum -c "${SOURCE_RECEIPT}") >/dev/null || \
  fail "source bundle changed"
(cd "${TRACE_ROOT}" && sha256sum -c manifest.json.sha256) >/dev/null || \
  fail "trace manifest changed"
[[ ! -e "${OUT}" ]] || fail "benchmark output exists"
TRACE_SHA=$(awk '{print $1}' "${TRACE_ROOT}/manifest.json.sha256")
export PYTHONPATH=${SOURCE_ROOT}:${SOURCE_ROOT}/MemNavData${PYTHONPATH:+:${PYTHONPATH}}
"${PY}" -u "${SOURCE_ROOT}/MemNavData/finalize_shared_online_novel_revisit.py" \
  --parts-root "${RUN_ROOT}/prepared/parts" --trace-root "${TRACE_ROOT}" \
  --expected-trace-manifest-sha "${TRACE_SHA}" \
  --expected-source-population 22 --out "${OUT}"
(cd "${OUT}" && sha256sum -c manifest.json.sha256) >/dev/null
[[ -f "${OUT}/SEALED" ]] || fail "benchmark was not sealed"
echo "COMPLETE ${OUT}"
