#!/usr/bin/env bash
# Independent raw-file recount after the primary report is sealed.
#SBATCH --job-name=nnr_verify
#SBATCH --output=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/%x_%j.out
#SBATCH --error=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/%x_%j.err
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
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
BENCH_ROOT=${RUN_ROOT}/prepared/benchmark
OUT=${RUN_ROOT}/independent_verification.json
export PYTHONPYCACHEPREFIX=${SLURM_TMPDIR:-/tmp}/nnr_verify_pycache_${SLURM_JOB_ID}
mkdir -p "${PYTHONPYCACHEPREFIX}"
fail() { echo "ABORT: $*" >&2; exit 2; }
[[ "$(sha256sum "${SOURCE_RECEIPT}" | awk '{print $1}')" == \
   "${EXPECTED_SOURCE_RECEIPT_SHA}" ]] || fail "source receipt changed"
(cd "${SOURCE_ROOT}" && sha256sum -c "${SOURCE_RECEIPT}") >/dev/null || \
  fail "source bundle changed"
(cd "${RUN_ROOT}" && sha256sum -c report.json.sha256) >/dev/null || \
  fail "report changed"
[[ ! -e "${OUT}" ]] || fail "verification exists"
MANIFEST_SHA=$(awk '{print $1}' "${BENCH_ROOT}/manifest.json.sha256")
export PYTHONPATH=${SOURCE_ROOT}:${SOURCE_ROOT}/MemNavData${PYTHONPATH:+:${PYTHONPATH}}
"${PY}" -u "${SOURCE_ROOT}/MemNavData/independent_verify_shared_online_nnr.py" \
  --benchmark-root "${BENCH_ROOT}" --run-root "${RUN_ROOT}" \
  --expected-manifest-sha "${MANIFEST_SHA}" --report "${RUN_ROOT}/report.json" \
  --out "${OUT}"
(cd "${RUN_ROOT}" && sha256sum -c independent_verification.json.sha256) >/dev/null
echo "COMPLETE ${OUT}"
