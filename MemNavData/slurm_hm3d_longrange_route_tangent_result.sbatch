#!/usr/bin/env bash
#SBATCH --job-name=h3RouteResult
#SBATCH --output=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/%x_%j.out
#SBATCH --error=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/%x_%j.err
#SBATCH --time=00:20:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --partition=cpu_short
#SBATCH --account=torch_pr_769_tandon_advanced

set -euo pipefail
umask 0022
export PYTHONDONTWRITEBYTECODE=1

: "${MODE:?set analyze or verify}" "${SOURCE_ROOT:?}" "${SOURCE_RECEIPT:?}"
: "${EXPECTED_SOURCE_RECEIPT_SHA:?}" "${RUN_ROOT:?}"

fail() { echo "ABORT: $*" >&2; exit 2; }
[[ "$(id -un)" == yz11502 ]] || fail "wrong HPC identity"
[[ "$(sha256sum "${SOURCE_RECEIPT}" | awk '{print $1}')" == \
   "${EXPECTED_SOURCE_RECEIPT_SHA}" ]] || fail "source receipt changed"
(cd "${SOURCE_ROOT}" && sha256sum -c --quiet "${SOURCE_RECEIPT}") || \
  fail "source bundle content changed"

PY=/scratch/lg154/conda-envs/memnav/bin/python
POPULATION=${RUN_ROOT}/fresh_population
RESULT=${RUN_ROOT}/result
export PYTHONPATH=${SOURCE_ROOT}:${SOURCE_ROOT}/MemNavData

case "${MODE}" in
  analyze)
    [[ ! -e "${RESULT}" ]] || fail "result directory already exists"
    mkdir "${RESULT}"
    "${PY}" -u \
      "${SOURCE_ROOT}/MemNavData/analyze_hm3d_longrange_route_tangent.py" \
      --population "${POPULATION}" --run-root "${RUN_ROOT}" \
      --out "${RESULT}/summary.json"
    (cd "${RESULT}" && sha256sum -c --quiet summary.json.sha256)
    ;;
  verify)
    [[ -f "${RESULT}/summary.json" ]] || fail "summary is missing"
    [[ ! -e "${RESULT}/independent_verification.json" ]] || \
      fail "result verification already exists"
    "${PY}" -u \
      "${SOURCE_ROOT}/MemNavData/independent_verify_hm3d_longrange_route_tangent_result.py" \
      --population "${POPULATION}" --run-root "${RUN_ROOT}" \
      --summary "${RESULT}/summary.json" \
      --out "${RESULT}/independent_verification.json"
    (cd "${RESULT}" && sha256sum -c --quiet \
      summary.json.sha256 independent_verification.json.sha256)
    chmod -R a-w "${RESULT}"
    ;;
  *)
    fail "invalid MODE=${MODE}"
    ;;
esac
