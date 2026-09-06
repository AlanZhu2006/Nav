#!/usr/bin/env bash
#SBATCH --job-name=h3RoutePop
#SBATCH --output=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/%x_%j.out
#SBATCH --error=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/%x_%j.err
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --partition=cpu_short
#SBATCH --account=torch_pr_769_tandon_advanced

set -euo pipefail
umask 0022
export PYTHONDONTWRITEBYTECODE=1

: "${MODE:?set freeze or verify}" "${SOURCE_ROOT:?}" "${SOURCE_RECEIPT:?}"
: "${EXPECTED_SOURCE_RECEIPT_SHA:?}" "${RUN_ROOT:?}"

fail() { echo "ABORT: $*" >&2; exit 2; }
[[ "$(id -un)" == yz11502 ]] || fail "wrong HPC identity"
[[ "$(sha256sum "${SOURCE_RECEIPT}" | awk '{print $1}')" == \
   "${EXPECTED_SOURCE_RECEIPT_SHA}" ]] || fail "source receipt changed"
(cd "${SOURCE_ROOT}" && sha256sum -c --quiet "${SOURCE_RECEIPT}") || \
  fail "source bundle content changed"

PY=/scratch/lg154/conda-envs/memnav/bin/python
PROTOCOL=${SOURCE_ROOT}/MemNavData/hm3d_longrange_route_tangent_freeze_protocol_v2_20260903.json
POPULATION=${RUN_ROOT}/fresh_population
export PYTHONPATH=${SOURCE_ROOT}:${SOURCE_ROOT}/MemNavData

case "${MODE}" in
  freeze)
    [[ -d "${RUN_ROOT}" ]] || fail "run root was not created by submission"
    [[ ! -e "${POPULATION}" ]] || fail "fresh population already exists"
    "${PY}" -u \
      "${SOURCE_ROOT}/MemNavData/freeze_hm3d_longrange_route_tangent_population.py" \
      --protocol "${PROTOCOL}" --out "${POPULATION}"
    ;;
  verify)
    [[ -d "${POPULATION}" ]] || fail "fresh population is missing"
    [[ ! -e "${POPULATION}/independent_verification.json" ]] || \
      fail "population verification already exists"
    "${PY}" -u \
      "${SOURCE_ROOT}/MemNavData/independent_verify_hm3d_longrange_route_tangent_population.py" \
      --protocol "${PROTOCOL}" --population "${POPULATION}" \
      --out "${POPULATION}/independent_verification.json"
    (cd "${POPULATION}" && sha256sum -c --quiet \
      population_receipt.json.sha256 independent_verification.json.sha256)
    (cd "${POPULATION}/role_pairs" && \
      sha256sum -c --quiet manifest.json.sha256)
    chmod -R a-w "${POPULATION}"
    ;;
  *)
    fail "invalid MODE=${MODE}"
    ;;
esac
