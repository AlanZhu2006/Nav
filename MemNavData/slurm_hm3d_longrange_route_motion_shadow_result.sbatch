#!/usr/bin/env bash
#SBATCH --job-name=h3ShadowResult
#SBATCH --output=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/%x_%j.out
#SBATCH --error=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/%x_%j.err
#SBATCH --time=00:10:00
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
PROTOCOL=${SOURCE_ROOT}/MemNavData/hm3d_longrange_route_motion_shadow_protocol_20260903.json
PROTOCOL_SHA=589af0882063bca63b47892938887c8a93e74477ccb68f2f9cfb02778da99943
RESULT=${RUN_ROOT}/result
[[ "$(sha256sum "${PROTOCOL}" | awk '{print $1}')" == "${PROTOCOL_SHA}" ]] || \
  fail "diagnostic protocol changed"
export PYTHONPATH=${SOURCE_ROOT}:${SOURCE_ROOT}/MemNavData

case "${MODE}" in
  analyze)
    [[ ! -e "${RESULT}" ]] || fail "result directory already exists"
    "${PY}" -u \
      "${SOURCE_ROOT}/MemNavData/analyze_hm3d_longrange_route_motion_shadow.py" \
      --run-root "${RUN_ROOT}" --protocol "${PROTOCOL}" \
      --expected-protocol-sha256 "${PROTOCOL_SHA}"
    (cd "${RESULT}" && sha256sum -c --quiet summary.json.sha256)
    ;;
  verify)
    [[ -f "${RESULT}/summary.json" ]] || fail "summary is missing"
    [[ ! -e "${RESULT}/independent_verification.json" ]] || \
      fail "result verification already exists"
    "${PY}" -u \
      "${SOURCE_ROOT}/MemNavData/independent_verify_hm3d_longrange_route_motion_shadow.py" \
      --run-root "${RUN_ROOT}" --protocol "${PROTOCOL}" \
      --expected-protocol-sha256 "${PROTOCOL_SHA}" \
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
