#!/usr/bin/env bash
#SBATCH --job-name=vintCECana
#SBATCH --output=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/%x_%j.out
#SBATCH --error=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/%x_%j.err
#SBATCH --time=00:20:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=cpu_short
#SBATCH --account=torch_pr_769_tandon_advanced

set -euo pipefail
umask 0022
export PYTHONDONTWRITEBYTECODE=1
: "${MODE:?pilot_aggregate, pilot_verify, formal_aggregate or formal_verify}"
: "${SOURCE_ROOT:?}" "${SOURCE_RECEIPT:?}" "${EXPECTED_SOURCE_RECEIPT_SHA:?}"
: "${FRESH_ROOT:?}" "${RUN_ROOT:?}"
MEMNAV_PY=/scratch/lg154/conda-envs/memnav/bin/python
BENCHMARK_MANIFEST=${FRESH_ROOT}/benchmarks/natural_direction/manifest.json
fail() { echo "ABORT: $*" >&2; exit 2; }

[[ "$(sha256sum "${SOURCE_RECEIPT}" | awk '{print $1}')" == \
   "${EXPECTED_SOURCE_RECEIPT_SHA}" ]] || fail "source receipt changed"
(cd "${SOURCE_ROOT}" && sha256sum -c --quiet "${SOURCE_RECEIPT}") || \
  fail "source bundle changed"
[[ "$(sha256sum "${BENCHMARK_MANIFEST}" | awk '{print $1}')" == \
   "aada40d25d01e9385df3ffdcaf37847f471b63c7be785a704eade961346a50b0" ]] || \
  fail "benchmark manifest changed"

case "${MODE}" in
  pilot_aggregate)
    PHASE_ROOT=${RUN_ROOT}/pilot
    [[ ! -e "${PHASE_ROOT}/pilot_summary.json" ]] || \
      fail "pilot summary already exists"
    env PYTHONPATH="${SOURCE_ROOT}" "${MEMNAV_PY}" -u \
      "${SOURCE_ROOT}/MemNavData/aggregate_vint_controller_native_hm3d.py" \
      --run-root "${PHASE_ROOT}" \
      --benchmark-manifest "${BENCHMARK_MANIFEST}" \
      --expected-histories 4 --expected-scenes 4 \
      --history-indices 0,7,14,21 \
      --claim-scope infrastructure_gate_performance_not_a_submission_gate \
      --out "${PHASE_ROOT}/pilot_summary.json"
    sha256sum "${PHASE_ROOT}/pilot_summary.json" \
      >"${PHASE_ROOT}/pilot_summary.json.sha256"
    ;;
  pilot_verify)
    PHASE_ROOT=${RUN_ROOT}/pilot
    [[ -f "${PHASE_ROOT}/pilot_summary.json" \
       && ! -e "${PHASE_ROOT}/pilot_independent_verification.json" ]] || \
      fail "pilot verification precondition failed"
    env PYTHONPATH="${SOURCE_ROOT}" "${MEMNAV_PY}" -u \
      "${SOURCE_ROOT}/MemNavData/independent_verify_vint_controller_native_hm3d.py" \
      --run-root "${PHASE_ROOT}" --summary "${PHASE_ROOT}/pilot_summary.json" \
      --benchmark-manifest "${BENCHMARK_MANIFEST}" \
      --out "${PHASE_ROOT}/pilot_independent_verification.json"
    sha256sum "${PHASE_ROOT}/pilot_independent_verification.json" \
      >"${PHASE_ROOT}/pilot_independent_verification.json.sha256"
    "${MEMNAV_PY}" - "${PHASE_ROOT}/pilot_independent_verification.json" <<'PY'
import json,sys
if json.load(open(sys.argv[1])).get("verified") is not True:
 raise SystemExit("pilot independent verification failed")
PY
    printf '%s\n' verified=true >"${PHASE_ROOT}/SEALED"
    chmod a-w "${PHASE_ROOT}/pilot_summary.json" \
      "${PHASE_ROOT}/pilot_summary.json.sha256" \
      "${PHASE_ROOT}/pilot_independent_verification.json" \
      "${PHASE_ROOT}/pilot_independent_verification.json.sha256" \
      "${PHASE_ROOT}/SEALED"
    ;;
  formal_aggregate)
    PHASE_ROOT=${RUN_ROOT}/formal
    [[ ! -e "${PHASE_ROOT}/formal_summary.json" ]] || \
      fail "formal summary already exists"
    env PYTHONPATH="${SOURCE_ROOT}" "${MEMNAV_PY}" -u \
      "${SOURCE_ROOT}/MemNavData/aggregate_vint_controller_native_hm3d.py" \
      --run-root "${PHASE_ROOT}" \
      --benchmark-manifest "${BENCHMARK_MANIFEST}" \
      --expected-histories 28 --expected-scenes 21 \
      --claim-scope conference_table_hm3d_within_vint_paired_result \
      --out "${PHASE_ROOT}/formal_summary.json"
    sha256sum "${PHASE_ROOT}/formal_summary.json" \
      >"${PHASE_ROOT}/formal_summary.json.sha256"
    ;;
  formal_verify)
    PHASE_ROOT=${RUN_ROOT}/formal
    [[ -f "${PHASE_ROOT}/formal_summary.json" \
       && ! -e "${PHASE_ROOT}/formal_independent_verification.json" ]] || \
      fail "formal verification precondition failed"
    env PYTHONPATH="${SOURCE_ROOT}" "${MEMNAV_PY}" -u \
      "${SOURCE_ROOT}/MemNavData/independent_verify_vint_controller_native_hm3d.py" \
      --run-root "${PHASE_ROOT}" --summary "${PHASE_ROOT}/formal_summary.json" \
      --benchmark-manifest "${BENCHMARK_MANIFEST}" \
      --out "${PHASE_ROOT}/formal_independent_verification.json"
    sha256sum "${PHASE_ROOT}/formal_independent_verification.json" \
      >"${PHASE_ROOT}/formal_independent_verification.json.sha256"
    "${MEMNAV_PY}" - "${PHASE_ROOT}/formal_independent_verification.json" <<'PY'
import json,sys
if json.load(open(sys.argv[1])).get("verified") is not True:
 raise SystemExit("formal independent verification failed")
PY
    printf '%s\n' verified=true >"${PHASE_ROOT}/SEALED"
    chmod a-w "${PHASE_ROOT}/formal_summary.json" \
      "${PHASE_ROOT}/formal_summary.json.sha256" \
      "${PHASE_ROOT}/formal_independent_verification.json" \
      "${PHASE_ROOT}/formal_independent_verification.json.sha256" \
      "${PHASE_ROOT}/SEALED"
    ;;
  *) fail "unknown MODE=${MODE}" ;;
esac
echo "COMPLETE mode=${MODE}"
