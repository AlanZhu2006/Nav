#!/usr/bin/env bash
#SBATCH --job-name=h3T1MetAna
#SBATCH --output=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/%x_%j.out
#SBATCH --error=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/%x_%j.err
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --partition=cpu_short
#SBATCH --account=torch_pr_769_tandon_advanced

set -euo pipefail
umask 0022
export PYTHONDONTWRITEBYTECODE=1

: "${MODE:?aggregate or verify}"
: "${TASK_ROOT:?}" "${TASK_RECEIPT:?}" "${EXPECTED_TASK_RECEIPT_SHA:?}"
: "${FORMAL_RUN_ROOT:?}" "${BENCH_ROOT:?}" "${EXPECTED_MANIFEST_SHA:?}"
PY=/scratch/lg154/conda-envs/memnav/bin/python
RUN_ROOT=${FORMAL_RUN_ROOT}/formal
SUMMARY=${RUN_ROOT}/metric_distance_revisit_summary.json
VERIFICATION=${RUN_ROOT}/metric_distance_revisit_independent_verification.json
fail() { echo "ABORT: $*" >&2; exit 2; }

[[ "$(sha256sum "${TASK_RECEIPT}" | awk '{print $1}')" == \
   "${EXPECTED_TASK_RECEIPT_SHA}" ]] || fail "task receipt changed"
(cd "${TASK_ROOT}" && sha256sum -c --quiet "${TASK_RECEIPT}") || \
  fail "task bundle changed"
export PYTHONPATH=${TASK_ROOT}:${TASK_ROOT}/MemNavData

case "${MODE}" in
  aggregate)
    [[ ! -e "${SUMMARY}" ]] || fail "summary already exists"
    "${PY}" -u \
      "${TASK_ROOT}/MemNavData/aggregate_hm3d_table1_metric_distance_revisit_pair.py" \
      --run-root "${RUN_ROOT}" --benchmark-root "${BENCH_ROOT}" \
      --expected-manifest-sha256 "${EXPECTED_MANIFEST_SHA}" \
      --out "${SUMMARY}"
    sha256sum "${SUMMARY}" >"${SUMMARY}.sha256"
    ;;
  verify)
    [[ -f "${SUMMARY}" && ! -e "${VERIFICATION}" ]] || \
      fail "verification precondition failed"
    "${PY}" -u \
      "${TASK_ROOT}/MemNavData/independent_verify_hm3d_table1_metric_distance_revisit_pair.py" \
      --run-root "${RUN_ROOT}" --benchmark-root "${BENCH_ROOT}" \
      --expected-manifest-sha256 "${EXPECTED_MANIFEST_SHA}" \
      --summary "${SUMMARY}" --out "${VERIFICATION}"
    sha256sum "${VERIFICATION}" >"${VERIFICATION}.sha256"
    "${PY}" - "${VERIFICATION}" <<'PY'
import json,sys
if json.load(open(sys.argv[1])).get("verified") is not True:
 raise SystemExit("independent verification failed")
PY
    printf '%s\n' verified=true >"${RUN_ROOT}/SEALED"
    chmod -R a-w "${RUN_ROOT}"
    ;;
  *) fail "invalid MODE=${MODE}" ;;
esac
echo "COMPLETE mode=${MODE}"
