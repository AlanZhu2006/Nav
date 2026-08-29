#!/usr/bin/env bash
#SBATCH --job-name=m3T1analysis
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
: "${TASK_ROOT:?}" "${TASK_RECEIPT:?}" "${EXPECTED_TASK_RECEIPT_SHA:?}"
: "${RUN_ROOT:?}" "${SOURCE_LEDGER:?}" "${EXPECTED_SOURCE_LEDGER_SHA:?}"
: "${MODE:?set MODE=finalize or verify}"

PY=/scratch/lg154/conda-envs/memnav/bin/python
fail() { echo "ABORT: $*" >&2; exit 2; }
[[ "$(sha256sum "${TASK_RECEIPT}" | awk '{print $1}')" == \
   "${EXPECTED_TASK_RECEIPT_SHA}" ]] || fail "task receipt changed"
(cd "${TASK_ROOT}" && sha256sum -c --quiet "${TASK_RECEIPT}") || \
  fail "task bundle changed"
[[ "$(sha256sum "${SOURCE_LEDGER}" | awk '{print $1}')" == \
   "${EXPECTED_SOURCE_LEDGER_SHA}" ]] || fail "source ledger changed"

case "${MODE}" in
  finalize)
    [[ ! -e "${RUN_ROOT}/population" ]] || fail "population output exists"
    PYTHONPATH="${TASK_ROOT}:${TASK_ROOT}/MemNavData" "${PY}" -u \
      "${TASK_ROOT}/MemNavData/finalize_mp3d_table1_new_query_population.py" \
      --construction-root "${RUN_ROOT}/construction/scenes" \
      --source-ledger "${SOURCE_LEDGER}" \
      --out "${RUN_ROOT}/population"
    chmod -R a-w "${RUN_ROOT}/population"
    ;;
  verify)
    [[ ! -e "${RUN_ROOT}/mp3d_table1_new_query_verification.json" ]] || \
      fail "verification output exists"
    PYTHONPATH="${TASK_ROOT}:${TASK_ROOT}/MemNavData" "${PY}" -u \
      "${TASK_ROOT}/MemNavData/verify_mp3d_table1_new_query_population.py" \
      --root "${RUN_ROOT}/population" --source-ledger "${SOURCE_LEDGER}" \
      --out "${RUN_ROOT}/mp3d_table1_new_query_verification.json"
    sha256sum "${RUN_ROOT}/mp3d_table1_new_query_verification.json" \
      >"${RUN_ROOT}/mp3d_table1_new_query_verification.json.sha256"
    chmod a-w "${RUN_ROOT}/mp3d_table1_new_query_verification.json" \
      "${RUN_ROOT}/mp3d_table1_new_query_verification.json.sha256"
    ;;
  *) fail "unknown MODE=${MODE}" ;;
esac
