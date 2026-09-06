#!/usr/bin/env bash
#SBATCH --job-name=h3DenseResult
#SBATCH --output=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/%x_%j.out
#SBATCH --error=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/%x_%j.err
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --partition=cpu_short
#SBATCH --account=torch_pr_769_tandon_advanced

set -euo pipefail
umask 0022
export PYTHONDONTWRITEBYTECODE=1
: "${SOURCE_ROOT:?}" "${SOURCE_RECEIPT:?}" \
  "${EXPECTED_SOURCE_RECEIPT_SHA:?}" "${RUN_ROOT:?}" "${MODE:?}"

fail() { echo "ABORT: $*" >&2; exit 2; }
[[ "$(id -un)" == yz11502 ]] || fail "wrong HPC identity"
[[ "$(sha256sum "${SOURCE_RECEIPT}" | awk '{print $1}')" == \
   "${EXPECTED_SOURCE_RECEIPT_SHA}" ]] || fail "source receipt changed"
(cd "${SOURCE_ROOT}" && sha256sum -c --quiet "${SOURCE_RECEIPT}") || \
  fail "source bundle changed"
PY=/scratch/lg154/conda-envs/memnav/bin/python
PROTOCOL=${SOURCE_ROOT}/MemNavData/hm3d_longrange_dense_query_gate_protocol_v2_20260903.json
case "${MODE}" in
  analyze)
    PYTHONPATH="${SOURCE_ROOT}:${SOURCE_ROOT}/MemNavData" "${PY}" \
      "${SOURCE_ROOT}/MemNavData/analyze_hm3d_longrange_dense_query_gate.py" \
      --run-root "${RUN_ROOT}" --protocol "${PROTOCOL}"
    ;;
  verify)
    PYTHONPATH="${SOURCE_ROOT}:${SOURCE_ROOT}/MemNavData" "${PY}" \
      "${SOURCE_ROOT}/MemNavData/independent_verify_hm3d_longrange_dense_query_gate.py" \
      --run-root "${RUN_ROOT}" --protocol "${PROTOCOL}"
    ;;
  *) fail "MODE must be analyze or verify" ;;
esac
