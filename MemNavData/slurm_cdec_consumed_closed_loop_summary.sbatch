#!/usr/bin/env bash
#SBATCH --job-name=cdec_cl_sum
#SBATCH --output=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/%x_%j.out
#SBATCH --error=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/%x_%j.err
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=cs
#SBATCH --qos=cpu48
#SBATCH --account=torch_pr_769_tandon_advanced
#SBATCH --export=ALL

set -euo pipefail
umask 0022
: "${SOURCE_ROOT:?set immutable source root}"
: "${SOURCE_RECEIPT:?set source receipt}"
: "${EXPECTED_SOURCE_RECEIPT_SHA:?set source receipt SHA}"
: "${RUN_ROOT:?set result root}"
PY=/scratch/lg154/conda-envs/memnav/bin/python
OUT=${RUN_ROOT}/report.json
[[ ! -e "${OUT}" ]] || { echo "ABORT: report exists" >&2; exit 2; }
[[ "$(sha256sum "${SOURCE_RECEIPT}" | awk '{print $1}')" == \
  "${EXPECTED_SOURCE_RECEIPT_SHA}" ]] || { echo "ABORT: source SHA" >&2; exit 2; }
(cd "${SOURCE_ROOT}" && sha256sum -c "${SOURCE_RECEIPT}") >/dev/null
for receipt in data_manifest.json trace_receipt.json preparation.json; do
  (cd "${RUN_ROOT}" && sha256sum -c "${receipt}.sha256") >/dev/null
done
export PYTHONPATH=${SOURCE_ROOT}${PYTHONPATH:+:${PYTHONPATH}}
"${PY}" -m pytest -q \
  "${SOURCE_ROOT}/MemNavData/test_summarize_cdec_consumed_closed_loop.py"
"${PY}" "${SOURCE_ROOT}/MemNavData/summarize_cdec_consumed_closed_loop.py" \
  --manifest "${RUN_ROOT}/data_manifest.json" \
  --trace-receipt "${RUN_ROOT}/trace_receipt.json" \
  --run-root "${RUN_ROOT}" --out "${OUT}"
sha256sum "${OUT}" > "${OUT}.sha256"
chmod a-w "${OUT}" "${OUT}.sha256"
echo "COMPLETE ${OUT}"
