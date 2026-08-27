#!/bin/bash
#SBATCH --job-name=f14MonoSum
#SBATCH --output=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/f14MonoSum_%j.out
#SBATCH --error=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/f14MonoSum_%j.err
#SBATCH --time=00:25:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=cpu_short
#SBATCH --account=torch_pr_769_tandon_advanced
#SBATCH --export=ALL

set -euo pipefail
: "${SOURCE_ROOT:?set SOURCE_ROOT}"
: "${RUN_ROOT:?set RUN_ROOT}"
: "${BENCH_ROOT:?set BENCH_ROOT}"
: "${SOURCE_RECEIPT:?set SOURCE_RECEIPT}"
: "${EXPECTED_SOURCE_RECEIPT_SHA:?set receipt SHA}"
MEMNAV_PY=/scratch/lg154/conda-envs/memnav/bin/python
MANIFEST_SHA=7468703a9efbb10e801ffdd226911f696a30fa9432ef9ab486d3134f6e40fe6a
[[ "$(sha256sum "${SOURCE_RECEIPT}" | awk '{print $1}')" == "${EXPECTED_SOURCE_RECEIPT_SHA}" ]]
(cd "${SOURCE_ROOT}" && sha256sum -c --quiet "${SOURCE_RECEIPT}")
mkdir -p "${RUN_ROOT}/POSTHOC"
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="${SOURCE_ROOT}" "${MEMNAV_PY}" \
  "${SOURCE_ROOT}/MemNavData/summarize_final14_mono_factorial.py" \
    --run-root "${RUN_ROOT}" --bench-root "${BENCH_ROOT}" \
    --expected-manifest-sha256 "${MANIFEST_SHA}" \
    --out "${RUN_ROOT}/POSTHOC/final14_mono_factorial_summary.json"
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="${SOURCE_ROOT}" "${MEMNAV_PY}" \
  "${SOURCE_ROOT}/MemNavData/independent_verify_final14_mono_factorial.py" \
    --run-root "${RUN_ROOT}" --bench-root "${BENCH_ROOT}" \
    --expected-manifest-sha256 "${MANIFEST_SHA}" \
    --summary "${RUN_ROOT}/POSTHOC/final14_mono_factorial_summary.json" \
    --out "${RUN_ROOT}/POSTHOC/final14_mono_factorial_independent_verification.json"
sha256sum "${RUN_ROOT}/POSTHOC/"*.json >"${RUN_ROOT}/POSTHOC/output_receipt.sha256"
chmod a-w "${RUN_ROOT}/POSTHOC/"*.json "${RUN_ROOT}/POSTHOC/output_receipt.sha256"
