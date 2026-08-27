#!/usr/bin/env bash
#SBATCH --job-name=f14ZeroAna
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
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPYCACHEPREFIX=${SLURM_TMPDIR:-/tmp}/f14_zero_analysis_${SLURM_JOB_ID}
: "${REPAIR_ROOT:?}" "${REPAIR_RECEIPT:?}"
: "${EXPECTED_REPAIR_RECEIPT_SHA:?}" "${BASE_SOURCE_ROOT:?}"
: "${BASE_SOURCE_RECEIPT:?}" "${EXPECTED_BASE_SOURCE_RECEIPT_SHA:?}"
: "${RUN_ROOT:?}" "${REFERENCE_ROOT:?}" "${BENCH_ROOT:?}"
[[ "$(sha256sum "${REPAIR_RECEIPT}" | awk '{print $1}')" == \
   "${EXPECTED_REPAIR_RECEIPT_SHA}" ]]
(cd "${REPAIR_ROOT}" && sha256sum -c --quiet "${REPAIR_RECEIPT}")
[[ "$(sha256sum "${BASE_SOURCE_RECEIPT}" | awk '{print $1}')" == \
   "${EXPECTED_BASE_SOURCE_RECEIPT_SHA}" ]]
(cd "${BASE_SOURCE_ROOT}" && \
  sha256sum -c --quiet "${BASE_SOURCE_RECEIPT}")

PY=/scratch/lg154/conda-envs/memnav/bin/python
OUT=${RUN_ROOT}/analysis
[[ ! -e "${OUT}" ]]
mkdir -p "${OUT}"
export PYTHONPATH=${REPAIR_ROOT}:${BASE_SOURCE_ROOT}
"${PY}" -m unittest -q MemNavData.test_final14_zero_depth
"${PY}" -u "${REPAIR_ROOT}/MemNavData/summarize_final14_zero_depth.py" \
  --run-root "${RUN_ROOT}" --reference-root "${REFERENCE_ROOT}" \
  --bench-root "${BENCH_ROOT}" \
  --expected-manifest-sha256 \
    7468703a9efbb10e801ffdd226911f696a30fa9432ef9ab486d3134f6e40fe6a \
  --out "${OUT}/final14_zero_depth_summary.json"
"${PY}" -u \
  "${REPAIR_ROOT}/MemNavData/independent_verify_final14_zero_depth.py" \
  --run-root "${RUN_ROOT}" --reference-root "${REFERENCE_ROOT}" \
  --bench-root "${BENCH_ROOT}" \
  --expected-manifest-sha256 \
    7468703a9efbb10e801ffdd226911f696a30fa9432ef9ab486d3134f6e40fe6a \
  --summary "${OUT}/final14_zero_depth_summary.json" \
  --out "${OUT}/final14_zero_depth_independent_verification.json"
(cd "${OUT}" && sha256sum -c --quiet \
  final14_zero_depth_summary.json.sha256 \
  final14_zero_depth_independent_verification.json.sha256)
chmod -R a-w "${OUT}"
