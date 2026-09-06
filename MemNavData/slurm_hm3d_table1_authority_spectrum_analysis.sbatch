#!/usr/bin/env bash
#SBATCH --job-name=h3AuthSpecAna
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
: "${TASK_ROOT:?}" "${RUNTIME_CLOSURE_ROOT:?}" "${FORMAL_RUN_ROOT:?}" "${BENCH_ROOT:?}"
: "${EXPECTED_MANIFEST_SHA:?}"
PY=/scratch/lg154/conda-envs/memnav/bin/python
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=${TASK_ROOT}:${TASK_ROOT}/MemNavData:${RUNTIME_CLOSURE_ROOT}:${RUNTIME_CLOSURE_ROOT}/MemNavData
out=${FORMAL_RUN_ROOT}/formal/POSTHOC
mkdir -p "${out}"
summary=${out}/hm3d_table1_authority_spectrum_summary.json
verification=${out}/hm3d_table1_authority_spectrum_independent_verification.json
"${PY}" "${TASK_ROOT}/MemNavData/summarize_hm3d_table1_authority_spectrum.py" \
  --run-root "${FORMAL_RUN_ROOT}/formal" --bench-root "${BENCH_ROOT}" \
  --expected-manifest-sha256 "${EXPECTED_MANIFEST_SHA}" \
  --expected-histories 28 --out "${summary}"
"${PY}" "${TASK_ROOT}/MemNavData/independent_verify_hm3d_table1_authority_spectrum.py" \
  --run-root "${FORMAL_RUN_ROOT}/formal" --bench-root "${BENCH_ROOT}" \
  --expected-manifest-sha256 "${EXPECTED_MANIFEST_SHA}" \
  --summary "${summary}" --out "${verification}"
