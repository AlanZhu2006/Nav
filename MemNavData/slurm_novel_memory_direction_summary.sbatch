#!/usr/bin/env bash
# Aggregate the complete consumed-development Novel causal control.
#SBATCH --job-name=novel_causal_sum
#SBATCH --output=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/%x_%j.out
#SBATCH --error=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/%x_%j.err
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=cpu_short
#SBATCH --account=torch_pr_769_tandon_advanced
#SBATCH --export=ALL

set -euo pipefail
umask 0022
: "${SOURCE_ROOT:?set immutable source root}"
: "${SOURCE_RECEIPT:?set source receipt}"
: "${EXPECTED_SOURCE_RECEIPT_SHA:?set receipt SHA}"
: "${RUN_ROOT:?set Novel-control run root}"
: "${CONTROL_MANIFEST:?set frozen control manifest}"
: "${EXPECTED_CONTROL_MANIFEST_SHA:?set control manifest SHA}"
PY=/scratch/lg154/conda-envs/memnav/bin/python

[[ "$(sha256sum "${SOURCE_RECEIPT}" | awk '{print $1}')" == \
   "${EXPECTED_SOURCE_RECEIPT_SHA}" ]]
(cd "${SOURCE_ROOT}" && sha256sum -c "${SOURCE_RECEIPT}") >/dev/null
[[ "$(sha256sum "${CONTROL_MANIFEST}" | awk '{print $1}')" == \
   "${EXPECTED_CONTROL_MANIFEST_SHA}" ]]
[[ ! -e "${RUN_ROOT}/novel_memory_direction_summary.json" ]]
export PYTHONPATH=${SOURCE_ROOT}:${SOURCE_ROOT}/MemNavData${PYTHONPATH:+:${PYTHONPATH}}
"${PY}" -u \
  "${SOURCE_ROOT}/MemNavData/summarize_novel_memory_direction_control.py" \
  --run-root "${RUN_ROOT}" --manifest "${CONTROL_MANIFEST}" \
  --out "${RUN_ROOT}/novel_memory_direction_summary.json"
