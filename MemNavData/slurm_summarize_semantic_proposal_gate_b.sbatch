#!/usr/bin/env bash
# Aggregate the complete 28-history consumed Gate-B comparison.
#SBATCH --job-name=cec_prop_b_sum
#SBATCH --output=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/%x_%j.out
#SBATCH --error=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/%x_%j.err
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --partition=cpu_short
#SBATCH --account=torch_pr_769_tandon_advanced
#SBATCH --export=ALL

set -euo pipefail
umask 0022
: "${SOURCE_ROOT:?set immutable task source root}"
: "${SOURCE_RECEIPT:?set task source receipt}"
: "${EXPECTED_SOURCE_RECEIPT_SHA:?set task receipt SHA}"
: "${RUN_ROOT:?set Gate-B run root}"
PY=/scratch/lg154/conda-envs/memnav/bin/python

[[ "$(sha256sum "${SOURCE_RECEIPT}" | awk '{print $1}')" == \
   "${EXPECTED_SOURCE_RECEIPT_SHA}" ]]
(cd "${SOURCE_ROOT}" && sha256sum -c "${SOURCE_RECEIPT}") >/dev/null
[[ ! -e "${RUN_ROOT}/semantic_proposal_gate_b_summary.json" ]]
export PYTHONPATH=${SOURCE_ROOT}:${SOURCE_ROOT}/MemNavData${PYTHONPATH:+:${PYTHONPATH}}
"${PY}" -u "${SOURCE_ROOT}/MemNavData/summarize_semantic_proposal_gate_b.py" \
  --input-root "${RUN_ROOT}/records" --expected-count 28 \
  --out "${RUN_ROOT}/semantic_proposal_gate_b_summary.json"
sha256sum "${RUN_ROOT}/semantic_proposal_gate_b_summary.json" \
  > "${RUN_ROOT}/semantic_proposal_gate_b_summary.json.sha256"
