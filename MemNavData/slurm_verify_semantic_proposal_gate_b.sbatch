#!/usr/bin/env bash
# Independently recompute Gate B from raw per-history completions.
#SBATCH --job-name=cec_prop_b_verify
#SBATCH --output=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/%x_%j.out
#SBATCH --error=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/%x_%j.err
#SBATCH --time=00:20:00
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
SUMMARY=${RUN_ROOT}/semantic_proposal_gate_b_summary.json
OUTPUT=${RUN_ROOT}/semantic_proposal_gate_b_independent_verification.json

[[ "$(sha256sum "${SOURCE_RECEIPT}" | awk '{print $1}')" == \
   "${EXPECTED_SOURCE_RECEIPT_SHA}" ]]
(cd "${SOURCE_ROOT}" && sha256sum -c "${SOURCE_RECEIPT}") >/dev/null
[[ "$(sha256sum "${SUMMARY}" | awk '{print $1}')" == \
   "$(awk '{print $1}' "${SUMMARY}.sha256")" ]]
[[ ! -e "${OUTPUT}" ]]
export PYTHONPATH=${SOURCE_ROOT}:${SOURCE_ROOT}/MemNavData${PYTHONPATH:+:${PYTHONPATH}}
"${PY}" -u "${SOURCE_ROOT}/MemNavData/verify_semantic_proposal_gate_b.py" \
  --input-root "${RUN_ROOT}/records" --summary "${SUMMARY}" \
  --expected-count 28 --output "${OUTPUT}"
sha256sum "${OUTPUT}" > "${OUTPUT}.sha256"
