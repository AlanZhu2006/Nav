#!/usr/bin/env bash
#SBATCH --job-name=cec_prop_sum
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
: "${SOURCE_ROOT:?set immutable source root}"
: "${SOURCE_RECEIPT:?set source receipt}"
: "${EXPECTED_SOURCE_RECEIPT_SHA:?set source receipt SHA}"
: "${RUN_ROOT:?set diagnostic run root}"
: "${EXPECTED_COUNT:?set frozen population size}"

PY=/scratch/lg154/conda-envs/memnav/bin/python
[[ "$(sha256sum "${SOURCE_RECEIPT}" | awk '{print $1}')" == \
   "${EXPECTED_SOURCE_RECEIPT_SHA}" ]]
(cd "${SOURCE_ROOT}" && sha256sum -c "${SOURCE_RECEIPT}") >/dev/null
[[ "${EXPECTED_COUNT}" =~ ^[1-9][0-9]*$ ]]
[[ ! -e "${RUN_ROOT}/summary.json" ]]
"${PY}" "${SOURCE_ROOT}/MemNavData/summarize_certified_proposal_counterfactual.py" \
  --input-root "${RUN_ROOT}/records" --expected-count "${EXPECTED_COUNT}" \
  --out "${RUN_ROOT}/summary.json"
sha256sum "${RUN_ROOT}/summary.json" > "${RUN_ROOT}/summary.json.sha256"
