#!/usr/bin/env bash
# Aggregate all frozen role-pair policy outcomes.
#SBATCH --job-name=paper_pair_sum
#SBATCH --output=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/%x_%j.out
#SBATCH --error=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/%x_%j.err
#SBATCH --time=00:45:00
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
: "${RUN_ROOT:?set paper run root}"
INCLUDE_LEARNED_PI3X=${INCLUDE_LEARNED_PI3X:-0}
PY=/scratch/lg154/conda-envs/memnav/bin/python

[[ "${INCLUDE_LEARNED_PI3X}" == 0 || "${INCLUDE_LEARNED_PI3X}" == 1 ]]

[[ "$(sha256sum "${SOURCE_RECEIPT}" | awk '{print $1}')" == \
   "${EXPECTED_SOURCE_RECEIPT_SHA}" ]]
(cd "${SOURCE_ROOT}" && sha256sum -c "${SOURCE_RECEIPT}") >/dev/null
[[ ! -e "${RUN_ROOT}/paper_role_pair_summary.json" ]]
export PYTHONPATH=${SOURCE_ROOT}:${SOURCE_ROOT}/MemNavData${PYTHONPATH:+:${PYTHONPATH}}
learned_args=()
if [[ "${INCLUDE_LEARNED_PI3X}" == 1 ]]; then
  learned_args=(--include-learned-pi3x)
fi
"${PY}" -u "${SOURCE_ROOT}/MemNavData/summarize_paper_role_pair_eval.py" \
  --root "${RUN_ROOT}" --out "${RUN_ROOT}/paper_role_pair_summary.json" \
  "${learned_args[@]}"
sha256sum "${RUN_ROOT}/paper_role_pair_summary.json" \
  > "${RUN_ROOT}/paper_role_pair_summary.json.sha256"
