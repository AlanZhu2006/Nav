#!/usr/bin/env bash
#SBATCH --job-name=h3T3Analysis
#SBATCH --output=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/%x_%j.out
#SBATCH --error=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/%x_%j.err
#SBATCH --time=00:20:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --partition=cpu_short
#SBATCH --account=torch_pr_769_tandon_advanced

set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
: "${MODE:?analyze or verify}" "${WRAPPER_ROOT:?}" "${RUN_ROOT:?}"
: "${WRAPPER_RECEIPT:?}" "${EXPECTED_WRAPPER_RECEIPT_SHA:?}"
[[ "$(sha256sum "${WRAPPER_RECEIPT}" | awk '{print $1}')" == \
   "${EXPECTED_WRAPPER_RECEIPT_SHA}" ]]
(cd "${WRAPPER_ROOT}" && sha256sum -c --quiet "${WRAPPER_RECEIPT}")
export PYTHONPATH=${WRAPPER_ROOT}:${WRAPPER_ROOT}/MemNavData
py=/scratch/lg154/conda-envs/memnav/bin/python
summary=${RUN_ROOT}/table3_result/summary.json
if [[ "${MODE}" == analyze ]]; then
  [[ "$(find "${RUN_ROOT}/evaluation/natural_direction" -mindepth 2 -maxdepth 2 -name completion.json -type f | wc -l)" -eq 48 ]]
  "${py}" -u "${WRAPPER_ROOT}/MemNavData/analyze_hm3d_table3_actual_mono.py" \
    --run-root "${RUN_ROOT}" --out "${summary}"
elif [[ "${MODE}" == verify ]]; then
  (cd "$(dirname "${summary}")" && sha256sum -c --quiet summary.json.sha256)
  "${py}" -u "${WRAPPER_ROOT}/MemNavData/independent_verify_hm3d_table3_actual_mono_result.py" \
    --run-root "${RUN_ROOT}" --summary "${summary}" \
    --out "${RUN_ROOT}/table3_result/independent_verification.json"
else
  echo "invalid MODE=${MODE}" >&2; exit 2
fi
