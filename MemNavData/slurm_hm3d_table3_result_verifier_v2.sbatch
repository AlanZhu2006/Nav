#!/usr/bin/env bash
#SBATCH --job-name=h3T3VerifyV2
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
: "${WRAPPER_ROOT:?}" "${WRAPPER_RECEIPT:?}" "${EXPECTED_WRAPPER_RECEIPT_SHA:?}"
: "${RUN_ROOT:?}"
[[ "$(sha256sum "${WRAPPER_RECEIPT}" | awk '{print $1}')" == \
   "${EXPECTED_WRAPPER_RECEIPT_SHA}" ]]
(cd "${WRAPPER_ROOT}" && sha256sum -c --quiet "${WRAPPER_RECEIPT}")
result=${RUN_ROOT}/table3_result
(cd "${result}" && sha256sum -c --quiet summary.json.sha256)
(cd "${result}" && sha256sum -c --quiet independent_verification.json.sha256)
py=/scratch/lg154/conda-envs/memnav/bin/python
"${py}" -u \
  "${WRAPPER_ROOT}/MemNavData/independent_verify_hm3d_table3_actual_mono_result.py" \
  --run-root "${RUN_ROOT}" --summary "${result}/summary.json" \
  --out "${result}/independent_verification_v2.json"
(cd "${result}" && sha256sum -c --quiet independent_verification_v2.json.sha256)
