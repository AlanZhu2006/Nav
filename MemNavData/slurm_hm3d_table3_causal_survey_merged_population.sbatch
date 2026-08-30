#!/usr/bin/env bash
#SBATCH --job-name=h3T3MergePop
#SBATCH --output=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/%x_%j.out
#SBATCH --error=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/%x_%j.err
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --partition=cpu_short
#SBATCH --account=torch_pr_769_tandon_advanced

set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
: "${MODE:?finalize or verify}" "${EXPANSION_EXEC_SOURCE_ROOT:?}"
: "${EXPANSION_EXEC_SOURCE_RECEIPT:?}" "${EXPECTED_EXPANSION_EXEC_SOURCE_RECEIPT_SHA:?}"
: "${BASE_RUN_ROOT:?}" "${BASE_CANDIDATE_PLAN:?}" "${BASE_SURVEY_PROTOCOL:?}"
: "${EXPANSION_RUN_ROOT:?}" "${EXPANSION_PLAN:?}"
: "${EXPANSION_CONSTRUCTION_PROTOCOL:?}" "${EXPANSION_PLAN_VERIFICATION:?}"
: "${EXPECTED_EXPANSION_CANDIDATE_COUNT:?}"
[[ "$(sha256sum "${EXPANSION_EXEC_SOURCE_RECEIPT}" | awk '{print $1}')" == \
   "${EXPECTED_EXPANSION_EXEC_SOURCE_RECEIPT_SHA}" ]]
(cd "${EXPANSION_EXEC_SOURCE_ROOT}" && \
  sha256sum -c --quiet "${EXPANSION_EXEC_SOURCE_RECEIPT}")
export PYTHONPATH=${EXPANSION_EXEC_SOURCE_ROOT}:${EXPANSION_EXEC_SOURCE_ROOT}/MemNavData
py=/scratch/lg154/conda-envs/memnav/bin/python
population=${EXPANSION_RUN_ROOT}/merged_query_population

if [[ "${MODE}" == finalize ]]; then
  [[ "$(find "${BASE_RUN_ROOT}/construction_fragments" -mindepth 2 -maxdepth 2 \
      -name completion.json -type f | wc -l)" -eq 125 ]]
  [[ "$(find "${EXPANSION_RUN_ROOT}/construction_fragments" -mindepth 2 -maxdepth 2 \
      -name completion.json -type f | wc -l)" \
      -eq "${EXPECTED_EXPANSION_CANDIDATE_COUNT}" ]]
  "${py}" -u \
    "${EXPANSION_EXEC_SOURCE_ROOT}/MemNavData/finalize_hm3d_table3_causal_survey_merged_population.py" \
    --base-run-root "${BASE_RUN_ROOT}" --base-plan "${BASE_CANDIDATE_PLAN}" \
    --base-protocol "${BASE_SURVEY_PROTOCOL}" \
    --expansion-run-root "${EXPANSION_RUN_ROOT}" \
    --expansion-plan "${EXPANSION_PLAN}" \
    --expansion-protocol "${EXPANSION_CONSTRUCTION_PROTOCOL}" \
    --expansion-plan-verification "${EXPANSION_PLAN_VERIFICATION}" \
    --out "${population}"
elif [[ "${MODE}" == verify ]]; then
  "${py}" -u \
    "${EXPANSION_EXEC_SOURCE_ROOT}/MemNavData/independent_verify_hm3d_table3_causal_survey_merged_population.py" \
    --population-root "${population}" \
    --base-run-root "${BASE_RUN_ROOT}" --base-plan "${BASE_CANDIDATE_PLAN}" \
    --base-protocol "${BASE_SURVEY_PROTOCOL}" \
    --expansion-run-root "${EXPANSION_RUN_ROOT}" \
    --expansion-plan "${EXPANSION_PLAN}" \
    --expansion-protocol "${EXPANSION_CONSTRUCTION_PROTOCOL}" \
    --expansion-plan-verification "${EXPANSION_PLAN_VERIFICATION}" \
    --out "${population}/independent_verification.json"
  (cd "${population}" && \
    sha256sum -c --quiet independent_verification.json.sha256)
  chmod -R a-w "${population}"
else
  echo "invalid MODE=${MODE}" >&2
  exit 2
fi
