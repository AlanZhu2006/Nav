#!/usr/bin/env bash
#SBATCH --job-name=h3T3SurveyPop
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
: "${MODE:?finalize or verify}" "${SURVEY_SOURCE_ROOT:?}"
: "${SURVEY_SOURCE_RECEIPT:?}" "${EXPECTED_SURVEY_SOURCE_RECEIPT_SHA:?}"
: "${RUN_ROOT:?}" "${CANDIDATE_PLAN:?}" "${SURVEY_PROTOCOL:?}"
[[ "$(sha256sum "${SURVEY_SOURCE_RECEIPT}" | awk '{print $1}')" == \
   "${EXPECTED_SURVEY_SOURCE_RECEIPT_SHA}" ]]
(cd "${SURVEY_SOURCE_ROOT}" && sha256sum -c --quiet "${SURVEY_SOURCE_RECEIPT}")
export PYTHONPATH=${SURVEY_SOURCE_ROOT}:${SURVEY_SOURCE_ROOT}/MemNavData
py=/scratch/lg154/conda-envs/memnav/bin/python
population=${RUN_ROOT}/query_population
if [[ "${MODE}" == finalize ]]; then
  [[ "$(find "${RUN_ROOT}/construction_fragments" -mindepth 2 -maxdepth 2 \
      -name completion.json -type f | wc -l)" -eq 125 ]]
  "${py}" -u \
    "${SURVEY_SOURCE_ROOT}/MemNavData/finalize_hm3d_table3_causal_survey_population.py" \
    --run-root "${RUN_ROOT}" --candidate-plan "${CANDIDATE_PLAN}" \
    --protocol "${SURVEY_PROTOCOL}" --out "${population}"
elif [[ "${MODE}" == verify ]]; then
  "${py}" -u \
    "${SURVEY_SOURCE_ROOT}/MemNavData/independent_verify_hm3d_table3_causal_survey_population.py" \
    --population-root "${population}" \
    --out "${population}/independent_verification.json"
  (cd "${population}" && sha256sum -c --quiet independent_verification.json.sha256)
  chmod -R a-w "${population}"
else
  echo "invalid MODE=${MODE}" >&2
  exit 2
fi
