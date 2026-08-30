#!/usr/bin/env bash
#SBATCH --job-name=h3T3SurveyResult
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
: "${MODE:?analyze or verify}" "${SURVEY_SOURCE_ROOT:?}" "${RUN_ROOT:?}"
: "${SURVEY_SOURCE_RECEIPT:?}" "${EXPECTED_SURVEY_SOURCE_RECEIPT_SHA:?}"
: "${EXPECTED_POPULATION_VERIFICATION_SHA:?}"
: "${EXPECTED_BENCHMARK_MANIFEST_SHA:?}"
POPULATION_RELATIVE_ROOT=${POPULATION_RELATIVE_ROOT:-query_population}
[[ "${POPULATION_RELATIVE_ROOT}" == query_population \
   || "${POPULATION_RELATIVE_ROOT}" == merged_query_population ]] || {
  echo "invalid Table-III population root" >&2; exit 2; }
[[ "$(sha256sum "${SURVEY_SOURCE_RECEIPT}" | awk '{print $1}')" == \
   "${EXPECTED_SURVEY_SOURCE_RECEIPT_SHA}" ]]
(cd "${SURVEY_SOURCE_ROOT}" && \
  sha256sum -c --quiet "${SURVEY_SOURCE_RECEIPT}")
population=${RUN_ROOT}/${POPULATION_RELATIVE_ROOT}
[[ "$(sha256sum "${population}/independent_verification.json" | awk '{print $1}')" == \
   "${EXPECTED_POPULATION_VERIFICATION_SHA}" ]]
[[ "$(sha256sum "${population}/role_pairs/manifest.json" | awk '{print $1}')" == \
   "${EXPECTED_BENCHMARK_MANIFEST_SHA}" ]]
export PYTHONPATH=${SURVEY_SOURCE_ROOT}:${SURVEY_SOURCE_ROOT}/MemNavData
py=/scratch/lg154/conda-envs/memnav/bin/python
summary=${RUN_ROOT}/table3_result/summary.json
if [[ "${MODE}" == analyze ]]; then
  [[ "$(find "${RUN_ROOT}/evaluation/natural_direction" \
      -mindepth 2 -maxdepth 2 -name completion.json -type f | wc -l)" -eq 48 ]]
  "${py}" -u \
    "${SURVEY_SOURCE_ROOT}/MemNavData/analyze_hm3d_table3_causal_survey.py" \
    --run-root "${RUN_ROOT}" \
    --population-relative-root "${POPULATION_RELATIVE_ROOT}" \
    --out "${summary}"
elif [[ "${MODE}" == verify ]]; then
  (cd "$(dirname "${summary}")" && \
    sha256sum -c --quiet summary.json.sha256)
  "${py}" -u \
    "${SURVEY_SOURCE_ROOT}/MemNavData/independent_verify_hm3d_table3_causal_survey_result.py" \
    --run-root "${RUN_ROOT}" --summary "${summary}" \
    --population-relative-root "${POPULATION_RELATIVE_ROOT}" \
    --out "${RUN_ROOT}/table3_result/independent_verification.json"
  (cd "${RUN_ROOT}/table3_result" && \
    sha256sum -c --quiet independent_verification.json.sha256)
  chmod -R a-w "${RUN_ROOT}/table3_result"
else
  echo "invalid MODE=${MODE}" >&2
  exit 2
fi
