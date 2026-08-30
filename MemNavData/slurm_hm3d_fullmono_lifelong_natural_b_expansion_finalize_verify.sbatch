#!/usr/bin/env bash
#SBATCH --job-name=h3NatBxPopV
#SBATCH --output=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/%x_%j.out
#SBATCH --error=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/%x_%j.err
#SBATCH --time=00:40:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --partition=cpu_short
#SBATCH --account=torch_pr_769_tandon_advanced

set -euo pipefail
: "${TASK_ROOT:?}" "${RUN_ROOT:?}" "${PROTOCOL:?}"
: "${TASK_RECEIPT:?}" "${EXPECTED_TASK_RECEIPT_SHA:?}"
: "${SERVER_SOURCE_ROOT:?}" "${SERVER_SOURCE_RECEIPT:?}" "${EXPECTED_SERVER_SOURCE_RECEIPT_SHA:?}"
: "${BASE_SOURCE_ROOT:?}" "${BASE_RECEIPT:?}" "${EXPECTED_BASE_RECEIPT_SHA:?}"
for spec in \
  "${TASK_RECEIPT}:${EXPECTED_TASK_RECEIPT_SHA}:${TASK_ROOT}" \
  "${SERVER_SOURCE_RECEIPT}:${EXPECTED_SERVER_SOURCE_RECEIPT_SHA}:${SERVER_SOURCE_ROOT}" \
  "${BASE_RECEIPT}:${EXPECTED_BASE_RECEIPT_SHA}:${BASE_SOURCE_ROOT}"; do
  IFS=: read -r receipt expected root <<<"${spec}"
  [[ "$(sha256sum "${receipt}" | awk '{print $1}')" == "${expected}" ]]
  (cd "${root}" && sha256sum -c --quiet "${receipt}")
done
[[ "$(sha256sum "${PROTOCOL}" | awk '{print $1}')" == \
   "${EXPECTED_PROTOCOL_SHA}" ]]
[[ "$(find "${RUN_ROOT}/prefix_fragments" -mindepth 2 -maxdepth 2 \
  -name completion.json -type f | wc -l)" -eq 84 ]]
py=/scratch/lg154/conda-envs/memnav/bin/python
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPYCACHEPREFIX=${SLURM_TMPDIR:-/tmp}/h3_natbx_popv_${SLURM_JOB_ID}
export PYTHONPATH=${TASK_ROOT}:${TASK_ROOT}/MemNavData:${SERVER_SOURCE_ROOT}:${SERVER_SOURCE_ROOT}/MemNavData:${BASE_SOURCE_ROOT}:${BASE_SOURCE_ROOT}/MemNavData
"${py}" -u "${TASK_ROOT}/MemNavData/finalize_hm3d_fullmono_lifelong_population.py" \
  --protocol "${PROTOCOL}" --ab-root "${RUN_ROOT}/ab_population" \
  --fragments "${RUN_ROOT}/prefix_fragments" --out "${RUN_ROOT}/population"
(cd "${RUN_ROOT}/population" && \
  sha256sum -c --quiet population.json.sha256 && \
  sha256sum -c --quiet POPULATION_FILES.sha256)
out=${RUN_ROOT}/independent_natural_b_expansion_population_verification.json
[[ ! -e "${out}" && ! -e "${out}.sha256" ]]
"${py}" -m unittest -q \
  test_independent_verify_hm3d_fullmono_lifelong_natural_v4_population
"${py}" -u \
  "${TASK_ROOT}/MemNavData/independent_verify_hm3d_fullmono_lifelong_natural_v4_population.py" \
  --run-root "${RUN_ROOT}" --protocol "${PROTOCOL}" --out "${out}"
(cd "${RUN_ROOT}" && sha256sum -c --quiet \
  independent_natural_b_expansion_population_verification.json.sha256)
chmod -R a-w "${RUN_ROOT}/factual_b" "${RUN_ROOT}/prefix_fragments" \
  "${RUN_ROOT}/population" "${out}" "${out}.sha256"
