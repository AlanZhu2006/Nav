#!/usr/bin/env bash
#SBATCH --job-name=h3lifeV4PopSeal
#SBATCH --output=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/%x_%j.out
#SBATCH --error=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/%x_%j.err
#SBATCH --time=00:20:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --partition=cpu_short
#SBATCH --account=torch_pr_769_tandon_advanced

set -euo pipefail
: "${TASK_ROOT:?}" "${SERVER_SOURCE_ROOT:?}" "${BASE_SOURCE_ROOT:?}"
: "${RUN_ROOT:?}" "${PROTOCOL:?}"
: "${TASK_RECEIPT:?}" "${EXPECTED_TASK_RECEIPT_SHA:?}"
: "${SERVER_SOURCE_RECEIPT:?}" "${EXPECTED_SERVER_SOURCE_RECEIPT_SHA:?}"
: "${BASE_RECEIPT:?}" "${EXPECTED_BASE_RECEIPT_SHA:?}"
[[ "$(sha256sum "${TASK_RECEIPT}" | awk '{print $1}')" == \
   "${EXPECTED_TASK_RECEIPT_SHA}" ]]
(cd "${TASK_ROOT}" && sha256sum -c --quiet "${TASK_RECEIPT}")
[[ "$(sha256sum "${SERVER_SOURCE_RECEIPT}" | awk '{print $1}')" == \
   "${EXPECTED_SERVER_SOURCE_RECEIPT_SHA}" ]]
(cd "${SERVER_SOURCE_ROOT}" && \
  sha256sum -c --quiet "${SERVER_SOURCE_RECEIPT}")
[[ "$(sha256sum "${BASE_RECEIPT}" | awk '{print $1}')" == \
   "${EXPECTED_BASE_RECEIPT_SHA}" ]]
(cd "${BASE_SOURCE_ROOT}" && sha256sum -c --quiet "${BASE_RECEIPT}")
[[ "$(find "${RUN_ROOT}/prefix_fragments" -mindepth 2 -maxdepth 2 \
  -name completion.json -type f | wc -l)" -eq 99 ]]
PY=/scratch/lg154/conda-envs/memnav/bin/python
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPYCACHEPREFIX=${SLURM_TMPDIR:-/tmp}/h3life_v4_pop_${SLURM_JOB_ID}
export PYTHONPATH=${TASK_ROOT}:${TASK_ROOT}/MemNavData:${SERVER_SOURCE_ROOT}:${SERVER_SOURCE_ROOT}/MemNavData:${BASE_SOURCE_ROOT}:${BASE_SOURCE_ROOT}/MemNavData
"${PY}" -u \
  "${TASK_ROOT}/MemNavData/finalize_hm3d_fullmono_lifelong_population.py" \
  --protocol "${PROTOCOL}" --ab-root "${RUN_ROOT}/ab_population" \
  --fragments "${RUN_ROOT}/prefix_fragments" \
  --out "${RUN_ROOT}/population"
(cd "${RUN_ROOT}/population" && \
  sha256sum -c --quiet population.json.sha256 && \
  sha256sum -c --quiet POPULATION_FILES.sha256)
chmod -R a-w "${RUN_ROOT}/factual_b" "${RUN_ROOT}/prefix_fragments" \
  "${RUN_ROOT}/population"
