#!/usr/bin/env bash
# Submit the 99-element prefix array only after the factual-B array has left
# gpu48.  This preserves the site's one-large-array-resident constraint.
#SBATCH --job-name=h3lifeV4BDefer
#SBATCH --output=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/%x_%j.out
#SBATCH --error=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/%x_%j.err
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=6G
#SBATCH --partition=cpu_short
#SBATCH --account=torch_pr_769_tandon_advanced

set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
: "${TASK_ROOT:?}" "${TASK_RECEIPT:?}" "${EXPECTED_TASK_RECEIPT_SHA:?}"
: "${VERIFY_ROOT:?}" "${VERIFY_RECEIPT:?}" "${EXPECTED_VERIFY_RECEIPT_SHA:?}"
: "${SERVER_SOURCE_ROOT:?}" "${SERVER_SOURCE_RECEIPT:?}"
: "${EXPECTED_SERVER_SOURCE_RECEIPT_SHA:?}"
: "${BASE_SOURCE_ROOT:?}" "${BASE_RECEIPT:?}" "${EXPECTED_BASE_RECEIPT_SHA:?}"
: "${RUN_ROOT:?}" "${PARENT_ROOT:?}" "${PROTOCOL:?}"
: "${EXPECTED_PROTOCOL_SHA:?}" "${SHARD_MANIFEST:?}"
: "${EXPECTED_SHARD_MANIFEST_SHA:?}" "${FACTUAL_B_ARRAY_JOB:?}"
: "${PREFIX_CONCURRENCY:?}"
[[ "${PREFIX_CONCURRENCY}" == 4 ]]
[[ "$(sha256sum "${TASK_RECEIPT}" | awk '{print $1}')" == \
   "${EXPECTED_TASK_RECEIPT_SHA}" ]]
(cd "${TASK_ROOT}" && sha256sum -c --quiet "${TASK_RECEIPT}")
[[ "$(sha256sum "${VERIFY_RECEIPT}" | awk '{print $1}')" == \
   "${EXPECTED_VERIFY_RECEIPT_SHA}" ]]
(cd "${VERIFY_ROOT}" && sha256sum -c --quiet "${VERIFY_RECEIPT}")
[[ "$(sha256sum "${SERVER_SOURCE_RECEIPT}" | awk '{print $1}')" == \
   "${EXPECTED_SERVER_SOURCE_RECEIPT_SHA}" ]]
(cd "${SERVER_SOURCE_ROOT}" && \
  sha256sum -c --quiet "${SERVER_SOURCE_RECEIPT}")
[[ "$(sha256sum "${BASE_RECEIPT}" | awk '{print $1}')" == \
   "${EXPECTED_BASE_RECEIPT_SHA}" ]]
(cd "${BASE_SOURCE_ROOT}" && sha256sum -c --quiet "${BASE_RECEIPT}")
[[ "$(sha256sum "${PROTOCOL}" | awk '{print $1}')" == \
   "${EXPECTED_PROTOCOL_SHA}" ]]
[[ "$(sha256sum "${SHARD_MANIFEST}" | awk '{print $1}')" == \
   "${EXPECTED_SHARD_MANIFEST_SHA}" ]]
(cd "$(dirname "${SHARD_MANIFEST}")" && \
  sha256sum -c --quiet "$(basename "${SHARD_MANIFEST}").sha256")

# Completeness only: no reached_B/support value is opened here.
[[ "$(find "${RUN_ROOT}/factual_b" -mindepth 2 -maxdepth 2 \
  -name completion.json -type f | wc -l)" -eq 99 ]]
while IFS= read -r sidecar; do
  (cd "$(dirname "${sidecar}")" && \
    sha256sum -c --quiet "$(basename "${sidecar}")")
done < <(find "${RUN_ROOT}/factual_b" -mindepth 2 -maxdepth 2 \
  -name completion.json.sha256 -type f | sort)
[[ "$(find "${RUN_ROOT}/factual_b" -mindepth 2 -maxdepth 2 \
  -name completion.json.sha256 -type f | wc -l)" -eq 99 ]]
[[ ! -e "${RUN_ROOT}/prefix_fragments" ]]
[[ ! -e "${RUN_ROOT}/population" ]]
[[ ! -e "${RUN_ROOT}/independent_natural_v4_population_verification.json" ]]

prefix=${TASK_ROOT}/MemNavData/slurm_hm3d_fullmono_lifelong_natural_v4_construct_prefix.sbatch
finalize_verify=${VERIFY_ROOT}/MemNavData/slurm_hm3d_fullmono_lifelong_natural_v4_finalize_and_verify_population.sbatch
safe=${TASK_ROOT}/MemNavData/slurm_safe_submit.sh
for path in "${prefix}" "${finalize_verify}" "${safe}"; do
  [[ -r "${path}" ]]
done
common="ALL,TASK_ROOT=${TASK_ROOT},SERVER_SOURCE_ROOT=${SERVER_SOURCE_ROOT},BASE_SOURCE_ROOT=${BASE_SOURCE_ROOT},RUN_ROOT=${RUN_ROOT},PARENT_ROOT=${PARENT_ROOT},PROTOCOL=${PROTOCOL},TASK_RECEIPT=${TASK_RECEIPT},EXPECTED_TASK_RECEIPT_SHA=${EXPECTED_TASK_RECEIPT_SHA},SERVER_SOURCE_RECEIPT=${SERVER_SOURCE_RECEIPT},EXPECTED_SERVER_SOURCE_RECEIPT_SHA=${EXPECTED_SERVER_SOURCE_RECEIPT_SHA},BASE_RECEIPT=${BASE_RECEIPT},EXPECTED_BASE_RECEIPT_SHA=${EXPECTED_BASE_RECEIPT_SHA},VERIFY_ROOT=${VERIFY_ROOT},VERIFY_RECEIPT=${VERIFY_RECEIPT},EXPECTED_VERIFY_RECEIPT_SHA=${EXPECTED_VERIFY_RECEIPT_SHA},EXPECTED_PROTOCOL_SHA=${EXPECTED_PROTOCOL_SHA}"
source "${safe}"
safe_sbatch --lint-fatal --test-only --qos=gpu48 --array=0 \
  --export="${common}" "${prefix}" >/dev/null
safe_sbatch --lint-fatal --test-only --partition=cpu_short \
  --export="${common}" "${finalize_verify}" >/dev/null
prefix_raw=$(safe_sbatch --lint-fatal --parsable --qos=gpu48 \
  --array="0-98%${PREFIX_CONCURRENCY}" --export="${common}" "${prefix}")
prefix_job=$(printf '%s\n' "${prefix_raw}" | tr -d '\r' | \
  awk -F';' '/^[0-9]+(;|$)/ {print $1; exit}')
[[ "${prefix_job}" =~ ^[0-9]+$ ]]
seal_raw=$(safe_sbatch --lint-fatal --parsable --partition=cpu_short \
  --dependency="afterany:${prefix_job}" --export="${common}" \
  "${finalize_verify}")
seal_job=$(printf '%s\n' "${seal_raw}" | tr -d '\r' | \
  awk -F';' '/^[0-9]+(;|$)/ {print $1; exit}')
[[ "${seal_job}" =~ ^[0-9]+$ ]]

receipt_root=${RUN_ROOT}/deferred_submission
mkdir -p "${receipt_root}"
receipt=${receipt_root}/natural_v4_prefix.json
[[ ! -e "${receipt}" && ! -e "${receipt}.sha256" ]]
/scratch/lg154/conda-envs/memnav/bin/python - "${receipt}" \
  "${FACTUAL_B_ARRAY_JOB}" "${prefix_job}" "${seal_job}" <<'PY'
import json, sys
path, factual_b, prefix, seal = sys.argv[1:]
payload = {
    "schema_version": "hm3d_fullmono_lifelong_natural_v4_deferred_prefix_v1_20260828",
    "factual_B_array_job": int(factual_b),
    "factual_B_completions": 99,
    "factual_B_navigation_outcomes_read_by_launcher": False,
    "prefix_array_job": int(prefix),
    "prefix_array": "0-98%4",
    "population_finalize_and_independent_verify_job": int(seal),
    "one_large_gpu_array_resident_per_stage": True,
    "C_B2_C2_jobs_submitted": False,
}
open(path, "x").write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
print(json.dumps(payload, sort_keys=True))
PY
sha256sum "${receipt}" >"${receipt}.sha256"
chmod a-w "${receipt}" "${receipt}.sha256"
