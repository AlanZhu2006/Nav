#!/usr/bin/env bash
#SBATCH --job-name=h3NatBxFinish
#SBATCH --output=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/%x_%j.out
#SBATCH --error=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/%x_%j.err
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --partition=cpu_short
#SBATCH --account=torch_pr_769_tandon_advanced

set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
: "${TASK_ROOT:?}" "${RUN_ROOT:?}" "${PROTOCOL:?}" "${EXPECTED_PROTOCOL_SHA:?}"
: "${TASK_RECEIPT:?}" "${EXPECTED_TASK_RECEIPT_SHA:?}"
: "${SERVER_SOURCE_ROOT:?}" "${SERVER_SOURCE_RECEIPT:?}" "${EXPECTED_SERVER_SOURCE_RECEIPT_SHA:?}"
: "${TABLE2_SERVER_SOURCE_ROOT:?}" "${TABLE2_SERVER_SOURCE_RECEIPT:?}"
: "${EXPECTED_TABLE2_SERVER_SOURCE_RECEIPT_SHA:?}"
: "${BASE_SOURCE_ROOT:?}" "${BASE_RECEIPT:?}" "${EXPECTED_BASE_RECEIPT_SHA:?}"
: "${PARENT_ROOT:?}" "${FACTUAL_B_ARRAY_JOB:?}" "${PREFIX_CONCURRENCY:?}"
[[ "${PREFIX_CONCURRENCY}" == 4 ]]
[[ "$(find "${RUN_ROOT}/factual_b" -mindepth 2 -maxdepth 2 \
  -name completion.json -type f | wc -l)" -eq 84 ]]
while IFS= read -r sidecar; do
  (cd "$(dirname "${sidecar}")" && sha256sum -c --quiet "$(basename "${sidecar}")")
done < <(find "${RUN_ROOT}/factual_b" -mindepth 2 -maxdepth 2 \
  -name completion.json.sha256 -type f | sort)
[[ "$(find "${RUN_ROOT}/factual_b" -mindepth 2 -maxdepth 2 \
  -name completion.json.sha256 -type f | wc -l)" -eq 84 ]]
[[ ! -e "${RUN_ROOT}/prefix_fragments" && ! -e "${RUN_ROOT}/population" ]]
prefix=${TASK_ROOT}/MemNavData/slurm_hm3d_fullmono_lifelong_natural_v4_construct_prefix.sbatch
finalize=${TASK_ROOT}/MemNavData/slurm_hm3d_fullmono_lifelong_natural_b_expansion_finalize_verify.sbatch
union_launch=${TASK_ROOT}/MemNavData/slurm_hm3d_fullmono_lifelong_population_union_launch_table2.sbatch
safe=${TASK_ROOT}/MemNavData/slurm_safe_submit.sh
common="ALL,TASK_ROOT=${TASK_ROOT},TASK_RECEIPT=${TASK_RECEIPT},EXPECTED_TASK_RECEIPT_SHA=${EXPECTED_TASK_RECEIPT_SHA},SERVER_SOURCE_ROOT=${SERVER_SOURCE_ROOT},SERVER_SOURCE_RECEIPT=${SERVER_SOURCE_RECEIPT},EXPECTED_SERVER_SOURCE_RECEIPT_SHA=${EXPECTED_SERVER_SOURCE_RECEIPT_SHA},TABLE2_SERVER_SOURCE_ROOT=${TABLE2_SERVER_SOURCE_ROOT},TABLE2_SERVER_SOURCE_RECEIPT=${TABLE2_SERVER_SOURCE_RECEIPT},EXPECTED_TABLE2_SERVER_SOURCE_RECEIPT_SHA=${EXPECTED_TABLE2_SERVER_SOURCE_RECEIPT_SHA},BASE_SOURCE_ROOT=${BASE_SOURCE_ROOT},BASE_RECEIPT=${BASE_RECEIPT},EXPECTED_BASE_RECEIPT_SHA=${EXPECTED_BASE_RECEIPT_SHA},RUN_ROOT=${RUN_ROOT},PARENT_ROOT=${PARENT_ROOT},PROTOCOL=${PROTOCOL},EXPECTED_PROTOCOL_SHA=${EXPECTED_PROTOCOL_SHA}"
source "${safe}"
safe_sbatch --lint-fatal --test-only --qos=gpu48 --time=00:45:00 \
  --array=0 --export="${common}" "${prefix}" >/dev/null
safe_sbatch --lint-fatal --test-only --partition=cpu_short \
  --export="${common}" "${finalize}" >/dev/null
safe_sbatch --lint-fatal --test-only --partition=cpu_short \
  --export="${common},PARENT_MANIFEST=${PARENT_ROOT}/sealed_inputs/parent_manifest.json" \
  "${union_launch}" >/dev/null
p_raw=$(safe_sbatch --lint-fatal --parsable --qos=gpu48 --time=00:45:00 \
  --array=0-83%4 --export="${common}" "${prefix}")
p_job=$(printf '%s\n' "${p_raw}" | tr -d '\r' | awk -F';' '/^[0-9]+(;|$)/ {print $1; exit}')
[[ "${p_job}" =~ ^[0-9]+$ ]]
v_raw=$(safe_sbatch --lint-fatal --parsable --partition=cpu_short \
  --dependency="afterany:${p_job}" --export="${common}" "${finalize}")
v_job=$(printf '%s\n' "${v_raw}" | tr -d '\r' | awk -F';' '/^[0-9]+(;|$)/ {print $1; exit}')
[[ "${v_job}" =~ ^[0-9]+$ ]]
u_raw=$(safe_sbatch --lint-fatal --parsable --partition=cpu_short \
  --dependency="afterok:${v_job}" --kill-on-invalid-dep=yes \
  --export="${common},PARENT_MANIFEST=${PARENT_ROOT}/sealed_inputs/parent_manifest.json" \
  "${union_launch}")
u_job=$(printf '%s\n' "${u_raw}" | tr -d '\r' | awk -F';' '/^[0-9]+(;|$)/ {print $1; exit}')
[[ "${u_job}" =~ ^[0-9]+$ ]]
receipt=${RUN_ROOT}/deferred_submission/expansion_prefix.json
/scratch/lg154/conda-envs/memnav/bin/python - "${receipt}" \
  "${FACTUAL_B_ARRAY_JOB}" "${p_job}" "${v_job}" "${u_job}" <<'PY'
import json,sys
path,b,prefix,verify,union=sys.argv[1:]
payload={'schema_version':'hm3d_natural_b_expansion_prefix_submission_v1_20260830',
 'factual_B_array_job':int(b),'factual_B_completions':84,
 'prefix_array_job':int(prefix),'prefix_array':'0-83%4',
 'population_finalize_and_verify_job':int(verify),
 'population_union_and_table2_launcher_job':int(union),
 'leg3_jobs_submitted':False,'navigation_outcomes_read_by_launcher':False}
open(path,'x').write(json.dumps(payload,indent=2,sort_keys=True)+'\n')
PY
sha256sum "${receipt}" >"${receipt}.sha256"
chmod a-w "${receipt}" "${receipt}.sha256"
