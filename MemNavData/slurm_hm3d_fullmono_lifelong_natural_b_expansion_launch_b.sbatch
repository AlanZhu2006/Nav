#!/usr/bin/env bash
#SBATCH --job-name=h3NatBxBLaunch
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
: "${TASK_ROOT:?}" "${TASK_RECEIPT:?}" "${EXPECTED_TASK_RECEIPT_SHA:?}"
: "${SERVER_SOURCE_ROOT:?}" "${SERVER_SOURCE_RECEIPT:?}" "${EXPECTED_SERVER_SOURCE_RECEIPT_SHA:?}"
: "${TABLE2_SERVER_SOURCE_ROOT:?}" "${TABLE2_SERVER_SOURCE_RECEIPT:?}"
: "${EXPECTED_TABLE2_SERVER_SOURCE_RECEIPT_SHA:?}"
: "${BASE_SOURCE_ROOT:?}" "${BASE_RECEIPT:?}" "${EXPECTED_BASE_RECEIPT_SHA:?}"
: "${RUN_ROOT:?}" "${PARENT_ROOT:?}" "${PROTOCOL:?}" "${EXPECTED_PROTOCOL_SHA:?}"
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
verification=${RUN_ROOT}/independent_natural_b_expansion_materialization_verification.json
(cd "${RUN_ROOT}" && sha256sum -c --quiet \
  independent_natural_b_expansion_materialization_verification.json.sha256)
/scratch/lg154/conda-envs/memnav/bin/python - "${verification}" <<'PY'
import json,sys
p=json.load(open(sys.argv[1]))
assert p['verified'] is True and p['factual_B_gate_verified'] is True
assert p['materialized_candidates'] == 84 and p['scene_clusters'] == 30
assert p['factual_B_executed'] is False and p['navigation_outcomes_read'] is False
PY
[[ ! -e "${RUN_ROOT}/factual_b" && ! -e "${RUN_ROOT}/factual_b_schedule" ]]
schedule=${RUN_ROOT}/factual_b_schedule
mkdir -p "${schedule}"
py=/scratch/lg154/conda-envs/memnav/bin/python
"${py}" -u "${TASK_ROOT}/MemNavData/build_hm3d_fullmono_lifelong_natural_v4_b_shards.py" \
  --manifest "${RUN_ROOT}/ab_population/role_pairs/manifest.json" \
  --maximum-histories-per-shard 2 --out "${schedule}/shards.json"
readarray -t values < <("${py}" - "${schedule}/shards.json" <<'PY'
import json,sys
p=json.load(open(sys.argv[1])); print(p['candidate_histories']); print(p['shard_count'])
assert p['source_recipient_histories'] == 53 and p['scene_clusters'] == 30
assert p['maximum_histories_per_shard'] == 2
assert p['all_candidates_partitioned_once'] is True
assert p['query_policy_outcomes_read'] is False and p['navigation_outcomes_read'] is False
PY
)
[[ "${values[0]}" == 84 && "${values[1]}" -ge 30 \
   && "${values[1]}" -le 84 ]]
chmod -R a-w "${schedule}"
shard_manifest=${schedule}/shards.json
shard_sha=$(sha256sum "${shard_manifest}" | awk '{print $1}')
collect=${TASK_ROOT}/MemNavData/slurm_hm3d_fullmono_lifelong_natural_v4_collect_b_shard.sbatch
finish=${TASK_ROOT}/MemNavData/slurm_hm3d_fullmono_lifelong_natural_b_expansion_finish.sbatch
safe=${TASK_ROOT}/MemNavData/slurm_safe_submit.sh
common="ALL,TASK_ROOT=${TASK_ROOT},TASK_RECEIPT=${TASK_RECEIPT},EXPECTED_TASK_RECEIPT_SHA=${EXPECTED_TASK_RECEIPT_SHA},SERVER_SOURCE_ROOT=${SERVER_SOURCE_ROOT},SERVER_SOURCE_RECEIPT=${SERVER_SOURCE_RECEIPT},EXPECTED_SERVER_SOURCE_RECEIPT_SHA=${EXPECTED_SERVER_SOURCE_RECEIPT_SHA},TABLE2_SERVER_SOURCE_ROOT=${TABLE2_SERVER_SOURCE_ROOT},TABLE2_SERVER_SOURCE_RECEIPT=${TABLE2_SERVER_SOURCE_RECEIPT},EXPECTED_TABLE2_SERVER_SOURCE_RECEIPT_SHA=${EXPECTED_TABLE2_SERVER_SOURCE_RECEIPT_SHA},BASE_SOURCE_ROOT=${BASE_SOURCE_ROOT},BASE_RECEIPT=${BASE_RECEIPT},EXPECTED_BASE_RECEIPT_SHA=${EXPECTED_BASE_RECEIPT_SHA},RUN_ROOT=${RUN_ROOT},PARENT_ROOT=${PARENT_ROOT},PROTOCOL=${PROTOCOL},EXPECTED_PROTOCOL_SHA=${EXPECTED_PROTOCOL_SHA},SHARD_MANIFEST=${shard_manifest},EXPECTED_SHARD_MANIFEST_SHA=${shard_sha}"
source "${safe}"
safe_sbatch --lint-fatal --test-only --qos=gpu48 --time=01:00:00 \
  --array=0 --export="${common},RUNTIME_ATTEMPT_PREFIX=expand" "${collect}" >/dev/null
safe_sbatch --lint-fatal --test-only --partition=cpu_short \
  --export="${common},FACTUAL_B_ARRAY_JOB=1,PREFIX_CONCURRENCY=4" "${finish}" >/dev/null
shard_last=$((values[1] - 1))
b_raw=$(safe_sbatch --lint-fatal --parsable --qos=gpu48 --time=01:00:00 \
  --array="0-${shard_last}%4" \
  --export="${common},RUNTIME_ATTEMPT_PREFIX=expand" "${collect}")
b_job=$(printf '%s\n' "${b_raw}" | tr -d '\r' | awk -F';' '/^[0-9]+(;|$)/ {print $1; exit}')
[[ "${b_job}" =~ ^[0-9]+$ ]]
f_raw=$(safe_sbatch --lint-fatal --parsable --partition=cpu_short \
  --dependency="afterany:${b_job}" --export="${common},FACTUAL_B_ARRAY_JOB=${b_job},PREFIX_CONCURRENCY=4" "${finish}")
f_job=$(printf '%s\n' "${f_raw}" | tr -d '\r' | awk -F';' '/^[0-9]+(;|$)/ {print $1; exit}')
[[ "${f_job}" =~ ^[0-9]+$ ]]
mkdir -p "${RUN_ROOT}/deferred_submission"
receipt=${RUN_ROOT}/deferred_submission/expansion_factual_b.json
"${py}" - "${receipt}" "${b_job}" "${f_job}" "${shard_sha}" \
  "${values[1]}" <<'PY'
import json,sys
path,b,finish,shard,shard_count=sys.argv[1:]
p={'schema_version':'hm3d_natural_b_expansion_factual_b_submission_v1_20260830',
   'candidate_histories':84,'maximum_histories_per_shard':2,
   'shard_count':int(shard_count),
   'factual_B_array_job':int(b),'post_B_launcher_job':int(finish),
   'shard_manifest_sha256':shard,'leg3_outcomes_read':False}
open(path,'x').write(json.dumps(p,indent=2,sort_keys=True)+'\n')
PY
sha256sum "${receipt}" >"${receipt}.sha256"
chmod a-w "${receipt}" "${receipt}.sha256"
