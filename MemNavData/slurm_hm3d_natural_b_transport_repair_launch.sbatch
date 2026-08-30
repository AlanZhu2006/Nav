#!/usr/bin/env bash
#SBATCH --job-name=h3NatBRepairLaunch
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
: "${WRAPPER_ROOT:?}" "${WRAPPER_RECEIPT:?}" "${EXPECTED_WRAPPER_RECEIPT_SHA:?}"
: "${TASK_ROOT:?}" "${RUN_ROOT:?}" "${PARENT_ROOT:?}" "${PROTOCOL:?}"
: "${SHARD_MANIFEST:?}" "${EXPECTED_SHARD_MANIFEST_SHA:?}"
: "${GPU_PARTITION:?}" "${REPAIR_TAG:?}" "${REPAIR_RUNTIME_PREFIX:?}"
[[ "${GPU_PARTITION}" =~ ^(h100_tandon|a100_tandon|h100_tandon,a100_tandon)$ ]]
[[ "${REPAIR_TAG}" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]]
[[ "${REPAIR_RUNTIME_PREFIX}" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]]
[[ "$(sha256sum "${WRAPPER_RECEIPT}" | awk '{print $1}')" == \
   "${EXPECTED_WRAPPER_RECEIPT_SHA}" ]]
(cd "${WRAPPER_ROOT}" && sha256sum -c --quiet "${WRAPPER_RECEIPT}")
repair_root=${RUN_ROOT}/${REPAIR_TAG}
archive=${RUN_ROOT}/failed_attempts/${REPAIR_TAG}
plan=${repair_root}/repair_plan.json
verify=${repair_root}/independent_verification.json
benchmark=${RUN_ROOT}/ab_population/role_pairs/manifest.json
benchmark_sha=$(sha256sum "${benchmark}" | awk '{print $1}')
mkdir -p "${repair_root}"
/scratch/lg154/conda-envs/memnav/bin/python -u \
  "${WRAPPER_ROOT}/MemNavData/freeze_hm3d_natural_b_transport_repair.py" \
  --run-root "${RUN_ROOT}" --benchmark-manifest "${benchmark}" \
  --expected-benchmark-sha256 "${benchmark_sha}" \
  --shard-manifest "${SHARD_MANIFEST}" \
  --expected-shard-sha256 "${EXPECTED_SHARD_MANIFEST_SHA}" \
  --archive-root "${archive}" --out "${plan}"
/scratch/lg154/conda-envs/memnav/bin/python -u \
  "${WRAPPER_ROOT}/MemNavData/independent_verify_hm3d_natural_b_transport_repair.py" \
  --run-root "${RUN_ROOT}" --benchmark-manifest "${benchmark}" \
  --repair-plan "${plan}" --archive-root "${archive}" --out "${verify}"
readarray -t counts < <(/scratch/lg154/conda-envs/memnav/bin/python - \
  "${plan}" "${verify}" <<'PY'
import json,sys
p=json.load(open(sys.argv[1])); v=json.load(open(sys.argv[2]))
assert v['verified'] is True and v['navigation_outcomes_read'] is False
assert p['fallback_completion_allowed'] is False
print(p['repair_group_count']); print(len(p['missing_history_indices']))
PY
)
(( counts[0] > 0 && counts[1] > 0 ))
common="ALL,WRAPPER_ROOT=${WRAPPER_ROOT},WRAPPER_RECEIPT=${WRAPPER_RECEIPT},EXPECTED_WRAPPER_RECEIPT_SHA=${EXPECTED_WRAPPER_RECEIPT_SHA},REPAIR_PLAN=${plan},TASK_ROOT=${TASK_ROOT},TASK_RECEIPT=${TASK_RECEIPT},EXPECTED_TASK_RECEIPT_SHA=${EXPECTED_TASK_RECEIPT_SHA},SERVER_SOURCE_ROOT=${SERVER_SOURCE_ROOT},SERVER_SOURCE_RECEIPT=${SERVER_SOURCE_RECEIPT},EXPECTED_SERVER_SOURCE_RECEIPT_SHA=${EXPECTED_SERVER_SOURCE_RECEIPT_SHA},TABLE2_SERVER_SOURCE_ROOT=${TABLE2_SERVER_SOURCE_ROOT},TABLE2_SERVER_SOURCE_RECEIPT=${TABLE2_SERVER_SOURCE_RECEIPT},EXPECTED_TABLE2_SERVER_SOURCE_RECEIPT_SHA=${EXPECTED_TABLE2_SERVER_SOURCE_RECEIPT_SHA},BASE_SOURCE_ROOT=${BASE_SOURCE_ROOT},BASE_RECEIPT=${BASE_RECEIPT},EXPECTED_BASE_RECEIPT_SHA=${EXPECTED_BASE_RECEIPT_SHA},RUN_ROOT=${RUN_ROOT},PARENT_ROOT=${PARENT_ROOT},PROTOCOL=${PROTOCOL},EXPECTED_PROTOCOL_SHA=${EXPECTED_PROTOCOL_SHA},REPAIR_RUNTIME_PREFIX=${REPAIR_RUNTIME_PREFIX}"
safe=${TASK_ROOT}/MemNavData/slurm_safe_submit.sh
repair=${WRAPPER_ROOT}/MemNavData/slurm_hm3d_natural_b_transport_repair_shard.sbatch
finish=${TASK_ROOT}/MemNavData/slurm_hm3d_fullmono_lifelong_natural_b_expansion_finish.sbatch
source "${safe}"
last=$((counts[0]-1))
safe_sbatch --lint-fatal --test-only --partition="${GPU_PARTITION}" --qos=gpu48 --time=01:00:00 \
  --array="0-${last}%4" --export="${common}" "${repair}" >/dev/null
r_raw=$(safe_sbatch --lint-fatal --parsable --partition="${GPU_PARTITION}" --qos=gpu48 --time=01:00:00 \
  --array="0-${last}%4" --export="${common}" "${repair}")
r_job=$(printf '%s\n' "${r_raw}" | tr -d '\r' | awk -F';' '/^[0-9]+(;|$)/ {print $1; exit}')
[[ "${r_job}" =~ ^[0-9]+$ ]]
f_raw=$(safe_sbatch --lint-fatal --parsable --partition=cpu_short \
  --dependency="afterok:${r_job}" \
  --export="${common},FACTUAL_B_ARRAY_JOB=${r_job},PREFIX_CONCURRENCY=4" "${finish}")
f_job=$(printf '%s\n' "${f_raw}" | tr -d '\r' | awk -F';' '/^[0-9]+(;|$)/ {print $1; exit}')
[[ "${f_job}" =~ ^[0-9]+$ ]]
receipt=${repair_root}/submission.json
/scratch/lg154/conda-envs/memnav/bin/python - "${receipt}" "${r_job}" "${f_job}" \
  "${counts[1]}" <<'PY'
import json,sys
path,repair,finish,missing=sys.argv[1:]
payload={'schema_version':'hm3d_natural_b_transport_repair_submission_v1_20260830',
 'repair_array_job':int(repair),'replacement_finish_job':int(finish),
 'missing_history_count':int(missing),'navigation_outcomes_read':False,
 'scientific_thresholds_changed':False,'fallback_completion_allowed':False}
open(path,'x').write(json.dumps(payload,indent=2,sort_keys=True)+'\n')
PY
sha256sum "${receipt}" >"${receipt}.sha256"
chmod -R a-w "${repair_root}" "${archive}"
