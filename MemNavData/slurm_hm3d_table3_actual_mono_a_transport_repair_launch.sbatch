#!/usr/bin/env bash
#SBATCH --job-name=h3T3ARepairLaunch
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
: "${TASK_ROOT:?}" "${RUN_ROOT:?}" "${TABLE3_PLAN:?}"
: "${EXPECTED_TABLE3_PLAN_SHA:?}" "${TABLE3_EXECUTION_PROTOCOL:?}"
: "${REPAIR_CONTRACT:?}" "${EXPECTED_REPAIR_CONTRACT_SHA:?}"
REPAIR_NAMESPACE=${REPAIR_NAMESPACE:-table3_a_transport_repair_20260830}
[[ "${REPAIR_NAMESPACE}" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]]
[[ "$(sha256sum "${WRAPPER_RECEIPT}" | awk '{print $1}')" == \
   "${EXPECTED_WRAPPER_RECEIPT_SHA}" ]]
(cd "${WRAPPER_ROOT}" && sha256sum -c --quiet "${WRAPPER_RECEIPT}")
[[ "$(sha256sum "${REPAIR_CONTRACT}" | awk '{print $1}')" == \
   "${EXPECTED_REPAIR_CONTRACT_SHA}" ]]
/scratch/lg154/conda-envs/memnav/bin/python - "${REPAIR_CONTRACT}" <<'PY'
import json,sys
p=json.load(open(sys.argv[1]))
assert p['schema_version']=='hm3d_table3_directed_geodesic_repair_contract_v1_20260830'
assert p['diagnosis']['navigation_outcomes_read'] is False
assert p['diagnosis']['query_policy_outcomes_read'] is False
assert p['repair']['candidate_identities_changed'] is False
assert p['repair']['scientific_thresholds_changed'] is False
assert p['repair']['fallback_completion_allowed'] is False
PY
repair_root=${RUN_ROOT}/${REPAIR_NAMESPACE}
archive=${RUN_ROOT}/failed_attempts/${REPAIR_NAMESPACE}
plan=${repair_root}/repair_plan.json
verify=${repair_root}/independent_verification.json
mkdir -p "${repair_root}"
/scratch/lg154/conda-envs/memnav/bin/python -u \
  "${WRAPPER_ROOT}/MemNavData/freeze_hm3d_table3_actual_mono_a_transport_repair.py" \
  --run-root "${RUN_ROOT}" --candidate-plan "${TABLE3_PLAN}" \
  --expected-candidate-plan-sha256 "${EXPECTED_TABLE3_PLAN_SHA}" \
  --archive-root "${archive}" --out "${plan}"
/scratch/lg154/conda-envs/memnav/bin/python -u \
  "${WRAPPER_ROOT}/MemNavData/independent_verify_hm3d_table3_actual_mono_a_transport_repair.py" \
  --run-root "${RUN_ROOT}" --candidate-plan "${TABLE3_PLAN}" \
  --expected-candidate-plan-sha256 "${EXPECTED_TABLE3_PLAN_SHA}" \
  --repair-plan "${plan}" --archive-root "${archive}" --out "${verify}"
readarray -t repair_values < <(/scratch/lg154/conda-envs/memnav/bin/python - \
  "${plan}" "${verify}" <<'PY'
import json,sys
p=json.load(open(sys.argv[1])); v=json.load(open(sys.argv[2]))
assert v['verified'] is True and v['navigation_outcomes_read'] is False
assert p['fallback_completion_allowed'] is False
missing=[int(i) for i in p['missing_history_indices']]
assert missing and missing==sorted(set(missing))
print(','.join(map(str,missing))); print(len(missing))
PY
)
[[ "${#repair_values[@]}" -eq 2 && -n "${repair_values[0]}" ]]
common="ALL,WRAPPER_ROOT=${WRAPPER_ROOT},WRAPPER_RECEIPT=${WRAPPER_RECEIPT},EXPECTED_WRAPPER_RECEIPT_SHA=${EXPECTED_WRAPPER_RECEIPT_SHA},REPAIR_PLAN=${plan},REPAIR_CONTRACT=${REPAIR_CONTRACT},EXPECTED_REPAIR_CONTRACT_SHA=${EXPECTED_REPAIR_CONTRACT_SHA},TASK_ROOT=${TASK_ROOT},TASK_RECEIPT=${TASK_RECEIPT},EXPECTED_TASK_RECEIPT_SHA=${EXPECTED_TASK_RECEIPT_SHA},SERVER_SOURCE_ROOT=${SERVER_SOURCE_ROOT},SERVER_SOURCE_RECEIPT=${SERVER_SOURCE_RECEIPT},EXPECTED_SERVER_SOURCE_RECEIPT_SHA=${EXPECTED_SERVER_SOURCE_RECEIPT_SHA},BASE_SOURCE_ROOT=${BASE_SOURCE_ROOT},BASE_RECEIPT=${BASE_RECEIPT},EXPECTED_BASE_RECEIPT_SHA=${EXPECTED_BASE_RECEIPT_SHA},RUN_ROOT=${RUN_ROOT},TABLE3_PLAN=${TABLE3_PLAN},EXPECTED_TABLE3_PLAN_SHA=${EXPECTED_TABLE3_PLAN_SHA},TABLE3_EXECUTION_PROTOCOL=${TABLE3_EXECUTION_PROTOCOL}"
safe=${TASK_ROOT}/MemNavData/slurm_safe_submit.sh
repair=${WRAPPER_ROOT}/MemNavData/slurm_hm3d_table3_actual_mono_a_transport_repair.sbatch
finish=${WRAPPER_ROOT}/MemNavData/slurm_hm3d_table3_actual_mono_a_transport_repair_finish.sbatch
source "${safe}"
array_spec=${repair_values[0]}
if (( repair_values[1] > 1 )); then array_spec=${array_spec}%4; fi
safe_sbatch --lint-fatal --test-only --partition=a100_tandon --qos=gpu48 \
  --time=02:00:00 --array="${array_spec}" --export="${common}" \
  "${repair}" >/dev/null
r_raw=$(safe_sbatch --lint-fatal --parsable --partition=a100_tandon --qos=gpu48 \
  --time=02:00:00 --array="${array_spec}" --export="${common}" "${repair}")
r_job=$(printf '%s\n' "${r_raw}" | tr -d '\r' | \
  awk -F';' '/^[0-9]+(;|$)/ {print $1; exit}')
[[ "${r_job}" =~ ^[0-9]+$ ]]
f_raw=$(safe_sbatch --lint-fatal --parsable --partition=cpu_short \
  --dependency="afterok:${r_job}" --kill-on-invalid-dep=yes \
  --export="${common},REPAIR_ARRAY_JOB=${r_job}" "${finish}")
f_job=$(printf '%s\n' "${f_raw}" | tr -d '\r' | \
  awk -F';' '/^[0-9]+(;|$)/ {print $1; exit}')
[[ "${f_job}" =~ ^[0-9]+$ ]]
receipt=${repair_root}/submission.json
/scratch/lg154/conda-envs/memnav/bin/python - "${receipt}" "${r_job}" \
  "${f_job}" "${repair_values[1]}" <<'PY'
import json,sys
path,repair,finish,missing=sys.argv[1:]
p={'schema_version':'hm3d_table3_actual_mono_a_transport_repair_submission_v1_20260830',
   'repair_array_job':int(repair),'repair_finish_job':int(finish),
   'missing_history_count':int(missing),'navigation_outcomes_read':False,
   'query_policy_outcomes_read':False,'scientific_thresholds_changed':False,
   'fallback_completion_allowed':False}
open(path,'x').write(json.dumps(p,indent=2,sort_keys=True)+'\n')
PY
sha256sum "${receipt}" >"${receipt}.sha256"
