#!/usr/bin/env bash
#SBATCH --job-name=m3T1monoNext
#SBATCH --output=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/%x_%j.out
#SBATCH --error=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/%x_%j.err
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --partition=cpu_short
#SBATCH --account=torch_pr_769_tandon_advanced

set -euo pipefail
umask 0022
: "${TASK_ROOT:?}" "${TASK_RECEIPT:?}" "${EXPECTED_TASK_RECEIPT_SHA:?}"
: "${RUN_ROOT:?}" "${BASE_SOURCE_LEDGER:?}" "${PROTOCOL:?}"
: "${EXPANSION_MANIFEST:?}" "${CONSTRUCT_CONCURRENCY:?}"

PY=/scratch/lg154/conda-envs/memnav/bin/python
fail() { echo "ABORT: $*" >&2; exit 2; }
[[ "${CONSTRUCT_CONCURRENCY}" =~ ^[1-9][0-9]*$ ]] || \
  fail "invalid construction concurrency"
[[ "$(sha256sum "${TASK_RECEIPT}" | awk '{print $1}')" == \
   "${EXPECTED_TASK_RECEIPT_SHA}" ]] || fail "task receipt changed"
(cd "${TASK_ROOT}" && sha256sum -c --quiet "${TASK_RECEIPT}") || \
  fail "task bundle changed"

sealed=${RUN_ROOT}/sealed_inputs
ledger=${sealed}/expanded_source_ledger.json
[[ ! -e "${ledger}" && ! -e "${ledger}.sha256" ]] || \
  fail "expanded source ledger already exists"
mkdir -p "${sealed}" "${RUN_ROOT}/construction/scenes" "${RUN_ROOT}/logs"
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="${TASK_ROOT}:${TASK_ROOT}/MemNavData" \
  "${PY}" -u \
  "${TASK_ROOT}/MemNavData/freeze_mp3d_table1_fullmono_expanded_source_ledger.py" \
    --base-source-ledger "${BASE_SOURCE_LEDGER}" \
    --protocol "${PROTOCOL}" --expansion-manifest "${EXPANSION_MANIFEST}" \
    --collection-root "${RUN_ROOT}" --out "${ledger}"
ledger_sha=$(sha256sum "${ledger}" | awk '{print $1}')
printf '%s  expanded_source_ledger.json\n' "${ledger_sha}" \
  >"${ledger}.sha256"
readarray -t counts < <("${PY}" - "${ledger}" <<'PY'
import json,sys
d=json.load(open(sys.argv[1]))
assert d['previous_goal_b_policy_outcomes_read'] is False
assert d['old_query_policy_outcomes_read'] is False
assert d['query_policy_outcomes_read'] is False
print(d['scene_count']); print(d['source_history_count'])
PY
)
[[ "${counts[0]}" == 36 && "${counts[1]}" == 104 ]] || \
  fail "expanded source population changed"
chmod -R a-w "${sealed}"

common="ALL,TASK_ROOT=${TASK_ROOT},TASK_RECEIPT=${TASK_RECEIPT},EXPECTED_TASK_RECEIPT_SHA=${EXPECTED_TASK_RECEIPT_SHA},RUN_ROOT=${RUN_ROOT},SOURCE_LEDGER=${ledger},EXPECTED_SOURCE_LEDGER_SHA=${ledger_sha}"
construct=${TASK_ROOT}/MemNavData/slurm_mp3d_table1_new_query_construct.sbatch
analysis=${TASK_ROOT}/MemNavData/slurm_mp3d_table1_new_query_analysis.sbatch
source "${TASK_ROOT}/MemNavData/slurm_safe_submit.sh"
safe_sbatch --lint-fatal --test-only --qos=gpu48 --array=0-35%"${CONSTRUCT_CONCURRENCY}" \
  --export="${common}" "${construct}" >/dev/null
safe_sbatch --lint-fatal --test-only --partition=cpu_short \
  --export="${common},MODE=finalize" "${analysis}" >/dev/null
safe_sbatch --lint-fatal --test-only --partition=cpu_short \
  --export="${common},MODE=verify" "${analysis}" >/dev/null

construction_raw=$(safe_sbatch --lint-fatal --parsable --qos=gpu48 \
  --array=0-35%"${CONSTRUCT_CONCURRENCY}" --export="${common}" "${construct}")
construction=${construction_raw%%;*}
finalize_raw=$(safe_sbatch --lint-fatal --parsable --partition=cpu_short \
  --dependency=afterok:"${construction}" --kill-on-invalid-dep=yes \
  --export="${common},MODE=finalize" "${analysis}")
finalize=${finalize_raw%%;*}
verify_raw=$(safe_sbatch --lint-fatal --parsable --partition=cpu_short \
  --dependency=afterok:"${finalize}" --kill-on-invalid-dep=yes \
  --export="${common},MODE=verify" "${analysis}")
verify=${verify_raw%%;*}
for id in "${construction}" "${finalize}" "${verify}"; do
  [[ "${id}" =~ ^[0-9]+$ ]] || fail "invalid downstream job id ${id}"
done

receipt=${RUN_ROOT}/deferred_submission.json
[[ ! -e "${receipt}" ]] || fail "deferred submission receipt exists"
"${PY}" - "${receipt}" "${ledger}" "${ledger_sha}" \
  "${construction}" "${finalize}" "${verify}" <<'PY'
import json,sys
path,ledger,sha,construction,finalize,verify=sys.argv[1:]
payload={
 'schema_version':'mp3d_table1_fullmono_source_expansion_deferred_v1_20260829',
 'scope':'outcome_blind_new_query_construction_only',
 'source_ledger':ledger,'source_ledger_sha256':sha,
 'source_histories':104,'scenes':36,
 'query_policy_outcomes_read_at_submission':False,
 'formal_controller_rollout_submitted':False,
 'jobs':{'construction_array':int(construction),
         'population_finalize':int(finalize),
         'independent_verification':int(verify)},
}
open(path,'x').write(json.dumps(payload,indent=2,sort_keys=True)+'\n')
PY
sha256sum "${receipt}" >"${receipt}.sha256"
chmod a-w "${receipt}" "${receipt}.sha256"
printf 'LEDGER=%s\nLEDGER_SHA=%s\nCONSTRUCTION=%s\nFINALIZE=%s\nVERIFY=%s\n' \
  "${ledger}" "${ledger_sha}" "${construction}" "${finalize}" "${verify}"
