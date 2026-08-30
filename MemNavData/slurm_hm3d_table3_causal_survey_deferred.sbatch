#!/usr/bin/env bash
# Dependency-held handoff for the powered Table-III causal-history-length study.
#
# MODE=expand is released only after the result-blind expansion-plan verifier.
# It submits every append-only expansion construction, seals the exact 16/16/16
# merged population, and then schedules MODE=query behind its independent
# verifier.  MODE=query submits one retained maximum-budget formal element as
# the full-stack gate, followed by the other 47 elements and the independent
# result recount.  Neither mode reads navigation outcomes before submission.
#SBATCH --job-name=h3T3Defer
#SBATCH --output=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/%x_%j.out
#SBATCH --error=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/%x_%j.err
#SBATCH --time=00:20:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --partition=cpu_short
#SBATCH --account=torch_pr_769_tandon_advanced
#SBATCH --export=ALL

set -euo pipefail
umask 0022
export PYTHONDONTWRITEBYTECODE=1

: "${DEFERRED_MODE:?expand, query_base, or query_merged}"
: "${CHAIN_SOURCE_ROOT:?}" "${CHAIN_SOURCE_RECEIPT:?}"
: "${EXPECTED_CHAIN_SOURCE_RECEIPT_SHA:?}"
: "${EXPECTED_REMOTE_USER:?}" "${EXPANSION_RUN_ROOT:?}"
: "${BASE_RUN_ROOT:?}" "${BASE_CANDIDATE_PLAN:?}" "${BASE_SURVEY_PROTOCOL:?}"
: "${BASE_POPULATION_VERIFY_JOB:?}"
: "${BASE_WRAPPER_ROOT:?}" "${BASE_WRAPPER_RECEIPT:?}"
: "${EXPECTED_BASE_WRAPPER_RECEIPT_SHA:?}"
: "${TASK_ROOT:?}" "${TASK_RECEIPT:?}" "${EXPECTED_TASK_RECEIPT_SHA:?}"
: "${SERVER_SOURCE_ROOT:?}" "${SERVER_SOURCE_RECEIPT:?}"
: "${EXPECTED_SERVER_SOURCE_RECEIPT_SHA:?}"
: "${BASE_SOURCE_ROOT:?}" "${BASE_RECEIPT:?}" "${EXPECTED_BASE_RECEIPT_SHA:?}"
: "${RUNTIME_CLOSURE_ROOT:?}" "${RUNTIME_CLOSURE_RECEIPT:?}"
: "${EXPECTED_RUNTIME_CLOSURE_RECEIPT_SHA:?}"

[[ "${DEFERRED_MODE}" == expand || "${DEFERRED_MODE}" == query_base \
   || "${DEFERRED_MODE}" == query_merged ]]
[[ "${BASE_POPULATION_VERIFY_JOB}" =~ ^[0-9]+$ ]]
[[ "$(id -un)" == "${EXPECTED_REMOTE_USER}" ]]
[[ "$(sha256sum "${CHAIN_SOURCE_RECEIPT}" | awk '{print $1}')" == \
   "${EXPECTED_CHAIN_SOURCE_RECEIPT_SHA}" ]]
(cd "${CHAIN_SOURCE_ROOT}" && sha256sum -c --quiet "${CHAIN_SOURCE_RECEIPT}")
[[ "$(sha256sum "${TASK_RECEIPT}" | awk '{print $1}')" == \
   "${EXPECTED_TASK_RECEIPT_SHA}" ]]
(cd "${TASK_ROOT}" && sha256sum -c --quiet "${TASK_RECEIPT}")
[[ "$(sha256sum "${RUNTIME_CLOSURE_RECEIPT}" | awk '{print $1}')" == \
   "${EXPECTED_RUNTIME_CLOSURE_RECEIPT_SHA}" ]]
(cd "${RUNTIME_CLOSURE_ROOT}" && \
  sha256sum -c --quiet "${RUNTIME_CLOSURE_RECEIPT}")

py=/scratch/lg154/conda-envs/memnav/bin/python
safe=${TASK_ROOT}/MemNavData/slurm_safe_submit.sh
deferred=${CHAIN_SOURCE_ROOT}/MemNavData/slurm_hm3d_table3_causal_survey_deferred.sbatch
receipt_root=${EXPANSION_RUN_ROOT}/deferred_submission
mkdir -p "${receipt_root}"

fail() { echo "ABORT: $*" >&2; exit 2; }
job_id() {
  local raw=$1 value=${1%%;*}
  [[ "${value}" =~ ^[0-9]+$ ]] || fail "invalid sbatch response: ${raw}"
  printf '%s' "${value}"
}
sha() { sha256sum "$1" | awk '{print $1}'; }

source "${safe}"

if [[ "${DEFERRED_MODE}" == expand ]]; then
  lock=${receipt_root}/expand.lock
  receipt=${receipt_root}/expand.json
  [[ ! -e "${receipt}" && ! -e "${receipt}.sha256" ]] || \
    fail "expansion handoff receipt already exists"
  mkdir "${lock}" || fail "expansion handoff already attempted"

  plan=${EXPANSION_RUN_ROOT}/expansion_plan/candidate_plan.json
  protocol=${EXPANSION_RUN_ROOT}/expansion_plan/construction_protocol.json
  verification=${EXPANSION_RUN_ROOT}/expansion_plan/independent_verification.json
  readarray -t state < <("${py}" - "${plan}" "${protocol}" "${verification}" <<'PY'
import hashlib,json,sys
sha=lambda p: hashlib.sha256(open(p,'rb').read()).hexdigest()
p=json.load(open(sys.argv[1])); c=json.load(open(sys.argv[2])); v=json.load(open(sys.argv[3]))
assert v['schema_version']=='hm3d_table3_causal_survey_expansion_plan_verification_v1_20260831'
assert v['verified'] is True
assert v['plan_sha256']==sha(sys.argv[1])
assert v['construction_protocol_sha256']==sha(sys.argv[2])
assert v['candidate_count']==p['candidate_count']==c['source_candidate_plan']['candidate_count']
assert v['query_policy_outcomes_read'] is False
assert v['navigation_policy_outcomes_read'] is False
assert v['query_policy_evaluation_authorized'] is False
assert v['base_candidates_deleted_or_replaced'] is False
print(p['candidate_count']); print(sha(sys.argv[1])); print(sha(sys.argv[2])); print(sha(sys.argv[3]))
print(','.join(p['deficient_bins']))
PY
  )
  [[ "${#state[@]}" -eq 5 && "${state[0]}" =~ ^[0-9]+$ ]] || \
    fail "verified expansion plan is unavailable"
  candidate_count=${state[0]}
  plan_sha=${state[1]}
  protocol_sha=${state[2]}
  verification_sha=${state[3]}
  deficient_bins=${state[4]}

  [[ "$(find "${BASE_RUN_ROOT}/construction_fragments" -mindepth 2 -maxdepth 2 \
      -name completion.json -type f | wc -l)" -eq 125 ]] || \
    fail "base construction is not 125/125"
  [[ ! -e "${EXPANSION_RUN_ROOT}/merged_query_population" ]] || \
    fail "merged population already exists"
  [[ ! -e "${EXPANSION_RUN_ROOT}/evaluation" ]] || \
    fail "query evaluation already exists"

  if [[ "${candidate_count}" -eq 0 ]]; then
    [[ -z "${deficient_bins}" ]] || \
      fail "empty expansion still reports deficient bins"
    continuation="ALL,DEFERRED_MODE=query_base,CHAIN_SOURCE_ROOT=${CHAIN_SOURCE_ROOT},CHAIN_SOURCE_RECEIPT=${CHAIN_SOURCE_RECEIPT},EXPECTED_CHAIN_SOURCE_RECEIPT_SHA=${EXPECTED_CHAIN_SOURCE_RECEIPT_SHA},EXPECTED_REMOTE_USER=${EXPECTED_REMOTE_USER},EXPANSION_RUN_ROOT=${EXPANSION_RUN_ROOT},BASE_RUN_ROOT=${BASE_RUN_ROOT},BASE_CANDIDATE_PLAN=${BASE_CANDIDATE_PLAN},BASE_SURVEY_PROTOCOL=${BASE_SURVEY_PROTOCOL},BASE_POPULATION_VERIFY_JOB=${BASE_POPULATION_VERIFY_JOB},BASE_WRAPPER_ROOT=${BASE_WRAPPER_ROOT},BASE_WRAPPER_RECEIPT=${BASE_WRAPPER_RECEIPT},EXPECTED_BASE_WRAPPER_RECEIPT_SHA=${EXPECTED_BASE_WRAPPER_RECEIPT_SHA},TASK_ROOT=${TASK_ROOT},TASK_RECEIPT=${TASK_RECEIPT},EXPECTED_TASK_RECEIPT_SHA=${EXPECTED_TASK_RECEIPT_SHA},SERVER_SOURCE_ROOT=${SERVER_SOURCE_ROOT},SERVER_SOURCE_RECEIPT=${SERVER_SOURCE_RECEIPT},EXPECTED_SERVER_SOURCE_RECEIPT_SHA=${EXPECTED_SERVER_SOURCE_RECEIPT_SHA},BASE_SOURCE_ROOT=${BASE_SOURCE_ROOT},BASE_RECEIPT=${BASE_RECEIPT},EXPECTED_BASE_RECEIPT_SHA=${EXPECTED_BASE_RECEIPT_SHA},RUNTIME_CLOSURE_ROOT=${RUNTIME_CLOSURE_ROOT},RUNTIME_CLOSURE_RECEIPT=${RUNTIME_CLOSURE_RECEIPT},EXPECTED_RUNTIME_CLOSURE_RECEIPT_SHA=${EXPECTED_RUNTIME_CLOSURE_RECEIPT_SHA},POPULATION_VERIFY_JOB=${BASE_POPULATION_VERIFY_JOB}"
    raw=$(safe_sbatch --lint-fatal --parsable --partition=cpu_short --time=00:20:00 \
      --dependency="afterok:${BASE_POPULATION_VERIFY_JOB}" \
      --kill-on-invalid-dep=yes --export="${continuation}" "${deferred}")
    query_launcher=$(job_id "${raw}")
    "${py}" - "${receipt}" "${query_launcher}" "${plan_sha}" \
      "${protocol_sha}" "${verification_sha}" <<'PY'
import json,sys
path,query,plan_sha,protocol_sha,verification_sha=sys.argv[1:]
p={'schema_version':'hm3d_table3_causal_survey_deferred_expansion_v1_20260831',
   'expansion_candidates':0,'deficient_bins':[],
   'base_population_used_directly':True,
   'jobs':{'formal_query_launcher':int(query)},
   'expansion_plan_sha256':plan_sha,'construction_protocol_sha256':protocol_sha,
   'expansion_plan_verification_sha256':verification_sha,
   'base_candidates_deleted_or_replaced':False,'query_policy_outcomes_read':False,
   'threshold_relaxation':False,'fallback_completion_allowed':False,
   'history_source':'controlled_causal_rgb_geodesic_survey'}
open(path,'x').write(json.dumps(p,indent=2,sort_keys=True)+'\n')
PY
    sha256sum "${receipt}" >"${receipt}.sha256"
    chmod a-w "${receipt}" "${receipt}.sha256"
    printf 'BASE_POPULATION_QUERY_LAUNCHER=%s\n' "${query_launcher}"
    exit 0
  fi

  construct=${CHAIN_SOURCE_ROOT}/MemNavData/slurm_hm3d_table3_causal_survey_expansion_construct.sbatch
  population=${CHAIN_SOURCE_ROOT}/MemNavData/slurm_hm3d_table3_causal_survey_merged_population.sbatch
  common="ALL,EXPANSION_EXEC_SOURCE_ROOT=${CHAIN_SOURCE_ROOT},EXPANSION_EXEC_SOURCE_RECEIPT=${CHAIN_SOURCE_RECEIPT},EXPECTED_EXPANSION_EXEC_SOURCE_RECEIPT_SHA=${EXPECTED_CHAIN_SOURCE_RECEIPT_SHA},BASE_WRAPPER_ROOT=${BASE_WRAPPER_ROOT},BASE_WRAPPER_RECEIPT=${BASE_WRAPPER_RECEIPT},EXPECTED_BASE_WRAPPER_RECEIPT_SHA=${EXPECTED_BASE_WRAPPER_RECEIPT_SHA},BASE_RUN_ROOT=${BASE_RUN_ROOT},BASE_CANDIDATE_PLAN=${BASE_CANDIDATE_PLAN},BASE_SURVEY_PROTOCOL=${BASE_SURVEY_PROTOCOL},EXPANSION_RUN_ROOT=${EXPANSION_RUN_ROOT},EXPANSION_PLAN=${plan},EXPECTED_EXPANSION_PLAN_SHA=${plan_sha},EXPANSION_CONSTRUCTION_PROTOCOL=${protocol},EXPECTED_EXPANSION_PROTOCOL_SHA=${protocol_sha},EXPANSION_PLAN_VERIFICATION=${verification},EXPECTED_EXPANSION_PLAN_VERIFICATION_SHA=${verification_sha},EXPECTED_EXPANSION_CANDIDATE_COUNT=${candidate_count}"
  array="0-$((candidate_count - 1))%4"

  safe_sbatch --lint-fatal --test-only --qos=gpu48 --time=01:00:00 \
    --array="${array}" --export="${common}" "${construct}" >/dev/null
  safe_sbatch --lint-fatal --test-only --partition=cpu_short --time=00:30:00 \
    --export="${common},MODE=finalize" "${population}" >/dev/null
  safe_sbatch --lint-fatal --test-only --partition=cpu_short --time=00:30:00 \
    --export="${common},MODE=verify" "${population}" >/dev/null

  raw=$(safe_sbatch --lint-fatal --parsable --qos=gpu48 --time=01:00:00 \
    --array="${array}" --export="${common}" "${construct}")
  construction=$(job_id "${raw}")
  raw=$(safe_sbatch --lint-fatal --parsable --partition=cpu_short --time=00:30:00 \
    --dependency="afterok:${construction}" --kill-on-invalid-dep=yes \
    --export="${common},MODE=finalize" "${population}")
  finalize=$(job_id "${raw}")
  raw=$(safe_sbatch --lint-fatal --parsable --partition=cpu_short --time=00:30:00 \
    --dependency="afterok:${finalize}" --kill-on-invalid-dep=yes \
    --export="${common},MODE=verify" "${population}")
  verify=$(job_id "${raw}")

  continuation="ALL,DEFERRED_MODE=query_merged,CHAIN_SOURCE_ROOT=${CHAIN_SOURCE_ROOT},CHAIN_SOURCE_RECEIPT=${CHAIN_SOURCE_RECEIPT},EXPECTED_CHAIN_SOURCE_RECEIPT_SHA=${EXPECTED_CHAIN_SOURCE_RECEIPT_SHA},EXPECTED_REMOTE_USER=${EXPECTED_REMOTE_USER},EXPANSION_RUN_ROOT=${EXPANSION_RUN_ROOT},BASE_RUN_ROOT=${BASE_RUN_ROOT},BASE_CANDIDATE_PLAN=${BASE_CANDIDATE_PLAN},BASE_SURVEY_PROTOCOL=${BASE_SURVEY_PROTOCOL},BASE_POPULATION_VERIFY_JOB=${BASE_POPULATION_VERIFY_JOB},BASE_WRAPPER_ROOT=${BASE_WRAPPER_ROOT},BASE_WRAPPER_RECEIPT=${BASE_WRAPPER_RECEIPT},EXPECTED_BASE_WRAPPER_RECEIPT_SHA=${EXPECTED_BASE_WRAPPER_RECEIPT_SHA},TASK_ROOT=${TASK_ROOT},TASK_RECEIPT=${TASK_RECEIPT},EXPECTED_TASK_RECEIPT_SHA=${EXPECTED_TASK_RECEIPT_SHA},SERVER_SOURCE_ROOT=${SERVER_SOURCE_ROOT},SERVER_SOURCE_RECEIPT=${SERVER_SOURCE_RECEIPT},EXPECTED_SERVER_SOURCE_RECEIPT_SHA=${EXPECTED_SERVER_SOURCE_RECEIPT_SHA},BASE_SOURCE_ROOT=${BASE_SOURCE_ROOT},BASE_RECEIPT=${BASE_RECEIPT},EXPECTED_BASE_RECEIPT_SHA=${EXPECTED_BASE_RECEIPT_SHA},RUNTIME_CLOSURE_ROOT=${RUNTIME_CLOSURE_ROOT},RUNTIME_CLOSURE_RECEIPT=${RUNTIME_CLOSURE_RECEIPT},EXPECTED_RUNTIME_CLOSURE_RECEIPT_SHA=${EXPECTED_RUNTIME_CLOSURE_RECEIPT_SHA},POPULATION_VERIFY_JOB=${verify}"
  raw=$(safe_sbatch --lint-fatal --parsable --partition=cpu_short --time=00:20:00 \
    --dependency="afterok:${verify}" --kill-on-invalid-dep=yes \
    --export="${continuation}" "${deferred}")
  query_launcher=$(job_id "${raw}")

  "${py}" - "${receipt}" "${candidate_count}" "${deficient_bins}" \
    "${construction}" "${finalize}" "${verify}" "${query_launcher}" \
    "${plan_sha}" "${protocol_sha}" "${verification_sha}" <<'PY'
import json,sys
(path,count,bins,construction,finalize,verify,query,plan_sha,protocol_sha,verification_sha)=sys.argv[1:]
p={'schema_version':'hm3d_table3_causal_survey_deferred_expansion_v1_20260831',
   'expansion_candidates':int(count),'deficient_bins':bins.split(',') if bins else [],
   'jobs':{'expansion_construction_array':int(construction),
           'merged_population_finalize':int(finalize),
           'merged_population_independent_verification':int(verify),
           'formal_query_launcher':int(query)},
   'expansion_plan_sha256':plan_sha,'construction_protocol_sha256':protocol_sha,
   'expansion_plan_verification_sha256':verification_sha,
   'base_candidates_deleted_or_replaced':False,'query_policy_outcomes_read':False,
   'threshold_relaxation':False,'fallback_completion_allowed':False,
   'history_source':'controlled_causal_rgb_geodesic_survey'}
open(path,'x').write(json.dumps(p,indent=2,sort_keys=True)+'\n')
PY
  sha256sum "${receipt}" >"${receipt}.sha256"
  chmod a-w "${receipt}" "${receipt}.sha256"
  printf 'EXPANSION=%s\nFINALIZE=%s\nVERIFY=%s\nQUERY_LAUNCHER=%s\n' \
    "${construction}" "${finalize}" "${verify}" "${query_launcher}"
  exit 0
fi

# Query modes: an independent population verifier is the only authority for
# policy submission.  Empty expansion means the original base population met
# every frozen gate; it is the primary path, not a fallback population.
: "${POPULATION_VERIFY_JOB:?}"
[[ "${POPULATION_VERIFY_JOB}" =~ ^[0-9]+$ ]]
query_kind=${DEFERRED_MODE#query_}
lock=${receipt_root}/query_${query_kind}.lock
receipt=${receipt_root}/query_${query_kind}.json
[[ ! -e "${receipt}" && ! -e "${receipt}.sha256" ]] || \
  fail "query handoff receipt already exists"
mkdir "${lock}" || fail "query handoff already attempted"

if [[ "${query_kind}" == base ]]; then
  query_run_root=${BASE_RUN_ROOT}
  population_relative_root=query_population
else
  query_run_root=${EXPANSION_RUN_ROOT}
  population_relative_root=merged_query_population
fi
population=${query_run_root}/${population_relative_root}
verification=${population}/independent_verification.json
manifest=${population}/role_pairs/manifest.json
(cd "${population}" && sha256sum -c --quiet \
  independent_verification.json.sha256 population_receipt.json.sha256)
(cd "${population}/role_pairs" && sha256sum -c --quiet manifest.json.sha256)

readarray -t sealed < <("${py}" - "${verification}" "${manifest}" \
  "${query_kind}" <<'PY'
import hashlib,json,math,sys
sha=lambda p: hashlib.sha256(open(p,'rb').read()).hexdigest()
v=json.load(open(sys.argv[1])); m=json.load(open(sys.argv[2]))
kind=sys.argv[3]
assert kind in ('base','merged')
expected=('hm3d_table3_causal_survey_population_verification_v1_20260830'
          if kind=='base' else
          'hm3d_table3_causal_survey_population_verification_v2_20260831')
assert v['schema_version']==expected
assert v['verified'] is True and v['formal_policy_evaluation_authorized'] is True
assert v['history_source']=='controlled_causal_rgb_geodesic_survey'
assert v['query_policy_outcomes_read'] is False and v['fallback_completion_allowed'] is False
assert v['histories_by_bin']=={'0_to_20_m':16,'20_to_30_m':16,'30_to_50_m':16}
assert all(int(n)>=10 for n in v['scene_clusters_by_bin'].values())
assert len(m['episodes'])==48 and sha(sys.argv[2])==v['benchmark_manifest_sha256']
if kind=='merged':
    assert v['base_candidates_preserved']==125 and v['expansion_candidates_appended']>0
    assert any(r['population_source']=='expansion' for r in m['episodes'])
budgets=[]
for index,row in enumerate(m['episodes']):
    ds=[float(q['geodesic_from_a_end_m']) for q in row['pairs'][0]['queries']]
    budget=max(600,math.ceil(2.5*max(ds)/0.0376)); assert budget<=3400
    budgets.append((budget,index))
gate_budget,gate_index=max(budgets)
remaining=[i for i in range(48) if i!=gate_index]
def compact(values):
    runs=[]; start=prev=values[0]
    for value in values[1:]:
        if value==prev+1: prev=value; continue
        runs.append(str(start) if start==prev else f'{start}-{prev}')
        start=prev=value
    runs.append(str(start) if start==prev else f'{start}-{prev}')
    return ','.join(runs)
print(sha(sys.argv[1])); print(sha(sys.argv[2])); print(gate_index); print(gate_budget)
print(compact(remaining)+'%4')
PY
)
[[ "${#sealed[@]}" -eq 5 && "${sealed[2]}" =~ ^([0-9]|[1-3][0-9]|4[0-7])$ \
   && "${sealed[3]}" =~ ^[0-9]+$ && "${sealed[4]}" =~ ^[0-9,-]+%4$ ]] || \
  fail "merged 16/16/16 seal failed"
population_sha=${sealed[0]}
manifest_sha=${sealed[1]}
gate_index=${sealed[2]}
gate_budget=${sealed[3]}
remaining_array=${sealed[4]}
[[ ! -e "${query_run_root}/evaluation" ]] || fail "evaluation already exists"
[[ ! -e "${query_run_root}/table3_result" ]] || fail "result already exists"

[[ "$(sha "${SERVER_SOURCE_RECEIPT}")" == "${EXPECTED_SERVER_SOURCE_RECEIPT_SHA}" ]]
[[ "$(sha "${BASE_RECEIPT}")" == "${EXPECTED_BASE_RECEIPT_SHA}" ]]
pair=${CHAIN_SOURCE_ROOT}/MemNavData/slurm_hm3d_table3_causal_survey_pair.sbatch
analysis=${CHAIN_SOURCE_ROOT}/MemNavData/slurm_hm3d_table3_causal_survey_analysis.sbatch
survey_protocol=${CHAIN_SOURCE_ROOT}/MemNavData/hm3d_table3_causal_survey_protocol_20260830.json
common="ALL,SURVEY_SOURCE_ROOT=${CHAIN_SOURCE_ROOT},SURVEY_SOURCE_RECEIPT=${CHAIN_SOURCE_RECEIPT},EXPECTED_SURVEY_SOURCE_RECEIPT_SHA=${EXPECTED_CHAIN_SOURCE_RECEIPT_SHA},TASK_ROOT=${TASK_ROOT},TASK_RECEIPT=${TASK_RECEIPT},EXPECTED_TASK_RECEIPT_SHA=${EXPECTED_TASK_RECEIPT_SHA},SERVER_SOURCE_ROOT=${SERVER_SOURCE_ROOT},SERVER_SOURCE_RECEIPT=${SERVER_SOURCE_RECEIPT},EXPECTED_SERVER_SOURCE_RECEIPT_SHA=${EXPECTED_SERVER_SOURCE_RECEIPT_SHA},BASE_SOURCE_ROOT=${BASE_SOURCE_ROOT},BASE_RECEIPT=${BASE_RECEIPT},EXPECTED_BASE_RECEIPT_SHA=${EXPECTED_BASE_RECEIPT_SHA},RUNTIME_CLOSURE_ROOT=${RUNTIME_CLOSURE_ROOT},RUNTIME_CLOSURE_RECEIPT=${RUNTIME_CLOSURE_RECEIPT},EXPECTED_RUNTIME_CLOSURE_RECEIPT_SHA=${EXPECTED_RUNTIME_CLOSURE_RECEIPT_SHA},RUN_ROOT=${query_run_root},CANDIDATE_PLAN=${manifest},EXPECTED_CANDIDATE_PLAN_SHA=${manifest_sha},SURVEY_PROTOCOL=${survey_protocol},POPULATION_RELATIVE_ROOT=${population_relative_root},EXPECTED_POPULATION_VERIFICATION_SHA=${population_sha},EXPECTED_BENCHMARK_MANIFEST_SHA=${manifest_sha}"

safe_sbatch --lint-fatal --test-only --qos=gpu48 --time=04:00:00 \
  --array="${gate_index}" --export="${common}" "${pair}" >/dev/null
safe_sbatch --lint-fatal --test-only --qos=gpu48 --time=04:00:00 \
  --array="${remaining_array}" --export="${common}" "${pair}" >/dev/null
safe_sbatch --lint-fatal --test-only --partition=cpu_short --time=00:20:00 \
  --export="${common},MODE=analyze" "${analysis}" >/dev/null
safe_sbatch --lint-fatal --test-only --partition=cpu_short --time=00:20:00 \
  --export="${common},MODE=verify" "${analysis}" >/dev/null

raw=$(safe_sbatch --lint-fatal --parsable --qos=gpu48 --time=04:00:00 \
  --array="${gate_index}" --dependency="afterok:${POPULATION_VERIFY_JOB}" \
  --kill-on-invalid-dep=yes --export="${common}" "${pair}")
gate=$(job_id "${raw}")
raw=$(safe_sbatch --lint-fatal --parsable --qos=gpu48 --time=04:00:00 \
  --array="${remaining_array}" --dependency="afterok:${gate}" \
  --kill-on-invalid-dep=yes --export="${common}" "${pair}")
remaining=$(job_id "${raw}")
raw=$(safe_sbatch --lint-fatal --parsable --partition=cpu_short --time=00:20:00 \
  --dependency="afterok:${remaining}" --kill-on-invalid-dep=yes \
  --export="${common},MODE=analyze" "${analysis}")
aggregate=$(job_id "${raw}")
raw=$(safe_sbatch --lint-fatal --parsable --partition=cpu_short --time=00:20:00 \
  --dependency="afterok:${aggregate}" --kill-on-invalid-dep=yes \
  --export="${common},MODE=verify" "${analysis}")
verify=$(job_id "${raw}")

"${py}" - "${receipt}" "${gate}" "${remaining}" "${aggregate}" "${verify}" \
  "${gate_index}" "${gate_budget}" "${remaining_array}" "${population_sha}" \
  "${manifest_sha}" "${query_kind}" "${query_run_root}" \
  "${population_relative_root}" <<'PY'
import json,sys
(path,gate,remaining,aggregate,verify,gate_index,gate_budget,remaining_array,
 population_sha,manifest_sha,kind,run,population_relative)=sys.argv[1:]
p={'schema_version':'hm3d_table3_causal_survey_deferred_query_v1_20260831',
   'jobs':{'paired_query_gate':int(gate),'paired_query_remaining_array':int(remaining),
           'analysis':int(aggregate),'independent_verification':int(verify)},
   'gate_population_index':int(gate_index),'gate_max_steps':int(gate_budget),
   'remaining_population_array':remaining_array,
   'population_independent_verification_sha256':population_sha,
   'benchmark_manifest_sha256':manifest_sha,
   'population_source':kind,'run_root':run,
   'population_relative_root':population_relative,
   'powered_histories':48,'formal_queries':96,'raw_arm_role_rows':192,
   'arms':['mono_native','mono_cec'],
   'length_bins':['0_to_20_m','20_to_30_m','30_to_50_m'],
   'history_source':'controlled_causal_rgb_geodesic_survey',
   'formal_gate_retained_in_final_population':True,
   'query_policy_outcomes_read_at_submission':False,'partial_results_allowed':False,
   'fallback_completion_allowed':False,'threshold_relaxation':False,
   'smoke_substitution':False,'base_candidates_deleted_or_replaced':False}
open(path,'x').write(json.dumps(p,indent=2,sort_keys=True)+'\n')
PY
sha256sum "${receipt}" >"${receipt}.sha256"
chmod a-w "${receipt}" "${receipt}.sha256"
printf 'GATE=%s\nREMAINING=%s\nANALYZE=%s\nVERIFY=%s\n' \
  "${gate}" "${remaining}" "${aggregate}" "${verify}"
