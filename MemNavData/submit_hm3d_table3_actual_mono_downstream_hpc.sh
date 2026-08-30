#!/usr/bin/env bash
# Submit the complete, fail-closed Table-III construction and policy DAG.
set -euo pipefail
umask 0022

ROOT=${ROOT:-/home/asus/Research/Nav-graph-blind}
SSH_ALIAS=${SSH_ALIAS:-alantorch}
LOCAL_MEMNAV_PY=${LOCAL_MEMNAV_PY:-/home/asus/miniconda3/envs/memnav/bin/python}
LOCAL_HAB_PY=${LOCAL_HAB_PY:-/home/asus/miniconda3/envs/habitat/bin/python}
A_RECEIPT=${A_RECEIPT:-${ROOT}/MemNavData/HM3D_TABLE3_ACTUAL_MONO_A_SIGABRT_REPAIR_SUBMISSION_20260830.json}
REPAIR_RECEIPT=${REPAIR_RECEIPT:-${ROOT}/MemNavData/HM3D_TABLE3_ACTUAL_MONO_A_DIRECTED_GEODESIC_REPAIR_SUBMISSION_20260830.json}
OUT_RECEIPT=${OUT_RECEIPT:-MemNavData/HM3D_TABLE3_ACTUAL_MONO_DOWNSTREAM_SUBMISSION_20260830.json}
CONSTRUCTION_JOB_OVERRIDE=${CONSTRUCTION_JOB_OVERRIDE:-}
FACTUAL_DEPENDENCY_OVERRIDE=${FACTUAL_DEPENDENCY_OVERRIDE:-}
TABLE3_PLAN=/scratch/yz11502/Research/Nav-axis-uturn-results/hm3d_table3_actual_mono_20260830/plan_20260830T061943Z_3b848b5c/candidate_plan.json
EXPECTED_TABLE3_PLAN_SHA=1b1d16dd2132adb32565604bcf99f4852fa36df66a22bec2121e8338ce40020d
TASK_ROOT=/scratch/yz11502/Research/Nav-axis-uturn-source-bundles/hm3d_lifelong_natural_b_expansion_execution_1f4979a7fd37d467
TASK_RECEIPT=${TASK_ROOT}/SOURCE_BUNDLE.sha256
EXPECTED_TASK_RECEIPT_SHA=1f4979a7fd37d46700011558063be34a8fba0a0b8746668469dba7e7955f4282
SERVER_SOURCE_ROOT=/scratch/yz11502/Research/Nav-axis-uturn-source-bundles/hm3d_fullmono_lifelong_375f0b6879b2ff87
SERVER_SOURCE_RECEIPT=${SERVER_SOURCE_ROOT}/SOURCE_BUNDLE.sha256
EXPECTED_SERVER_SOURCE_RECEIPT_SHA=375f0b6879b2ff87b7019dae4727880d1b03fd3185a1862e6239942a76b5bcc8
BASE_SOURCE_ROOT=/scratch/yz11502/Research/Nav-axis-uturn-source-bundles/final14_mono_factorial_5690569a4373f2d2
BASE_RECEIPT=${BASE_SOURCE_ROOT}/source_inputs.sha256
EXPECTED_BASE_RECEIPT_SHA=5690569a4373f2d2768671418f0c604c4a03aa4b0ffe01baf70b288af03ba216
REMOTE_BUNDLES=/scratch/yz11502/Research/Nav-axis-uturn-source-bundles
SSH_CONTROL_PATH=${SSH_CONTROL_PATH:-$(ssh -G "${SSH_ALIAS}" 2>/dev/null | awk '$1=="controlpath"{v=$2} END{print v}')}

cd "${ROOT}"
fail() { echo "ABORT: $*" >&2; exit 2; }
remote() { timeout 300 ssh -n -T -o BatchMode=yes -o ControlMaster=no -S "${SSH_CONTROL_PATH}" "${SSH_ALIAS}" "$@"; }
job_id() { tr -d '\r' | awk -F';' '/^[0-9]+(;|$)/ {print $1; exit}'; }
[[ -x "${LOCAL_MEMNAV_PY}" && -x "${LOCAL_HAB_PY}" && -S "${SSH_CONTROL_PATH}" ]] || fail "local prerequisite missing"
timeout 15 ssh -O check -S "${SSH_CONTROL_PATH}" "${SSH_ALIAS}" >/dev/null 2>&1 || fail "shared SSH unavailable"
[[ -f "${A_RECEIPT}" ]] || fail "missing factual-A submission receipt"
[[ -f "${REPAIR_RECEIPT}" ]] || fail "missing directed-geodesic repair receipt"
readarray -t factual < <("${LOCAL_MEMNAV_PY}" - "${A_RECEIPT}" <<'PY'
import hashlib,json,sys
p=json.load(open(sys.argv[1]))
assert p['schema_version']=='hm3d_table3_actual_mono_a_submission_v3_20260830'
assert p['candidate_count']==125 and p['all_frozen_reserves_submitted'] is True
assert p['factual_A_outcomes_read_at_submission'] is False
assert p['query_policy_outcomes_read_at_submission'] is False
assert p['threshold_relaxation'] is False and p['fallback_completion_allowed'] is False
print(p['first_formal_candidate_job'])
print(p['factual_A_remainder_array_job'])
print(p['run_root'])
print(hashlib.sha256(open(sys.argv[1],'rb').read()).hexdigest())
PY
)
[[ "${#factual[@]}" -eq 4 && "${factual[0]}" =~ ^[0-9]+$ \
   && "${factual[1]}" =~ ^[0-9]+$ ]] || fail "invalid factual-A receipt"
factual_gate=${factual[0]}; factual_array=${factual[1]}; run_root=${factual[2]}
factual_receipt_sha=${factual[3]}
readarray -t repair < <("${LOCAL_MEMNAV_PY}" - "${REPAIR_RECEIPT}" <<'PY'
import hashlib,json,sys
p=json.load(open(sys.argv[1]))
assert p['schema_version']=='hm3d_table3_actual_mono_a_directed_geodesic_repair_launcher_submission_v1_20260830'
assert p['navigation_outcomes_read_at_submission'] is False
assert p['query_policy_outcomes_read_at_submission'] is False
assert p['scientific_thresholds_changed'] is False
assert p['fallback_completion_allowed'] is False
print(p['original_factual_A_array_job'])
print(p['repair_finish_job'])
print(p['repair_array_job'])
print(p['repair_namespace'])
print(hashlib.sha256(open(sys.argv[1],'rb').read()).hexdigest())
PY
)
[[ "${#repair[@]}" -eq 5 && "${repair[0]}" =~ ^[0-9]+$ \
   && "${repair[1]}" =~ ^[0-9]+$ && "${repair[2]}" =~ ^[0-9]+$ \
   && "${repair[3]}" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]] || \
  fail "invalid directed repair receipt"
[[ "${factual_array}" == "${repair[0]}" ]] || \
  fail "factual-A receipt and directed repair bind different arrays"
repair_finish=${repair[1]}; repair_array=${repair[2]}
repair_namespace=${repair[3]}; repair_receipt_sha=${repair[4]}
factual_dependency=${repair_finish}
if [[ -n "${FACTUAL_DEPENDENCY_OVERRIDE}" ]]; then
  [[ "${FACTUAL_DEPENDENCY_OVERRIDE}" =~ ^[0-9]+$ ]] || \
    fail "bad factual dependency override"
  [[ "${FACTUAL_DEPENDENCY_OVERRIDE}" == "${repair_finish}" ]] || \
    fail "factual dependency override is not the receipt-bound repair verifier"
  factual_dependency=${FACTUAL_DEPENDENCY_OVERRIDE}
fi
repair_completion=${run_root}/${repair_namespace}/completion.json
repair_completion_sha=$(remote "set -euo pipefail
test -f '${repair_completion}' -a -f '${repair_completion}.sha256'
digest=\$(sha256sum '${repair_completion}' | awk '{print \$1}')
test \"\$(cat '${repair_completion}.sha256')\" = \"\${digest}  completion.json\"
/scratch/lg154/conda-envs/memnav/bin/python - '${repair_completion}' \
  '${EXPECTED_TABLE3_PLAN_SHA}' '${repair_array}' <<'PY'
import json,sys
p=json.load(open(sys.argv[1]))
assert p['schema_version']=='hm3d_table3_actual_mono_a_transport_repair_completion_v1_20260830'
assert p['status']=='complete'
assert p['candidate_plan_sha256']==sys.argv[2]
assert p['repair_array_job']==int(sys.argv[3])
assert p['candidate_count']==125 and p['completion_receipts_verified']==125
assert len(p['completion_receipt_digests'])==125
assert p['completion_payloads_deserialized'] is False
assert p['navigation_outcomes_read'] is False
assert p['query_policy_outcomes_read'] is False
assert p['scientific_thresholds_changed'] is False
assert p['fallback_completion_allowed'] is False
PY
printf '%s\\n' \"\${digest}\"")
[[ "${repair_completion_sha}" =~ ^[0-9a-f]{64}$ ]] || \
  fail "directed repair completion receipt did not verify"

files=(
  MemNavData/analyze_hm3d_table3_actual_mono.py
  MemNavData/arrival_shadow.py
  MemNavData/audit_hm3d_table3_length_role_pairs.py
  MemNavData/audit_shared_online_double_revisit.py
  MemNavData/audit_shared_online_role_pairs.py
  MemNavData/bearing_diagnostics.py
  MemNavData/build_shared_online_double_revisit.py
  MemNavData/build_shared_online_role_pairs.py
  MemNavData/cec_authority_receipt.py
  MemNavData/cec_bearing_alignment.py
  MemNavData/cec_handoff_contract.py
  MemNavData/collect_hm3d_table3_actual_mono_a.py
  MemNavData/construct_hm3d_table3_actual_mono_role_pair.py
  MemNavData/controller_portability_contract.py
  MemNavData/deterministic_eval_protocol.py
  MemNavData/eval_2leg_habitat.py
  MemNavData/eval_shared_online_role_pairs.py
  MemNavData/final14_mono_factorial.py
  MemNavData/finalize_hm3d_table3_actual_mono_population.py
  MemNavData/generate_twoleg.py
  MemNavData/hm3d_fullmono_mixed_role.py
  MemNavData/hm3d_table3_actual_mono_execution_protocol_20260830.json
  MemNavData/hm3d_table3_actual_mono_protocol_20260830.json
  MemNavData/hm3d_table3_length_contract.py
  MemNavData/independent_verify_hm3d_table3_actual_mono_population.py
  MemNavData/independent_verify_hm3d_table3_actual_mono_result.py
  MemNavData/materialize_online_a_traces.py
  MemNavData/mdtec_raw_depth_gate_d.py
  MemNavData/navdp_goal_switch.py
  MemNavData/revisit_action_shadow.py
  MemNavData/revisit_bearing_adapter.py
  MemNavData/run_final14_mono_factorial_episode.py
  MemNavData/run_hm3d_fullmono_query_history.py
  MemNavData/run_hm3d_fullmono_server_scene.sh
  MemNavData/slurm_hm3d_table3_actual_mono_analysis.sbatch
  MemNavData/slurm_hm3d_table3_actual_mono_construct.sbatch
  MemNavData/slurm_hm3d_table3_actual_mono_pair.sbatch
  MemNavData/slurm_port_pair.sh
  MemNavData/slurm_hm3d_table3_actual_mono_population.sbatch
  MemNavData/shared_online_double_revisit_runtime.py
  MemNavData/shared_online_role_pair_contract.py
  MemNavData/terminal_uturn.py
  MemNavData/test_hm3d_table3_actual_mono_collection.py
  MemNavData/test_hm3d_table3_length_contract.py
  MemNavData/visual_yaw_refinement.py
  MemNavData/xnavdp_revisit_contract.py
)
for path in "${files[@]}"; do [[ -f "${path}" && ! -L "${path}" ]] || fail "missing ${path}"; done
"${LOCAL_MEMNAV_PY}" -m pytest -q \
  MemNavData/test_hm3d_table3_actual_mono_collection.py \
  MemNavData/test_hm3d_table3_length_contract.py \
  MemNavData/test_hm3d_table1_navdp_transport_contract.py
"${LOCAL_HAB_PY}" -m py_compile \
  MemNavData/construct_hm3d_table3_actual_mono_role_pair.py \
  MemNavData/eval_2leg_habitat.py MemNavData/eval_shared_online_role_pairs.py \
  MemNavData/generate_twoleg.py MemNavData/run_hm3d_fullmono_query_history.py
"${LOCAL_MEMNAV_PY}" -m py_compile \
  MemNavData/analyze_hm3d_table3_actual_mono.py \
  MemNavData/audit_hm3d_table3_length_role_pairs.py \
  MemNavData/finalize_hm3d_table3_actual_mono_population.py \
  MemNavData/hm3d_table3_length_contract.py \
  MemNavData/independent_verify_hm3d_table3_actual_mono_population.py \
  MemNavData/independent_verify_hm3d_table3_actual_mono_result.py
bash -n MemNavData/run_hm3d_fullmono_server_scene.sh \
  MemNavData/slurm_hm3d_table3_actual_mono_construct.sbatch \
  MemNavData/slurm_hm3d_table3_actual_mono_population.sbatch \
  MemNavData/slurm_hm3d_table3_actual_mono_pair.sbatch \
  MemNavData/slurm_hm3d_table3_actual_mono_analysis.sbatch

scratch=$(mktemp -d /tmp/h3_table3_downstream.XXXXXX)
trap 'rm -rf -- "${scratch}"' EXIT
mkdir -p "${scratch}/root"
for path in "${files[@]}"; do
  mkdir -p "${scratch}/root/$(dirname "${path}")"
  cp -p "${path}" "${scratch}/root/${path}"
done
(cd "${scratch}/root" && find . -type f ! -name SOURCE_BUNDLE.sha256 -print0 | sort -z | xargs -0 sha256sum >SOURCE_BUNDLE.sha256 && sha256sum -c --quiet SOURCE_BUNDLE.sha256)
receipt_sha=$(sha256sum "${scratch}/root/SOURCE_BUNDLE.sha256" | awk '{print $1}')
wrapper_root=${REMOTE_BUNDLES}/hm3d_table3_actual_mono_downstream_${receipt_sha:0:16}

remote "set -euo pipefail
test \"\$(sha256sum '${TASK_RECEIPT}' | awk '{print \$1}')\" = '${EXPECTED_TASK_RECEIPT_SHA}'
test \"\$(sha256sum '${SERVER_SOURCE_RECEIPT}' | awk '{print \$1}')\" = '${EXPECTED_SERVER_SOURCE_RECEIPT_SHA}'
test \"\$(sha256sum '${BASE_RECEIPT}' | awk '{print \$1}')\" = '${EXPECTED_BASE_RECEIPT_SHA}'
test \"\$(sha256sum '${TABLE3_PLAN}' | awk '{print \$1}')\" = '${EXPECTED_TABLE3_PLAN_SHA}'
test -d '${run_root}'"
if remote "test -d '${wrapper_root}'"; then
  remote "test \"\$(sha256sum '${wrapper_root}/SOURCE_BUNDLE.sha256' | awk '{print \$1}')\" = '${receipt_sha}'; cd '${wrapper_root}'; sha256sum -c --quiet SOURCE_BUNDLE.sha256"
else
  stage=${wrapper_root}.partial.$$
  remote "mkdir -p '${stage}'"
  rsync -a --chmod=Du=rwx,Dgo=rx,Fu=rw,Fgo=r \
    -e "ssh -o BatchMode=yes -o ControlMaster=no -S ${SSH_CONTROL_PATH}" \
    "${scratch}/root/" "${SSH_ALIAS}:${stage}/"
  remote "cd '${stage}'; sha256sum -c --quiet SOURCE_BUNDLE.sha256; chmod -R a-w '${stage}'; mv '${stage}' '${wrapper_root}'"
fi
remote "singularity exec -B /scratch/lg154 -B /scratch/yz11502 \
  /share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif env \
  PYTHONDONTWRITEBYTECODE=1 \
  PYTHONPATH='${wrapper_root}:${wrapper_root}/MemNavData:${TASK_ROOT}:${TASK_ROOT}/MemNavData:${SERVER_SOURCE_ROOT}:${SERVER_SOURCE_ROOT}/MemNavData:${BASE_SOURCE_ROOT}:${BASE_SOURCE_ROOT}/MemNavData:/scratch/yz11502/Research/Nav-axis-uturn/InternNav/src/diffusion-policy:/scratch/lg154/conda-envs/habitat/lib/python3.9/site-packages/pip/_vendor' \
  /scratch/lg154/conda-envs/habitat/bin/python \
  '${wrapper_root}/MemNavData/eval_2leg_habitat.py' --help >/dev/null"

table3_protocol=${wrapper_root}/MemNavData/hm3d_table3_actual_mono_protocol_20260830.json
execution_protocol=${wrapper_root}/MemNavData/hm3d_table3_actual_mono_execution_protocol_20260830.json
common="ALL,WRAPPER_ROOT=${wrapper_root},WRAPPER_RECEIPT=${wrapper_root}/SOURCE_BUNDLE.sha256,EXPECTED_WRAPPER_RECEIPT_SHA=${receipt_sha},TASK_ROOT=${TASK_ROOT},TASK_RECEIPT=${TASK_RECEIPT},EXPECTED_TASK_RECEIPT_SHA=${EXPECTED_TASK_RECEIPT_SHA},SERVER_SOURCE_ROOT=${SERVER_SOURCE_ROOT},SERVER_SOURCE_RECEIPT=${SERVER_SOURCE_RECEIPT},EXPECTED_SERVER_SOURCE_RECEIPT_SHA=${EXPECTED_SERVER_SOURCE_RECEIPT_SHA},BASE_SOURCE_ROOT=${BASE_SOURCE_ROOT},BASE_RECEIPT=${BASE_RECEIPT},EXPECTED_BASE_RECEIPT_SHA=${EXPECTED_BASE_RECEIPT_SHA},RUN_ROOT=${run_root},TABLE3_PLAN=${TABLE3_PLAN},EXPECTED_TABLE3_PLAN_SHA=${EXPECTED_TABLE3_PLAN_SHA},TABLE3_PROTOCOL=${table3_protocol},TABLE3_EXECUTION_PROTOCOL=${execution_protocol}"
safe=${TASK_ROOT}/MemNavData/slurm_safe_submit.sh
construct=${wrapper_root}/MemNavData/slurm_hm3d_table3_actual_mono_construct.sbatch
population=${wrapper_root}/MemNavData/slurm_hm3d_table3_actual_mono_population.sbatch
pair=${wrapper_root}/MemNavData/slurm_hm3d_table3_actual_mono_pair.sbatch
analysis=${wrapper_root}/MemNavData/slurm_hm3d_table3_actual_mono_analysis.sbatch

remote "source '${safe}'; safe_sbatch --lint-fatal --test-only --qos=gpu48 --time=01:00:00 --array=0-124%4 --export='${common}' '${construct}' >/dev/null"
remote "source '${safe}'; safe_sbatch --lint-fatal --test-only --partition=cpu_short --time=00:20:00 --export='${common},MODE=finalize' '${population}' >/dev/null"
remote "source '${safe}'; safe_sbatch --lint-fatal --test-only --partition=cpu_short --time=00:20:00 --export='${common},MODE=verify' '${population}' >/dev/null"
remote "source '${safe}'; safe_sbatch --lint-fatal --test-only --qos=gpu48 --time=04:00:00 --array=0-47%4 --export='${common}' '${pair}' >/dev/null"
remote "source '${safe}'; safe_sbatch --lint-fatal --test-only --partition=cpu_short --time=00:20:00 --export='${common},MODE=analyze' '${analysis}' >/dev/null"
remote "source '${safe}'; safe_sbatch --lint-fatal --test-only --partition=cpu_short --time=00:20:00 --export='${common},MODE=verify' '${analysis}' >/dev/null"

if [[ -n "${CONSTRUCTION_JOB_OVERRIDE}" ]]; then
  [[ "${CONSTRUCTION_JOB_OVERRIDE}" =~ ^[0-9]+$ ]] || fail "bad construction override"
  remote "scontrol show job '${CONSTRUCTION_JOB_OVERRIDE}' >/dev/null"
  construct_job=${CONSTRUCTION_JOB_OVERRIDE}
else
  raw=$(remote "source '${safe}'; safe_sbatch --lint-fatal --parsable --qos=gpu48 --time=01:00:00 --array=0-124%4 --export='${common}' '${construct}'")
  construct_job=$(printf '%s\n' "${raw}" | job_id)
fi
[[ "${construct_job}" =~ ^[0-9]+$ ]] || fail "bad construction job id"
raw=$(remote "source '${safe}'; safe_sbatch --lint-fatal --parsable --partition=cpu_short --time=00:20:00 --dependency='afterok:${construct_job}' --kill-on-invalid-dep=yes --export='${common},MODE=finalize' '${population}'")
finalize_job=$(printf '%s\n' "${raw}" | job_id)
[[ "${finalize_job}" =~ ^[0-9]+$ ]] || fail "bad population-finalize job id"
raw=$(remote "source '${safe}'; safe_sbatch --lint-fatal --parsable --partition=cpu_short --time=00:20:00 --dependency='afterok:${finalize_job}' --kill-on-invalid-dep=yes --export='${common},MODE=verify' '${population}'")
population_verify_job=$(printf '%s\n' "${raw}" | job_id)
[[ "${population_verify_job}" =~ ^[0-9]+$ ]] || fail "bad population-verifier job id"
raw=$(remote "source '${safe}'; safe_sbatch --lint-fatal --parsable --qos=gpu48 --time=04:00:00 --array=0-47%4 --dependency='afterok:${population_verify_job}' --kill-on-invalid-dep=yes --export='${common}' '${pair}'")
pair_job=$(printf '%s\n' "${raw}" | job_id)
[[ "${pair_job}" =~ ^[0-9]+$ ]] || fail "bad paired-evaluation job id"
raw=$(remote "source '${safe}'; safe_sbatch --lint-fatal --parsable --partition=cpu_short --time=00:20:00 --dependency='afterok:${pair_job}' --kill-on-invalid-dep=yes --export='${common},MODE=analyze' '${analysis}'")
analysis_job=$(printf '%s\n' "${raw}" | job_id)
[[ "${analysis_job}" =~ ^[0-9]+$ ]] || fail "bad analysis job id"
raw=$(remote "source '${safe}'; safe_sbatch --lint-fatal --parsable --partition=cpu_short --time=00:20:00 --dependency='afterok:${analysis_job}' --kill-on-invalid-dep=yes --export='${common},MODE=verify' '${analysis}'")
result_verify_job=$(printf '%s\n' "${raw}" | job_id)
[[ "${result_verify_job}" =~ ^[0-9]+$ ]] || fail "bad result-verifier job id"

receipt=${OUT_RECEIPT}
[[ ! -e "${receipt}" ]] || fail "downstream submission receipt exists"
"${LOCAL_MEMNAV_PY}" - "${receipt}" "${factual_gate}" "${factual_array}" \
  "${construct_job}" "${finalize_job}" "${population_verify_job}" \
  "${pair_job}" "${analysis_job}" "${result_verify_job}" \
  "${run_root}" "${wrapper_root}" "${receipt_sha}" \
  "${CONSTRUCTION_JOB_OVERRIDE:+1}" "${factual_dependency}" \
  "${A_RECEIPT}" "${factual_receipt_sha}" \
  "${REPAIR_RECEIPT}" "${repair_receipt_sha}" \
  "${repair_completion}" "${repair_completion_sha}" <<'PY'
import json,sys
(path,gate,factual,construct,finalize,popverify,pair,analysis,resultverify,
 run,bundle,bundle_sha,reused,factual_completion,
 factual_receipt,factual_receipt_sha,repair_receipt,repair_receipt_sha,
 repair_completion,repair_completion_sha)=sys.argv[1:]
p={
 'schema_version':'hm3d_table3_actual_mono_downstream_submission_v2_20260830',
 'factual_A_gate_job':int(gate),'factual_A_array_job':int(factual),
 'factual_completion_dependency_job':int(factual_completion),
 'factual_A_submission_receipt':factual_receipt,
 'factual_A_submission_receipt_sha256':factual_receipt_sha,
 'directed_repair_submission_receipt':repair_receipt,
 'directed_repair_submission_receipt_sha256':repair_receipt_sha,
 'directed_repair_completion_receipt':repair_completion,
 'directed_repair_completion_receipt_sha256':repair_completion_sha,
 'construction_array_job':int(construct),'construction_job_reused':reused=='1',
 'population_finalize_job':int(finalize),
 'population_independent_verification_job':int(popverify),
 'paired_query_array_job':int(pair),'analysis_job':int(analysis),
 'result_independent_verification_job':int(resultverify),
 'run_root':run,'wrapper_bundle':bundle,'wrapper_bundle_sha256':bundle_sha,
 'frozen_candidates':125,'powered_histories':48,'formal_queries':96,
 'arms':['mono_native','mono_cec'],'length_bins':['0_to_20_m','20_to_30_m','30_to_50_m'],
 'all_dependencies_afterok':True,'query_policy_outcomes_read_at_submission':False,
 'partial_results_allowed':False,'fallback_completion_allowed':False,
 'threshold_relaxation':False,'smoke_substitution':False,
}
open(path,'x').write(json.dumps(p,indent=2,sort_keys=True)+'\n')
print(json.dumps(p,indent=2,sort_keys=True))
PY
