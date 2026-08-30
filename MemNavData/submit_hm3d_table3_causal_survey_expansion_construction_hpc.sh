#!/usr/bin/env bash
# After the independently verified expansion plan exists, render every appended
# candidate and independently gate the combined 16/16/16 population.
set -euo pipefail
umask 0022

ROOT=${ROOT:-/home/asus/Research/Nav-graph-blind}
SSH_ALIAS=${SSH_ALIAS:-alantorch}
EXPECTED_SSH_USER=${EXPECTED_SSH_USER:-yz11502}
LOCAL_MEMNAV_PY=${LOCAL_MEMNAV_PY:-/home/asus/miniconda3/envs/memnav/bin/python}
LOCAL_HAB_PY=${LOCAL_HAB_PY:-/home/asus/miniconda3/envs/habitat/bin/python}
PLAN_SUBMISSION=${PLAN_SUBMISSION:-MemNavData/HM3D_TABLE3_CAUSAL_SURVEY_EXPANSION_PLAN_SUBMISSION_20260831.json}
OUT_RECEIPT=${OUT_RECEIPT:-MemNavData/HM3D_TABLE3_CAUSAL_SURVEY_EXPANSION_CONSTRUCTION_SUBMISSION_20260831.json}
BASE_RUN_ROOT=/scratch/yz11502/Research/Nav-axis-uturn-results/hm3d_table3_causal_survey_20260830/formal_20260830T150042Z_3d811e0f
BASE_CANDIDATE_PLAN=/scratch/yz11502/Research/Nav-axis-uturn-results/hm3d_table3_actual_mono_20260830/plan_20260830T061943Z_3b848b5c/candidate_plan.json
BASE_SURVEY_PROTOCOL=/scratch/yz11502/Research/Nav-axis-uturn-source-bundles/hm3d_table3_causal_survey_3d811e0f63298480/MemNavData/hm3d_table3_causal_survey_protocol_20260830.json
BASE_WRAPPER_ROOT=/scratch/yz11502/Research/Nav-axis-uturn-source-bundles/hm3d_table3_actual_mono_downstream_2d8d08ff5a65da0a
BASE_WRAPPER_RECEIPT=${BASE_WRAPPER_ROOT}/SOURCE_BUNDLE.sha256
EXPECTED_BASE_WRAPPER_RECEIPT_SHA=2d8d08ff5a65da0ad00e7372fd756174b88958ed4cf74076e6605c951b38c3fe
TASK_ROOT=/scratch/yz11502/Research/Nav-axis-uturn-source-bundles/hm3d_lifelong_natural_b_expansion_execution_1f4979a7fd37d467
TASK_RECEIPT=${TASK_ROOT}/SOURCE_BUNDLE.sha256
EXPECTED_TASK_RECEIPT_SHA=1f4979a7fd37d46700011558063be34a8fba0a0b8746668469dba7e7955f4282
REMOTE_BUNDLES=/scratch/yz11502/Research/Nav-axis-uturn-source-bundles
SSH_CONTROL_PATH=${SSH_CONTROL_PATH:-$(ssh -G "${SSH_ALIAS}" 2>/dev/null | awk '$1=="controlpath"{v=$2} END{print v}')}

cd "${ROOT}"
fail() { echo "ABORT: $*" >&2; exit 2; }
remote() {
  timeout 300 ssh -n -T -o BatchMode=yes -o ControlMaster=no \
    -S "${SSH_CONTROL_PATH}" "${SSH_ALIAS}" "$@"
}
job_id() { tr -d '\r' | awk -F';' '/^[0-9]+(;|$)/ {print $1; exit}'; }
[[ -x "${LOCAL_MEMNAV_PY}" && -x "${LOCAL_HAB_PY}" \
   && -S "${SSH_CONTROL_PATH}" ]] || fail "local prerequisite missing"
timeout 15 ssh -O check -S "${SSH_CONTROL_PATH}" "${SSH_ALIAS}" \
  >/dev/null 2>&1 || fail "shared SSH unavailable"
remote_user=$(timeout 15 ssh -n -tt -o BatchMode=yes -o ControlMaster=no \
  -S "${SSH_CONTROL_PATH}" "${SSH_ALIAS}" 'id -un' 2>/dev/null \
  | tr -d '\r' | tail -n 1)
[[ "${remote_user}" == "${EXPECTED_SSH_USER}" ]] \
  || fail "shared SSH identity is ${remote_user:-unavailable}, expected ${EXPECTED_SSH_USER}"
[[ ! -e "${OUT_RECEIPT}" ]] || fail "submission receipt already exists"

readarray -t frozen < <("${LOCAL_MEMNAV_PY}" - "${PLAN_SUBMISSION}" <<'PY'
import json,sys
p=json.load(open(sys.argv[1]))
assert p['schema_version']=='hm3d_table3_causal_survey_expansion_plan_submission_v1_20260831'
assert p['query_policy_jobs_submitted'] is False
assert p['query_policy_outcomes_read'] is False
assert p['navigation_policy_outcomes_read'] is False
assert p['base_candidates_deleted_or_replaced'] is False
print(p['run_root'])
print(p['source_bundle'])
print(p['source_bundle_sha256'])
PY
)
[[ "${#frozen[@]}" -eq 3 ]] || fail "invalid expansion-plan submission"
run_root=${frozen[0]}
plan_source_root=${frozen[1]}
plan_source_sha=${frozen[2]}
plan=${run_root}/expansion_plan/candidate_plan.json
construction_protocol=${run_root}/expansion_plan/construction_protocol.json
plan_verification=${run_root}/expansion_plan/independent_verification.json

readarray -t plan_state < <(remote "/scratch/lg154/conda-envs/memnav/bin/python - '${plan}' '${construction_protocol}' '${plan_verification}' <<'PY'
import hashlib,json,sys
plan,protocol,verification=sys.argv[1:]
sha=lambda p: hashlib.sha256(open(p,'rb').read()).hexdigest()
p=json.load(open(plan)); c=json.load(open(protocol)); v=json.load(open(verification))
assert v['schema_version']=='hm3d_table3_causal_survey_expansion_plan_verification_v1_20260831'
assert v['verified'] is True
assert v['plan_sha256']==sha(plan)
assert v['construction_protocol_sha256']==sha(protocol)
assert v['candidate_count']==p['candidate_count']==c['source_candidate_plan']['candidate_count']
assert v['query_policy_outcomes_read'] is False
assert v['navigation_policy_outcomes_read'] is False
assert v['query_policy_evaluation_authorized'] is False
print(p['candidate_count'])
print(sha(plan)); print(sha(protocol)); print(sha(verification))
print(','.join(p['deficient_bins']))
PY")
[[ "${#plan_state[@]}" -eq 5 ]] || fail "expansion plan is not verified"
candidate_count=${plan_state[0]}
plan_sha=${plan_state[1]}
protocol_sha=${plan_state[2]}
plan_verification_sha=${plan_state[3]}
deficient_bins=${plan_state[4]}
[[ "${candidate_count}" =~ ^[1-9][0-9]*$ ]] \
  || fail "verified expansion is empty; original population must be used"

files=(
  MemNavData/analyze_hm3d_table3_actual_mono.py
  MemNavData/audit_hm3d_table3_length_role_pairs.py
  MemNavData/bundle_selftest.sh
  MemNavData/construct_hm3d_table3_causal_survey_role_pair.py
  MemNavData/eval_shared_online_role_pairs.py
  MemNavData/finalize_hm3d_table3_causal_survey_merged_population.py
  MemNavData/hm3d_table3_causal_survey_contract.py
  MemNavData/hm3d_table3_length_contract.py
  MemNavData/independent_verify_hm3d_table3_actual_mono_result.py
  MemNavData/independent_verify_hm3d_table3_causal_survey_merged_population.py
  MemNavData/run_hm3d_fullmono_query_history.py
  MemNavData/run_hm3d_fullmono_server_scene.sh
  MemNavData/slurm_hm3d_table3_causal_survey_expansion_construct.sbatch
  MemNavData/slurm_hm3d_table3_causal_survey_merged_population.sbatch
  MemNavData/test_audit_hm3d_table3_actual_mono_constructibility.py
  MemNavData/test_hm3d_table3_causal_survey_expansion.py
  MemNavData/test_hm3d_table3_length_contract.py
)
for path in "${files[@]}"; do
  [[ -f "${path}" && ! -L "${path}" ]] || fail "missing ${path}"
done
"${LOCAL_MEMNAV_PY}" -m pytest -q \
  MemNavData/test_hm3d_table3_causal_survey_expansion.py \
  MemNavData/test_hm3d_table3_length_contract.py \
  MemNavData/test_audit_hm3d_table3_actual_mono_constructibility.py
"${LOCAL_HAB_PY}" -m py_compile \
  MemNavData/construct_hm3d_table3_causal_survey_role_pair.py \
  MemNavData/eval_shared_online_role_pairs.py \
  MemNavData/run_hm3d_fullmono_query_history.py
"${LOCAL_MEMNAV_PY}" -m py_compile \
  MemNavData/audit_hm3d_table3_length_role_pairs.py \
  MemNavData/finalize_hm3d_table3_causal_survey_merged_population.py \
  MemNavData/hm3d_table3_length_contract.py \
  MemNavData/independent_verify_hm3d_table3_causal_survey_merged_population.py
bash -n MemNavData/run_hm3d_fullmono_server_scene.sh \
  MemNavData/slurm_hm3d_table3_causal_survey_expansion_construct.sbatch \
  MemNavData/slurm_hm3d_table3_causal_survey_merged_population.sbatch "$0"

scratch=$(mktemp -d /tmp/h3_table3_expansion_exec.XXXXXX)
trap 'rm -r -- "${scratch}"' EXIT
mkdir -p "${scratch}/root"
for path in "${files[@]}"; do
  mkdir -p "${scratch}/root/$(dirname "${path}")"
  cp -p "${path}" "${scratch}/root/${path}"
done
SELFTEST_BUNDLE_SUBPATHS=MemNavData \
  bash "${scratch}/root/MemNavData/bundle_selftest.sh" "${scratch}/root" \
  <(printf '%s import %s\n' \
    "${LOCAL_MEMNAV_PY}" MemNavData.finalize_hm3d_table3_causal_survey_merged_population \
    "${LOCAL_MEMNAV_PY}" MemNavData.independent_verify_hm3d_table3_causal_survey_merged_population)
(cd "${scratch}/root" && \
  find . -type f ! -name SOURCE_BUNDLE.sha256 -print0 | sort -z | \
  xargs -0 sha256sum >SOURCE_BUNDLE.sha256 && \
  sha256sum -c --quiet SOURCE_BUNDLE.sha256)
receipt_sha=$(sha256sum "${scratch}/root/SOURCE_BUNDLE.sha256" | awk '{print $1}')
source_root=${REMOTE_BUNDLES}/hm3d_table3_causal_survey_expansion_exec_${receipt_sha:0:16}
source_receipt=${source_root}/SOURCE_BUNDLE.sha256

remote "set -euo pipefail
test \"\$(id -un)\" = '${EXPECTED_SSH_USER}'
test \"\$(sha256sum '${TASK_RECEIPT}' | awk '{print \$1}')\" = '${EXPECTED_TASK_RECEIPT_SHA}'
cd '${TASK_ROOT}'; sha256sum -c --quiet SOURCE_BUNDLE.sha256
test \"\$(sha256sum '${plan_source_root}/SOURCE_BUNDLE.sha256' | awk '{print \$1}')\" = '${plan_source_sha}'
cd '${plan_source_root}'; sha256sum -c --quiet SOURCE_BUNDLE.sha256
test \"\$(sha256sum '${BASE_WRAPPER_RECEIPT}' | awk '{print \$1}')\" = '${EXPECTED_BASE_WRAPPER_RECEIPT_SHA}'
cd '${BASE_WRAPPER_ROOT}'; sha256sum -c --quiet SOURCE_BUNDLE.sha256"
if remote "test -d '${source_root}'"; then
  remote "set -euo pipefail
test \"\$(id -un)\" = '${EXPECTED_SSH_USER}'
test \"\$(sha256sum '${source_receipt}' | awk '{print \$1}')\" = '${receipt_sha}'
cd '${source_root}'; sha256sum -c --quiet SOURCE_BUNDLE.sha256
PYTHONPATH='${source_root}:${source_root}/MemNavData' PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 /scratch/lg154/conda-envs/memnav/bin/python -c 'import MemNavData.finalize_hm3d_table3_causal_survey_merged_population, MemNavData.independent_verify_hm3d_table3_causal_survey_merged_population'"
else
  stage=${source_root}.partial.$$
  remote "set -euo pipefail; test \"\$(id -un)\" = '${EXPECTED_SSH_USER}'; mkdir -p '${stage}'"
  rsync -a --chmod=Du=rwx,Dgo=rx,Fu=rw,Fgo=r \
    -e "ssh -o BatchMode=yes -o ControlMaster=no -S ${SSH_CONTROL_PATH}" \
    "${scratch}/root/" "${SSH_ALIAS}:${stage}/"
  remote "set -euo pipefail
test \"\$(id -un)\" = '${EXPECTED_SSH_USER}'
cd '${stage}'; sha256sum -c --quiet SOURCE_BUNDLE.sha256
SELFTEST_BUNDLE_SUBPATHS=MemNavData bash '${stage}/MemNavData/bundle_selftest.sh' '${stage}' <(printf '%s import %s\\n' /scratch/lg154/conda-envs/memnav/bin/python MemNavData.finalize_hm3d_table3_causal_survey_merged_population /scratch/lg154/conda-envs/memnav/bin/python MemNavData.independent_verify_hm3d_table3_causal_survey_merged_population)
cd '${stage}'; sha256sum -c --quiet SOURCE_BUNDLE.sha256
singularity exec -B /scratch/lg154 -B /scratch/yz11502 /share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH='${stage}:${stage}/MemNavData:${BASE_WRAPPER_ROOT}:${BASE_WRAPPER_ROOT}/MemNavData' /scratch/lg154/conda-envs/habitat/bin/python '${stage}/MemNavData/construct_hm3d_table3_causal_survey_role_pair.py' --help >/dev/null
chmod -R a-w '${stage}'; mv '${stage}' '${source_root}'"
fi

safe=${TASK_ROOT}/MemNavData/slurm_safe_submit.sh
construct=${source_root}/MemNavData/slurm_hm3d_table3_causal_survey_expansion_construct.sbatch
population=${source_root}/MemNavData/slurm_hm3d_table3_causal_survey_merged_population.sbatch
common="ALL,EXPANSION_EXEC_SOURCE_ROOT=${source_root},EXPANSION_EXEC_SOURCE_RECEIPT=${source_receipt},EXPECTED_EXPANSION_EXEC_SOURCE_RECEIPT_SHA=${receipt_sha},BASE_WRAPPER_ROOT=${BASE_WRAPPER_ROOT},BASE_WRAPPER_RECEIPT=${BASE_WRAPPER_RECEIPT},EXPECTED_BASE_WRAPPER_RECEIPT_SHA=${EXPECTED_BASE_WRAPPER_RECEIPT_SHA},BASE_RUN_ROOT=${BASE_RUN_ROOT},BASE_CANDIDATE_PLAN=${BASE_CANDIDATE_PLAN},BASE_SURVEY_PROTOCOL=${BASE_SURVEY_PROTOCOL},EXPANSION_RUN_ROOT=${run_root},EXPANSION_PLAN=${plan},EXPECTED_EXPANSION_PLAN_SHA=${plan_sha},EXPANSION_CONSTRUCTION_PROTOCOL=${construction_protocol},EXPECTED_EXPANSION_PROTOCOL_SHA=${protocol_sha},EXPANSION_PLAN_VERIFICATION=${plan_verification},EXPECTED_EXPANSION_PLAN_VERIFICATION_SHA=${plan_verification_sha},EXPECTED_EXPANSION_CANDIDATE_COUNT=${candidate_count}"
array="0-$((candidate_count - 1))%4"
remote "set -euo pipefail; test \"\$(id -un)\" = '${EXPECTED_SSH_USER}'; source '${safe}'; safe_sbatch --lint-fatal --test-only --qos=gpu48 --time=01:00:00 --array='${array}' --export='${common}' '${construct}' >/dev/null"
remote "set -euo pipefail; test \"\$(id -un)\" = '${EXPECTED_SSH_USER}'; source '${safe}'; safe_sbatch --lint-fatal --test-only --partition=cpu_short --time=00:30:00 --export='${common},MODE=finalize' '${population}' >/dev/null"
remote "set -euo pipefail; test \"\$(id -un)\" = '${EXPECTED_SSH_USER}'; source '${safe}'; safe_sbatch --lint-fatal --test-only --partition=cpu_short --time=00:30:00 --export='${common},MODE=verify' '${population}' >/dev/null"

raw=$(remote "set -euo pipefail; test \"\$(id -un)\" = '${EXPECTED_SSH_USER}'; source '${safe}'; safe_sbatch --lint-fatal --parsable --qos=gpu48 --time=01:00:00 --array='${array}' --export='${common}' '${construct}'")
construct_job=$(printf '%s\n' "${raw}" | job_id)
[[ "${construct_job}" =~ ^[0-9]+$ ]] || fail "invalid expansion construction job"
raw=$(remote "set -euo pipefail; test \"\$(id -un)\" = '${EXPECTED_SSH_USER}'; source '${safe}'; safe_sbatch --lint-fatal --parsable --partition=cpu_short --time=00:30:00 --dependency='afterok:${construct_job}' --kill-on-invalid-dep=yes --export='${common},MODE=finalize' '${population}'")
finalize_job=$(printf '%s\n' "${raw}" | job_id)
[[ "${finalize_job}" =~ ^[0-9]+$ ]] || fail "invalid merged finalizer job"
raw=$(remote "set -euo pipefail; test \"\$(id -un)\" = '${EXPECTED_SSH_USER}'; source '${safe}'; safe_sbatch --lint-fatal --parsable --partition=cpu_short --time=00:30:00 --dependency='afterok:${finalize_job}' --kill-on-invalid-dep=yes --export='${common},MODE=verify' '${population}'")
verify_job=$(printf '%s\n' "${raw}" | job_id)
[[ "${verify_job}" =~ ^[0-9]+$ ]] || fail "invalid merged population verifier job"

"${LOCAL_MEMNAV_PY}" - "${OUT_RECEIPT}" "${construct_job}" "${finalize_job}" \
  "${verify_job}" "${run_root}" "${source_root}" "${receipt_sha}" \
  "${candidate_count}" "${deficient_bins}" <<'PY'
import json,sys
path,construct,finalize,verify,run,source,source_sha,count,bins=sys.argv[1:]
payload={
  'schema_version':'hm3d_table3_causal_survey_expansion_construction_submission_v1_20260831',
  'expansion_construction_array_job':int(construct),
  'merged_population_finalize_job':int(finalize),
  'merged_population_independent_verification_job':int(verify),
  'run_root':run,'source_bundle':source,'source_bundle_sha256':source_sha,
  'expansion_candidates':int(count),'deficient_bins':bins.split(',') if bins else [],
  'merged_population':'merged_query_population',
  'base_candidates_deleted_or_replaced':False,
  'query_policy_jobs_submitted':False,'query_policy_outcomes_read':False,
  'threshold_relaxation':False,'fallback_completion_allowed':False,
}
open(path,'x').write(json.dumps(payload,indent=2,sort_keys=True)+'\n')
print(json.dumps(payload,indent=2,sort_keys=True))
PY
