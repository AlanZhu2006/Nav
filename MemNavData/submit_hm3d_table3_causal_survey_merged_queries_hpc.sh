#!/usr/bin/env bash
# Submit the formal 48-history/96-query Table-III evaluation from the
# independently verified append-only base+expansion population.
set -euo pipefail
umask 0022

ROOT=${ROOT:-/home/asus/Research/Nav-graph-blind}
SSH_ALIAS=${SSH_ALIAS:-alantorch}
EXPECTED_SSH_USER=${EXPECTED_SSH_USER:-yz11502}
LOCAL_MEMNAV_PY=${LOCAL_MEMNAV_PY:-/home/asus/miniconda3/envs/memnav/bin/python}
LOCAL_HAB_PY=${LOCAL_HAB_PY:-/home/asus/miniconda3/envs/habitat/bin/python}
CONSTRUCTION_RECEIPT=${CONSTRUCTION_RECEIPT:-${ROOT}/MemNavData/HM3D_TABLE3_CAUSAL_SURVEY_EXPANSION_CONSTRUCTION_SUBMISSION_20260831.json}
OUT_RECEIPT=${OUT_RECEIPT:-MemNavData/HM3D_TABLE3_CAUSAL_SURVEY_MERGED_QUERY_SUBMISSION_20260831.json}
TASK_ROOT=/scratch/yz11502/Research/Nav-axis-uturn-source-bundles/hm3d_lifelong_natural_b_expansion_execution_1f4979a7fd37d467
TASK_RECEIPT=${TASK_ROOT}/SOURCE_BUNDLE.sha256
EXPECTED_TASK_RECEIPT_SHA=1f4979a7fd37d46700011558063be34a8fba0a0b8746668469dba7e7955f4282
SERVER_SOURCE_ROOT=/scratch/yz11502/Research/Nav-axis-uturn-source-bundles/hm3d_fullmono_lifelong_375f0b6879b2ff87
SERVER_SOURCE_RECEIPT=${SERVER_SOURCE_ROOT}/SOURCE_BUNDLE.sha256
EXPECTED_SERVER_SOURCE_RECEIPT_SHA=375f0b6879b2ff87b7019dae4727880d1b03fd3185a1862e6239942a76b5bcc8
BASE_SOURCE_ROOT=/scratch/yz11502/Research/Nav-axis-uturn-source-bundles/final14_mono_factorial_5690569a4373f2d2
BASE_RECEIPT=${BASE_SOURCE_ROOT}/source_inputs.sha256
EXPECTED_BASE_RECEIPT_SHA=5690569a4373f2d2768671418f0c604c4a03aa4b0ffe01baf70b288af03ba216
RUNTIME_CLOSURE_ROOT=/scratch/yz11502/Research/Nav-axis-uturn-source-bundles/hm3d_table1_navdp_authority_transaction_repair_82e71f19ee7f4e52
RUNTIME_CLOSURE_RECEIPT=${RUNTIME_CLOSURE_ROOT}/SOURCE_BUNDLE.sha256
EXPECTED_RUNTIME_CLOSURE_RECEIPT_SHA=82e71f19ee7f4e5233fae499633ce5a233c9c036bb41b9e2bf7d4f0f18effd7d
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
  >/dev/null 2>&1 || fail "authoritative shared SSH master unavailable"
remote_user=$(timeout 15 ssh -n -tt -o BatchMode=yes -o ControlMaster=no \
  -S "${SSH_CONTROL_PATH}" "${SSH_ALIAS}" 'id -un' 2>/dev/null \
  | tr -d '\r' | tail -n 1)
[[ "${remote_user}" == "${EXPECTED_SSH_USER}" ]] \
  || fail "shared SSH identity is ${remote_user:-unavailable}, expected ${EXPECTED_SSH_USER}"
[[ -f "${CONSTRUCTION_RECEIPT}" ]] || fail "merged construction receipt missing"
[[ ! -e "${OUT_RECEIPT}" ]] || fail "merged query receipt already exists"

readarray -t frozen < <("${LOCAL_MEMNAV_PY}" - "${CONSTRUCTION_RECEIPT}" <<'PY'
import hashlib,json,sys
p=json.load(open(sys.argv[1]))
assert p['schema_version']=='hm3d_table3_causal_survey_expansion_construction_submission_v1_20260831'
assert p['base_candidates_deleted_or_replaced'] is False
assert p['query_policy_jobs_submitted'] is False
assert p['query_policy_outcomes_read'] is False
assert p['threshold_relaxation'] is False
assert p['fallback_completion_allowed'] is False
assert p['expansion_candidates'] > 0
print(p['merged_population_independent_verification_job'])
print(p['run_root'])
print(p['source_bundle'])
print(p['source_bundle_sha256'])
print(hashlib.sha256(open(sys.argv[1],'rb').read()).hexdigest())
PY
)
[[ "${#frozen[@]}" -eq 5 && "${frozen[0]}" =~ ^[0-9]+$ \
   && "${frozen[3]}" =~ ^[0-9a-f]{64}$ \
   && "${frozen[4]}" =~ ^[0-9a-f]{64}$ ]] || fail "bad merged construction receipt"
population_verify_job=${frozen[0]}
run_root=${frozen[1]}
construction_source=${frozen[2]}
construction_source_sha=${frozen[3]}
construction_receipt_sha=${frozen[4]}
population=${run_root}/merged_query_population
verification=${population}/independent_verification.json
manifest=${population}/role_pairs/manifest.json

readarray -t sealed < <(remote "set -euo pipefail
test \"\$(id -un)\" = '${EXPECTED_SSH_USER}'
test \"\$(sha256sum '${construction_source}/SOURCE_BUNDLE.sha256' | awk '{print \$1}')\" = '${construction_source_sha}'
cd '${construction_source}'; sha256sum -c --quiet SOURCE_BUNDLE.sha256
test -f '${verification}' -a -f '${verification}.sha256'
test -f '${manifest}' -a -f '${manifest}.sha256'
test -f '${population}/population_receipt.json' -a -f '${population}/population_receipt.json.sha256'
cd '${population}'; sha256sum -c --quiet independent_verification.json.sha256 population_receipt.json.sha256
cd '${population}/role_pairs'; sha256sum -c --quiet manifest.json.sha256
/scratch/lg154/conda-envs/memnav/bin/python - '${verification}' '${manifest}' <<'PY'
import hashlib,json,sys
import math
sha=lambda p: hashlib.sha256(open(p,'rb').read()).hexdigest()
v=json.load(open(sys.argv[1])); m=json.load(open(sys.argv[2]))
assert v['schema_version']=='hm3d_table3_causal_survey_population_verification_v2_20260831'
assert v['verified'] is True and v['formal_policy_evaluation_authorized'] is True
assert v['history_source']=='controlled_causal_rgb_geodesic_survey'
assert v['query_policy_outcomes_read'] is False
assert v['fallback_completion_allowed'] is False
assert v['base_candidates_preserved']==125
assert v['expansion_candidates_appended']>0
assert v['histories_by_bin']=={'0_to_20_m':16,'20_to_30_m':16,'30_to_50_m':16}
assert all(int(n)>=10 for n in v['scene_clusters_by_bin'].values())
assert len(m['episodes'])==48
assert {row['population_source'] for row in m['episodes']} <= {'base','expansion'}
assert any(row['population_source']=='expansion' for row in m['episodes'])
assert sha(sys.argv[2])==v['benchmark_manifest_sha256']
budgets=[]
for index,row in enumerate(m['episodes']):
    distances=[float(q['geodesic_from_a_end_m']) for q in row['pairs'][0]['queries']]
    budget=max(600,math.ceil(2.5*max(distances)/0.0376))
    assert budget <= 3400
    budgets.append((budget,index))
gate_budget,gate_index=max(budgets)
remaining=[i for i in range(48) if i != gate_index]
def compact(values):
    runs=[]; start=previous=values[0]
    for value in values[1:]:
        if value == previous + 1:
            previous=value; continue
        runs.append(str(start) if start == previous else f'{start}-{previous}')
        start=previous=value
    runs.append(str(start) if start == previous else f'{start}-{previous}')
    return ','.join(runs)
print(sha(sys.argv[1])); print(sha(sys.argv[2]))
print(gate_index); print(gate_budget); print(compact(remaining)+'%4')
PY
test ! -e '${run_root}/evaluation'
test ! -e '${run_root}/table3_result'
")
[[ "${#sealed[@]}" -eq 5 && "${sealed[0]}" =~ ^[0-9a-f]{64}$ \
   && "${sealed[1]}" =~ ^[0-9a-f]{64}$ \
   && "${sealed[2]}" =~ ^([0-9]|[1-3][0-9]|4[0-7])$ \
   && "${sealed[3]}" =~ ^[0-9]+$ \
   && "${sealed[4]}" =~ ^[0-9,-]+%4$ ]] || fail "merged population seal failed"
population_verification_sha=${sealed[0]}
benchmark_manifest_sha=${sealed[1]}
gate_index=${sealed[2]}
gate_budget=${sealed[3]}
remaining_array=${sealed[4]}

files=(
  MemNavData/analyze_hm3d_table3_causal_survey.py
  MemNavData/audit_hm3d_table3_length_role_pairs.py
  MemNavData/bundle_selftest.sh
  MemNavData/cec_handoff_contract.py
  MemNavData/certified_relocalization_runtime.py
  MemNavData/controller_portability_contract.py
  MemNavData/eval_shared_online_role_pairs.py
  MemNavData/hm3d_table3_causal_survey_contract.py
  MemNavData/hm3d_table3_causal_survey_protocol_20260830.json
  MemNavData/hm3d_table3_length_contract.py
  MemNavData/independent_verify_hm3d_table3_causal_survey_result.py
  MemNavData/monocular_depth_runtime.py
  MemNavData/run_hm3d_fullmono_query_history.py
  MemNavData/run_hm3d_fullmono_server_scene.sh
  MemNavData/slurm_hm3d_table3_causal_survey_analysis.sbatch
  MemNavData/slurm_hm3d_table3_causal_survey_pair.sbatch
  MemNavData/slurm_port_pair.sh
  MemNavData/test_hm3d_table3_causal_survey_expansion.py
  MemNavData/test_hm3d_table3_causal_survey_submission.py
  MemNavData/test_hm3d_table3_length_contract.py
)
for path in "${files[@]}"; do
  [[ -f "${path}" && ! -L "${path}" ]] || fail "missing ${path}"
done
"${LOCAL_MEMNAV_PY}" -m pytest -q \
  MemNavData/test_hm3d_table3_causal_survey_expansion.py \
  MemNavData/test_hm3d_table3_length_contract.py \
  MemNavData/test_hm3d_table3_causal_survey_submission.py
"${LOCAL_HAB_PY}" -m py_compile \
  MemNavData/eval_shared_online_role_pairs.py \
  MemNavData/run_hm3d_fullmono_query_history.py
"${LOCAL_MEMNAV_PY}" -m py_compile \
  MemNavData/analyze_hm3d_table3_causal_survey.py \
  MemNavData/cec_handoff_contract.py \
  MemNavData/certified_relocalization_runtime.py \
  MemNavData/controller_portability_contract.py \
  MemNavData/hm3d_table3_length_contract.py \
  MemNavData/independent_verify_hm3d_table3_causal_survey_result.py \
  MemNavData/monocular_depth_runtime.py
bash -n MemNavData/run_hm3d_fullmono_server_scene.sh \
  MemNavData/slurm_hm3d_table3_causal_survey_pair.sbatch \
  MemNavData/slurm_hm3d_table3_causal_survey_analysis.sbatch "$0"

scratch=$(mktemp -d /tmp/h3_table3_merged_query.XXXXXX)
trap 'rm -r -- "${scratch}"' EXIT
mkdir -p "${scratch}/root"
for path in "${files[@]}"; do
  mkdir -p "${scratch}/root/$(dirname "${path}")"
  cp -p "${path}" "${scratch}/root/${path}"
done
SELFTEST_BUNDLE_SUBPATHS=MemNavData \
  bash "${scratch}/root/MemNavData/bundle_selftest.sh" "${scratch}/root" \
  <(printf '%s import %s\n' \
    "${LOCAL_MEMNAV_PY}" MemNavData.analyze_hm3d_table3_causal_survey \
    "${LOCAL_MEMNAV_PY}" MemNavData.independent_verify_hm3d_table3_causal_survey_result)
(cd "${scratch}/root" && \
  find . -type f ! -name SOURCE_BUNDLE.sha256 -print0 | sort -z | \
  xargs -0 sha256sum >SOURCE_BUNDLE.sha256 && \
  sha256sum -c --quiet SOURCE_BUNDLE.sha256)
receipt_sha=$(sha256sum "${scratch}/root/SOURCE_BUNDLE.sha256" | awk '{print $1}')
source_root=${REMOTE_BUNDLES}/hm3d_table3_causal_survey_merged_query_${receipt_sha:0:16}
source_receipt=${source_root}/SOURCE_BUNDLE.sha256
if remote "test -d '${source_root}'"; then
  remote "set -euo pipefail
test \"\$(id -un)\" = '${EXPECTED_SSH_USER}'
test \"\$(sha256sum '${source_receipt}' | awk '{print \$1}')\" = '${receipt_sha}'
cd '${source_root}'; sha256sum -c --quiet SOURCE_BUNDLE.sha256
PYTHONPATH='${source_root}:${source_root}/MemNavData' PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 /scratch/lg154/conda-envs/memnav/bin/python -c 'import MemNavData.analyze_hm3d_table3_causal_survey, MemNavData.independent_verify_hm3d_table3_causal_survey_result'"
else
  stage=${source_root}.partial.$$
  remote "set -euo pipefail; test \"\$(id -un)\" = '${EXPECTED_SSH_USER}'; mkdir -p '${stage}'"
  rsync -a --chmod=Du=rwx,Dgo=rx,Fu=rw,Fgo=r \
    -e "ssh -o BatchMode=yes -o ControlMaster=no -S ${SSH_CONTROL_PATH}" \
    "${scratch}/root/" "${SSH_ALIAS}:${stage}/"
  remote "set -euo pipefail
test \"\$(id -un)\" = '${EXPECTED_SSH_USER}'
cd '${stage}'; sha256sum -c --quiet SOURCE_BUNDLE.sha256
SELFTEST_BUNDLE_SUBPATHS=MemNavData bash '${stage}/MemNavData/bundle_selftest.sh' '${stage}' <(printf '%s import %s\\n' /scratch/lg154/conda-envs/memnav/bin/python MemNavData.analyze_hm3d_table3_causal_survey /scratch/lg154/conda-envs/memnav/bin/python MemNavData.independent_verify_hm3d_table3_causal_survey_result)
cd '${stage}'; sha256sum -c --quiet SOURCE_BUNDLE.sha256
chmod -R a-w '${stage}'; mv '${stage}' '${source_root}'"
fi

protocol=${source_root}/MemNavData/hm3d_table3_causal_survey_protocol_20260830.json
pair=${source_root}/MemNavData/slurm_hm3d_table3_causal_survey_pair.sbatch
analysis=${source_root}/MemNavData/slurm_hm3d_table3_causal_survey_analysis.sbatch
# The merged manifest is also the authoritative scene-index namespace.  The
# original 54-scene candidate plan cannot bound append-only capacity scenes.
candidate_plan=${manifest}
candidate_plan_sha=${benchmark_manifest_sha}
common="ALL,SURVEY_SOURCE_ROOT=${source_root},SURVEY_SOURCE_RECEIPT=${source_receipt},EXPECTED_SURVEY_SOURCE_RECEIPT_SHA=${receipt_sha},TASK_ROOT=${TASK_ROOT},TASK_RECEIPT=${TASK_RECEIPT},EXPECTED_TASK_RECEIPT_SHA=${EXPECTED_TASK_RECEIPT_SHA},SERVER_SOURCE_ROOT=${SERVER_SOURCE_ROOT},SERVER_SOURCE_RECEIPT=${SERVER_SOURCE_RECEIPT},EXPECTED_SERVER_SOURCE_RECEIPT_SHA=${EXPECTED_SERVER_SOURCE_RECEIPT_SHA},BASE_SOURCE_ROOT=${BASE_SOURCE_ROOT},BASE_RECEIPT=${BASE_RECEIPT},EXPECTED_BASE_RECEIPT_SHA=${EXPECTED_BASE_RECEIPT_SHA},RUNTIME_CLOSURE_ROOT=${RUNTIME_CLOSURE_ROOT},RUNTIME_CLOSURE_RECEIPT=${RUNTIME_CLOSURE_RECEIPT},EXPECTED_RUNTIME_CLOSURE_RECEIPT_SHA=${EXPECTED_RUNTIME_CLOSURE_RECEIPT_SHA},RUN_ROOT=${run_root},CANDIDATE_PLAN=${candidate_plan},EXPECTED_CANDIDATE_PLAN_SHA=${candidate_plan_sha},SURVEY_PROTOCOL=${protocol},POPULATION_RELATIVE_ROOT=merged_query_population,EXPECTED_POPULATION_VERIFICATION_SHA=${population_verification_sha},EXPECTED_BENCHMARK_MANIFEST_SHA=${benchmark_manifest_sha}"
safe=${TASK_ROOT}/MemNavData/slurm_safe_submit.sh

remote "set -euo pipefail
test \"\$(id -un)\" = '${EXPECTED_SSH_USER}'
test \"\$(sha256sum '${TASK_RECEIPT}' | awk '{print \$1}')\" = '${EXPECTED_TASK_RECEIPT_SHA}'
test \"\$(sha256sum '${SERVER_SOURCE_RECEIPT}' | awk '{print \$1}')\" = '${EXPECTED_SERVER_SOURCE_RECEIPT_SHA}'
test \"\$(sha256sum '${BASE_RECEIPT}' | awk '{print \$1}')\" = '${EXPECTED_BASE_RECEIPT_SHA}'
test \"\$(sha256sum '${RUNTIME_CLOSURE_RECEIPT}' | awk '{print \$1}')\" = '${EXPECTED_RUNTIME_CLOSURE_RECEIPT_SHA}'
cd '${RUNTIME_CLOSURE_ROOT}'; sha256sum -c --quiet SOURCE_BUNDLE.sha256
test \"\$(sha256sum '${candidate_plan}' | awk '{print \$1}')\" = '${candidate_plan_sha}'
singularity exec -B /scratch/lg154 -B /scratch/yz11502 /share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH='${source_root}:${source_root}/MemNavData:${RUNTIME_CLOSURE_ROOT}:${RUNTIME_CLOSURE_ROOT}/MemNavData:${TASK_ROOT}:${TASK_ROOT}/MemNavData:${SERVER_SOURCE_ROOT}:${SERVER_SOURCE_ROOT}/MemNavData:${BASE_SOURCE_ROOT}:${BASE_SOURCE_ROOT}/MemNavData:/scratch/yz11502/Research/Nav-axis-uturn/InternNav/src/diffusion-policy:/scratch/lg154/conda-envs/habitat/lib/python3.9/site-packages/pip/_vendor' /scratch/lg154/conda-envs/habitat/bin/python '${source_root}/MemNavData/eval_shared_online_role_pairs.py' --help >/dev/null
singularity exec -B /scratch/lg154 -B /scratch/yz11502 /share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH='${source_root}:${source_root}/MemNavData:${RUNTIME_CLOSURE_ROOT}:${RUNTIME_CLOSURE_ROOT}/MemNavData:${TASK_ROOT}:${TASK_ROOT}/MemNavData:${SERVER_SOURCE_ROOT}:${SERVER_SOURCE_ROOT}/MemNavData:${BASE_SOURCE_ROOT}:${BASE_SOURCE_ROOT}/MemNavData' /scratch/lg154/conda-envs/habitat/bin/python '${source_root}/MemNavData/run_hm3d_fullmono_query_history.py' --help >/dev/null
source '${safe}'
safe_sbatch --lint-fatal --test-only --qos=gpu48 --time=04:00:00 --array='${gate_index}' --export='${common}' '${pair}' >/dev/null
safe_sbatch --lint-fatal --test-only --qos=gpu48 --time=04:00:00 --array='${remaining_array}' --export='${common}' '${pair}' >/dev/null
safe_sbatch --lint-fatal --test-only --partition=cpu_short --time=00:20:00 --export='${common},MODE=analyze' '${analysis}' >/dev/null
safe_sbatch --lint-fatal --test-only --partition=cpu_short --time=00:20:00 --export='${common},MODE=verify' '${analysis}' >/dev/null"

remote "set -euo pipefail
singularity exec -B /scratch/lg154 -B /scratch/yz11502 \
  /share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif \
  env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH='${source_root}:${source_root}/MemNavData:${RUNTIME_CLOSURE_ROOT}:${RUNTIME_CLOSURE_ROOT}/MemNavData:${TASK_ROOT}:${TASK_ROOT}/MemNavData:${SERVER_SOURCE_ROOT}:${SERVER_SOURCE_ROOT}/MemNavData:${BASE_SOURCE_ROOT}:${BASE_SOURCE_ROOT}/MemNavData' \
  /scratch/lg154/conda-envs/memnav/bin/python - '${source_root}' '${RUNTIME_CLOSURE_ROOT}' <<'PY'
from pathlib import Path
import sys
from MemNavData import cec_handoff_contract as handoff
from MemNavData import certified_relocalization_runtime as certified
from MemNavData import monocular_depth_runtime as mono
from MemNavData import lingbot_pnp_localization as pnp
root=Path(sys.argv[1]).resolve(); closure=Path(sys.argv[2]).resolve()
assert Path(handoff.__file__).resolve()==root/'MemNavData/cec_handoff_contract.py'
assert Path(certified.__file__).resolve()==root/'MemNavData/certified_relocalization_runtime.py'
assert Path(mono.__file__).resolve()==root/'MemNavData/monocular_depth_runtime.py'
assert Path(pnp.__file__).resolve()==closure/'MemNavData/lingbot_pnp_localization.py'
assert 'default_authority_policy' in certified.runtime_contract()
for name in ('bind_monocular_depth_transaction','monocular_depth_transaction_token',
             'validate_monocular_depth_transaction'):
    assert callable(getattr(mono,name))
print('runtime_provenance_verified=true')
PY"

raw=$(remote "set -euo pipefail; test \"\$(id -un)\" = '${EXPECTED_SSH_USER}'; source '${safe}'; safe_sbatch --lint-fatal --parsable --qos=gpu48 --time=04:00:00 --array='${gate_index}' --dependency='afterok:${population_verify_job}' --kill-on-invalid-dep=yes --export='${common}' '${pair}'")
gate_job=$(printf '%s\n' "${raw}" | job_id)
[[ "${gate_job}" =~ ^[0-9]+$ ]] || fail "bad paired-query gate job id"
raw=$(remote "set -euo pipefail; test \"\$(id -un)\" = '${EXPECTED_SSH_USER}'; source '${safe}'; safe_sbatch --lint-fatal --parsable --qos=gpu48 --time=04:00:00 --array='${remaining_array}' --dependency='afterok:${gate_job}' --kill-on-invalid-dep=yes --export='${common}' '${pair}'")
remaining_job=$(printf '%s\n' "${raw}" | job_id)
[[ "${remaining_job}" =~ ^[0-9]+$ ]] || fail "bad paired-query remainder job id"
raw=$(remote "set -euo pipefail; test \"\$(id -un)\" = '${EXPECTED_SSH_USER}'; source '${safe}'; safe_sbatch --lint-fatal --parsable --partition=cpu_short --time=00:20:00 --dependency='afterok:${remaining_job}' --kill-on-invalid-dep=yes --export='${common},MODE=analyze' '${analysis}'")
analysis_job=$(printf '%s\n' "${raw}" | job_id)
[[ "${analysis_job}" =~ ^[0-9]+$ ]] || fail "bad analysis job id"
raw=$(remote "set -euo pipefail; test \"\$(id -un)\" = '${EXPECTED_SSH_USER}'; source '${safe}'; safe_sbatch --lint-fatal --parsable --partition=cpu_short --time=00:20:00 --dependency='afterok:${analysis_job}' --kill-on-invalid-dep=yes --export='${common},MODE=verify' '${analysis}'")
result_verify_job=$(printf '%s\n' "${raw}" | job_id)
[[ "${result_verify_job}" =~ ^[0-9]+$ ]] || fail "bad result-verifier job id"

"${LOCAL_MEMNAV_PY}" - "${OUT_RECEIPT}" "${gate_job}" "${remaining_job}" \
  "${gate_index}" "${gate_budget}" "${remaining_array}" "${analysis_job}" \
  "${result_verify_job}" "${run_root}" "${source_root}" "${receipt_sha}" \
  "${population_verify_job}" "${population_verification_sha}" \
  "${benchmark_manifest_sha}" "${construction_receipt_sha}" <<'PY'
import json,sys
(path,gate_job,remaining_job,gate_index,gate_budget,remaining_array,
 analysis,verify,run,source,source_sha,pop_job,pop_sha,
 manifest_sha,construction_sha)=sys.argv[1:]
p={
 'schema_version':'hm3d_table3_causal_survey_merged_query_submission_v1_20260831',
 'paired_query_gate_job':int(gate_job),
 'paired_query_remaining_array_job':int(remaining_job),
 'gate_population_index':int(gate_index),'gate_max_steps':int(gate_budget),
 'remaining_population_array':remaining_array,
 'formal_gate_retained_in_final_population':True,
 'analysis_job':int(analysis),
 'result_independent_verification_job':int(verify),'run_root':run,
 'source_bundle':source,'source_bundle_sha256':source_sha,
 'population_relative_root':'merged_query_population',
 'population_schema':'hm3d_table3_causal_survey_population_verification_v2_20260831',
 'population_independent_verification_job':int(pop_job),
 'population_independent_verification_sha256':pop_sha,
 'benchmark_manifest_sha256':manifest_sha,
 'expansion_construction_submission_receipt_sha256':construction_sha,
 'powered_histories':48,'formal_queries':96,'raw_arm_role_rows':192,
 'arms':['mono_native','mono_cec'],
 'length_bins':['0_to_20_m','20_to_30_m','30_to_50_m'],
 'history_source':'controlled_causal_rgb_geodesic_survey',
 'all_dependencies_afterok':True,'query_policy_outcomes_read_at_submission':False,
 'partial_results_allowed':False,'fallback_completion_allowed':False,
 'threshold_relaxation':False,'smoke_substitution':False,
 'base_candidates_deleted_or_replaced':False,
}
open(path,'x').write(json.dumps(p,indent=2,sort_keys=True)+'\n')
print(json.dumps(p,indent=2,sort_keys=True))
PY
