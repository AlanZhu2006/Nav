#!/usr/bin/env bash
# Stage and submit the dependency-held Table-III expansion/query handoff.
set -euo pipefail
umask 0022

ROOT=${ROOT:-/home/asus/Research/Nav-graph-blind}
SSH_ALIAS=${SSH_ALIAS:-alantorch}
EXPECTED_SSH_USER=${EXPECTED_SSH_USER:-yz11502}
LOCAL_MEMNAV_PY=${LOCAL_MEMNAV_PY:-/home/asus/miniconda3/envs/memnav/bin/python}
LOCAL_HAB_PY=${LOCAL_HAB_PY:-/home/asus/miniconda3/envs/habitat/bin/python}
PLAN_SUBMISSION=${PLAN_SUBMISSION:-MemNavData/HM3D_TABLE3_CAUSAL_SURVEY_EXPANSION_PLAN_SUBMISSION_20260831.json}
BASE_CONSTRUCTION_SUBMISSION=${BASE_CONSTRUCTION_SUBMISSION:-MemNavData/HM3D_TABLE3_CAUSAL_SURVEY_CONSTRUCTION_SUBMISSION_20260830.json}
OUT_RECEIPT=${OUT_RECEIPT:-MemNavData/HM3D_TABLE3_CAUSAL_SURVEY_DEFERRED_V2_SUBMISSION_20260831.json}
BASE_RUN_ROOT=/scratch/yz11502/Research/Nav-axis-uturn-results/hm3d_table3_causal_survey_20260830/formal_20260830T150042Z_3d811e0f
BASE_CANDIDATE_PLAN=/scratch/yz11502/Research/Nav-axis-uturn-results/hm3d_table3_actual_mono_20260830/plan_20260830T061943Z_3b848b5c/candidate_plan.json
BASE_SURVEY_PROTOCOL=/scratch/yz11502/Research/Nav-axis-uturn-source-bundles/hm3d_table3_causal_survey_3d811e0f63298480/MemNavData/hm3d_table3_causal_survey_protocol_20260830.json
BASE_WRAPPER_ROOT=/scratch/yz11502/Research/Nav-axis-uturn-source-bundles/hm3d_table3_actual_mono_downstream_2d8d08ff5a65da0a
BASE_WRAPPER_RECEIPT=${BASE_WRAPPER_ROOT}/SOURCE_BUNDLE.sha256
EXPECTED_BASE_WRAPPER_RECEIPT_SHA=2d8d08ff5a65da0ad00e7372fd756174b88958ed4cf74076e6605c951b38c3fe
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
remote_user=$(remote 'id -un' | tr -d '\r' | tail -n 1)
[[ "${remote_user}" == "${EXPECTED_SSH_USER}" ]] || \
  fail "shared SSH identity is ${remote_user:-unavailable}"
[[ ! -e "${OUT_RECEIPT}" ]] || fail "deferred submission receipt exists"

readarray -t frozen < <("${LOCAL_MEMNAV_PY}" - "${PLAN_SUBMISSION}" \
  "${BASE_CONSTRUCTION_SUBMISSION}" <<'PY'
import json,sys
p=json.load(open(sys.argv[1]))
assert p['schema_version']=='hm3d_table3_causal_survey_expansion_plan_submission_v1_20260831'
assert p['base_candidates_deleted_or_replaced'] is False
assert p['query_policy_jobs_submitted'] is False
assert p['query_policy_outcomes_read'] is False
assert p['navigation_policy_outcomes_read'] is False
assert p['threshold_relaxation'] is False and p['fallback_completion_allowed'] is False
print(p['expansion_plan_independent_verification_job'])
print(p['run_root']); print(p['source_bundle']); print(p['source_bundle_sha256'])
b=json.load(open(sys.argv[2]))
assert b['schema_version']=='hm3d_table3_causal_survey_construction_submission_v1_20260830'
assert b['frozen_candidates']==125 and b['query_policy_jobs_submitted'] is False
assert b['query_policy_outcomes_read_at_submission'] is False
assert b['threshold_relaxation'] is False and b['fallback_completion_allowed'] is False
print(b['population_independent_verification_job'])
PY
)
[[ "${#frozen[@]}" -eq 5 && "${frozen[0]}" =~ ^[0-9]+$ \
   && "${frozen[3]}" =~ ^[0-9a-f]{64}$ \
   && "${frozen[4]}" =~ ^[0-9]+$ ]] || fail "bad upstream submission receipts"
plan_verify_job=${frozen[0]}
run_root=${frozen[1]}
plan_source=${frozen[2]}
plan_source_sha=${frozen[3]}
base_population_verify_job=${frozen[4]}

files=(
  MemNavData/analyze_hm3d_table3_actual_mono.py
  MemNavData/analyze_hm3d_table3_causal_survey.py
  MemNavData/audit_hm3d_table3_length_role_pairs.py
  MemNavData/bundle_selftest.sh
  MemNavData/cec_handoff_contract.py
  MemNavData/certified_relocalization_runtime.py
  MemNavData/construct_hm3d_table3_causal_survey_role_pair.py
  MemNavData/controller_portability_contract.py
  MemNavData/eval_shared_online_role_pairs.py
  MemNavData/finalize_hm3d_table3_causal_survey_merged_population.py
  MemNavData/hm3d_table3_causal_survey_contract.py
  MemNavData/hm3d_table3_causal_survey_protocol_20260830.json
  MemNavData/hm3d_table3_length_contract.py
  MemNavData/independent_verify_hm3d_table3_actual_mono_result.py
  MemNavData/independent_verify_hm3d_table3_causal_survey_merged_population.py
  MemNavData/independent_verify_hm3d_table3_causal_survey_result.py
  MemNavData/monocular_depth_runtime.py
  MemNavData/run_hm3d_fullmono_query_history.py
  MemNavData/run_hm3d_fullmono_server_scene.sh
  MemNavData/slurm_hm3d_table3_causal_survey_analysis.sbatch
  MemNavData/slurm_hm3d_table3_causal_survey_deferred.sbatch
  MemNavData/slurm_hm3d_table3_causal_survey_expansion_construct.sbatch
  MemNavData/slurm_hm3d_table3_causal_survey_merged_population.sbatch
  MemNavData/slurm_hm3d_table3_causal_survey_pair.sbatch
  MemNavData/slurm_port_pair.sh
  MemNavData/test_audit_hm3d_table3_actual_mono_constructibility.py
  MemNavData/test_cec_handoff_contract.py
  MemNavData/test_certified_relocalization_runtime.py
  MemNavData/test_hm3d_table3_causal_survey_expansion.py
  MemNavData/test_hm3d_table3_causal_survey_submission.py
  MemNavData/test_hm3d_table3_length_contract.py
  MemNavData/test_monocular_depth_runtime.py
)
for path in "${files[@]}"; do
  [[ -f "${path}" && ! -L "${path}" ]] || fail "missing ${path}"
done
"${LOCAL_MEMNAV_PY}" -m pytest -q \
  MemNavData/test_hm3d_table3_causal_survey_expansion.py \
  MemNavData/test_hm3d_table3_causal_survey_submission.py \
  MemNavData/test_hm3d_table3_length_contract.py \
  MemNavData/test_audit_hm3d_table3_actual_mono_constructibility.py \
  MemNavData/test_cec_handoff_contract.py \
  MemNavData/test_certified_relocalization_runtime.py \
  MemNavData/test_monocular_depth_runtime.py
"${LOCAL_HAB_PY}" -m py_compile \
  MemNavData/construct_hm3d_table3_causal_survey_role_pair.py \
  MemNavData/eval_shared_online_role_pairs.py \
  MemNavData/run_hm3d_fullmono_query_history.py
"${LOCAL_MEMNAV_PY}" -m py_compile \
  MemNavData/analyze_hm3d_table3_causal_survey.py \
  MemNavData/finalize_hm3d_table3_causal_survey_merged_population.py \
  MemNavData/independent_verify_hm3d_table3_causal_survey_merged_population.py \
  MemNavData/independent_verify_hm3d_table3_causal_survey_result.py
"${LOCAL_MEMNAV_PY}" -m py_compile \
  MemNavData/cec_handoff_contract.py \
  MemNavData/certified_relocalization_runtime.py \
  MemNavData/controller_portability_contract.py \
  MemNavData/monocular_depth_runtime.py
bash -n MemNavData/run_hm3d_fullmono_server_scene.sh \
  MemNavData/slurm_hm3d_table3_causal_survey_analysis.sbatch \
  MemNavData/slurm_hm3d_table3_causal_survey_deferred.sbatch \
  MemNavData/slurm_hm3d_table3_causal_survey_expansion_construct.sbatch \
  MemNavData/slurm_hm3d_table3_causal_survey_merged_population.sbatch \
  MemNavData/slurm_hm3d_table3_causal_survey_pair.sbatch "$0"

scratch=$(mktemp -d /tmp/h3_table3_deferred.XXXXXX)
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
    "${LOCAL_MEMNAV_PY}" MemNavData.independent_verify_hm3d_table3_causal_survey_merged_population \
    "${LOCAL_MEMNAV_PY}" MemNavData.analyze_hm3d_table3_causal_survey \
    "${LOCAL_MEMNAV_PY}" MemNavData.independent_verify_hm3d_table3_causal_survey_result)
(cd "${scratch}/root" && \
  find . -type f ! -name SOURCE_BUNDLE.sha256 -print0 | sort -z | \
  xargs -0 sha256sum >SOURCE_BUNDLE.sha256 && \
  sha256sum -c --quiet SOURCE_BUNDLE.sha256)
receipt_sha=$(sha256sum "${scratch}/root/SOURCE_BUNDLE.sha256" | awk '{print $1}')
source_root=${REMOTE_BUNDLES}/hm3d_table3_causal_survey_deferred_${receipt_sha:0:16}
source_receipt=${source_root}/SOURCE_BUNDLE.sha256

remote "set -euo pipefail
test \"\$(id -un)\" = '${EXPECTED_SSH_USER}'
test \"\$(sha256sum '${plan_source}/SOURCE_BUNDLE.sha256' | awk '{print \$1}')\" = '${plan_source_sha}'
cd '${plan_source}'; sha256sum -c --quiet SOURCE_BUNDLE.sha256
for spec in \
  '${TASK_RECEIPT}:${EXPECTED_TASK_RECEIPT_SHA}:${TASK_ROOT}' \
  '${BASE_WRAPPER_RECEIPT}:${EXPECTED_BASE_WRAPPER_RECEIPT_SHA}:${BASE_WRAPPER_ROOT}' \
  '${SERVER_SOURCE_RECEIPT}:${EXPECTED_SERVER_SOURCE_RECEIPT_SHA}:${SERVER_SOURCE_ROOT}' \
  '${BASE_RECEIPT}:${EXPECTED_BASE_RECEIPT_SHA}:${BASE_SOURCE_ROOT}' \
  '${RUNTIME_CLOSURE_RECEIPT}:${EXPECTED_RUNTIME_CLOSURE_RECEIPT_SHA}:${RUNTIME_CLOSURE_ROOT}'; do
  IFS=: read -r receipt expected root <<<\"\${spec}\"
  test \"\$(sha256sum \"\${receipt}\" | awk '{print \$1}')\" = \"\${expected}\"
  cd \"\${root}\"; sha256sum -c --quiet \"\${receipt}\"
done"

if remote "test -d '${source_root}'"; then
  remote "set -euo pipefail
test \"\$(id -un)\" = '${EXPECTED_SSH_USER}'
test \"\$(sha256sum '${source_receipt}' | awk '{print \$1}')\" = '${receipt_sha}'
cd '${source_root}'; sha256sum -c --quiet SOURCE_BUNDLE.sha256"
else
  stage=${source_root}.partial.$$
  remote "set -euo pipefail; mkdir -p '${stage}'"
  rsync -a --chmod=Du=rwx,Dgo=rx,Fu=rw,Fgo=r \
    -e "ssh -o BatchMode=yes -o ControlMaster=no -S ${SSH_CONTROL_PATH}" \
    "${scratch}/root/" "${SSH_ALIAS}:${stage}/"
  remote "set -euo pipefail
test \"\$(id -un)\" = '${EXPECTED_SSH_USER}'
cd '${stage}'; sha256sum -c --quiet SOURCE_BUNDLE.sha256
SELFTEST_BUNDLE_SUBPATHS=MemNavData bash '${stage}/MemNavData/bundle_selftest.sh' '${stage}' <(printf '%s import %s\\n' \
  /scratch/lg154/conda-envs/memnav/bin/python MemNavData.finalize_hm3d_table3_causal_survey_merged_population \
  /scratch/lg154/conda-envs/memnav/bin/python MemNavData.independent_verify_hm3d_table3_causal_survey_merged_population \
  /scratch/lg154/conda-envs/memnav/bin/python MemNavData.analyze_hm3d_table3_causal_survey \
  /scratch/lg154/conda-envs/memnav/bin/python MemNavData.independent_verify_hm3d_table3_causal_survey_result)
chmod -R a-w '${stage}'; mv '${stage}' '${source_root}'"
fi

remote "set -euo pipefail
singularity exec -B /scratch/lg154 -B /scratch/yz11502 \
  /share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif \
  env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH='${source_root}:${source_root}/MemNavData:${BASE_WRAPPER_ROOT}:${BASE_WRAPPER_ROOT}/MemNavData' \
  /scratch/lg154/conda-envs/habitat/bin/python \
  '${source_root}/MemNavData/construct_hm3d_table3_causal_survey_role_pair.py' --help >/dev/null
singularity exec -B /scratch/lg154 -B /scratch/yz11502 \
  /share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif \
  env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH='${source_root}:${source_root}/MemNavData:${RUNTIME_CLOSURE_ROOT}:${RUNTIME_CLOSURE_ROOT}/MemNavData:${TASK_ROOT}:${TASK_ROOT}/MemNavData:${SERVER_SOURCE_ROOT}:${SERVER_SOURCE_ROOT}/MemNavData:${BASE_SOURCE_ROOT}:${BASE_SOURCE_ROOT}/MemNavData:/scratch/yz11502/Research/Nav-axis-uturn/InternNav/src/diffusion-policy:/scratch/lg154/conda-envs/habitat/lib/python3.9/site-packages/pip/_vendor' \
  /scratch/lg154/conda-envs/habitat/bin/python \
  '${source_root}/MemNavData/eval_shared_online_role_pairs.py' --help >/dev/null"

remote "singularity exec -B /scratch/lg154 -B /scratch/yz11502 \
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

deferred=${source_root}/MemNavData/slurm_hm3d_table3_causal_survey_deferred.sbatch
common="ALL,DEFERRED_MODE=expand,CHAIN_SOURCE_ROOT=${source_root},CHAIN_SOURCE_RECEIPT=${source_receipt},EXPECTED_CHAIN_SOURCE_RECEIPT_SHA=${receipt_sha},EXPECTED_REMOTE_USER=${EXPECTED_SSH_USER},EXPANSION_RUN_ROOT=${run_root},BASE_RUN_ROOT=${BASE_RUN_ROOT},BASE_CANDIDATE_PLAN=${BASE_CANDIDATE_PLAN},BASE_SURVEY_PROTOCOL=${BASE_SURVEY_PROTOCOL},BASE_POPULATION_VERIFY_JOB=${base_population_verify_job},BASE_WRAPPER_ROOT=${BASE_WRAPPER_ROOT},BASE_WRAPPER_RECEIPT=${BASE_WRAPPER_RECEIPT},EXPECTED_BASE_WRAPPER_RECEIPT_SHA=${EXPECTED_BASE_WRAPPER_RECEIPT_SHA},TASK_ROOT=${TASK_ROOT},TASK_RECEIPT=${TASK_RECEIPT},EXPECTED_TASK_RECEIPT_SHA=${EXPECTED_TASK_RECEIPT_SHA},SERVER_SOURCE_ROOT=${SERVER_SOURCE_ROOT},SERVER_SOURCE_RECEIPT=${SERVER_SOURCE_RECEIPT},EXPECTED_SERVER_SOURCE_RECEIPT_SHA=${EXPECTED_SERVER_SOURCE_RECEIPT_SHA},BASE_SOURCE_ROOT=${BASE_SOURCE_ROOT},BASE_RECEIPT=${BASE_RECEIPT},EXPECTED_BASE_RECEIPT_SHA=${EXPECTED_BASE_RECEIPT_SHA},RUNTIME_CLOSURE_ROOT=${RUNTIME_CLOSURE_ROOT},RUNTIME_CLOSURE_RECEIPT=${RUNTIME_CLOSURE_RECEIPT},EXPECTED_RUNTIME_CLOSURE_RECEIPT_SHA=${EXPECTED_RUNTIME_CLOSURE_RECEIPT_SHA}"
safe=${TASK_ROOT}/MemNavData/slurm_safe_submit.sh
remote "source '${safe}'; safe_sbatch --lint-fatal --test-only --partition=cpu_short --time=00:20:00 --export='${common}' '${deferred}' >/dev/null"
raw=$(remote "source '${safe}'; safe_sbatch --lint-fatal --parsable --partition=cpu_short --time=00:20:00 --dependency='afterok:${plan_verify_job}' --kill-on-invalid-dep=yes --export='${common}' '${deferred}'")
launcher=$(printf '%s\n' "${raw}" | job_id)
[[ "${launcher}" =~ ^[0-9]+$ ]] || fail "bad deferred launcher job id"

"${LOCAL_MEMNAV_PY}" - "${OUT_RECEIPT}" "${launcher}" "${plan_verify_job}" \
  "${base_population_verify_job}" "${run_root}" "${source_root}" \
  "${receipt_sha}" <<'PY'
import json,sys
path,launcher,plan_verify,base_verify,run,source,source_sha=sys.argv[1:]
p={'schema_version':'hm3d_table3_causal_survey_deferred_submission_v1_20260831',
   'deferred_expansion_launcher_job':int(launcher),
   'dependency_expansion_plan_independent_verification_job':int(plan_verify),
   'base_population_independent_verification_job':int(base_verify),
   'run_root':run,'source_bundle':source,'source_bundle_sha256':source_sha,
   'downstream_target':{'histories':48,'queries':96,'raw_arm_role_rows':192},
   'history_source':'controlled_causal_rgb_geodesic_survey',
   'query_policy_jobs_submitted':False,'query_policy_outcomes_read':False,
   'base_candidates_deleted_or_replaced':False,'threshold_relaxation':False,
   'fallback_completion_allowed':False,'smoke_substitution':False}
open(path,'x').write(json.dumps(p,indent=2,sort_keys=True)+'\n')
print(json.dumps(p,indent=2,sort_keys=True))
PY
