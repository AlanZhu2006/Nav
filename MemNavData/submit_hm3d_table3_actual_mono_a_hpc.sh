#!/usr/bin/env bash
# Submit all frozen Table-III factual Goal-A histories; no query policy runs.
set -euo pipefail
umask 0022

ROOT=${ROOT:-/home/asus/Research/Nav-graph-blind}
SSH_ALIAS=${SSH_ALIAS:-alantorch}
LOCAL_MEMNAV_PY=${LOCAL_MEMNAV_PY:-/home/asus/miniconda3/envs/memnav/bin/python}
LOCAL_HAB_PY=${LOCAL_HAB_PY:-/home/asus/miniconda3/envs/habitat/bin/python}
TASK_ROOT=/scratch/yz11502/Research/Nav-axis-uturn-source-bundles/hm3d_lifelong_natural_b_expansion_execution_1f4979a7fd37d467
TASK_RECEIPT=${TASK_ROOT}/SOURCE_BUNDLE.sha256
EXPECTED_TASK_RECEIPT_SHA=1f4979a7fd37d46700011558063be34a8fba0a0b8746668469dba7e7955f4282
SERVER_SOURCE_ROOT=/scratch/yz11502/Research/Nav-axis-uturn-source-bundles/hm3d_fullmono_lifelong_375f0b6879b2ff87
SERVER_SOURCE_RECEIPT=${SERVER_SOURCE_ROOT}/SOURCE_BUNDLE.sha256
EXPECTED_SERVER_SOURCE_RECEIPT_SHA=375f0b6879b2ff87b7019dae4727880d1b03fd3185a1862e6239942a76b5bcc8
BASE_SOURCE_ROOT=/scratch/yz11502/Research/Nav-axis-uturn-source-bundles/final14_mono_factorial_5690569a4373f2d2
BASE_RECEIPT=${BASE_SOURCE_ROOT}/source_inputs.sha256
EXPECTED_BASE_RECEIPT_SHA=5690569a4373f2d2768671418f0c604c4a03aa4b0ffe01baf70b288af03ba216
TABLE3_PLAN=/scratch/yz11502/Research/Nav-axis-uturn-results/hm3d_table3_actual_mono_20260830/plan_20260830T061943Z_3b848b5c/candidate_plan.json
EXPECTED_TABLE3_PLAN_SHA=1b1d16dd2132adb32565604bcf99f4852fa36df66a22bec2121e8338ce40020d
REMOTE_BUNDLES=/scratch/yz11502/Research/Nav-axis-uturn-source-bundles
REMOTE_RESULTS=/scratch/yz11502/Research/Nav-axis-uturn-results/hm3d_table3_actual_mono_execution_20260830
SSH_CONTROL_PATH=${SSH_CONTROL_PATH:-$(ssh -G "${SSH_ALIAS}" 2>/dev/null | awk '$1=="controlpath"{v=$2} END{print v}')}

cd "${ROOT}"
fail() { echo "ABORT: $*" >&2; exit 2; }
remote() { timeout 300 ssh -n -T -o BatchMode=yes -o ControlMaster=no -S "${SSH_CONTROL_PATH}" "${SSH_ALIAS}" "$@"; }
job_id() { tr -d '\r' | awk -F';' '/^[0-9]+(;|$)/ {print $1; exit}'; }
[[ -x "${LOCAL_MEMNAV_PY}" && -x "${LOCAL_HAB_PY}" && -S "${SSH_CONTROL_PATH}" ]] || fail "local prerequisite missing"
timeout 15 ssh -O check -S "${SSH_CONTROL_PATH}" "${SSH_ALIAS}" >/dev/null 2>&1 || fail "shared SSH unavailable"
files=(
  MemNavData/collect_hm3d_table3_actual_mono_a.py
  MemNavData/hm3d_table3_actual_mono_execution_protocol_20260830.json
  MemNavData/run_hm3d_fullmono_server_scene.sh
  MemNavData/slurm_hm3d_table3_actual_mono_a.sbatch
  MemNavData/test_hm3d_table3_actual_mono_collection.py
)
for path in "${files[@]}"; do [[ -f "${path}" && ! -L "${path}" ]] || fail "missing ${path}"; done
"${LOCAL_MEMNAV_PY}" -m pytest -q MemNavData/test_hm3d_table3_actual_mono_collection.py MemNavData/test_hm3d_table1_navdp_transport_contract.py
"${LOCAL_HAB_PY}" -m py_compile MemNavData/collect_hm3d_table3_actual_mono_a.py
bash -n MemNavData/run_hm3d_fullmono_server_scene.sh MemNavData/slurm_hm3d_table3_actual_mono_a.sbatch
python -m json.tool MemNavData/hm3d_table3_actual_mono_execution_protocol_20260830.json >/dev/null
scratch=$(mktemp -d /tmp/h3_table3_actual_a.XXXXXX)
trap 'rm -rf -- "${scratch}"' EXIT
mkdir -p "${scratch}/root"
for path in "${files[@]}"; do mkdir -p "${scratch}/root/$(dirname "${path}")"; cp -p "${path}" "${scratch}/root/${path}"; done
(cd "${scratch}/root" && find . -type f ! -name SOURCE_BUNDLE.sha256 -print0 | sort -z | xargs -0 sha256sum >SOURCE_BUNDLE.sha256 && sha256sum -c --quiet SOURCE_BUNDLE.sha256)
receipt_sha=$(sha256sum "${scratch}/root/SOURCE_BUNDLE.sha256" | awk '{print $1}')
wrapper_root=${REMOTE_BUNDLES}/hm3d_table3_actual_mono_a_${receipt_sha:0:16}
run_tag=formal_$(date -u +%Y%m%dT%H%M%SZ)_${receipt_sha:0:8}
run_root=${REMOTE_RESULTS}/${run_tag}
remote "set -euo pipefail
test \"\$(sha256sum '${TASK_RECEIPT}' | awk '{print \$1}')\" = '${EXPECTED_TASK_RECEIPT_SHA}'
test \"\$(sha256sum '${SERVER_SOURCE_RECEIPT}' | awk '{print \$1}')\" = '${EXPECTED_SERVER_SOURCE_RECEIPT_SHA}'
test \"\$(sha256sum '${BASE_RECEIPT}' | awk '{print \$1}')\" = '${EXPECTED_BASE_RECEIPT_SHA}'
test \"\$(sha256sum '${TABLE3_PLAN}' | awk '{print \$1}')\" = '${EXPECTED_TABLE3_PLAN_SHA}'
test ! -e '${run_root}'"
if remote "test -d '${wrapper_root}'"; then
  remote "test \"\$(sha256sum '${wrapper_root}/SOURCE_BUNDLE.sha256' | awk '{print \$1}')\" = '${receipt_sha}'; cd '${wrapper_root}'; sha256sum -c --quiet SOURCE_BUNDLE.sha256"
else
  stage=${wrapper_root}.partial.$$
  remote "mkdir -p '${stage}'"
  rsync -a --chmod=Du=rwx,Dgo=rx,Fu=rw,Fgo=r -e "ssh -o BatchMode=yes -o ControlMaster=no -S ${SSH_CONTROL_PATH}" "${scratch}/root/" "${SSH_ALIAS}:${stage}/"
  remote "cd '${stage}'; sha256sum -c --quiet SOURCE_BUNDLE.sha256; chmod -R a-w '${stage}'; mv '${stage}' '${wrapper_root}'"
fi
protocol=${wrapper_root}/MemNavData/hm3d_table3_actual_mono_execution_protocol_20260830.json
remote "mkdir -p '${run_root}/sealed_inputs' '${run_root}/factual_a' '${run_root}/carriers' /scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs; cp '${protocol}' '${run_root}/sealed_inputs/'; sha256sum '${TABLE3_PLAN}' '${protocol}' '${wrapper_root}/SOURCE_BUNDLE.sha256' >'${run_root}/sealed_inputs/source_inputs.sha256'; chmod -R a-w '${run_root}/sealed_inputs'"
common="ALL,WRAPPER_ROOT=${wrapper_root},WRAPPER_RECEIPT=${wrapper_root}/SOURCE_BUNDLE.sha256,EXPECTED_WRAPPER_RECEIPT_SHA=${receipt_sha},TASK_ROOT=${TASK_ROOT},TASK_RECEIPT=${TASK_RECEIPT},EXPECTED_TASK_RECEIPT_SHA=${EXPECTED_TASK_RECEIPT_SHA},SERVER_SOURCE_ROOT=${SERVER_SOURCE_ROOT},SERVER_SOURCE_RECEIPT=${SERVER_SOURCE_RECEIPT},EXPECTED_SERVER_SOURCE_RECEIPT_SHA=${EXPECTED_SERVER_SOURCE_RECEIPT_SHA},BASE_SOURCE_ROOT=${BASE_SOURCE_ROOT},BASE_RECEIPT=${BASE_RECEIPT},EXPECTED_BASE_RECEIPT_SHA=${EXPECTED_BASE_RECEIPT_SHA},RUN_ROOT=${run_root},TABLE3_PLAN=${TABLE3_PLAN},TABLE3_EXECUTION_PROTOCOL=${protocol}"
safe=${TASK_ROOT}/MemNavData/slurm_safe_submit.sh
sbatch=${wrapper_root}/MemNavData/slurm_hm3d_table3_actual_mono_a.sbatch
remote "source '${safe}'; safe_sbatch --lint-fatal --test-only --qos=gpu48 --time=02:00:00 --array=0 --export='${common}' '${sbatch}' >/dev/null"
remote "source '${safe}'; safe_sbatch --lint-fatal --test-only --qos=gpu48 --time=02:00:00 --array=1-124%4 --export='${common}' '${sbatch}' >/dev/null"
gate_raw=$(remote "source '${safe}'; safe_sbatch --lint-fatal --parsable --qos=gpu48 --time=02:00:00 --array=0 --export='${common}' '${sbatch}'")
gate_job=$(printf '%s\n' "${gate_raw}" | job_id)
[[ "${gate_job}" =~ ^[0-9]+$ ]] || fail "bad first formal candidate job id"
raw=$(remote "source '${safe}'; safe_sbatch --lint-fatal --parsable --qos=gpu48 --time=02:00:00 --array=1-124%4 --dependency='afterok:${gate_job}' --kill-on-invalid-dep=yes --export='${common}' '${sbatch}'")
job=$(printf '%s\n' "${raw}" | job_id)
[[ "${job}" =~ ^[0-9]+$ ]] || fail "bad remaining factual Goal-A job id"
receipt=MemNavData/HM3D_TABLE3_ACTUAL_MONO_A_SUBMISSION_20260830.json
[[ ! -e "${receipt}" ]] || fail "submission receipt exists"
"${LOCAL_MEMNAV_PY}" - "${receipt}" "${gate_job}" "${job}" "${run_root}" "${wrapper_root}" "${receipt_sha}" <<'PY'
import json,sys
path,gate,job,run,bundle,bundle_sha=sys.argv[1:]
p={'schema_version':'hm3d_table3_actual_mono_a_submission_v1_20260830',
 'first_formal_candidate_job':int(gate),'factual_A_remainder_array_job':int(job),
 'arrays':['0','1-124%4'],'time_limit':'02:00:00',
 'run_root':run,'wrapper_bundle':bundle,'wrapper_bundle_sha256':bundle_sha,
 'candidate_plan_sha256':'1b1d16dd2132adb32565604bcf99f4852fa36df66a22bec2121e8338ce40020d',
 'candidate_count':125,'all_frozen_reserves_submitted':True,
 'factual_A_outcomes_read_at_submission':False,
 'query_policy_outcomes_read_at_submission':False,
 'threshold_relaxation':False,'fallback_completion_allowed':False}
open(path,'x').write(json.dumps(p,indent=2,sort_keys=True)+'\n')
print(json.dumps(p,indent=2,sort_keys=True))
PY
