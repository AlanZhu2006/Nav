#!/usr/bin/env bash
# Submit an outcome-blind exact repair after the original Table-III A array.
set -euo pipefail
umask 0022

ROOT=${ROOT:-/home/asus/Research/Nav-graph-blind}
SSH_ALIAS=${SSH_ALIAS:-alantorch}
LOCAL_MEMNAV_PY=${LOCAL_MEMNAV_PY:-/home/asus/miniconda3/envs/memnav/bin/python}
LOCAL_HAB_PY=${LOCAL_HAB_PY:-/home/asus/miniconda3/envs/habitat/bin/python}
ORIGINAL_ARRAY_JOB=16596273
REPAIR_NAMESPACE=${REPAIR_NAMESPACE:-table3_a_directed_geodesic_repair_20260830}
SUBMISSION_RECEIPT=${SUBMISSION_RECEIPT:-MemNavData/HM3D_TABLE3_ACTUAL_MONO_A_DIRECTED_GEODESIC_REPAIR_SUBMISSION_20260830.json}
RUN_ROOT=/scratch/yz11502/Research/Nav-axis-uturn-results/hm3d_table3_actual_mono_execution_20260830/formal_20260830T080030Z_8ff97ca6
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
SSH_CONTROL_PATH=${SSH_CONTROL_PATH:-$(ssh -G "${SSH_ALIAS}" 2>/dev/null | awk '$1=="controlpath"{v=$2} END{print v}')}

cd "${ROOT}"
fail() { echo "ABORT: $*" >&2; exit 2; }
remote() { timeout 300 ssh -n -T -o BatchMode=yes -o ControlMaster=no -S "${SSH_CONTROL_PATH}" "${SSH_ALIAS}" "$@"; }
job_id() { tr -d '\r' | awk -F';' '/^[0-9]+(;|$)/ {print $1; exit}'; }
[[ -x "${LOCAL_MEMNAV_PY}" && -x "${LOCAL_HAB_PY}" && -S "${SSH_CONTROL_PATH}" ]] || fail "local prerequisite missing"
timeout 15 ssh -O check -S "${SSH_CONTROL_PATH}" "${SSH_ALIAS}" >/dev/null 2>&1 || fail "shared SSH unavailable"
files=(
  MemNavData/arrival_shadow.py
  MemNavData/audit_shared_online_double_revisit.py
  MemNavData/audit_shared_online_role_pairs.py
  MemNavData/bearing_diagnostics.py
  MemNavData/cec_authority_receipt.py
  MemNavData/cec_bearing_alignment.py
  MemNavData/cec_handoff_contract.py
  MemNavData/collect_hm3d_table3_actual_mono_a.py
  MemNavData/controller_portability_contract.py
  MemNavData/deterministic_eval_protocol.py
  MemNavData/eval_2leg_habitat.py
  MemNavData/final14_mono_factorial.py
  MemNavData/freeze_hm3d_table3_actual_mono_a_transport_repair.py
  MemNavData/generate_twoleg.py
  MemNavData/hm3d_fullmono_mixed_role.py
  MemNavData/hm3d_table3_actual_mono_execution_protocol_20260830.json
  MemNavData/hm3d_table3_directed_geodesic_repair_contract_20260830.json
  MemNavData/independent_verify_hm3d_table3_actual_mono_a_transport_repair.py
  MemNavData/materialize_online_a_traces.py
  MemNavData/mdtec_raw_depth_gate_d.py
  MemNavData/navdp_goal_switch.py
  MemNavData/revisit_action_shadow.py
  MemNavData/revisit_bearing_adapter.py
  MemNavData/run_hm3d_fullmono_server_scene.sh
  MemNavData/shared_online_double_revisit_runtime.py
  MemNavData/slurm_hm3d_table3_actual_mono_a_transport_repair.sbatch
  MemNavData/slurm_hm3d_table3_actual_mono_a_transport_repair_finish.sbatch
  MemNavData/slurm_hm3d_table3_actual_mono_a_transport_repair_launch.sbatch
  MemNavData/slurm_port_pair.sh
  MemNavData/terminal_uturn.py
  MemNavData/test_hm3d_table3_actual_mono_a_transport_repair.py
  MemNavData/visual_yaw_refinement.py
  MemNavData/xnavdp_revisit_contract.py
)
for path in "${files[@]}"; do [[ -f "${path}" && ! -L "${path}" ]] || fail "missing ${path}"; done
"${LOCAL_MEMNAV_PY}" -m pytest -q \
  MemNavData/test_hm3d_table3_actual_mono_a_transport_repair.py \
  MemNavData/test_hm3d_table3_actual_mono_collection.py \
  MemNavData/test_hm3d_table1_navdp_transport_contract.py
"${LOCAL_HAB_PY}" -m py_compile \
  MemNavData/collect_hm3d_table3_actual_mono_a.py \
  MemNavData/freeze_hm3d_table3_actual_mono_a_transport_repair.py \
  MemNavData/independent_verify_hm3d_table3_actual_mono_a_transport_repair.py
bash -n MemNavData/run_hm3d_fullmono_server_scene.sh \
  MemNavData/slurm_hm3d_table3_actual_mono_a_transport_repair.sbatch \
  MemNavData/slurm_hm3d_table3_actual_mono_a_transport_repair_finish.sbatch \
  MemNavData/slurm_hm3d_table3_actual_mono_a_transport_repair_launch.sbatch
scratch=$(mktemp -d /tmp/h3_table3_a_repair.XXXXXX)
trap 'rm -rf -- "${scratch}"' EXIT
mkdir -p "${scratch}/root"
for path in "${files[@]}"; do mkdir -p "${scratch}/root/$(dirname "${path}")"; cp -p "${path}" "${scratch}/root/${path}"; done
(cd "${scratch}/root" && find . -type f ! -name SOURCE_BUNDLE.sha256 -print0 | sort -z | xargs -0 sha256sum >SOURCE_BUNDLE.sha256 && sha256sum -c --quiet SOURCE_BUNDLE.sha256)
receipt_sha=$(sha256sum "${scratch}/root/SOURCE_BUNDLE.sha256" | awk '{print $1}')
wrapper_root=${REMOTE_BUNDLES}/hm3d_table3_actual_mono_a_transport_repair_${receipt_sha:0:16}
remote "set -euo pipefail
test \"\$(sha256sum '${TASK_RECEIPT}' | awk '{print \$1}')\" = '${EXPECTED_TASK_RECEIPT_SHA}'
test \"\$(sha256sum '${SERVER_SOURCE_RECEIPT}' | awk '{print \$1}')\" = '${EXPECTED_SERVER_SOURCE_RECEIPT_SHA}'
test \"\$(sha256sum '${BASE_RECEIPT}' | awk '{print \$1}')\" = '${EXPECTED_BASE_RECEIPT_SHA}'
test \"\$(sha256sum '${TABLE3_PLAN}' | awk '{print \$1}')\" = '${EXPECTED_TABLE3_PLAN_SHA}'
test -d '${RUN_ROOT}'"
if remote "test -d '${wrapper_root}'"; then
  remote "test \"\$(sha256sum '${wrapper_root}/SOURCE_BUNDLE.sha256' | awk '{print \$1}')\" = '${receipt_sha}'; cd '${wrapper_root}'; sha256sum -c --quiet SOURCE_BUNDLE.sha256"
else
  stage=${wrapper_root}.partial.$$
  remote "mkdir -p '${stage}'"
  rsync -a --chmod=Du=rwx,Dgo=rx,Fu=rw,Fgo=r -e "ssh -o BatchMode=yes -o ControlMaster=no -S ${SSH_CONTROL_PATH}" "${scratch}/root/" "${SSH_ALIAS}:${stage}/"
  remote "cd '${stage}'; sha256sum -c --quiet SOURCE_BUNDLE.sha256; chmod -R a-w '${stage}'; mv '${stage}' '${wrapper_root}'"
fi
execution=${wrapper_root}/MemNavData/hm3d_table3_actual_mono_execution_protocol_20260830.json
repair_contract=${wrapper_root}/MemNavData/hm3d_table3_directed_geodesic_repair_contract_20260830.json
repair_contract_sha=$(sha256sum MemNavData/hm3d_table3_directed_geodesic_repair_contract_20260830.json | awk '{print $1}')
common="ALL,WRAPPER_ROOT=${wrapper_root},WRAPPER_RECEIPT=${wrapper_root}/SOURCE_BUNDLE.sha256,EXPECTED_WRAPPER_RECEIPT_SHA=${receipt_sha},TASK_ROOT=${TASK_ROOT},TASK_RECEIPT=${TASK_RECEIPT},EXPECTED_TASK_RECEIPT_SHA=${EXPECTED_TASK_RECEIPT_SHA},SERVER_SOURCE_ROOT=${SERVER_SOURCE_ROOT},SERVER_SOURCE_RECEIPT=${SERVER_SOURCE_RECEIPT},EXPECTED_SERVER_SOURCE_RECEIPT_SHA=${EXPECTED_SERVER_SOURCE_RECEIPT_SHA},BASE_SOURCE_ROOT=${BASE_SOURCE_ROOT},BASE_RECEIPT=${BASE_RECEIPT},EXPECTED_BASE_RECEIPT_SHA=${EXPECTED_BASE_RECEIPT_SHA},RUN_ROOT=${RUN_ROOT},TABLE3_PLAN=${TABLE3_PLAN},EXPECTED_TABLE3_PLAN_SHA=${EXPECTED_TABLE3_PLAN_SHA},TABLE3_EXECUTION_PROTOCOL=${execution},REPAIR_NAMESPACE=${REPAIR_NAMESPACE},REPAIR_CONTRACT=${repair_contract},EXPECTED_REPAIR_CONTRACT_SHA=${repair_contract_sha}"
safe=${TASK_ROOT}/MemNavData/slurm_safe_submit.sh
launcher=${wrapper_root}/MemNavData/slurm_hm3d_table3_actual_mono_a_transport_repair_launch.sbatch
remote "source '${safe}'; safe_sbatch --lint-fatal --test-only --partition=cpu_short --dependency='afterany:${ORIGINAL_ARRAY_JOB}' --export='${common}' '${launcher}' >/dev/null"
raw=$(remote "source '${safe}'; safe_sbatch --lint-fatal --parsable --partition=cpu_short --dependency='afterany:${ORIGINAL_ARRAY_JOB}' --export='${common}' '${launcher}'")
job=$(printf '%s\n' "${raw}" | job_id)
[[ "${job}" =~ ^[0-9]+$ ]] || fail "bad repair launcher job id"
receipt=${SUBMISSION_RECEIPT}
[[ ! -e "${receipt}" ]] || fail "submission receipt exists"
"${LOCAL_MEMNAV_PY}" - "${receipt}" "${job}" "${wrapper_root}" \
  "${receipt_sha}" "${REPAIR_NAMESPACE}" "${repair_contract_sha}" <<'PY'
import json,sys
path,job,bundle,bundle_sha,namespace,contract_sha=sys.argv[1:]
p={'schema_version':'hm3d_table3_actual_mono_a_directed_geodesic_repair_launcher_submission_v1_20260830',
   'original_factual_A_array_job':16596273,'repair_launcher_job':int(job),
   'wrapper_bundle':bundle,'wrapper_bundle_sha256':bundle_sha,
   'repair_namespace':namespace,'repair_contract_sha256':contract_sha,
   'supersedes_cancelled_repair_launcher_job':16597086,
   'navigation_outcomes_read_at_submission':False,
   'query_policy_outcomes_read_at_submission':False,
   'scientific_thresholds_changed':False,'fallback_completion_allowed':False}
open(path,'x').write(json.dumps(p,indent=2,sort_keys=True)+'\n')
print(json.dumps(p,indent=2,sort_keys=True))
PY
