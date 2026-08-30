#!/usr/bin/env bash
set -euo pipefail
umask 0022

ROOT=${ROOT:-/home/asus/Research/Nav-graph-blind}
SSH_ALIAS=${SSH_ALIAS:-alantorch}
OLD_TASK_ROOT=/scratch/yz11502/Research/Nav-axis-uturn-source-bundles/hm3d_lifelong_natural_b_expansion_execution_1f4979a7fd37d467
OLD_TASK_RECEIPT_SHA=1f4979a7fd37d46700011558063be34a8fba0a0b8746668469dba7e7955f4282
RUN_ROOT=/scratch/yz11502/Research/Nav-axis-uturn-results/hm3d_fullmono_lifelong_natural_b_expansion_execution_20260830/formal_20260830T045416Z_1f4979a7
PARENT_ROOT=/scratch/yz11502/Research/Nav-axis-uturn-results/hm3d_fresh_fullmono_mixed_role_20260820/formal_20260820T143609Z_e6dd44c6
BASE_SOURCE_ROOT=/scratch/yz11502/Research/Nav-axis-uturn-source-bundles/final14_mono_factorial_5690569a4373f2d2
BASE_RECEIPT=${BASE_SOURCE_ROOT}/source_inputs.sha256
EXPECTED_BASE_RECEIPT_SHA=5690569a4373f2d2768671418f0c604c4a03aa4b0ffe01baf70b288af03ba216
SERVER_SOURCE_ROOT=/scratch/yz11502/Research/Nav-axis-uturn-source-bundles/hm3d_fullmono_lifelong_375f0b6879b2ff87
SERVER_SOURCE_RECEIPT=${SERVER_SOURCE_ROOT}/SOURCE_BUNDLE.sha256
EXPECTED_SERVER_SOURCE_RECEIPT_SHA=375f0b6879b2ff87b7019dae4727880d1b03fd3185a1862e6239942a76b5bcc8
TABLE2_SERVER_SOURCE_ROOT=/scratch/yz11502/Research/Nav-axis-uturn-source-bundles/hm3d_table1_navdp_authority_transaction_718661db1733d5de
TABLE2_SERVER_SOURCE_RECEIPT=${TABLE2_SERVER_SOURCE_ROOT}/SOURCE_BUNDLE.sha256
EXPECTED_TABLE2_SERVER_SOURCE_RECEIPT_SHA=718661db1733d5de16cd86687eec880a8d02fc5ae5ca982e1ab7d5bde5e96f7d
PROTOCOL_REL=MemNavData/hm3d_fullmono_lifelong_natural_b_expansion_execution_protocol_20260830.json
EXPECTED_PROTOCOL_SHA=28101fe2574e9ea428306dbf12932cb3da7cf3c24d88f82f573c1cb3209d9edd
PARENT_LAUNCH_JOB=16591458
MATERIALIZATION_VERIFY_JOB=16591452
REMOTE_BUNDLES=/scratch/yz11502/Research/Nav-axis-uturn-source-bundles
SSH_CONTROL_PATH=${SSH_CONTROL_PATH:-$(ssh -G "${SSH_ALIAS}" 2>/dev/null | awk '$1=="controlpath"{v=$2} END{print v}')}

cd "${ROOT}"
fail() { echo "ABORT: $*" >&2; exit 2; }
remote() {
  timeout 300 ssh -n -T -o BatchMode=yes -o ControlMaster=no \
    -S "${SSH_CONTROL_PATH}" "${SSH_ALIAS}" "$@"
}
[[ -S "${SSH_CONTROL_PATH}" ]] || fail "shared SSH socket missing"
bash -n MemNavData/slurm_hm3d_fullmono_lifelong_natural_b_expansion_launch_b.sbatch
python -m json.tool \
  MemNavData/hm3d_natural_b_expansion_launcher_shard_repair_20260830.json >/dev/null
source MemNavData/slurm_safe_submit.sh
lint_sbatch_template \
  MemNavData/slurm_hm3d_fullmono_lifelong_natural_b_expansion_launch_b.sbatch

preflight=$(remote "set -euo pipefail
test \"\$(sha256sum '${OLD_TASK_ROOT}/SOURCE_BUNDLE.sha256' | awk '{print \$1}')\" = '${OLD_TASK_RECEIPT_SHA}'
state=\$(sacct -j '${PARENT_LAUNCH_JOB}' -X -n -o State | awk 'NF{print \$1;exit}')
elapsed=\$(sacct -j '${PARENT_LAUNCH_JOB}' -X -n -o ElapsedRaw | awk 'NF{print \$1;exit}')
test \"\${state}\" = CANCELLED+ -o \"\${state}\" = CANCELLED
test \"\${elapsed}\" = 0
test ! -e '${RUN_ROOT}/factual_b'
test ! -e '${RUN_ROOT}/factual_b_schedule'
test ! -e '${RUN_ROOT}/prefix_fragments'
test ! -e '${RUN_ROOT}/population'
echo PREFLIGHT_OK")
[[ "${preflight}" == *PREFLIGHT_OK* ]] || fail "repair preflight failed"

stage=${REMOTE_BUNDLES}/hm3d_lifelong_natbx_launcher_repair.partial.$$
remote "test ! -e '${stage}'; cp -a '${OLD_TASK_ROOT}' '${stage}'; chmod -R u+w '${stage}'; rm '${stage}/SOURCE_BUNDLE.sha256'"
timeout 180 rsync -a --chmod=Fu=rw,Fgo=r \
  -e "ssh -o BatchMode=yes -o ControlMaster=no -S ${SSH_CONTROL_PATH}" \
  MemNavData/slurm_hm3d_fullmono_lifelong_natural_b_expansion_launch_b.sbatch \
  MemNavData/hm3d_natural_b_expansion_launcher_shard_repair_20260830.json \
  "${SSH_ALIAS}:${stage}/MemNavData/"
receipt_sha=$(remote "set -euo pipefail
cd '${stage}'
find . -type f ! -name SOURCE_BUNDLE.sha256 -print0 | sort -z | xargs -0 sha256sum >SOURCE_BUNDLE.sha256
sha256sum -c --quiet SOURCE_BUNDLE.sha256
sha256sum SOURCE_BUNDLE.sha256 | awk '{print \$1}'" | tr -d '\r')
[[ "${receipt_sha}" =~ ^[0-9a-f]{64}$ ]] || fail "bad repair receipt"
task_root=${REMOTE_BUNDLES}/hm3d_lifelong_natbx_launcher_repair_${receipt_sha:0:16}
remote "test ! -e '${task_root}'; chmod -R a-w '${stage}'; mv '${stage}' '${task_root}'"
task_receipt=${task_root}/SOURCE_BUNDLE.sha256
protocol=${task_root}/${PROTOCOL_REL}
common="ALL,TASK_ROOT=${task_root},TASK_RECEIPT=${task_receipt},EXPECTED_TASK_RECEIPT_SHA=${receipt_sha},SERVER_SOURCE_ROOT=${SERVER_SOURCE_ROOT},SERVER_SOURCE_RECEIPT=${SERVER_SOURCE_RECEIPT},EXPECTED_SERVER_SOURCE_RECEIPT_SHA=${EXPECTED_SERVER_SOURCE_RECEIPT_SHA},TABLE2_SERVER_SOURCE_ROOT=${TABLE2_SERVER_SOURCE_ROOT},TABLE2_SERVER_SOURCE_RECEIPT=${TABLE2_SERVER_SOURCE_RECEIPT},EXPECTED_TABLE2_SERVER_SOURCE_RECEIPT_SHA=${EXPECTED_TABLE2_SERVER_SOURCE_RECEIPT_SHA},BASE_SOURCE_ROOT=${BASE_SOURCE_ROOT},BASE_RECEIPT=${BASE_RECEIPT},EXPECTED_BASE_RECEIPT_SHA=${EXPECTED_BASE_RECEIPT_SHA},RUN_ROOT=${RUN_ROOT},PARENT_ROOT=${PARENT_ROOT},PROTOCOL=${protocol},EXPECTED_PROTOCOL_SHA=${EXPECTED_PROTOCOL_SHA}"
launch=${task_root}/MemNavData/slurm_hm3d_fullmono_lifelong_natural_b_expansion_launch_b.sbatch
safe=${task_root}/MemNavData/slurm_safe_submit.sh
remote "source '${safe}'; safe_sbatch --lint-fatal --test-only --partition=cpu_short --export='${common}' '${launch}' >/dev/null"
raw=$(remote "source '${safe}'; safe_sbatch --lint-fatal --parsable --partition=cpu_short --dependency='afterok:${MATERIALIZATION_VERIFY_JOB}' --kill-on-invalid-dep=yes --export='${common}' '${launch}'")
job=$(printf '%s\n' "${raw}" | tr -d '\r' | awk -F';' '/^[0-9]+(;|$)/{print $1;exit}')
[[ "${job}" =~ ^[0-9]+$ ]] || fail "bad replacement job"

receipt=MemNavData/HM3D_NATURAL_B_EXPANSION_LAUNCHER_REPAIR_SUBMISSION_20260830.json
[[ ! -e "${receipt}" ]] || fail "local repair receipt exists"
python - "${receipt}" "${task_root}" "${receipt_sha}" "${job}" <<'PY'
import json,sys
path,root,sha,job=sys.argv[1:]
p={'schema_version':'hm3d_natural_b_expansion_launcher_repair_submission_v1_20260830',
   'replaces_cancelled_zero_elapsed_job':16591458,
   'depends_on_materialization_verifier':16591452,
   'replacement_launcher_job':int(job),'task_bundle':root,
   'task_receipt_sha256':sha,'factual_B_outputs_before_repair':0,
   'navigation_outcomes_read':False,'fallback_completion_allowed':False}
open(path,'x').write(json.dumps(p,indent=2,sort_keys=True)+'\n')
print(json.dumps(p,indent=2,sort_keys=True))
PY
timeout 180 rsync -a --chmod=Fugo=r \
  -e "ssh -o BatchMode=yes -o ControlMaster=no -S ${SSH_CONTROL_PATH}" \
  "${receipt}" "${SSH_ALIAS}:${RUN_ROOT}/launcher_repair_submission.json"
remote "cd '${RUN_ROOT}'; sha256sum launcher_repair_submission.json >launcher_repair_submission.json.sha256; chmod a-w launcher_repair_submission.json launcher_repair_submission.json.sha256"
