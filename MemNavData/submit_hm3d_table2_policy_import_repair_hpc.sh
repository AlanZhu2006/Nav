#!/usr/bin/env bash
# Exact infrastructure repair for the Table-II policy chain after the first
# A100 smoke exposed an immutable-overlay import omission.  The scientific
# population and policy contract are unchanged; a fresh run root prevents any
# partial smoke artifact from entering the formal evaluation.
set -euo pipefail
umask 0022

ROOT=${ROOT:-/home/asus/Research/Nav-graph-blind}
SSH_ALIAS=${SSH_ALIAS:-alantorch}
LOCAL_PY=${LOCAL_PY:-/home/asus/miniconda3/envs/memnav/bin/python}
REMOTE_BUNDLES=/scratch/yz11502/Research/Nav-axis-uturn-source-bundles
TASK_ROOT=/scratch/yz11502/Research/Nav-axis-uturn-source-bundles/hm3d_lifelong_natural_b_expansion_execution_1f4979a7fd37d467
TASK_RECEIPT=${TASK_ROOT}/SOURCE_BUNDLE.sha256
EXPECTED_TASK_RECEIPT_SHA=1f4979a7fd37d46700011558063be34a8fba0a0b8746668469dba7e7955f4282
BASE_SOURCE_ROOT=/scratch/yz11502/Research/Nav-axis-uturn-source-bundles/final14_mono_factorial_5690569a4373f2d2
BASE_RECEIPT=${BASE_SOURCE_ROOT}/source_inputs.sha256
EXPECTED_BASE_RECEIPT_SHA=5690569a4373f2d2768671418f0c604c4a03aa4b0ffe01baf70b288af03ba216
SERVER_SOURCE_ROOT=/scratch/yz11502/Research/Nav-axis-uturn-source-bundles/hm3d_table1_navdp_authority_transaction_718661db1733d5de
SERVER_SOURCE_RECEIPT=${SERVER_SOURCE_ROOT}/SOURCE_BUNDLE.sha256
EXPECTED_SERVER_SOURCE_RECEIPT_SHA=718661db1733d5de16cd86687eec880a8d02fc5ae5ca982e1ab7d5bde5e96f7d
SOURCE_RUN_ROOT=/scratch/yz11502/Research/Nav-axis-uturn-results/hm3d_fullmono_lifelong_natural_b_expansion_execution_20260830/formal_20260830T045416Z_1f4979a7/table2_source_union
TABLE2_RUN_ROOT=/scratch/yz11502/Research/Nav-axis-uturn-results/hm3d_fullmono_lifelong_natural_b_expansion_execution_20260830/formal_20260830T045416Z_1f4979a7/table2_leg3_power
PROTOCOL=${SOURCE_RUN_ROOT}/hm3d_table2_leg3_power_protocol.json
EXPECTED_PROTOCOL_SHA=28352498de740add233b783caad79ac2665f13313ec49912a72ec1db5a6a69b0
PARENT_MANIFEST=/scratch/yz11502/Research/Nav-axis-uturn-results/hm3d_fresh_fullmono_mixed_role_20260820/formal_20260820T143609Z_e6dd44c6/sealed_inputs/parent_manifest.json
CONSTRUCTION_VERIFY_JOB=${CONSTRUCTION_VERIFY_JOB:-16599079}
PREVIOUS_FAILED_SMOKE_JOB=${PREVIOUS_FAILED_SMOKE_JOB:-16601072}
POLICY_RUN_NAME=${POLICY_RUN_NAME:-policy_import_repair_v1}
OUT_RECEIPT=${OUT_RECEIPT:-MemNavData/HM3D_TABLE2_POLICY_IMPORT_REPAIR_SUBMISSION_20260830.json}
SSH_CONTROL_PATH=${SSH_CONTROL_PATH:-$(ssh -G "${SSH_ALIAS}" 2>/dev/null | awk '$1=="controlpath"{v=$2} END{print v}')}

cd "${ROOT}"
fail() { echo "ABORT: $*" >&2; exit 2; }
remote() {
  timeout 300 ssh -n -T -o BatchMode=yes -o ControlMaster=no \
    -S "${SSH_CONTROL_PATH}" "${SSH_ALIAS}" "$@"
}
job_id() { tr -d '\r' | awk -F';' '/^[0-9]+(;|$)/ {print $1; exit}'; }

[[ -x "${LOCAL_PY}" && -S "${SSH_CONTROL_PATH}" ]] || \
  fail "local Python or authoritative shared SSH master missing"
timeout 15 ssh -O check -S "${SSH_CONTROL_PATH}" "${SSH_ALIAS}" \
  >/dev/null 2>&1 || fail "shared SSH unavailable"
[[ ! -e "${OUT_RECEIPT}" ]] || fail "submission receipt already exists"
[[ "${POLICY_RUN_NAME}" =~ ^policy[A-Za-z0-9_.-]*$ ]] || fail "bad run name"

files=(
  MemNavData/slurm_hm3d_table2_leg3_power_policy_launch.sbatch
  MemNavData/slurm_hm3d_table2_leg3_navdp_pair.sbatch
  MemNavData/slurm_hm3d_table2_meeting_result.sbatch
  MemNavData/independent_verify_hm3d_table2_meeting_result.py
  MemNavData/run_hm3d_fullmono_server_scene.sh
  MemNavData/slurm_port_pair.sh
  MemNavData/cec_handoff_contract.py
  MemNavData/controller_portability_contract.py
  MemNavData/submit_hm3d_table2_policy_import_repair_hpc.sh
)
for path in "${files[@]}"; do
  [[ -f "${path}" && ! -L "${path}" ]] || fail "missing physical ${path}"
done
bash -n \
  MemNavData/slurm_hm3d_table2_leg3_power_policy_launch.sbatch \
  MemNavData/slurm_hm3d_table2_leg3_navdp_pair.sbatch \
  MemNavData/slurm_hm3d_table2_meeting_result.sbatch \
  MemNavData/run_hm3d_fullmono_server_scene.sh \
  MemNavData/slurm_port_pair.sh \
  MemNavData/submit_hm3d_table2_policy_import_repair_hpc.sh
"${LOCAL_PY}" -m py_compile \
  MemNavData/cec_handoff_contract.py \
  MemNavData/controller_portability_contract.py \
  MemNavData/independent_verify_hm3d_table2_meeting_result.py
"${LOCAL_PY}" -c \
  'from MemNavData.cec_handoff_contract import verify_handoff_packet_envelope'

scratch=$(mktemp -d /tmp/h3_table2_policy_import_repair.XXXXXX)
cleanup() { rm -rf -- "${scratch}"; }
trap cleanup EXIT
mkdir -p "${scratch}/root/MemNavData"
for path in "${files[@]}"; do cp -p "${path}" "${scratch}/root/${path}"; done
(
  cd "${scratch}/root"
  find . -type f ! -name SOURCE_BUNDLE.sha256 -print0 | sort -z | \
    xargs -0 sha256sum >SOURCE_BUNDLE.sha256
  sha256sum -c --quiet SOURCE_BUNDLE.sha256
)
wrapper_sha=$(sha256sum "${scratch}/root/SOURCE_BUNDLE.sha256" | awk '{print $1}')
wrapper_root=${REMOTE_BUNDLES}/hm3d_table2_policy_import_repair_${wrapper_sha:0:16}
construction=${TABLE2_RUN_ROOT}/hm3d_table2_leg3_construction_verification.json
new_policy_root=${TABLE2_RUN_ROOT}/${POLICY_RUN_NAME}
old_formal=${TABLE2_RUN_ROOT}/policy/formal/navdp

remote_identity=$(remote 'id -un' | tr -d '\r')
[[ "${remote_identity}" == yz11502 ]] || fail "wrong remote identity"
remote "set -euo pipefail
test \"\$(sha256sum '${TASK_RECEIPT}' | awk '{print \$1}')\" = '${EXPECTED_TASK_RECEIPT_SHA}'
cd '${TASK_ROOT}'; sha256sum -c --quiet '${TASK_RECEIPT}'
test \"\$(sha256sum '${BASE_RECEIPT}' | awk '{print \$1}')\" = '${EXPECTED_BASE_RECEIPT_SHA}'
test \"\$(sha256sum '${SERVER_SOURCE_RECEIPT}' | awk '{print \$1}')\" = '${EXPECTED_SERVER_SOURCE_RECEIPT_SHA}'
test \"\$(sha256sum '${PROTOCOL}' | awk '{print \$1}')\" = '${EXPECTED_PROTOCOL_SHA}'
cd '${TABLE2_RUN_ROOT}'; sha256sum -c --quiet hm3d_table2_leg3_construction_verification.json.sha256
test \"\$(sacct -X -n -P -j '${CONSTRUCTION_VERIFY_JOB}' --format=State | head -1 | cut -d'|' -f1)\" = COMPLETED
test \"\$(sacct -X -n -P -j '${PREVIOUS_FAILED_SMOKE_JOB}' --format=State | head -1 | cut -d'|' -f1)\" = FAILED
test ! -e '${old_formal}'
test ! -e '${new_policy_root}'"

if remote "test -d '${wrapper_root}'"; then
  remote "test \"\$(sha256sum '${wrapper_root}/SOURCE_BUNDLE.sha256' | awk '{print \$1}')\" = '${wrapper_sha}'; cd '${wrapper_root}'; sha256sum -c --quiet SOURCE_BUNDLE.sha256"
else
  stage=${wrapper_root}.partial.$$
  remote "test ! -e '${stage}'; mkdir -p '${stage}'"
  timeout 180 rsync -a --timeout=60 \
    --chmod=Du=rwx,Dgo=rx,Fu=rw,Fgo=r \
    -e "ssh -o BatchMode=yes -o ControlMaster=no -S ${SSH_CONTROL_PATH}" \
    "${scratch}/root/" "${SSH_ALIAS}:${stage}/"
  remote "cd '${stage}'; sha256sum -c --quiet SOURCE_BUNDLE.sha256; chmod -R a-w '${stage}'; mv '${stage}' '${wrapper_root}'"
fi

# Exercise the exact evaluator import surface inside the production container
# before allocating another GPU.  This specifically catches the omission that
# killed job 16601072 before any formal policy row was written.
remote "singularity exec -B /scratch/lg154 -B /scratch/yz11502 \
  /share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif env \
  PYTHONDONTWRITEBYTECODE=1 \
  PYTHONPATH='${wrapper_root}:${wrapper_root}/MemNavData:${TASK_ROOT}:${TASK_ROOT}/MemNavData:${SERVER_SOURCE_ROOT}:${SERVER_SOURCE_ROOT}/MemNavData:${BASE_SOURCE_ROOT}:${BASE_SOURCE_ROOT}/MemNavData:/scratch/yz11502/Research/Nav-axis-uturn/InternNav/src/diffusion-policy:/scratch/lg154/conda-envs/habitat/lib/python3.9/site-packages/pip/_vendor' \
  /scratch/lg154/conda-envs/habitat/bin/python \
  '${TASK_ROOT}/MemNavData/eval_shared_online_role_pairs.py' --help >/dev/null"

construction_sha=$(remote "sha256sum '${construction}' | awk '{print \$1}'" | tr -d '\r')
[[ "${construction_sha}" =~ ^[0-9a-f]{64}$ ]] || fail "bad construction SHA"
launcher=${wrapper_root}/MemNavData/slurm_hm3d_table2_leg3_power_policy_launch.sbatch
safe=${TASK_ROOT}/MemNavData/slurm_safe_submit.sh
common="ALL,TASK_ROOT=${TASK_ROOT},TASK_RECEIPT=${TASK_RECEIPT},EXPECTED_TASK_RECEIPT_SHA=${EXPECTED_TASK_RECEIPT_SHA},BASE_SOURCE_ROOT=${BASE_SOURCE_ROOT},BASE_RECEIPT=${BASE_RECEIPT},EXPECTED_BASE_RECEIPT_SHA=${EXPECTED_BASE_RECEIPT_SHA},SERVER_SOURCE_ROOT=${SERVER_SOURCE_ROOT},SERVER_SOURCE_RECEIPT=${SERVER_SOURCE_RECEIPT},EXPECTED_SERVER_SOURCE_RECEIPT_SHA=${EXPECTED_SERVER_SOURCE_RECEIPT_SHA},SOURCE_RUN_ROOT=${SOURCE_RUN_ROOT},TABLE2_RUN_ROOT=${TABLE2_RUN_ROOT},RUN_ROOT=${TABLE2_RUN_ROOT},PROTOCOL=${PROTOCOL},PARENT_MANIFEST=${PARENT_MANIFEST},POLICY_GPU_PARTITION=a100_tandon,POLICY_RUN_NAME=${POLICY_RUN_NAME},POLICY_LAUNCHER_ROOT=${wrapper_root},POLICY_LAUNCHER_RECEIPT=${wrapper_root}/SOURCE_BUNDLE.sha256,EXPECTED_POLICY_LAUNCHER_RECEIPT_SHA=${wrapper_sha}"

remote "source '${safe}'; safe_sbatch --lint-fatal --test-only --partition=cpu_short --time=00:15:00 --export='${common}' '${launcher}' >/dev/null"
raw=$(remote "source '${safe}'; safe_sbatch --lint-fatal --parsable --job-name=h3T2PolicyImportRepair --partition=cpu_short --time=00:15:00 --export='${common}' '${launcher}'")
launcher_job=$(printf '%s\n' "${raw}" | job_id)
[[ "${launcher_job}" =~ ^[0-9]+$ ]] || fail "bad repair launcher job"

"${LOCAL_PY}" - "${OUT_RECEIPT}" "${wrapper_root}" "${wrapper_sha}" \
  "${construction_sha}" "${PREVIOUS_FAILED_SMOKE_JOB}" "${launcher_job}" \
  "${new_policy_root}" <<'PY'
import json,sys
path,bundle,bundle_sha,construction_sha,failed,launcher,run_root=sys.argv[1:]
p={
 "schema_version":"hm3d_table2_policy_import_repair_submission_v1_20260830",
 "scope":"pre-formal immutable-overlay import repair",
 "wrapper_bundle":bundle,"wrapper_receipt_sha256":bundle_sha,
 "construction_verification_sha256":construction_sha,
 "superseded_failed_smoke_job":int(failed),
 "replacement_launcher_job":int(launcher),"fresh_policy_run_root":run_root,
 "missing_modules_added":[
   "MemNavData.cec_handoff_contract",
   "MemNavData.controller_portability_contract"],
 "production_container_evaluator_import_preflight":True,
 "completed_construction_job_verified_before_dependency_free_submission":True,
 "dynamic_lifetime_held_port_allocator":True,
 "previous_formal_policy_rows":0,
 "scientific_protocol_changed":False,"population_changed":False,
 "arms_changed":False,"budgets_changed":False,"thresholds_changed":False,
 "rows_deleted":False,"fallback_allowed":False,
 "partial_policy_outcomes_read_at_submission":False,
}
open(path,"x").write(json.dumps(p,indent=2,sort_keys=True)+"\n")
print(json.dumps(p,indent=2,sort_keys=True))
PY
printf 'REPAIR_LAUNCHER=%s\nPOLICY_RUN_ROOT=%s\n' \
  "${launcher_job}" "${new_policy_root}"
