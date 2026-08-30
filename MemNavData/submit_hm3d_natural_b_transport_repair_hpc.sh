#!/usr/bin/env bash
# Submit an exact, outcome-blind repair after the original factual-B array.
set -euo pipefail
umask 0022

ROOT=${ROOT:-/home/asus/Research/Nav-graph-blind}
SSH_ALIAS=${SSH_ALIAS:-alantorch}
LOCAL_PY=${LOCAL_PY:-/home/asus/miniconda3/envs/memnav/bin/python}
ORIGINAL_ARRAY_JOB=16592875
RUN_ROOT=/scratch/yz11502/Research/Nav-axis-uturn-results/hm3d_fullmono_lifelong_natural_b_expansion_execution_20260830/formal_20260830T045416Z_1f4979a7
TASK_ROOT=/scratch/yz11502/Research/Nav-axis-uturn-source-bundles/hm3d_lifelong_natural_b_expansion_execution_1f4979a7fd37d467
TASK_RECEIPT=${TASK_ROOT}/SOURCE_BUNDLE.sha256
EXPECTED_TASK_RECEIPT_SHA=1f4979a7fd37d46700011558063be34a8fba0a0b8746668469dba7e7955f4282
SERVER_SOURCE_ROOT=/scratch/yz11502/Research/Nav-axis-uturn-source-bundles/hm3d_fullmono_lifelong_375f0b6879b2ff87
SERVER_SOURCE_RECEIPT=${SERVER_SOURCE_ROOT}/SOURCE_BUNDLE.sha256
EXPECTED_SERVER_SOURCE_RECEIPT_SHA=375f0b6879b2ff87b7019dae4727880d1b03fd3185a1862e6239942a76b5bcc8
TABLE2_SERVER_SOURCE_ROOT=/scratch/yz11502/Research/Nav-axis-uturn-source-bundles/hm3d_table1_navdp_authority_transaction_718661db1733d5de
TABLE2_SERVER_SOURCE_RECEIPT=${TABLE2_SERVER_SOURCE_ROOT}/SOURCE_BUNDLE.sha256
EXPECTED_TABLE2_SERVER_SOURCE_RECEIPT_SHA=718661db1733d5de16cd86687eec880a8d02fc5ae5ca982e1ab7d5bde5e96f7d
BASE_SOURCE_ROOT=/scratch/yz11502/Research/Nav-axis-uturn-source-bundles/final14_mono_factorial_5690569a4373f2d2
BASE_RECEIPT=${BASE_SOURCE_ROOT}/source_inputs.sha256
EXPECTED_BASE_RECEIPT_SHA=5690569a4373f2d2768671418f0c604c4a03aa4b0ffe01baf70b288af03ba216
PARENT_ROOT=/scratch/yz11502/Research/Nav-axis-uturn-results/hm3d_fresh_fullmono_mixed_role_20260820/formal_20260820T143609Z_e6dd44c6
PROTOCOL=${TASK_ROOT}/MemNavData/hm3d_fullmono_lifelong_natural_b_expansion_execution_protocol_20260830.json
EXPECTED_PROTOCOL_SHA=28101fe2574e9ea428306dbf12932cb3da7cf3c24d88f82f573c1cb3209d9edd
SHARD_MANIFEST=${RUN_ROOT}/factual_b_schedule/shards.json
EXPECTED_SHARD_MANIFEST_SHA=bdc9048b0a1421ba5b405feff6fc1ca2bb8515a50c96c5c99babb5045fe4f3ec
REMOTE_BUNDLES=/scratch/yz11502/Research/Nav-axis-uturn-source-bundles
SSH_CONTROL_PATH=${SSH_CONTROL_PATH:-$(ssh -G "${SSH_ALIAS}" 2>/dev/null | awk '$1=="controlpath"{v=$2} END{print v}')}

cd "${ROOT}"
fail() { echo "ABORT: $*" >&2; exit 2; }
remote() { timeout 300 ssh -n -T -o BatchMode=yes -o ControlMaster=no -S "${SSH_CONTROL_PATH}" "${SSH_ALIAS}" "$@"; }
job_id() { tr -d '\r' | awk -F';' '/^[0-9]+(;|$)/ {print $1; exit}'; }
[[ -x "${LOCAL_PY}" && -S "${SSH_CONTROL_PATH}" ]] || fail "local prerequisite missing"
timeout 15 ssh -O check -S "${SSH_CONTROL_PATH}" "${SSH_ALIAS}" >/dev/null 2>&1 || fail "shared SSH unavailable"
files=(
  MemNavData/freeze_hm3d_natural_b_transport_repair.py
  MemNavData/independent_verify_hm3d_natural_b_transport_repair.py
  MemNavData/run_hm3d_fullmono_server_scene.sh
  MemNavData/slurm_hm3d_natural_b_transport_repair_shard.sbatch
  MemNavData/slurm_hm3d_natural_b_transport_repair_launch.sbatch
  MemNavData/test_hm3d_natural_b_transport_repair.py
)
for path in "${files[@]}"; do [[ -f "${path}" && ! -L "${path}" ]] || fail "missing ${path}"; done
"${LOCAL_PY}" -m pytest -q MemNavData/test_hm3d_natural_b_transport_repair.py MemNavData/test_hm3d_table1_navdp_transport_contract.py
bash -n MemNavData/run_hm3d_fullmono_server_scene.sh MemNavData/slurm_hm3d_natural_b_transport_repair_shard.sbatch MemNavData/slurm_hm3d_natural_b_transport_repair_launch.sbatch
scratch=$(mktemp -d /tmp/h3_natb_transport.XXXXXX)
trap 'rm -rf -- "${scratch}"' EXIT
mkdir -p "${scratch}/root"
for path in "${files[@]}"; do mkdir -p "${scratch}/root/$(dirname "${path}")"; cp -p "${path}" "${scratch}/root/${path}"; done
(cd "${scratch}/root" && find . -type f ! -name SOURCE_BUNDLE.sha256 -print0 | sort -z | xargs -0 sha256sum >SOURCE_BUNDLE.sha256 && sha256sum -c --quiet SOURCE_BUNDLE.sha256)
receipt_sha=$(sha256sum "${scratch}/root/SOURCE_BUNDLE.sha256" | awk '{print $1}')
wrapper_root=${REMOTE_BUNDLES}/hm3d_natural_b_transport_repair_${receipt_sha:0:16}
if remote "test -d '${wrapper_root}'"; then
  remote "test \"\$(sha256sum '${wrapper_root}/SOURCE_BUNDLE.sha256' | awk '{print \$1}')\" = '${receipt_sha}'; cd '${wrapper_root}'; sha256sum -c --quiet SOURCE_BUNDLE.sha256"
else
  stage=${wrapper_root}.partial.$$
  remote "mkdir -p '${stage}'"
  rsync -a --chmod=Du=rwx,Dgo=rx,Fu=rw,Fgo=r -e "ssh -o BatchMode=yes -o ControlMaster=no -S ${SSH_CONTROL_PATH}" "${scratch}/root/" "${SSH_ALIAS}:${stage}/"
  remote "cd '${stage}'; sha256sum -c --quiet SOURCE_BUNDLE.sha256; chmod -R a-w '${stage}'; mv '${stage}' '${wrapper_root}'"
fi
common="ALL,WRAPPER_ROOT=${wrapper_root},WRAPPER_RECEIPT=${wrapper_root}/SOURCE_BUNDLE.sha256,EXPECTED_WRAPPER_RECEIPT_SHA=${receipt_sha},TASK_ROOT=${TASK_ROOT},TASK_RECEIPT=${TASK_RECEIPT},EXPECTED_TASK_RECEIPT_SHA=${EXPECTED_TASK_RECEIPT_SHA},SERVER_SOURCE_ROOT=${SERVER_SOURCE_ROOT},SERVER_SOURCE_RECEIPT=${SERVER_SOURCE_RECEIPT},EXPECTED_SERVER_SOURCE_RECEIPT_SHA=${EXPECTED_SERVER_SOURCE_RECEIPT_SHA},TABLE2_SERVER_SOURCE_ROOT=${TABLE2_SERVER_SOURCE_ROOT},TABLE2_SERVER_SOURCE_RECEIPT=${TABLE2_SERVER_SOURCE_RECEIPT},EXPECTED_TABLE2_SERVER_SOURCE_RECEIPT_SHA=${EXPECTED_TABLE2_SERVER_SOURCE_RECEIPT_SHA},BASE_SOURCE_ROOT=${BASE_SOURCE_ROOT},BASE_RECEIPT=${BASE_RECEIPT},EXPECTED_BASE_RECEIPT_SHA=${EXPECTED_BASE_RECEIPT_SHA},RUN_ROOT=${RUN_ROOT},PARENT_ROOT=${PARENT_ROOT},PROTOCOL=${PROTOCOL},EXPECTED_PROTOCOL_SHA=${EXPECTED_PROTOCOL_SHA},SHARD_MANIFEST=${SHARD_MANIFEST},EXPECTED_SHARD_MANIFEST_SHA=${EXPECTED_SHARD_MANIFEST_SHA}"
safe=${TASK_ROOT}/MemNavData/slurm_safe_submit.sh
launcher=${wrapper_root}/MemNavData/slurm_hm3d_natural_b_transport_repair_launch.sbatch
remote "source '${safe}'; safe_sbatch --lint-fatal --test-only --partition=cpu_short --dependency='afterany:${ORIGINAL_ARRAY_JOB}' --export='${common}' '${launcher}' >/dev/null"
raw=$(remote "source '${safe}'; safe_sbatch --lint-fatal --parsable --partition=cpu_short --dependency='afterany:${ORIGINAL_ARRAY_JOB}' --export='${common}' '${launcher}'")
job=$(printf '%s\n' "${raw}" | job_id)
[[ "${job}" =~ ^[0-9]+$ ]] || fail "bad repair launcher job id"
receipt=MemNavData/HM3D_NATURAL_B_TRANSPORT_REPAIR_SUBMISSION_20260830.json
[[ ! -e "${receipt}" ]] || fail "submission receipt exists"
"${LOCAL_PY}" - "${receipt}" "${job}" "${wrapper_root}" "${receipt_sha}" "${EXPECTED_SHARD_MANIFEST_SHA}" <<'PY'
import json,sys
path,job,bundle,bundle_sha,shard_sha=sys.argv[1:]
p={'schema_version':'hm3d_natural_b_transport_repair_launcher_submission_v1_20260830',
 'original_factual_B_array_job':16592875,'repair_launcher_job':int(job),
 'wrapper_bundle':bundle,'wrapper_bundle_sha256':bundle_sha,
 'factual_B_shard_manifest_sha256':shard_sha,
 'navigation_outcomes_read_at_submission':False,
 'scientific_thresholds_changed':False,'fallback_completion_allowed':False}
open(path,'x').write(json.dumps(p,indent=2,sort_keys=True)+'\n')
print(json.dumps(p,indent=2,sort_keys=True))
PY
