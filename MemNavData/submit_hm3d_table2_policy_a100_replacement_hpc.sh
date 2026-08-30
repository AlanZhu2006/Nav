#!/usr/bin/env bash
# Replace a still-pending Table-II policy launcher with an immutable A100-only
# launcher.  The wrapper also carries the post-seal conference waterfall
# verifier; construction, population, arms, budgets, thresholds, and the
# result-blind policy analysis remain byte-identical.
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
REPLACED_LAUNCHER_JOB=${REPLACED_LAUNCHER_JOB:-16600350}
OUT_RECEIPT=${OUT_RECEIPT:-MemNavData/HM3D_TABLE2_POLICY_MEETING_V2_SUBMISSION_20260830.json}
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

files=(
  MemNavData/slurm_hm3d_table2_leg3_power_policy_launch.sbatch
  MemNavData/slurm_hm3d_table2_meeting_result.sbatch
  MemNavData/independent_verify_hm3d_table2_meeting_result.py
  MemNavData/submit_hm3d_table2_policy_a100_replacement_hpc.sh
)
for path in "${files[@]}"; do
  [[ -f "${path}" && ! -L "${path}" ]] || fail "missing physical ${path}"
done
bash -n "${files[@]}"
grep -q -- '--partition="${POLICY_GPU_PARTITION}"' \
  MemNavData/slurm_hm3d_table2_leg3_power_policy_launch.sbatch || \
  fail "launcher does not bind the policy partition"

scratch=$(mktemp -d /tmp/h3_table2_policy_a100.XXXXXX)
cleanup() { rm -rf -- "${scratch}"; }
trap cleanup EXIT
mkdir -p "${scratch}/root/MemNavData"
for path in "${files[@]}"; do
  cp -p "${path}" "${scratch}/root/${path}"
done
(
  cd "${scratch}/root"
  find . -type f ! -name SOURCE_BUNDLE.sha256 -print0 | sort -z | \
    xargs -0 sha256sum >SOURCE_BUNDLE.sha256
  sha256sum -c --quiet SOURCE_BUNDLE.sha256
)
wrapper_sha=$(sha256sum "${scratch}/root/SOURCE_BUNDLE.sha256" | awk '{print $1}')
wrapper_root=${REMOTE_BUNDLES}/hm3d_table2_policy_a100_${wrapper_sha:0:16}

remote_identity=$(remote 'id -un' | tr -d '\r')
[[ "${remote_identity}" == yz11502 ]] || fail "wrong remote identity"
remote "set -euo pipefail
test \"\$(sha256sum '${TASK_RECEIPT}' | awk '{print \$1}')\" = '${EXPECTED_TASK_RECEIPT_SHA}'
cd '${TASK_ROOT}'; sha256sum -c --quiet '${TASK_RECEIPT}'
test \"\$(sha256sum '${BASE_RECEIPT}' | awk '{print \$1}')\" = '${EXPECTED_BASE_RECEIPT_SHA}'
test \"\$(sha256sum '${SERVER_SOURCE_RECEIPT}' | awk '{print \$1}')\" = '${EXPECTED_SERVER_SOURCE_RECEIPT_SHA}'
test \"\$(sha256sum '${PROTOCOL}' | awk '{print \$1}')\" = '${EXPECTED_PROTOCOL_SHA}'
test ! -e '${TABLE2_RUN_ROOT}/policy_submission.json'
test \"\$(squeue -h -j '${REPLACED_LAUNCHER_JOB}' -o '%T')\" = PENDING
test \"\$(squeue -h -j '${CONSTRUCTION_VERIFY_JOB}' -o '%T')\" = PENDING"

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

launcher=${wrapper_root}/MemNavData/slurm_hm3d_table2_leg3_power_policy_launch.sbatch
safe=${TASK_ROOT}/MemNavData/slurm_safe_submit.sh
common="ALL,TASK_ROOT=${TASK_ROOT},TASK_RECEIPT=${TASK_RECEIPT},EXPECTED_TASK_RECEIPT_SHA=${EXPECTED_TASK_RECEIPT_SHA},BASE_SOURCE_ROOT=${BASE_SOURCE_ROOT},BASE_RECEIPT=${BASE_RECEIPT},EXPECTED_BASE_RECEIPT_SHA=${EXPECTED_BASE_RECEIPT_SHA},SERVER_SOURCE_ROOT=${SERVER_SOURCE_ROOT},SERVER_SOURCE_RECEIPT=${SERVER_SOURCE_RECEIPT},EXPECTED_SERVER_SOURCE_RECEIPT_SHA=${EXPECTED_SERVER_SOURCE_RECEIPT_SHA},SOURCE_RUN_ROOT=${SOURCE_RUN_ROOT},TABLE2_RUN_ROOT=${TABLE2_RUN_ROOT},RUN_ROOT=${TABLE2_RUN_ROOT},PROTOCOL=${PROTOCOL},PARENT_MANIFEST=${PARENT_MANIFEST},POLICY_GPU_PARTITION=a100_tandon,POLICY_LAUNCHER_ROOT=${wrapper_root},POLICY_LAUNCHER_RECEIPT=${wrapper_root}/SOURCE_BUNDLE.sha256,EXPECTED_POLICY_LAUNCHER_RECEIPT_SHA=${wrapper_sha}"

remote "source '${safe}'; safe_sbatch --lint-fatal --test-only --partition=cpu_short --time=00:15:00 --dependency='afterok:${CONSTRUCTION_VERIFY_JOB}' --kill-on-invalid-dep=yes --export='${common}' '${launcher}' >/dev/null"
raw=$(remote "source '${safe}'; safe_sbatch --lint-fatal --parsable --job-name=h3T2PolicyA100 --partition=cpu_short --time=00:15:00 --dependency='afterok:${CONSTRUCTION_VERIFY_JOB}' --kill-on-invalid-dep=yes --export='${common}' '${launcher}'")
replacement_job=$(printf '%s\n' "${raw}" | job_id)
[[ "${replacement_job}" =~ ^[0-9]+$ ]] || fail "bad replacement launcher job"

# The obsolete launcher is still dependency-held.  Cancel it only after the
# immutable replacement has been accepted by Slurm.
remote "set -euo pipefail
test \"\$(squeue -h -j '${replacement_job}' -o '%T')\" = PENDING
test \"\$(squeue -h -j '${REPLACED_LAUNCHER_JOB}' -o '%T')\" = PENDING
scancel '${REPLACED_LAUNCHER_JOB}'
for _ in 1 2 3 4 5; do
  state=\$(sacct -X -n -P -j '${REPLACED_LAUNCHER_JOB}' --format=State | head -1 | cut -d'|' -f1)
  [[ \${state} == CANCELLED* ]] && exit 0
  sleep 1
done
exit 2"

"${LOCAL_PY}" - "${OUT_RECEIPT}" "${wrapper_root}" "${wrapper_sha}" \
  "${CONSTRUCTION_VERIFY_JOB}" "${REPLACED_LAUNCHER_JOB}" \
  "${replacement_job}" <<'PY'
import json,sys
path,bundle,bundle_sha,verify,old,new=sys.argv[1:]
p={
 "schema_version":"hm3d_table2_policy_a100_replacement_submission_v2_20260830",
 "scope":"infrastructure-only partition binding before policy evaluation",
 "post_seal_conference_waterfall_verifier_included":True,
 "wrapper_bundle":bundle,"wrapper_receipt_sha256":bundle_sha,
 "construction_independent_verify_job":int(verify),
 "replaced_pending_launcher_job":int(old),
 "replacement_launcher_job":int(new),
 "policy_gpu_partition":"a100_tandon",
 "scientific_protocol_changed":False,
 "population_changed":False,"arms_changed":False,"budgets_changed":False,
 "thresholds_changed":False,"rows_deleted":False,"fallback_allowed":False,
 "query_policy_outcomes_read_at_submission":False,
 "replacement_submitted_before_obsolete_launcher_cancelled":True,
}
open(path,"x").write(json.dumps(p,indent=2,sort_keys=True)+"\n")
print(json.dumps(p,indent=2,sort_keys=True))
PY
printf 'REPLACEMENT_LAUNCHER=%s\nOBSOLETE_LAUNCHER=%s\n' \
  "${replacement_job}" "${REPLACED_LAUNCHER_JOB}"
