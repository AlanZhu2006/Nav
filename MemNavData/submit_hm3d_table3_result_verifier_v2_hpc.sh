#!/usr/bin/env bash
# Submit strengthened raw-row verification without rerunning any policy arm.
set -euo pipefail
umask 0022

ROOT=${ROOT:-/home/asus/Research/Nav-graph-blind}
SSH_ALIAS=${SSH_ALIAS:-alantorch}
LOCAL_MEMNAV_PY=${LOCAL_MEMNAV_PY:-/home/asus/miniconda3/envs/memnav/bin/python}
DOWNSTREAM_RECEIPT=${DOWNSTREAM_RECEIPT:-${ROOT}/MemNavData/HM3D_TABLE3_ACTUAL_MONO_DOWNSTREAM_SUBMISSION_20260830.json}
TASK_ROOT=/scratch/yz11502/Research/Nav-axis-uturn-source-bundles/hm3d_lifelong_natural_b_expansion_execution_1f4979a7fd37d467
SAFE=${TASK_ROOT}/MemNavData/slurm_safe_submit.sh
REMOTE_BUNDLES=/scratch/yz11502/Research/Nav-axis-uturn-source-bundles
SSH_CONTROL_PATH=${SSH_CONTROL_PATH:-$(ssh -G "${SSH_ALIAS}" 2>/dev/null | awk '$1=="controlpath"{v=$2} END{print v}')}

cd "${ROOT}"
fail() { echo "ABORT: $*" >&2; exit 2; }
remote() { timeout 300 ssh -n -T -o BatchMode=yes -o ControlMaster=no -S "${SSH_CONTROL_PATH}" "${SSH_ALIAS}" "$@"; }
job_id() { tr -d '\r' | awk -F';' '/^[0-9]+(;|$)/ {print $1; exit}'; }
[[ -x "${LOCAL_MEMNAV_PY}" && -S "${SSH_CONTROL_PATH}" ]] || fail "local prerequisite missing"
timeout 15 ssh -O check -S "${SSH_CONTROL_PATH}" "${SSH_ALIAS}" >/dev/null 2>&1 || fail "shared SSH unavailable"
readarray -t upstream < <("${LOCAL_MEMNAV_PY}" - "${DOWNSTREAM_RECEIPT}" <<'PY'
import json,sys
p=json.load(open(sys.argv[1]))
assert p['schema_version']=='hm3d_table3_actual_mono_downstream_submission_v1_20260830'
assert p['partial_results_allowed'] is False and p['fallback_completion_allowed'] is False
print(p['result_independent_verification_job'])
print(p['run_root'])
PY
)
[[ "${#upstream[@]}" -eq 2 && "${upstream[0]}" =~ ^[0-9]+$ ]] || fail "invalid upstream receipt"
old_verifier=${upstream[0]}; run_root=${upstream[1]}
files=(
  MemNavData/independent_verify_hm3d_table3_actual_mono_result.py
  MemNavData/slurm_hm3d_table3_result_verifier_v2.sbatch
)
for path in "${files[@]}"; do [[ -f "${path}" && ! -L "${path}" ]] || fail "missing ${path}"; done
"${LOCAL_MEMNAV_PY}" -m py_compile "${files[0]}"
bash -n "${files[1]}"
scratch=$(mktemp -d /tmp/h3_table3_verify_v2.XXXXXX)
trap 'rm -rf -- "${scratch}"' EXIT
mkdir -p "${scratch}/root/MemNavData"
for path in "${files[@]}"; do cp -p "${path}" "${scratch}/root/${path}"; done
(cd "${scratch}/root" && find . -type f ! -name SOURCE_BUNDLE.sha256 -print0 | sort -z | xargs -0 sha256sum >SOURCE_BUNDLE.sha256 && sha256sum -c --quiet SOURCE_BUNDLE.sha256)
receipt_sha=$(sha256sum "${scratch}/root/SOURCE_BUNDLE.sha256" | awk '{print $1}')
wrapper_root=${REMOTE_BUNDLES}/hm3d_table3_result_verifier_v2_${receipt_sha:0:16}
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
common="ALL,WRAPPER_ROOT=${wrapper_root},WRAPPER_RECEIPT=${wrapper_root}/SOURCE_BUNDLE.sha256,EXPECTED_WRAPPER_RECEIPT_SHA=${receipt_sha},RUN_ROOT=${run_root}"
sbatch=${wrapper_root}/MemNavData/slurm_hm3d_table3_result_verifier_v2.sbatch
remote "source '${SAFE}'; safe_sbatch --lint-fatal --test-only --partition=cpu_short --time=00:20:00 --export='${common}' '${sbatch}' >/dev/null"
raw=$(remote "source '${SAFE}'; safe_sbatch --lint-fatal --parsable --partition=cpu_short --time=00:20:00 --dependency='afterok:${old_verifier}' --kill-on-invalid-dep=yes --export='${common}' '${sbatch}'")
job=$(printf '%s\n' "${raw}" | job_id)
[[ "${job}" =~ ^[0-9]+$ ]] || fail "bad strengthened verifier job id"
receipt=MemNavData/HM3D_TABLE3_RESULT_VERIFIER_V2_SUBMISSION_20260830.json
[[ ! -e "${receipt}" ]] || fail "strengthened verifier receipt exists"
"${LOCAL_MEMNAV_PY}" - "${receipt}" "${job}" "${old_verifier}" \
  "${run_root}" "${wrapper_root}" "${receipt_sha}" <<'PY'
import json,sys
path,job,upstream,run,bundle,bundle_sha=sys.argv[1:]
p={'schema_version':'hm3d_table3_result_verifier_v2_submission_v1_20260830',
   'job_id':int(job),'depends_afterok_on':int(upstream),'run_root':run,
   'wrapper_bundle':bundle,'wrapper_bundle_sha256':bundle_sha,
   'policy_rerun':False,'raw_rows_expected':192,
   'verifies':['SR','SPL','paired_gain_loss','exact_McNemar','authorization_counts'],
   'partial_results_allowed':False,'fallback_completion_allowed':False}
open(path,'x').write(json.dumps(p,indent=2,sort_keys=True)+'\n')
print(json.dumps(p,indent=2,sort_keys=True))
PY
