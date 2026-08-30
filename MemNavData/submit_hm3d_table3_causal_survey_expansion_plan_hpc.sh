#!/usr/bin/env bash
# Submit the append-only Table-III expansion plan after all base construction
# receipts exist. This stage reads no query-policy outcome and runs no policy.
set -euo pipefail
umask 0022

ROOT=${ROOT:-/home/asus/Research/Nav-graph-blind}
SSH_ALIAS=${SSH_ALIAS:-alantorch}
LOCAL_MEMNAV_PY=${LOCAL_MEMNAV_PY:-/home/asus/miniconda3/envs/memnav/bin/python}
OUT_RECEIPT=${OUT_RECEIPT:-MemNavData/HM3D_TABLE3_CAUSAL_SURVEY_EXPANSION_PLAN_SUBMISSION_20260831.json}
BASE_SUBMISSION=${BASE_SUBMISSION:-MemNavData/HM3D_TABLE3_CAUSAL_SURVEY_CONSTRUCTION_SUBMISSION_20260830.json}
BASE_PLAN=/scratch/yz11502/Research/Nav-axis-uturn-results/hm3d_table3_actual_mono_20260830/plan_20260830T061943Z_3b848b5c/candidate_plan.json
EXPECTED_BASE_PLAN_SHA=1b1d16dd2132adb32565604bcf99f4852fa36df66a22bec2121e8338ce40020d
BASE_PROTOCOL=/scratch/yz11502/Research/Nav-axis-uturn-source-bundles/hm3d_table3_causal_survey_3d811e0f63298480/MemNavData/hm3d_table3_causal_survey_protocol_20260830.json
EXPECTED_BASE_PROTOCOL_SHA=0f2f3b02e2e2bb3253bdcb386f501ec293533e4bf0014f6d853146d8e29723ea
CAPACITY_ROOT=/scratch/yz11502/Research/Nav-axis-uturn-results/hm3d_table3_navmesh_capacity_replenishment_20260831/capacity_replenishment_20260830T160351Z_7d4e0756
EXPECTED_CAPACITY_SUMMARY_SHA=9faa519882bd0641e6ded9b6d2042e333dfc2c352007197392d9334974820c4d
EXPECTED_CAPACITY_VERIFY_SHA=b21b16ddd9acd941748177e5b927441e94f7387f8c8cc6219ed8f63caa977190
TASK_ROOT=/scratch/yz11502/Research/Nav-axis-uturn-source-bundles/hm3d_lifelong_natural_b_expansion_execution_1f4979a7fd37d467
REMOTE_BUNDLES=/scratch/yz11502/Research/Nav-axis-uturn-source-bundles
REMOTE_RESULTS=/scratch/yz11502/Research/Nav-axis-uturn-results/hm3d_table3_causal_survey_expansion_20260831
SSH_CONTROL_PATH=${SSH_CONTROL_PATH:-$(ssh -G "${SSH_ALIAS}" 2>/dev/null | awk '$1=="controlpath"{v=$2} END{print v}')}

cd "${ROOT}"
fail() { echo "ABORT: $*" >&2; exit 2; }
remote() {
  timeout 300 ssh -n -T -o BatchMode=yes -o ControlMaster=no \
    -S "${SSH_CONTROL_PATH}" "${SSH_ALIAS}" "$@"
}
job_id() { tr -d '\r' | awk -F';' '/^[0-9]+(;|$)/ {print $1; exit}'; }
[[ -x "${LOCAL_MEMNAV_PY}" && -S "${SSH_CONTROL_PATH}" ]] \
  || fail "local prerequisite missing"
timeout 15 ssh -O check -S "${SSH_CONTROL_PATH}" "${SSH_ALIAS}" \
  >/dev/null 2>&1 || fail "shared SSH unavailable"
[[ ! -e "${OUT_RECEIPT}" ]] || fail "submission receipt already exists"

base_job=$("${LOCAL_MEMNAV_PY}" - "${BASE_SUBMISSION}" <<'PY'
import json,sys
p=json.load(open(sys.argv[1]))
assert p['schema_version']=='hm3d_table3_causal_survey_construction_submission_v1_20260830'
assert p['frozen_candidates']==125
assert p['query_policy_jobs_submitted'] is False
assert p['query_policy_outcomes_read_at_submission'] is False
assert p['fallback_completion_allowed'] is False
print(p['construction_array_job'])
PY
)
[[ "${base_job}" =~ ^[0-9]+$ ]] || fail "invalid base construction job"

files=(
  MemNavData/freeze_hm3d_table3_causal_survey_expansion_plan.py
  MemNavData/hm3d_table3_causal_survey_expansion_selection_protocol_20260831.json
  MemNavData/independent_verify_hm3d_table3_causal_survey_expansion_plan.py
  MemNavData/slurm_hm3d_table3_causal_survey_expansion_plan.sbatch
  MemNavData/test_hm3d_table3_causal_survey_expansion.py
)
for path in "${files[@]}"; do
  [[ -f "${path}" && ! -L "${path}" ]] || fail "missing ${path}"
done
"${LOCAL_MEMNAV_PY}" -m pytest -q \
  MemNavData/test_hm3d_table3_causal_survey_expansion.py
"${LOCAL_MEMNAV_PY}" -m py_compile \
  MemNavData/freeze_hm3d_table3_causal_survey_expansion_plan.py \
  MemNavData/independent_verify_hm3d_table3_causal_survey_expansion_plan.py
bash -n MemNavData/slurm_hm3d_table3_causal_survey_expansion_plan.sbatch "$0"
python -m json.tool \
  MemNavData/hm3d_table3_causal_survey_expansion_selection_protocol_20260831.json \
  >/dev/null

scratch=$(mktemp -d /tmp/h3_table3_expansion_plan.XXXXXX)
trap 'rm -r -- "${scratch}"' EXIT
mkdir -p "${scratch}/root"
for path in "${files[@]}"; do
  mkdir -p "${scratch}/root/$(dirname "${path}")"
  cp -p "${path}" "${scratch}/root/${path}"
done
(cd "${scratch}/root" && \
  find . -type f ! -name SOURCE_BUNDLE.sha256 -print0 | sort -z | \
  xargs -0 sha256sum >SOURCE_BUNDLE.sha256 && \
  sha256sum -c --quiet SOURCE_BUNDLE.sha256)
receipt_sha=$(sha256sum "${scratch}/root/SOURCE_BUNDLE.sha256" | awk '{print $1}')
source_root=${REMOTE_BUNDLES}/hm3d_table3_causal_survey_expansion_plan_${receipt_sha:0:16}
source_receipt=${source_root}/SOURCE_BUNDLE.sha256
stamp=$(date -u +%Y%m%dT%H%M%SZ)
run_root=${REMOTE_RESULTS}/expansion_${stamp}_${receipt_sha:0:8}

remote "set -euo pipefail
test \"\$(sha256sum '${BASE_PLAN}' | awk '{print \$1}')\" = '${EXPECTED_BASE_PLAN_SHA}'
test \"\$(sha256sum '${BASE_PROTOCOL}' | awk '{print \$1}')\" = '${EXPECTED_BASE_PROTOCOL_SHA}'
test \"\$(sha256sum '${CAPACITY_ROOT}/formal/capacity_summary.json' | awk '{print \$1}')\" = '${EXPECTED_CAPACITY_SUMMARY_SHA}'
test \"\$(sha256sum '${CAPACITY_ROOT}/formal/independent_capacity_verification.json' | awk '{print \$1}')\" = '${EXPECTED_CAPACITY_VERIFY_SHA}'"
if remote "test -d '${source_root}'"; then
  remote "test \"\$(sha256sum '${source_receipt}' | awk '{print \$1}')\" = '${receipt_sha}'; cd '${source_root}'; sha256sum -c --quiet SOURCE_BUNDLE.sha256"
else
  stage=${source_root}.partial.$$
  remote "mkdir -p '${stage}'"
  rsync -a --chmod=Du=rwx,Dgo=rx,Fu=rw,Fgo=r \
    -e "ssh -o BatchMode=yes -o ControlMaster=no -S ${SSH_CONTROL_PATH}" \
    "${scratch}/root/" "${SSH_ALIAS}:${stage}/"
  remote "cd '${stage}'; sha256sum -c --quiet SOURCE_BUNDLE.sha256; chmod -R a-w '${stage}'; mv '${stage}' '${source_root}'"
fi
remote "mkdir -p '${run_root}'"

selection=${source_root}/MemNavData/hm3d_table3_causal_survey_expansion_selection_protocol_20260831.json
safe=${TASK_ROOT}/MemNavData/slurm_safe_submit.sh
job=${source_root}/MemNavData/slurm_hm3d_table3_causal_survey_expansion_plan.sbatch
common="ALL,EXPANSION_SOURCE_ROOT=${source_root},EXPANSION_SOURCE_RECEIPT=${source_receipt},EXPECTED_EXPANSION_SOURCE_RECEIPT_SHA=${receipt_sha},EXPANSION_RUN_ROOT=${run_root},EXPANSION_SELECTION_PROTOCOL=${selection}"
remote "source '${safe}'; safe_sbatch --lint-fatal --test-only --partition=cpu_short --time=00:20:00 --export='${common},MODE=freeze' '${job}' >/dev/null"
remote "source '${safe}'; safe_sbatch --lint-fatal --test-only --partition=cpu_short --time=00:20:00 --export='${common},MODE=verify' '${job}' >/dev/null"

raw=$(remote "source '${safe}'; safe_sbatch --lint-fatal --parsable --partition=cpu_short --time=00:20:00 --dependency='afterok:${base_job}' --kill-on-invalid-dep=yes --export='${common},MODE=freeze' '${job}'")
freeze_job=$(printf '%s\n' "${raw}" | job_id)
[[ "${freeze_job}" =~ ^[0-9]+$ ]] || fail "invalid expansion freeze job"
raw=$(remote "source '${safe}'; safe_sbatch --lint-fatal --parsable --partition=cpu_short --time=00:20:00 --dependency='afterok:${freeze_job}' --kill-on-invalid-dep=yes --export='${common},MODE=verify' '${job}'")
verify_job=$(printf '%s\n' "${raw}" | job_id)
[[ "${verify_job}" =~ ^[0-9]+$ ]] || fail "invalid expansion verifier job"

"${LOCAL_MEMNAV_PY}" - "${OUT_RECEIPT}" "${base_job}" "${freeze_job}" \
  "${verify_job}" "${run_root}" "${source_root}" "${receipt_sha}" <<'PY'
import json,sys
path,base,freeze,verify,run,source,source_sha=sys.argv[1:]
payload={
  'schema_version':'hm3d_table3_causal_survey_expansion_plan_submission_v1_20260831',
  'base_construction_array_job':int(base),
  'expansion_plan_freeze_job':int(freeze),
  'expansion_plan_independent_verification_job':int(verify),
  'run_root':run,'source_bundle':source,'source_bundle_sha256':source_sha,
  'plan':'expansion_plan/candidate_plan.json',
  'construction_protocol':'expansion_plan/construction_protocol.json',
  'independent_verification':'expansion_plan/independent_verification.json',
  'base_candidates_deleted_or_replaced':False,
  'query_policy_jobs_submitted':False,
  'query_policy_outcomes_read':False,
  'navigation_policy_outcomes_read':False,
  'threshold_relaxation':False,'fallback_completion_allowed':False,
}
open(path,'x').write(json.dumps(payload,indent=2,sort_keys=True)+'\n')
print(json.dumps(payload,indent=2,sort_keys=True))
PY
