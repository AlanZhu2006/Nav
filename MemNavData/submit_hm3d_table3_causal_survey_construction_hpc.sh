#!/usr/bin/env bash
# Submit the result-blind construction and independent population gate for the
# controlled causal-RGB Table-III range analysis.  No query policy is loaded.
set -euo pipefail
umask 0022

ROOT=${ROOT:-/home/asus/Research/Nav-graph-blind}
SSH_ALIAS=${SSH_ALIAS:-alantorch}
LOCAL_MEMNAV_PY=${LOCAL_MEMNAV_PY:-/home/asus/miniconda3/envs/memnav/bin/python}
LOCAL_HAB_PY=${LOCAL_HAB_PY:-/home/asus/miniconda3/envs/habitat/bin/python}
OUT_RECEIPT=${OUT_RECEIPT:-MemNavData/HM3D_TABLE3_CAUSAL_SURVEY_CONSTRUCTION_SUBMISSION_20260830.json}
CANDIDATE_PLAN=/scratch/yz11502/Research/Nav-axis-uturn-results/hm3d_table3_actual_mono_20260830/plan_20260830T061943Z_3b848b5c/candidate_plan.json
EXPECTED_CANDIDATE_PLAN_SHA=1b1d16dd2132adb32565604bcf99f4852fa36df66a22bec2121e8338ce40020d
BASE_WRAPPER_ROOT=/scratch/yz11502/Research/Nav-axis-uturn-source-bundles/hm3d_table3_actual_mono_downstream_2d8d08ff5a65da0a
BASE_WRAPPER_RECEIPT=${BASE_WRAPPER_ROOT}/SOURCE_BUNDLE.sha256
EXPECTED_BASE_WRAPPER_RECEIPT_SHA=2d8d08ff5a65da0ad00e7372fd756174b88958ed4cf74076e6605c951b38c3fe
TASK_ROOT=/scratch/yz11502/Research/Nav-axis-uturn-source-bundles/hm3d_lifelong_natural_b_expansion_execution_1f4979a7fd37d467
REMOTE_BUNDLES=/scratch/yz11502/Research/Nav-axis-uturn-source-bundles
REMOTE_RESULTS=/scratch/yz11502/Research/Nav-axis-uturn-results/hm3d_table3_causal_survey_20260830
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
  >/dev/null 2>&1 || fail "shared SSH unavailable"
[[ ! -e "${OUT_RECEIPT}" ]] || fail "submission receipt already exists"

files=(
  MemNavData/analyze_hm3d_table3_actual_mono.py
  MemNavData/audit_hm3d_table3_length_role_pairs.py
  MemNavData/construct_hm3d_table3_causal_survey_role_pair.py
  MemNavData/eval_shared_online_role_pairs.py
  MemNavData/finalize_hm3d_table3_causal_survey_population.py
  MemNavData/hm3d_table3_causal_survey_contract.py
  MemNavData/hm3d_table3_causal_survey_protocol_20260830.json
  MemNavData/hm3d_table3_length_contract.py
  MemNavData/independent_verify_hm3d_table3_actual_mono_result.py
  MemNavData/independent_verify_hm3d_table3_causal_survey_population.py
  MemNavData/run_hm3d_fullmono_query_history.py
  MemNavData/run_hm3d_fullmono_server_scene.sh
  MemNavData/slurm_hm3d_table3_causal_survey_analysis.sbatch
  MemNavData/slurm_hm3d_table3_causal_survey_construct.sbatch
  MemNavData/slurm_hm3d_table3_causal_survey_pair.sbatch
  MemNavData/slurm_hm3d_table3_causal_survey_population.sbatch
  MemNavData/test_audit_hm3d_table3_actual_mono_constructibility.py
  MemNavData/test_hm3d_table3_length_contract.py
)
for path in "${files[@]}"; do
  [[ -f "${path}" && ! -L "${path}" ]] || fail "missing ${path}"
done

"${LOCAL_MEMNAV_PY}" -m pytest -q \
  MemNavData/test_hm3d_table3_length_contract.py \
  MemNavData/test_audit_hm3d_table3_actual_mono_constructibility.py
"${LOCAL_HAB_PY}" -m py_compile \
  MemNavData/construct_hm3d_table3_causal_survey_role_pair.py \
  MemNavData/eval_shared_online_role_pairs.py \
  MemNavData/run_hm3d_fullmono_query_history.py
"${LOCAL_MEMNAV_PY}" -m py_compile \
  MemNavData/analyze_hm3d_table3_actual_mono.py \
  MemNavData/audit_hm3d_table3_length_role_pairs.py \
  MemNavData/finalize_hm3d_table3_causal_survey_population.py \
  MemNavData/hm3d_table3_causal_survey_contract.py \
  MemNavData/hm3d_table3_length_contract.py \
  MemNavData/independent_verify_hm3d_table3_actual_mono_result.py \
  MemNavData/independent_verify_hm3d_table3_causal_survey_population.py
bash -n \
  MemNavData/run_hm3d_fullmono_server_scene.sh \
  MemNavData/slurm_hm3d_table3_causal_survey_analysis.sbatch \
  MemNavData/slurm_hm3d_table3_causal_survey_construct.sbatch \
  MemNavData/slurm_hm3d_table3_causal_survey_pair.sbatch \
  MemNavData/slurm_hm3d_table3_causal_survey_population.sbatch \
  "$0"
python -m json.tool \
  MemNavData/hm3d_table3_causal_survey_protocol_20260830.json >/dev/null

scratch=$(mktemp -d /tmp/h3_table3_survey_formal.XXXXXX)
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
source_root=${REMOTE_BUNDLES}/hm3d_table3_causal_survey_${receipt_sha:0:16}
source_receipt=${source_root}/SOURCE_BUNDLE.sha256
stamp=$(date -u +%Y%m%dT%H%M%SZ)
run_root=${REMOTE_RESULTS}/formal_${stamp}_${receipt_sha:0:8}

remote "set -euo pipefail
test \"\$(sha256sum '${BASE_WRAPPER_RECEIPT}' | awk '{print \$1}')\" = '${EXPECTED_BASE_WRAPPER_RECEIPT_SHA}'
cd '${BASE_WRAPPER_ROOT}'; sha256sum -c --quiet SOURCE_BUNDLE.sha256
test \"\$(sha256sum '${CANDIDATE_PLAN}' | awk '{print \$1}')\" = '${EXPECTED_CANDIDATE_PLAN_SHA}'"
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

protocol=${source_root}/MemNavData/hm3d_table3_causal_survey_protocol_20260830.json
common="ALL,SURVEY_SOURCE_ROOT=${source_root},SURVEY_SOURCE_RECEIPT=${source_receipt},EXPECTED_SURVEY_SOURCE_RECEIPT_SHA=${receipt_sha},BASE_WRAPPER_ROOT=${BASE_WRAPPER_ROOT},BASE_WRAPPER_RECEIPT=${BASE_WRAPPER_RECEIPT},EXPECTED_BASE_WRAPPER_RECEIPT_SHA=${EXPECTED_BASE_WRAPPER_RECEIPT_SHA},RUN_ROOT=${run_root},CANDIDATE_PLAN=${CANDIDATE_PLAN},EXPECTED_CANDIDATE_PLAN_SHA=${EXPECTED_CANDIDATE_PLAN_SHA},SURVEY_PROTOCOL=${protocol}"
safe=${TASK_ROOT}/MemNavData/slurm_safe_submit.sh
construct=${source_root}/MemNavData/slurm_hm3d_table3_causal_survey_construct.sbatch
population=${source_root}/MemNavData/slurm_hm3d_table3_causal_survey_population.sbatch

remote "source '${safe}'; safe_sbatch --lint-fatal --test-only --qos=gpu48 --time=01:00:00 --array=0-124%4 --export='${common}' '${construct}' >/dev/null"
remote "source '${safe}'; safe_sbatch --lint-fatal --test-only --partition=cpu_short --time=00:20:00 --export='${common},MODE=finalize' '${population}' >/dev/null"
remote "source '${safe}'; safe_sbatch --lint-fatal --test-only --partition=cpu_short --time=00:20:00 --export='${common},MODE=verify' '${population}' >/dev/null"

raw=$(remote "source '${safe}'; safe_sbatch --lint-fatal --parsable --qos=gpu48 --time=01:00:00 --array=0-124%4 --export='${common}' '${construct}'")
construct_job=$(printf '%s\n' "${raw}" | job_id)
[[ "${construct_job}" =~ ^[0-9]+$ ]] || fail "bad construction job id"
raw=$(remote "source '${safe}'; safe_sbatch --lint-fatal --parsable --partition=cpu_short --time=00:20:00 --dependency='afterok:${construct_job}' --kill-on-invalid-dep=yes --export='${common},MODE=finalize' '${population}'")
finalize_job=$(printf '%s\n' "${raw}" | job_id)
[[ "${finalize_job}" =~ ^[0-9]+$ ]] || fail "bad population finalizer job id"
raw=$(remote "source '${safe}'; safe_sbatch --lint-fatal --parsable --partition=cpu_short --time=00:20:00 --dependency='afterok:${finalize_job}' --kill-on-invalid-dep=yes --export='${common},MODE=verify' '${population}'")
verify_job=$(printf '%s\n' "${raw}" | job_id)
[[ "${verify_job}" =~ ^[0-9]+$ ]] || fail "bad population verifier job id"

"${LOCAL_MEMNAV_PY}" - "${OUT_RECEIPT}" "${construct_job}" \
  "${finalize_job}" "${verify_job}" "${run_root}" "${source_root}" \
  "${receipt_sha}" <<'PY'
import json,sys
path,construct,finalize,verify,run,source,source_sha=sys.argv[1:]
payload={
  'schema_version':'hm3d_table3_causal_survey_construction_submission_v1_20260830',
  'construction_array_job':int(construct),
  'population_finalize_job':int(finalize),
  'population_independent_verification_job':int(verify),
  'run_root':run,'source_bundle':source,
  'source_bundle_sha256':source_sha,
  'frozen_candidates':125,
  'population_gate':{'histories_per_bin':16,'scene_clusters_per_bin':10,
                     'maximum_histories_per_scene_per_bin':2},
  'query_policy_jobs_submitted':False,
  'query_policy_outcomes_read_at_submission':False,
  'threshold_relaxation':False,'fallback_completion_allowed':False,
  'history_source':'controlled_causal_rgb_geodesic_survey',
}
open(path,'x').write(json.dumps(payload,indent=2,sort_keys=True)+'\n')
print(json.dumps(payload,indent=2,sort_keys=True))
PY
