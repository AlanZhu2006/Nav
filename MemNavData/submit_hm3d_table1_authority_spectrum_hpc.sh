#!/usr/bin/env bash
# Build, verify, stage, and submit the HM3D Table-I authority spectrum.
set -euo pipefail
umask 0022

ROOT=${ROOT:-/home/asus/Research/Nav-graph-blind}
SSH_ALIAS=${SSH_ALIAS:-alantorch}
EXPECTED_SSH_USER=${EXPECTED_SSH_USER:-yz11502}
CONCURRENCY=${CONCURRENCY:-4}
SUBMIT=${SUBMIT:-1}
SUBMISSION_RECEIPT=${SUBMISSION_RECEIPT:-MemNavData/HM3D_TABLE1_AUTHORITY_SPECTRUM_RETRY2_SUBMISSION_20260904.json}
LOCAL_PY=${LOCAL_PY:-/home/asus/miniconda3/envs/memnav/bin/python}
REMOTE_PY=/scratch/lg154/conda-envs/memnav/bin/python
REMOTE_BUNDLES=/scratch/yz11502/Research/Nav-axis-uturn-source-bundles
REMOTE_RESULTS=/scratch/yz11502/Research/Nav-axis-uturn-results/hm3d_table1_authority_spectrum_20260904

CONSTRUCTION_RUN=/scratch/yz11502/Research/Nav-axis-uturn-results/hm3d_table1_fresh_query_reserve_20260829/construction_20260828T212552Z_bb757914
BENCH_ROOT=${CONSTRUCTION_RUN}/population/natural_direction
CONSTRUCTION_VERIFICATION=${CONSTRUCTION_RUN}/hm3d_table1_fresh_query_verification.json
EXPECTED_CONSTRUCTION_VERIFICATION_SHA=2a7b8f86f61a6f55762640dcbaef4b975539ec3d93cfb06649bddd6fa4c96dc8
EXPECTED_MANIFEST_SHA=f82dbcbc6255219aae94b6d77bffdfa454f36835cf803a70df5cf8616193ad01
PARENT_MANIFEST=/scratch/yz11502/Research/Nav-axis-uturn-results/hm3d_fresh_fullmono_mixed_role_20260820/formal_20260820T143609Z_e6dd44c6/sealed_inputs/parent_manifest.json
EXPECTED_PARENT_MANIFEST_SHA=a96a0b96fab7b7b47709b36cb8eeb9410b42b09f095f87ef01304a68de716dd5

BASE_SOURCE_ROOT=/scratch/yz11502/Research/Nav-axis-uturn-source-bundles/final14_mono_factorial_5690569a4373f2d2
BASE_RECEIPT=${BASE_SOURCE_ROOT}/source_inputs.sha256
EXPECTED_BASE_RECEIPT_SHA=5690569a4373f2d2768671418f0c604c4a03aa4b0ffe01baf70b288af03ba216
SERVER_SOURCE_ROOT=/scratch/yz11502/Research/Nav-axis-uturn-source-bundles/hm3d_table1_navdp_authority_transaction_718661db1733d5de
SERVER_SOURCE_RECEIPT=${SERVER_SOURCE_ROOT}/SOURCE_BUNDLE.sha256
EXPECTED_SERVER_SOURCE_RECEIPT_SHA=718661db1733d5de16cd86687eec880a8d02fc5ae5ca982e1ab7d5bde5e96f7d
RUNTIME_CLOSURE_ROOT=/scratch/yz11502/Research/Nav-axis-uturn-source-bundles/hm3d_table1_navdp_authority_transaction_repair_51ee9a4ca063c7f1
RUNTIME_CLOSURE_RECEIPT=${RUNTIME_CLOSURE_ROOT}/SOURCE_BUNDLE.sha256
EXPECTED_RUNTIME_CLOSURE_RECEIPT_SHA=51ee9a4ca063c7f1125a69dd226918ea9a8bd6404c379df58b28e2b296e68eec

cd "${ROOT}"
fail() { echo "ABORT: $*" >&2; exit 2; }
[[ "${CONCURRENCY}" =~ ^[1-9][0-9]*$ ]] || fail "bad concurrency"
[[ "${SUBMIT}" =~ ^[01]$ ]] || fail "SUBMIT must be 0 or 1"
[[ -x "${LOCAL_PY}" ]] || fail "local memnav Python missing"

SSH_CONTROL_PATH=${SSH_CONTROL_PATH:-$(
  ssh -G "${SSH_ALIAS}" 2>/dev/null |
    awk '$1=="controlpath"{value=$2} END{print value}'
)}
[[ -S "${SSH_CONTROL_PATH}" ]] || fail "shared SSH socket missing"
timeout 15 ssh -O check -S "${SSH_CONTROL_PATH}" "${SSH_ALIAS}" \
  >/dev/null 2>&1 || fail "shared SSH master is not responsive"
remote() {
  timeout 180 ssh -n -tt -o BatchMode=yes -o ControlMaster=no \
    -S "${SSH_CONTROL_PATH}" "${SSH_ALIAS}" "$@"
}
job_id() { tr -d '\r' | awk -F';' '/^[0-9]+(;|$)/ {print $1; exit}'; }

files=(
  MemNavData/HM3D_TABLE1_AUTHORITY_SPECTRUM_PROTOCOL_20260904.md
  MemNavData/hm3d_table1_authority_spectrum_bundle_manifest_20260904.json
  MemNavData/hm3d_table1_authority_spectrum_protocol_20260904.json
  MemNavData/hm3d_table1_authority_spectrum.py
  MemNavData/run_hm3d_table1_authority_spectrum_history.py
  MemNavData/run_hm3d_fullmono_query_history.py
  MemNavData/summarize_hm3d_table1_authority_spectrum.py
  MemNavData/independent_verify_hm3d_table1_authority_spectrum.py
  MemNavData/test_hm3d_table1_authority_spectrum.py
  MemNavData/run_hm3d_fullmono_server_scene.sh
  MemNavData/slurm_port_pair.sh
  MemNavData/test_slurm_port_pair.sh
  MemNavData/slurm_hm3d_table1_authority_spectrum.sbatch
  MemNavData/slurm_hm3d_table1_authority_spectrum_analysis.sbatch
)
for path in "${files[@]}"; do
  [[ -f "${path}" && ! -L "${path}" ]] || fail "missing physical ${path}"
done

export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=${ROOT}:${ROOT}/MemNavData
"${LOCAL_PY}" -m pytest -q -p no:cacheprovider \
  MemNavData/test_hm3d_table1_authority_spectrum.py \
  MemNavData/test_final14_authority_ablation.py
"${LOCAL_PY}" -m py_compile \
  MemNavData/hm3d_table1_authority_spectrum.py \
  MemNavData/run_hm3d_table1_authority_spectrum_history.py \
  MemNavData/summarize_hm3d_table1_authority_spectrum.py \
  MemNavData/independent_verify_hm3d_table1_authority_spectrum.py
"${LOCAL_PY}" -m json.tool \
  MemNavData/hm3d_table1_authority_spectrum_protocol_20260904.json >/dev/null
bash -n \
  MemNavData/run_hm3d_fullmono_server_scene.sh \
  MemNavData/slurm_hm3d_table1_authority_spectrum.sbatch \
  MemNavData/slurm_hm3d_table1_authority_spectrum_analysis.sbatch \
  MemNavData/submit_hm3d_table1_authority_spectrum_hpc.sh
bash MemNavData/test_slurm_port_pair.sh

echo '[gate] shared SSH identity and frozen population'
remote_identity=$(remote 'id -un' | tr -d '\r')
[[ "${remote_identity}" == "${EXPECTED_SSH_USER}" ]] || \
  fail "wrong remote identity: ${remote_identity:-unavailable}"
remote "set -euo pipefail
test \"\$(sha256sum '${CONSTRUCTION_VERIFICATION}' | awk '{print \$1}')\" = '${EXPECTED_CONSTRUCTION_VERIFICATION_SHA}'
test \"\$(sha256sum '${BENCH_ROOT}/manifest.json' | awk '{print \$1}')\" = '${EXPECTED_MANIFEST_SHA}'
test \"\$(sha256sum '${PARENT_MANIFEST}' | awk '{print \$1}')\" = '${EXPECTED_PARENT_MANIFEST_SHA}'
python - '${CONSTRUCTION_VERIFICATION}' <<'PY'
import json,sys
x=json.load(open(sys.argv[1]))
assert x['verified'] is True and x['formal_policy_evaluation_authorized'] is True
assert x['histories']==28 and x['scene_clusters']==21 and x['queries']==56
assert x['policy_outcomes_read'] is False
PY
for spec in \
 '${BASE_SOURCE_ROOT}:${BASE_RECEIPT}:${EXPECTED_BASE_RECEIPT_SHA}' \
 '${SERVER_SOURCE_ROOT}:${SERVER_SOURCE_RECEIPT}:${EXPECTED_SERVER_SOURCE_RECEIPT_SHA}' \
 '${RUNTIME_CLOSURE_ROOT}:${RUNTIME_CLOSURE_RECEIPT}:${EXPECTED_RUNTIME_CLOSURE_RECEIPT_SHA}'; do
 IFS=: read -r root receipt expected <<<\"\${spec}\"
 test \"\$(sha256sum \"\${receipt}\" | awk '{print \$1}')\" = \"\${expected}\"
 (cd \"\${root}\" && sha256sum -c --quiet \"\${receipt}\")
done"

staging=$(mktemp -d /tmp/h3_authority_spectrum.XXXXXX)
cleanup() { rm -rf -- "${staging}"; }
trap cleanup EXIT
for path in "${files[@]}"; do
  mkdir -p "${staging}/$(dirname "${path}")"
  cp --preserve=mode,timestamps "${path}" "${staging}/${path}"
done
cp "${staging}/MemNavData/hm3d_table1_authority_spectrum_bundle_manifest_20260904.json" \
  "${staging}/source_bundle_manifest.json"
(
  cd "${staging}"
  find . -type f ! -name SOURCE_BUNDLE.sha256 -print0 | sort -z | \
    xargs -0 sha256sum >SOURCE_BUNDLE.sha256
  sha256sum -c --quiet SOURCE_BUNDLE.sha256
)
task_sha=$(sha256sum "${staging}/SOURCE_BUNDLE.sha256" | awk '{print $1}')
task_root=${REMOTE_BUNDLES}/hm3d_table1_authority_spectrum_${task_sha:0:16}
task_stage=${task_root}.partial.$$
run_root=${REMOTE_RESULTS}/formal_${task_sha:0:16}

echo '[stage] immutable task bundle'
if remote "test -d '${task_root}'"; then
  remote "test \"\$(sha256sum '${task_root}/SOURCE_BUNDLE.sha256' | awk '{print \$1}')\" = '${task_sha}' && cd '${task_root}' && sha256sum -c --quiet SOURCE_BUNDLE.sha256"
else
  remote "test ! -e '${task_stage}' && mkdir -p '${task_stage}'"
  timeout 240 rsync -a --partial \
    --chmod=Du=rwx,Dgo=rx,Fu=rw,Fgo=r \
    -e "ssh -o BatchMode=yes -o ControlMaster=no -S ${SSH_CONTROL_PATH}" \
    "${staging}/" "${SSH_ALIAS}:${task_stage}/"
  remote "cd '${task_stage}' && sha256sum -c --quiet SOURCE_BUNDLE.sha256 && chmod -R a-w '${task_stage}' && mv '${task_stage}' '${task_root}'"
fi
task_receipt=${task_root}/SOURCE_BUNDLE.sha256

echo '[gate] remote imports, route dry-runs, and Slurm allocation'
remote "PYTHONDONTWRITEBYTECODE=1 PYTHONPATH='${task_root}:${task_root}/MemNavData:${RUNTIME_CLOSURE_ROOT}:${RUNTIME_CLOSURE_ROOT}/MemNavData' '${REMOTE_PY}' -m pytest -q -p no:cacheprovider '${task_root}/MemNavData/test_hm3d_table1_authority_spectrum.py'"
remote "PYTHONDONTWRITEBYTECODE=1 PYTHONPATH='${task_root}:${task_root}/MemNavData:${RUNTIME_CLOSURE_ROOT}:${RUNTIME_CLOSURE_ROOT}/MemNavData' '${REMOTE_PY}' -c 'from pathlib import Path; import MemNavData.run_hm3d_table1_authority_spectrum_history as x; import MemNavData.run_hm3d_fullmono_query_history as b; assert x.SCHEMA.endswith(\"20260904\"); assert hasattr(b, \"SCHEMAS\") and \"goal_a\" in b.SCHEMAS; assert Path(b.__file__).resolve() == Path(\"${task_root}/MemNavData/run_hm3d_fullmono_query_history.py\").resolve()'"

common="ALL,TASK_ROOT=${task_root},TASK_RECEIPT=${task_receipt},EXPECTED_TASK_RECEIPT_SHA=${task_sha},BASE_SOURCE_ROOT=${BASE_SOURCE_ROOT},BASE_RECEIPT=${BASE_RECEIPT},EXPECTED_BASE_RECEIPT_SHA=${EXPECTED_BASE_RECEIPT_SHA},SERVER_SOURCE_ROOT=${SERVER_SOURCE_ROOT},SERVER_SOURCE_RECEIPT=${SERVER_SOURCE_RECEIPT},EXPECTED_SERVER_SOURCE_RECEIPT_SHA=${EXPECTED_SERVER_SOURCE_RECEIPT_SHA},RUNTIME_CLOSURE_ROOT=${RUNTIME_CLOSURE_ROOT},RUNTIME_CLOSURE_RECEIPT=${RUNTIME_CLOSURE_RECEIPT},EXPECTED_RUNTIME_CLOSURE_RECEIPT_SHA=${EXPECTED_RUNTIME_CLOSURE_RECEIPT_SHA},BENCH_ROOT=${BENCH_ROOT},PARENT_MANIFEST=${PARENT_MANIFEST},FORMAL_RUN_ROOT=${run_root},CONSTRUCTION_VERIFICATION=${CONSTRUCTION_VERIFICATION},EXPECTED_CONSTRUCTION_VERIFICATION_SHA=${EXPECTED_CONSTRUCTION_VERIFICATION_SHA},EXPECTED_MANIFEST_SHA=${EXPECTED_MANIFEST_SHA}"
gpu_script=${task_root}/MemNavData/slurm_hm3d_table1_authority_spectrum.sbatch
analysis_script=${task_root}/MemNavData/slurm_hm3d_table1_authority_spectrum_analysis.sbatch
remote "sbatch --test-only --array=0 --export='${common},PHASE=smoke' '${gpu_script}' >/dev/null"
remote "sbatch --test-only --array=0-27%${CONCURRENCY} --export='${common},PHASE=formal' '${gpu_script}' >/dev/null"
remote "sbatch --test-only --export='${common}' '${analysis_script}' >/dev/null"

if [[ "${SUBMIT}" == 0 ]]; then
  printf 'PREPARED_ONLY=1\nTASK_ROOT=%s\nRUN_ROOT=%s\n' "${task_root}" "${run_root}"
  exit 0
fi

echo '[submit] smoke -> formal history array -> independent analysis'
remote "test ! -e '${run_root}' && mkdir -p '${run_root}/sealed_inputs' /scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs
cp '${BENCH_ROOT}/manifest.json' '${CONSTRUCTION_VERIFICATION}' '${PARENT_MANIFEST}' '${task_root}/MemNavData/HM3D_TABLE1_AUTHORITY_SPECTRUM_PROTOCOL_20260904.md' '${task_root}/MemNavData/hm3d_table1_authority_spectrum_protocol_20260904.json' '${run_root}/sealed_inputs/'
sha256sum '${run_root}/sealed_inputs/'* >'${run_root}/sealed_inputs/inputs.sha256'
chmod -R a-w '${run_root}/sealed_inputs'"
smoke_job=$(remote "sbatch --parsable --array=0 --export='${common},PHASE=smoke' '${gpu_script}'" | job_id)
formal_job=$(remote "sbatch --parsable --array=0-27%${CONCURRENCY} --dependency=afterok:${smoke_job} --kill-on-invalid-dep=yes --export='${common},PHASE=formal' '${gpu_script}'" | job_id)
analysis_job=$(remote "sbatch --parsable --dependency=afterok:${formal_job} --kill-on-invalid-dep=yes --export='${common}' '${analysis_script}'" | job_id)
for value in "${smoke_job}" "${formal_job}" "${analysis_job}"; do
  [[ "${value}" =~ ^[0-9]+$ ]] || fail "invalid Slurm job id: ${value}"
done

receipt=${SUBMISSION_RECEIPT}
[[ ! -e "${receipt}" ]] || fail "submission receipt already exists"
"${LOCAL_PY}" - "${receipt}" "${run_root}" "${task_root}" "${task_sha}" \
  "${smoke_job}" "${formal_job}" "${analysis_job}" <<'PY'
import json,sys
path,run,bundle,digest,smoke,formal,analysis=sys.argv[1:]
payload={
 "schema_version":"hm3d_table1_authority_spectrum_submission_v1_20260904",
 "run_root":run,"task_bundle":bundle,"task_receipt_sha256":digest,
 "histories":28,"scene_clusters":21,"queries_per_arm":56,
 "fresh_confirmation":False,
 "native_and_strict_cec_outcomes_known_before_design":True,
 "raw_and_finite_witness_outcomes_known_before_submission":False,
 "supersedes_failed_jobs":{
   "attempt0":{"smoke":16923257,"formal_array":16923258,
               "analysis":16923259},
   "attempt1":{"smoke":16924414,"formal_array":16924419,
               "analysis":16924421}},
 "retry_reason":"bundle-local fullmono runner pinned after legacy namespace fallback lacked the SCHEMAS contract",
 "jobs":{"smoke":int(smoke),"formal_array":int(formal),"analysis":int(analysis)},
}
open(path,"x").write(json.dumps(payload,indent=2,sort_keys=True)+"\n")
PY
printf 'SUBMITTED=1\nSMOKE_JOB=%s\nFORMAL_JOB=%s\nANALYSIS_JOB=%s\nRUN_ROOT=%s\n' \
  "${smoke_job}" "${formal_job}" "${analysis_job}" "${run_root}"
