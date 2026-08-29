#!/usr/bin/env bash
# Additively repair the result-blind NavDP branch after the transaction server
# overlay exposed an incomplete import closure.  The frozen population, policy,
# checkpoints, thresholds, seeds, budgets, and ViNT jobs are not resubmitted.
set -euo pipefail
umask 0022

ROOT=${ROOT:-/home/asus/Research/Nav-graph-blind}
SSH_ALIAS=${SSH_ALIAS:-alantorch}
LOCAL_MEMNAV_PY=/home/asus/miniconda3/envs/memnav/bin/python
LOCAL_HAB_PY=/home/asus/miniconda3/envs/habitat/bin/python
REMOTE_BUNDLES=/scratch/yz11502/Research/Nav-axis-uturn-source-bundles
FORMAL_RUN_ROOT=/scratch/yz11502/Research/Nav-axis-uturn-results/hm3d_table1_controller_portability_20260829/formal_20260828T231109Z
CONSTRUCTION_RUN=/scratch/yz11502/Research/Nav-axis-uturn-results/hm3d_table1_fresh_query_reserve_20260829/construction_20260828T212552Z_bb757914
SOURCE_RUN_ROOT=/scratch/yz11502/Research/Nav-axis-uturn-results/hm3d_fresh_fullmono_mixed_role_20260820/formal_20260820T143609Z_e6dd44c6
NAVDP_BASE_SOURCE_ROOT=/scratch/yz11502/Research/Nav-axis-uturn-source-bundles/final14_mono_factorial_5690569a4373f2d2
NAVDP_BASE_RECEIPT=${NAVDP_BASE_SOURCE_ROOT}/source_inputs.sha256
EXPECTED_NAVDP_BASE_RECEIPT_SHA=5690569a4373f2d2768671418f0c604c4a03aa4b0ffe01baf70b288af03ba216
NAVDP_SERVER_SOURCE_ROOT=/scratch/yz11502/Research/Nav-axis-uturn-source-bundles/hm3d_fullmono_transaction_repair_67e1132783ce2cb1
NAVDP_SERVER_RECEIPT=${NAVDP_SERVER_SOURCE_ROOT}/SOURCE_BUNDLE.sha256
EXPECTED_NAVDP_SERVER_RECEIPT_SHA=05ce401aac8c2e7e31e8a8d820613d30b3a03856a35c8750085b93d5a1539a97
ORIGINAL_SUBMISSION=${FORMAL_RUN_ROOT}/submission.json
EXPECTED_ORIGINAL_SUBMISSION_SHA=e5693544f3431150a4162ccfb854ea001671bbfeeeaef21c5911d9239bf2b407
PREVIOUS_REPAIR_SUBMISSION=${FORMAL_RUN_ROOT}/repairs/navdp_transaction_v1/submission.json
EXPECTED_PREVIOUS_REPAIR_SUBMISSION_SHA=a5b19eaab1a76eae0e1c2ac4f71305b12a34cf626cdee5816b8c081dcaaf7f86
OLD_NAVDP_SMOKE_JOB=16527714
VINT_VERIFY_JOB=16526759
NAVDP_CONCURRENCY=${NAVDP_CONCURRENCY:-2}
REPAIR_KEY=navdp_transaction_v2
REPAIR_ROOT=${FORMAL_RUN_ROOT}/repairs/${REPAIR_KEY}
NAVDP_SMOKE_RUN_ROOT=${FORMAL_RUN_ROOT}/smoke/${REPAIR_KEY}
HAB_REQUESTS_VENDOR=/scratch/lg154/conda-envs/habitat/lib/python3.9/site-packages/pip/_vendor
SSH_CONTROL_PATH=${SSH_CONTROL_PATH:-$(ssh -G "${SSH_ALIAS}" 2>/dev/null | awk '$1=="controlpath"{value=$2} END{print value}')}
cd "${ROOT}"

fail() { echo "ABORT: $*" >&2; exit 2; }
remote() {
  timeout 180 ssh -n -tt -o BatchMode=yes -o ControlMaster=no \
    -S "${SSH_CONTROL_PATH}" "${SSH_ALIAS}" "$@"
}

[[ -S "${SSH_CONTROL_PATH}" ]] || fail "authoritative SSH master missing"
[[ "${NAVDP_CONCURRENCY}" =~ ^[1-9][0-9]*$ ]] || \
  fail "invalid NavDP concurrency"

required=(
  MemNavData/HM3D_TABLE1_CONTROLLER_PORTABILITY_PROTOCOL_20260829.md
  MemNavData/HM3D_TABLE1_NAVDP_TRANSACTION_REPAIR_20260829.md
  MemNavData/hm3d_table1_controller_portability_protocol_20260829.json
  MemNavData/hm3d_fullmono_mixed_role_protocol_20260820.json
  MemNavData/run_hm3d_fullmono_query_history.py
  MemNavData/run_hm3d_fullmono_server_scene.sh
  MemNavData/slurm_hm3d_table1_navdp_pair.sbatch
  MemNavData/slurm_hm3d_table1_navdp_analysis.sbatch
  MemNavData/slurm_hm3d_table1_controller_seal.sbatch
  MemNavData/aggregate_hm3d_table1_navdp_pair.py
  MemNavData/independent_verify_hm3d_table1_navdp_pair.py
  MemNavData/test_hm3d_table1_navdp_pair.py
  MemNavData/test_hm3d_table1_navdp_transport_contract.py
  MemNavData/submit_hm3d_table1_navdp_transport_repair_hpc.sh
)
for path in "${required[@]}"; do
  [[ -f "${path}" && ! -L "${path}" ]] || fail "missing physical ${path}"
done

export PYTHONPATH=${ROOT}:${ROOT}/MemNavData${PYTHONPATH:+:${PYTHONPATH}}
"${LOCAL_HAB_PY}" -m py_compile \
  MemNavData/run_hm3d_fullmono_query_history.py \
  MemNavData/eval_shared_online_role_pairs.py \
  MemNavData/eval_2leg_habitat.py
"${LOCAL_MEMNAV_PY}" -m py_compile \
  MemNavData/aggregate_hm3d_table1_navdp_pair.py \
  MemNavData/independent_verify_hm3d_table1_navdp_pair.py
"${LOCAL_MEMNAV_PY}" -m pytest -q \
  MemNavData/test_hm3d_table1_navdp_pair.py \
  MemNavData/test_hm3d_table1_navdp_transport_contract.py \
  MemNavData/test_hm3d_fullmono_mixed_role.py \
  MemNavData/test_monocular_depth_runtime.py
bash -n \
  MemNavData/run_hm3d_fullmono_server_scene.sh \
  MemNavData/slurm_hm3d_table1_navdp_pair.sbatch \
  MemNavData/slurm_hm3d_table1_navdp_analysis.sbatch \
  MemNavData/slurm_hm3d_table1_controller_seal.sbatch \
  MemNavData/submit_hm3d_table1_navdp_transport_repair_hpc.sh
source MemNavData/slurm_safe_submit.sh
for script in \
  MemNavData/slurm_hm3d_table1_navdp_pair.sbatch \
  MemNavData/slurm_hm3d_table1_navdp_analysis.sbatch \
  MemNavData/slurm_hm3d_table1_controller_seal.sbatch; do
  lint_sbatch_template "${script}" || fail "sbatch lint failed: ${script}"
done

construction_verification=${CONSTRUCTION_RUN}/hm3d_table1_fresh_query_verification.json
bench_root=${CONSTRUCTION_RUN}/population/natural_direction
parent_manifest=${SOURCE_RUN_ROOT}/sealed_inputs/parent_manifest.json
old_eval_log=${FORMAL_RUN_ROOT}/smoke/navdp_transaction_v1/runtime/smoke_0/logs/server_navdp.log
final_navdp_root=${FORMAL_RUN_ROOT}/formal/navdp
final_receipt=${FORMAL_RUN_ROOT}/hm3d_table1_controller_portability_receipt.json

echo '[gate] result-blind failure classification and frozen inputs'
readarray -t gate < <(remote "python - '${construction_verification}' '${bench_root}/manifest.json' '${ORIGINAL_SUBMISSION}' <<'PY'
import hashlib,json,sys
verification=json.load(open(sys.argv[1])); manifest=sys.argv[2]
submission=json.load(open(sys.argv[3]))
if verification.get('verified') is not True:
 raise SystemExit('construction verifier did not pass')
if verification.get('formal_policy_evaluation_authorized') is not True:
 raise SystemExit('construction evaluation authorization missing')
digest=hashlib.sha256(open(manifest,'rb').read()).hexdigest()
if digest != verification.get('benchmark_manifest_sha256'):
 raise SystemExit('verified benchmark manifest changed')
if submission.get('partial_policy_outcomes_read_at_submission') is not False:
 raise SystemExit('original result-blind receipt is invalid')
print(verification['histories']); print(verification['scene_clusters'])
print(digest); print(hashlib.sha256(open(sys.argv[1],'rb').read()).hexdigest())
PY" | tr -d '\r')
[[ "${#gate[@]}" -eq 4 ]] || fail "construction gate receipt incomplete"
histories=${gate[0]}; scenes=${gate[1]}
manifest_sha=${gate[2]}; construction_verification_sha=${gate[3]}
[[ "${histories}" == 28 && "${scenes}" == 21 ]] || \
  fail "frozen denominator changed"

remote "set -euo pipefail
test \"\$(sha256sum '${ORIGINAL_SUBMISSION}' | awk '{print \$1}')\" = '${EXPECTED_ORIGINAL_SUBMISSION_SHA}'
test \"\$(sha256sum '${PREVIOUS_REPAIR_SUBMISSION}' | awk '{print \$1}')\" = '${EXPECTED_PREVIOUS_REPAIR_SUBMISSION_SHA}'
test \"\$(sha256sum '${NAVDP_BASE_RECEIPT}' | awk '{print \$1}')\" = '${EXPECTED_NAVDP_BASE_RECEIPT_SHA}'
cd '${NAVDP_BASE_SOURCE_ROOT}' && sha256sum -c --quiet '${NAVDP_BASE_RECEIPT}'
test \"\$(sha256sum '${NAVDP_SERVER_RECEIPT}' | awk '{print \$1}')\" = '${EXPECTED_NAVDP_SERVER_RECEIPT_SHA}'
cd '${NAVDP_SERVER_SOURCE_ROOT}' && sha256sum -c --quiet '${NAVDP_SERVER_RECEIPT}'
grep -q \"No module named 'depth_anything'\" '${old_eval_log}'
test ! -e '${final_navdp_root}'
test ! -e '${NAVDP_SMOKE_RUN_ROOT}'
test ! -e '${REPAIR_ROOT}'
test ! -e '${final_receipt}'
test \"\$(sha256sum '${parent_manifest}' | awk '{print \$1}')\" = a96a0b96fab7b7b47709b36cb8eeb9410b42b09f095f87ef01304a68de716dd5"

vint_state=$(remote "sacct -X -n -j '${VINT_VERIFY_JOB}' --format=State | awk 'NF {print \$1; exit}'" | tr -d '\r')
case "${vint_state}" in
  PENDING|RUNNING|COMPLETED) ;;
  *) fail "ViNT verifier job is not a valid afterok dependency: ${vint_state}" ;;
esac
old_navdp_state=$(remote "sacct -X -n -j '${OLD_NAVDP_SMOKE_JOB}' --format=State | awk 'NF {print \$1; exit}'" | tr -d '\r')
[[ "${old_navdp_state}" == FAILED ]] || \
  fail "old NavDP smoke no longer has the audited failed state"

echo '[stage] immutable repair evaluator bundle'
staging=$(mktemp -d)
cleanup() { rm -rf -- "${staging}"; }
trap cleanup EXIT
mkdir -p "${staging}/MemNavData"
while IFS= read -r -d '' path; do
  cp --preserve=mode,timestamps "${path}" \
    "${staging}/MemNavData/$(basename "${path}")"
done < <(find "${ROOT}/MemNavData" -maxdepth 1 -type f -name '*.py' -print0)
for path in "${required[@]}"; do
  case "${path}" in
    *.py) ;;
    *) cp --preserve=mode,timestamps "${path}" \
         "${staging}/MemNavData/$(basename "${path}")" ;;
  esac
done
local_head=$(git -C "${ROOT}" rev-parse HEAD)
"${LOCAL_MEMNAV_PY}" - "${staging}" "${local_head}" "${manifest_sha}" \
  "${construction_verification_sha}" "${EXPECTED_ORIGINAL_SUBMISSION_SHA}" \
  "${EXPECTED_PREVIOUS_REPAIR_SUBMISSION_SHA}" <<'PY'
import hashlib,json,sys
root,head,manifest,verification,original,previous=sys.argv[1:]
base=__import__('pathlib').Path(root); files={}
for path in sorted(base.rglob('*')):
 if path.is_symlink(): raise SystemExit('bundle symlink: '+str(path))
 if path.is_file() and path.name not in {'SOURCE_BUNDLE.sha256','source_bundle_manifest.json'}:
  files[path.relative_to(base).as_posix()]=hashlib.sha256(path.read_bytes()).hexdigest()
payload={
 'schema_version':'hm3d_table1_navdp_transport_repair_bundle_v2_20260829',
 'local_git_head_context':head,
 'benchmark_manifest_sha256':manifest,
 'construction_verification_sha256':verification,
 'original_submission_sha256':original,
 'previous_repair_submission_sha256':previous,
 'repair_scope':'server_overlay_dependency_closure_only',
 'server_overlay_receipt_sha256':'05ce401aac8c2e7e31e8a8d820613d30b3a03856a35c8750085b93d5a1539a97',
 'scientific_factors_changed':False,
 'partial_policy_outcomes_read':False,
 'files':files,
}
(base/'source_bundle_manifest.json').write_text(json.dumps(payload,indent=2,sort_keys=True)+'\n')
PY
(
  cd "${staging}"
  find . -type f ! -name SOURCE_BUNDLE.sha256 -print0 | sort -z | \
    xargs -0 sha256sum >SOURCE_BUNDLE.sha256
  sha256sum -c SOURCE_BUNDLE.sha256 >/dev/null
)
task_receipt_sha=$(sha256sum "${staging}/SOURCE_BUNDLE.sha256" | awk '{print $1}')
task_root=${REMOTE_BUNDLES}/hm3d_table1_navdp_transport_repair_${task_receipt_sha:0:16}
task_stage=${task_root}.partial.$$

if remote "test -d '${task_root}'"; then
  remote "test \"\$(sha256sum '${task_root}/SOURCE_BUNDLE.sha256' | awk '{print \$1}')\" = '${task_receipt_sha}' && cd '${task_root}' && sha256sum -c --quiet SOURCE_BUNDLE.sha256"
else
  remote "test ! -e '${task_stage}' && mkdir -p '${task_stage}'"
  rsync -a --chmod=Du=rwx,Dgo=rx,Fu=rw,Fgo=r \
    -e "ssh -o BatchMode=yes -o ControlMaster=no -S ${SSH_CONTROL_PATH}" \
    "${staging}/" "${SSH_ALIAS}:${task_stage}/"
  remote "cd '${task_stage}' && sha256sum -c --quiet SOURCE_BUNDLE.sha256 && chmod -R a-w '${task_stage}' && mv '${task_stage}' '${task_root}'"
fi

task_receipt=${task_root}/SOURCE_BUNDLE.sha256
protocol=${task_root}/MemNavData/hm3d_fullmono_mixed_role_protocol_20260820.json
common="ALL,TASK_ROOT=${task_root},TASK_RECEIPT=${task_receipt},EXPECTED_TASK_RECEIPT_SHA=${task_receipt_sha},FORMAL_RUN_ROOT=${FORMAL_RUN_ROOT},BENCH_ROOT=${bench_root},CONSTRUCTION_VERIFICATION=${construction_verification},EXPECTED_CONSTRUCTION_VERIFICATION_SHA=${construction_verification_sha}"
nav_common="${common},BASE_SOURCE_ROOT=${NAVDP_BASE_SOURCE_ROOT},BASE_RECEIPT=${NAVDP_BASE_RECEIPT},EXPECTED_BASE_RECEIPT_SHA=${EXPECTED_NAVDP_BASE_RECEIPT_SHA},SERVER_SOURCE_ROOT=${NAVDP_SERVER_SOURCE_ROOT},SERVER_SOURCE_RECEIPT=${NAVDP_SERVER_RECEIPT},EXPECTED_SERVER_SOURCE_RECEIPT_SHA=${EXPECTED_NAVDP_SERVER_RECEIPT_SHA},SOURCE_RUN_ROOT=${SOURCE_RUN_ROOT},PARENT_MANIFEST=${parent_manifest},PROTOCOL=${protocol},NAVDP_SMOKE_RUN_ROOT=${NAVDP_SMOKE_RUN_ROOT}"
nav_pair=${task_root}/MemNavData/slurm_hm3d_table1_navdp_pair.sbatch
nav_analysis=${task_root}/MemNavData/slurm_hm3d_table1_navdp_analysis.sbatch
seal=${task_root}/MemNavData/slurm_hm3d_table1_controller_seal.sbatch

echo '[gate] runtime interface and Slurm test-only'
remote "test -r '${HAB_REQUESTS_VENDOR}/requests/__init__.py' && singularity exec --nv -B /scratch/lg154 -B /scratch/yz11502 /share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH='${task_root}:${task_root}/MemNavData:${NAVDP_SERVER_SOURCE_ROOT}:${NAVDP_SERVER_SOURCE_ROOT}/MemNavData:${NAVDP_BASE_SOURCE_ROOT}:${NAVDP_BASE_SOURCE_ROOT}/MemNavData:${HAB_REQUESTS_VENDOR}' /scratch/lg154/conda-envs/habitat/bin/python -c 'from pathlib import Path; [compile(Path(path).read_bytes(),path,\"exec\") for path in (\"${task_root}/MemNavData/eval_shared_online_role_pairs.py\",\"${task_root}/MemNavData/eval_2leg_habitat.py\",\"${NAVDP_SERVER_SOURCE_ROOT}/NavDP/baselines/memnav/memnav_server.py\",\"${NAVDP_SERVER_SOURCE_ROOT}/NavDP/baselines/navdp/navdp_server.py\")]' && grep -q 'def append_request_frame' '${NAVDP_SERVER_SOURCE_ROOT}/NavDP/baselines/memnav/memnav_server.py' && grep -q 'require_monocular_depth_transaction' '${NAVDP_SERVER_SOURCE_ROOT}/NavDP/baselines/navdp/navdp_server.py'"
remote "singularity exec --nv -B /scratch/lg154 -B /scratch/yz11502 /share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH='${NAVDP_SERVER_SOURCE_ROOT}/NavDP/baselines/navdp:${NAVDP_BASE_SOURCE_ROOT}/NavDP/baselines/navdp:${task_root}:${task_root}/MemNavData:${NAVDP_SERVER_SOURCE_ROOT}:${NAVDP_SERVER_SOURCE_ROOT}/MemNavData:${NAVDP_BASE_SOURCE_ROOT}:${NAVDP_BASE_SOURCE_ROOT}/MemNavData' /scratch/lg154/conda-envs/memnav/bin/python -c 'import policy_backbone; from depth_anything.depth_anything_v2.dpt import DepthAnythingV2'"
remote "sbatch --test-only --array=0 --export='${nav_common},PHASE=smoke' '${nav_pair}' >/dev/null"
remote "sbatch --test-only --array=0-53%${NAVDP_CONCURRENCY} --export='${nav_common},PHASE=formal' '${nav_pair}' >/dev/null"
remote "sbatch --test-only --export='${common},MODE=aggregate' '${nav_analysis}' >/dev/null"
remote "sbatch --test-only --export='${common},MODE=verify' '${nav_analysis}' >/dev/null"
remote "sbatch --test-only --export='${common}' '${seal}' >/dev/null"

remote "mkdir -p '${REPAIR_ROOT}'"
echo '[submit] additive NavDP repair chain; existing ViNT chain is untouched'
nav_smoke_raw=$(remote "sbatch --parsable --array=0 --export='${nav_common},PHASE=smoke' '${nav_pair}'" | tr -d '\r')
nav_smoke=${nav_smoke_raw%%;*}
nav_formal_raw=$(remote "sbatch --parsable --array=0-53%${NAVDP_CONCURRENCY} --dependency=afterok:${nav_smoke} --kill-on-invalid-dep=yes --export='${nav_common},PHASE=formal' '${nav_pair}'" | tr -d '\r')
nav_formal=${nav_formal_raw%%;*}
nav_aggregate_raw=$(remote "sbatch --parsable --dependency=afterok:${nav_formal} --kill-on-invalid-dep=yes --export='${common},MODE=aggregate' '${nav_analysis}'" | tr -d '\r')
nav_aggregate=${nav_aggregate_raw%%;*}
nav_verify_raw=$(remote "sbatch --parsable --dependency=afterok:${nav_aggregate} --kill-on-invalid-dep=yes --export='${common},MODE=verify' '${nav_analysis}'" | tr -d '\r')
nav_verify=${nav_verify_raw%%;*}
seal_raw=$(remote "sbatch --parsable --dependency=afterok:${nav_verify}:${VINT_VERIFY_JOB} --kill-on-invalid-dep=yes --export='${common}' '${seal}'" | tr -d '\r')
seal_job=${seal_raw%%;*}
for id in "${nav_smoke}" "${nav_formal}" "${nav_aggregate}" \
          "${nav_verify}" "${seal_job}"; do
  [[ "${id}" =~ ^[0-9]+$ ]] || fail "invalid submitted job id: ${id}"
done

receipt=MemNavData/HM3D_TABLE1_NAVDP_TRANSACTION_REPAIR2_SUBMISSION_20260829.json
[[ ! -e "${receipt}" ]] || fail "local repair receipt already exists"
"${LOCAL_MEMNAV_PY}" - "${receipt}" "${FORMAL_RUN_ROOT}" "${REPAIR_ROOT}" \
  "${task_root}" "${task_receipt_sha}" "${construction_verification_sha}" \
  "${manifest_sha}" "${nav_smoke}" "${nav_formal}" "${nav_aggregate}" \
  "${nav_verify}" "${VINT_VERIFY_JOB}" "${seal_job}" <<'PY'
import json,sys
(path,run,repair,bundle,bundle_sha,construction_sha,manifest_sha,
 nav_smoke,nav_formal,nav_aggregate,nav_verify,vint_verify,seal)=sys.argv[1:]
payload={
 'schema_version':'hm3d_table1_navdp_transport_repair_submission_v2_20260829',
 'run_root':run,'repair_root':repair,'task_bundle':bundle,
 'task_receipt_sha256':bundle_sha,
 'construction_verification_sha256':construction_sha,
 'benchmark_manifest_sha256':manifest_sha,
 'original_submission_sha256':'e5693544f3431150a4162ccfb854ea001671bbfeeeaef21c5911d9239bf2b407',
 'previous_repair_submission_sha256':'a5b19eaab1a76eae0e1c2ac4f71305b12a34cf626cdee5816b8c081dcaaf7f86',
 'superseded_navdp_smoke_job':16527714,
 'failure_class':'server_overlay_dependency_closure_missing_depth_anything',
 'server_overlay_receipt_sha256':'05ce401aac8c2e7e31e8a8d820613d30b3a03856a35c8750085b93d5a1539a97',
 'partial_policy_outcomes_read_before_repair':False,
 'scientific_guards':{
  'population_changed':False,'checkpoint_changed':False,
  'threshold_seed_budget_or_controller_changed':False,
  'vint_branch_resubmitted':False,
 },
 'jobs':{
  'navdp_smoke':int(nav_smoke),'navdp_formal':int(nav_formal),
  'navdp_aggregate':int(nav_aggregate),'navdp_verify':int(nav_verify),
  'existing_vint_verify':int(vint_verify),
  'replacement_controller_portability_seal':int(seal),
 },
}
open(path,'x').write(json.dumps(payload,indent=2,sort_keys=True)+'\n')
print(json.dumps(payload,indent=2,sort_keys=True))
PY
scp -q -o BatchMode=yes -o ControlMaster=no -o ControlPath="${SSH_CONTROL_PATH}" \
  "${ROOT}/${receipt}" "${SSH_ALIAS}:${REPAIR_ROOT}/submission.json"
remote "sha256sum '${REPAIR_ROOT}/submission.json' >'${REPAIR_ROOT}/submission.json.sha256' && chmod a-w '${REPAIR_ROOT}/submission.json' '${REPAIR_ROOT}/submission.json.sha256'"
printf 'FORMAL_RUN_ROOT=%s\nREPAIR_ROOT=%s\nTASK_ROOT=%s\nNAVDP_SMOKE=%s\nNAVDP_FORMAL=%s\nNAVDP_VERIFY=%s\nVINT_VERIFY=%s\nSEAL=%s\n' \
  "${FORMAL_RUN_ROOT}" "${REPAIR_ROOT}" "${task_root}" "${nav_smoke}" \
  "${nav_formal}" "${nav_verify}" "${VINT_VERIFY_JOB}" "${seal_job}"
