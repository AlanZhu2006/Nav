#!/usr/bin/env bash
# Freeze source receipts and submit construction only; no controller rollout.
set -euo pipefail
umask 0022

ROOT=${ROOT:-/home/asus/Research/Nav-graph-blind}
SSH_ALIAS=${SSH_ALIAS:-alantorch}
LOCAL_MEMNAV_PY=/home/asus/miniconda3/envs/memnav/bin/python
LOCAL_HAB_PY=/home/asus/miniconda3/envs/habitat/bin/python
REMOTE_BUNDLES=/scratch/yz11502/Research/Nav-axis-uturn-source-bundles
REMOTE_RESULTS=/scratch/yz11502/Research/Nav-axis-uturn-results/mp3d_table1_new_query_20260829
SOURCE_RUN_ROOT=/scratch/yz11502/Research/Nav-axis-uturn-results/mdtec_monocular_cec_composition_20260819/formal_20260819T055600Z_624f9fa9
SOURCE_MANIFEST=/scratch/yz11502/Research/Nav-axis-uturn-results/revisit_fresh_confirmation_20260811/fresh160_v3_attempt600_20260811T2000/data_manifest.json
EXPECTED_SOURCE_MANIFEST_SHA=8013fa2a768d84638a9f9ecc50df46dda67ebb79250d14ad0a8087ac52fd33e5
CONSTRUCT_CONCURRENCY=${CONSTRUCT_CONCURRENCY:-2}
RUN_TAG=${RUN_TAG:-construction_$(date -u +%Y%m%dT%H%M%SZ)}
SSH_CONTROL_PATH=${SSH_CONTROL_PATH:-$(ssh -G "${SSH_ALIAS}" 2>/dev/null | awk '$1=="controlpath"{value=$2} END{print value}')}
cd "${ROOT}"

fail() { echo "ABORT: $*" >&2; exit 2; }
remote() {
  timeout 180 ssh -n -tt -o BatchMode=yes -o ControlMaster=no \
    -S "${SSH_CONTROL_PATH}" "${SSH_ALIAS}" "$@"
}
[[ -S "${SSH_CONTROL_PATH}" ]] || fail "authoritative SSH master missing"
[[ "${CONSTRUCT_CONCURRENCY}" =~ ^[1-9][0-9]*$ ]] || \
  fail "invalid construction concurrency"
[[ "${RUN_TAG}" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]] || fail "invalid run tag"

required=(
  MemNavData/MP3D_TABLE1_NEW_QUERY_PROTOCOL_20260829.md
  MemNavData/mp3d_table1_new_query_protocol_20260829.json
  MemNavData/mp3d_table1_new_query_contract.py
  MemNavData/freeze_mp3d_table1_source_ledger.py
  MemNavData/build_mp3d_table1_new_query_scene.py
  MemNavData/finalize_mp3d_table1_new_query_population.py
  MemNavData/verify_mp3d_table1_new_query_population.py
  MemNavData/materialize_hm3d_fullmono_online_a.py
  MemNavData/materialize_online_a_traces.py
  MemNavData/build_final14_role_pair_scene.py
  MemNavData/build_shared_online_double_revisit.py
  MemNavData/shared_online_role_pair_contract.py
  MemNavData/audit_shared_online_role_pairs.py
  MemNavData/test_mp3d_table1_new_query_contract.py
  MemNavData/slurm_mp3d_table1_new_query_construct.sbatch
  MemNavData/slurm_mp3d_table1_new_query_analysis.sbatch
  MemNavData/submit_mp3d_table1_new_query_construction_hpc.sh
)
for path in "${required[@]}"; do
  [[ -f "${path}" && ! -L "${path}" ]] || fail "missing physical ${path}"
done

"${LOCAL_MEMNAV_PY}" -m json.tool \
  MemNavData/mp3d_table1_new_query_protocol_20260829.json >/dev/null
PYTHONPATH="${ROOT}:${ROOT}/MemNavData" "${LOCAL_MEMNAV_PY}" -m py_compile \
  MemNavData/mp3d_table1_new_query_contract.py \
  MemNavData/freeze_mp3d_table1_source_ledger.py \
  MemNavData/finalize_mp3d_table1_new_query_population.py \
  MemNavData/verify_mp3d_table1_new_query_population.py
PYTHONPATH="${ROOT}:${ROOT}/MemNavData" "${LOCAL_MEMNAV_PY}" -m pytest -q \
  MemNavData/test_mp3d_table1_new_query_contract.py
PYTHONPATH="${ROOT}:${ROOT}/MemNavData" "${LOCAL_HAB_PY}" -m py_compile \
  MemNavData/materialize_hm3d_fullmono_online_a.py \
  MemNavData/build_mp3d_table1_new_query_scene.py
bash -n \
  MemNavData/slurm_mp3d_table1_new_query_construct.sbatch \
  MemNavData/slurm_mp3d_table1_new_query_analysis.sbatch \
  MemNavData/submit_mp3d_table1_new_query_construction_hpc.sh

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
"${LOCAL_MEMNAV_PY}" - "${staging}" "${local_head}" <<'PY'
import hashlib,json,sys
from pathlib import Path
root=Path(sys.argv[1]); files={}
for path in sorted(root.rglob('*')):
 if path.is_symlink(): raise SystemExit('bundle symlink: '+str(path))
 if path.is_file() and path.name not in {'SOURCE_BUNDLE.sha256','source_bundle_manifest.json'}:
  files[path.relative_to(root).as_posix()]=hashlib.sha256(path.read_bytes()).hexdigest()
payload={
 'schema_version':'mp3d_table1_new_query_bundle_v1_20260829',
 'local_git_head_context':sys.argv[2],
 'scope':'construction_only_no_controller_outcomes',
 'dataset':'MP3D','fresh_scene':False,'fresh_history':False,'new_query':True,
 'files':files,
}
(root/'source_bundle_manifest.json').write_text(json.dumps(payload,indent=2,sort_keys=True)+'\n')
PY
(
  cd "${staging}"
  find . -type f ! -name SOURCE_BUNDLE.sha256 -print0 | sort -z | \
    xargs -0 sha256sum >SOURCE_BUNDLE.sha256
  sha256sum -c SOURCE_BUNDLE.sha256 >/dev/null
)
task_receipt_sha=$(sha256sum "${staging}/SOURCE_BUNDLE.sha256" | awk '{print $1}')
bundle_key=${task_receipt_sha:0:16}
task_root=${REMOTE_BUNDLES}/mp3d_table1_new_query_${bundle_key}
task_stage=${task_root}.partial.$$
run_root=${REMOTE_RESULTS}/${RUN_TAG}_${bundle_key:0:8}
smoke_root=${REMOTE_RESULTS}/${RUN_TAG}_${bundle_key:0:8}_smoke

echo '[gate] verify shared SSH identity and immutable source inputs'
remote_identity=$(remote 'id -un' | tr -d '\r')
[[ "${remote_identity}" == yz11502 ]] || fail "wrong remote identity"
remote "set -euo pipefail
test \"\$(sha256sum '${SOURCE_MANIFEST}' | awk '{print \$1}')\" = '${EXPECTED_SOURCE_MANIFEST_SHA}'
test -d '${SOURCE_RUN_ROOT}/scenes'
test -r '${SOURCE_RUN_ROOT}/POSTHOC/mdtec_cec_composition_summary.json'
test -r '${SOURCE_RUN_ROOT}/POSTHOC/mdtec_cec_composition_independent_verification.json'
test -r '${SOURCE_RUN_ROOT}/POSTHOC/output_receipt.sha256'
test -x /scratch/lg154/conda-envs/habitat/bin/python
test -x /scratch/lg154/conda-envs/memnav/bin/python
test -r /share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif"

echo '[stage] verify or upload immutable source bundle'
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
protocol=${task_root}/MemNavData/mp3d_table1_new_query_protocol_20260829.json
echo '[stage] freeze source ledger before any construction'
remote "set -euo pipefail
test ! -e '${run_root}' && test ! -e '${smoke_root}'
mkdir -p '${run_root}/construction/scenes' '${run_root}/sealed_inputs' '${run_root}/logs' '${smoke_root}/construction/scenes' '${smoke_root}/logs' /scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs
cp '${protocol}' '${run_root}/sealed_inputs/'
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH='${task_root}:${task_root}/MemNavData' /scratch/lg154/conda-envs/memnav/bin/python '${task_root}/MemNavData/freeze_mp3d_table1_source_ledger.py' --source-run-root '${SOURCE_RUN_ROOT}' --source-manifest '${SOURCE_MANIFEST}' --out '${run_root}/sealed_inputs/source_ledger.json'
sha256sum '${run_root}/sealed_inputs/source_ledger.json' >'${run_root}/sealed_inputs/source_ledger.json.sha256'
chmod -R a-w '${run_root}/sealed_inputs'"
readarray -t ledger_gate < <(remote "python - '${run_root}/sealed_inputs/source_ledger.json' <<'PY'
import hashlib,json,sys
p=sys.argv[1]; d=json.load(open(p))
if d.get('previous_goal_b_policy_outcomes_read') is not False: raise SystemExit(2)
if d.get('query_policy_outcomes_read') is not False: raise SystemExit(2)
print(hashlib.sha256(open(p,'rb').read()).hexdigest())
print(d['scene_count']); print(sum(len(s['episodes']) for s in d['scenes']))
PY" | tr -d '\r')
[[ "${#ledger_gate[@]}" -eq 3 ]] || fail "source ledger gate incomplete"
source_ledger_sha=${ledger_gate[0]}
[[ "${ledger_gate[1]}" == 20 && "${ledger_gate[2]}" == 40 ]] || \
  fail "source ledger population changed"
source_ledger=${run_root}/sealed_inputs/source_ledger.json

echo '[gate] remote import and Slurm test-only'
remote "singularity exec --nv -B /scratch/lg154 -B /scratch/yz11502 /share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH='${task_root}:${task_root}/MemNavData' /scratch/lg154/conda-envs/habitat/bin/python -c 'import build_mp3d_table1_new_query_scene,materialize_hm3d_fullmono_online_a'"
common="ALL,TASK_ROOT=${task_root},TASK_RECEIPT=${task_receipt},EXPECTED_TASK_RECEIPT_SHA=${task_receipt_sha},SOURCE_LEDGER=${source_ledger},EXPECTED_SOURCE_LEDGER_SHA=${source_ledger_sha}"
construct=${task_root}/MemNavData/slurm_mp3d_table1_new_query_construct.sbatch
analysis=${task_root}/MemNavData/slurm_mp3d_table1_new_query_analysis.sbatch
remote "sbatch --test-only --array=0 --export='${common},RUN_ROOT=${smoke_root}' '${construct}' >/dev/null"
remote "sbatch --test-only --array=0-19%${CONSTRUCT_CONCURRENCY} --export='${common},RUN_ROOT=${run_root}' '${construct}' >/dev/null"
remote "sbatch --test-only --export='${common},RUN_ROOT=${run_root},MODE=finalize' '${analysis}' >/dev/null"
remote "sbatch --test-only --export='${common},RUN_ROOT=${run_root},MODE=verify' '${analysis}' >/dev/null"

echo '[submit] construction-only smoke -> formal -> finalize -> verifier'
smoke_raw=$(remote "sbatch --parsable --job-name=m3T1smoke --array=0 --export='${common},RUN_ROOT=${smoke_root}' '${construct}'" | tr -d '\r')
smoke_id=${smoke_raw%%;*}
formal_raw=$(remote "sbatch --parsable --job-name=m3T1construct --array=0-19%${CONSTRUCT_CONCURRENCY} --dependency=afterok:${smoke_id} --kill-on-invalid-dep=yes --export='${common},RUN_ROOT=${run_root}' '${construct}'" | tr -d '\r')
formal_id=${formal_raw%%;*}
finalize_raw=$(remote "sbatch --parsable --job-name=m3T1finalize --dependency=afterok:${formal_id} --kill-on-invalid-dep=yes --export='${common},RUN_ROOT=${run_root},MODE=finalize' '${analysis}'" | tr -d '\r')
finalize_id=${finalize_raw%%;*}
verify_raw=$(remote "sbatch --parsable --job-name=m3T1verify --dependency=afterok:${finalize_id} --kill-on-invalid-dep=yes --export='${common},RUN_ROOT=${run_root},MODE=verify' '${analysis}'" | tr -d '\r')
verify_id=${verify_raw%%;*}
for id in "${smoke_id}" "${formal_id}" "${finalize_id}" "${verify_id}"; do
  [[ "${id}" =~ ^[0-9]+$ ]] || fail "invalid job id ${id}"
done

receipt=MemNavData/MP3D_TABLE1_NEW_QUERY_SUBMISSION_20260829.json
[[ ! -e "${receipt}" ]] || fail "local submission receipt already exists"
"${LOCAL_MEMNAV_PY}" - "${receipt}" "${run_root}" "${smoke_root}" \
  "${task_root}" "${task_receipt_sha}" "${source_ledger_sha}" \
  "${smoke_id}" "${formal_id}" "${finalize_id}" "${verify_id}" <<'PY'
import json,sys
(path,run,smoke,bundle,bundle_sha,ledger_sha,smoke_job,formal,finalize,verify)=sys.argv[1:]
payload={
 'schema_version':'mp3d_table1_new_query_submission_v1_20260829',
 'scope':'construction_only_reused_scene_history_new_query',
 'run_root':run,'smoke_root':smoke,'task_bundle':bundle,
 'task_receipt_sha256':bundle_sha,'source_ledger_sha256':ledger_sha,
 'previous_goal_b_policy_outcomes_read_at_submission':False,
 'query_policy_outcomes_read_at_submission':False,
 'formal_policy_evaluation_submitted':False,
 'jobs':{'construction_smoke':int(smoke_job),'construction_array':int(formal),
         'population_finalize':int(finalize),'independent_verification':int(verify)},
}
open(path,'x').write(json.dumps(payload,indent=2,sort_keys=True)+'\n')
print(json.dumps(payload,indent=2,sort_keys=True))
PY
scp -q -o BatchMode=yes -o ControlMaster=no -o ControlPath="${SSH_CONTROL_PATH}" \
  "${ROOT}/${receipt}" "${SSH_ALIAS}:${run_root}/submission.json"
remote "sha256sum '${run_root}/submission.json' >'${run_root}/submission.json.sha256' && chmod a-w '${run_root}/submission.json' '${run_root}/submission.json.sha256'"
printf 'RUN_ROOT=%s\nTASK_ROOT=%s\nSMOKE=%s\nCONSTRUCTION=%s\nFINALIZE=%s\nVERIFY=%s\n' \
  "${run_root}" "${task_root}" "${smoke_id}" "${formal_id}" \
  "${finalize_id}" "${verify_id}"
