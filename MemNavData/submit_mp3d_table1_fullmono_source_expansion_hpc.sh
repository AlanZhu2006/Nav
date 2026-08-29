#!/usr/bin/env bash
# Submit full-mono Goal-A source expansion, then construction-only verification.
set -euo pipefail
umask 0022

ROOT=${ROOT:-/home/asus/Research/Nav-graph-blind}
SSH_ALIAS=${SSH_ALIAS:-alantorch}
LOCAL_MEMNAV_PY=/home/asus/miniconda3/envs/memnav/bin/python
LOCAL_HAB_PY=/home/asus/miniconda3/envs/habitat/bin/python
REMOTE_BUNDLES=/scratch/yz11502/Research/Nav-axis-uturn-source-bundles
REMOTE_RESULTS=/scratch/yz11502/Research/Nav-axis-uturn-results/mp3d_table1_fullmono_source_expansion_20260829
BASE_SOURCE_LEDGER=/scratch/yz11502/Research/Nav-axis-uturn-results/mp3d_table1_new_query_20260829/construction_20260829T050401Z_6813d501/sealed_inputs/source_ledger.json
EXPECTED_BASE_SOURCE_LEDGER_SHA=8b19af79300d63ba81b8efdf76498c22357a4d840f73236863ec0e44b606031e
NAVDP_BASE_SOURCE_ROOT=/scratch/yz11502/Research/Nav-axis-uturn-source-bundles/final14_mono_factorial_5690569a4373f2d2
NAVDP_BASE_RECEIPT=${NAVDP_BASE_SOURCE_ROOT}/source_inputs.sha256
EXPECTED_NAVDP_BASE_RECEIPT_SHA=5690569a4373f2d2768671418f0c604c4a03aa4b0ffe01baf70b288af03ba216
NAVDP_SERVER_SOURCE_ROOT=/scratch/yz11502/Research/Nav-axis-uturn-source-bundles/hm3d_table1_navdp_authority_transaction_718661db1733d5de
NAVDP_SERVER_RECEIPT=${NAVDP_SERVER_SOURCE_ROOT}/SOURCE_BUNDLE.sha256
EXPECTED_NAVDP_SERVER_RECEIPT_SHA=718661db1733d5de16cd86687eec880a8d02fc5ae5ca982e1ab7d5bde5e96f7d
SOURCE_CONCURRENCY=${SOURCE_CONCURRENCY:-2}
CONSTRUCT_CONCURRENCY=${CONSTRUCT_CONCURRENCY:-2}
RUN_TAG=${RUN_TAG:-source_expansion_$(date -u +%Y%m%dT%H%M%SZ)}
SSH_CONTROL_PATH=${SSH_CONTROL_PATH:-$(ssh -G "${SSH_ALIAS}" 2>/dev/null | awk '$1=="controlpath"{value=$2} END{print value}')}
cd "${ROOT}"

fail() { echo "ABORT: $*" >&2; exit 2; }
remote() {
  timeout 180 ssh -n -tt -o BatchMode=yes -o ControlMaster=no \
    -S "${SSH_CONTROL_PATH}" "${SSH_ALIAS}" "$@"
}
[[ -S "${SSH_CONTROL_PATH}" ]] || fail "authoritative SSH master missing"
for value in "${SOURCE_CONCURRENCY}" "${CONSTRUCT_CONCURRENCY}"; do
  [[ "${value}" =~ ^[1-9][0-9]*$ ]] || fail "invalid concurrency ${value}"
done
[[ "${RUN_TAG}" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]] || fail "invalid run tag"

protocol=MemNavData/mp3d_table1_fullmono_source_expansion_protocol_20260829.json
source_manifest=.diagnostics/paper_power_expansion_freeze_20260814_pre_result/paper_power_expansion_manifest.json
required=(
  "${protocol}"
  MemNavData/MP3D_TABLE1_FULLMONO_SOURCE_EXPANSION_PROTOCOL_20260829.md
  MemNavData/collect_mp3d_table1_fullmono_goal_a.py
  MemNavData/freeze_mp3d_table1_fullmono_expanded_source_ledger.py
  MemNavData/mp3d_table1_new_query_contract.py
  MemNavData/build_mp3d_table1_new_query_scene.py
  MemNavData/finalize_mp3d_table1_new_query_population.py
  MemNavData/verify_mp3d_table1_new_query_population.py
  MemNavData/run_hm3d_fullmono_server_scene.sh
  MemNavData/slurm_mp3d_table1_fullmono_source.sbatch
  MemNavData/slurm_mp3d_table1_fullmono_source_deferred.sbatch
  MemNavData/slurm_mp3d_table1_new_query_construct.sbatch
  MemNavData/slurm_mp3d_table1_new_query_analysis.sbatch
  MemNavData/slurm_safe_submit.sh
  MemNavData/test_mp3d_table1_fullmono_source_expansion.py
  MemNavData/test_mp3d_table1_new_query_contract.py
  "${source_manifest}"
)
for path in "${required[@]}"; do
  [[ -f "${path}" && ! -L "${path}" ]] || fail "missing physical ${path}"
done

"${LOCAL_MEMNAV_PY}" -m json.tool "${protocol}" >/dev/null
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="${ROOT}:${ROOT}/MemNavData" \
  "${LOCAL_MEMNAV_PY}" -m pytest -p no:cacheprovider -q \
    MemNavData/test_mp3d_table1_fullmono_source_expansion.py \
    MemNavData/test_mp3d_table1_new_query_contract.py \
    MemNavData/test_hm3d_table1_navdp_transport_contract.py
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="${ROOT}:${ROOT}/MemNavData" \
  "${LOCAL_HAB_PY}" -m py_compile \
    MemNavData/collect_mp3d_table1_fullmono_goal_a.py \
    MemNavData/freeze_mp3d_table1_fullmono_expanded_source_ledger.py \
    MemNavData/build_mp3d_table1_new_query_scene.py
bash -n \
  MemNavData/run_hm3d_fullmono_server_scene.sh \
  MemNavData/slurm_mp3d_table1_fullmono_source.sbatch \
  MemNavData/slurm_mp3d_table1_fullmono_source_deferred.sbatch \
  MemNavData/slurm_mp3d_table1_new_query_construct.sbatch \
  MemNavData/slurm_mp3d_table1_new_query_analysis.sbatch \
  MemNavData/submit_mp3d_table1_fullmono_source_expansion_hpc.sh
source MemNavData/slurm_safe_submit.sh
for script in \
  MemNavData/slurm_mp3d_table1_fullmono_source.sbatch \
  MemNavData/slurm_mp3d_table1_fullmono_source_deferred.sbatch \
  MemNavData/slurm_mp3d_table1_new_query_construct.sbatch \
  MemNavData/slurm_mp3d_table1_new_query_analysis.sbatch; do
  lint_sbatch_template "${script}" || fail "sbatch lint failed: ${script}"
done

staging=$(mktemp -d)
cleanup() { rm -rf -- "${staging}"; }
trap cleanup EXIT
mkdir -p "${staging}/MemNavData" "${staging}/sealed_inputs"
while IFS= read -r -d '' path; do
  cp --preserve=mode,timestamps "${path}" \
    "${staging}/MemNavData/$(basename "${path}")"
done < <(find "${ROOT}/MemNavData" -maxdepth 1 -type f -name '*.py' -print0)
for path in "${required[@]}"; do
  case "${path}" in
    MemNavData/*.py) ;;
    "${source_manifest}")
      cp --preserve=mode,timestamps "${path}" \
        "${staging}/sealed_inputs/paper_power_expansion_manifest.json" ;;
    MemNavData/*)
      cp --preserve=mode,timestamps "${path}" \
        "${staging}/MemNavData/$(basename "${path}")" ;;
  esac
done
local_head=$(git rev-parse HEAD)
"${LOCAL_MEMNAV_PY}" - "${staging}" "${local_head}" <<'PY'
import hashlib,json,sys
from pathlib import Path
root=Path(sys.argv[1]); files={}
for path in sorted(root.rglob('*')):
 if path.is_symlink(): raise SystemExit('bundle symlink: '+str(path))
 if path.is_file() and path.name not in {'SOURCE_BUNDLE.sha256','source_bundle_manifest.json'}:
  files[path.relative_to(root).as_posix()]=hashlib.sha256(path.read_bytes()).hexdigest()
payload={
 'schema_version':'mp3d_table1_fullmono_source_expansion_bundle_v1_20260829',
 'local_git_head_context':sys.argv[2],
 'scope':'fullmono_source_collection_then_outcome_blind_query_construction',
 'policy_query_outcomes_read':False,'formal_controller_rollout_included':False,
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
task_root=${REMOTE_BUNDLES}/mp3d_table1_fullmono_source_expansion_${bundle_key}
task_stage=${task_root}.partial.$$
run_root=${REMOTE_RESULTS}/${RUN_TAG}_${bundle_key:0:8}
smoke_root=${REMOTE_RESULTS}/${RUN_TAG}_${bundle_key:0:8}_smoke

echo '[gate] shared SSH identity and frozen remote sources'
remote_identity=$(remote 'id -un' | tr -d '\r')
[[ "${remote_identity}" == yz11502 ]] || fail "wrong remote identity"
remote "set -euo pipefail
test \"\$(sha256sum '${BASE_SOURCE_LEDGER}' | awk '{print \$1}')\" = '${EXPECTED_BASE_SOURCE_LEDGER_SHA}'
test \"\$(sha256sum '${NAVDP_BASE_RECEIPT}' | awk '{print \$1}')\" = '${EXPECTED_NAVDP_BASE_RECEIPT_SHA}'
test \"\$(sha256sum '${NAVDP_SERVER_RECEIPT}' | awk '{print \$1}')\" = '${EXPECTED_NAVDP_SERVER_RECEIPT_SHA}'
(cd '${NAVDP_BASE_SOURCE_ROOT}' && sha256sum -c --quiet '${NAVDP_BASE_RECEIPT}')
(cd '${NAVDP_SERVER_SOURCE_ROOT}' && sha256sum -c --quiet '${NAVDP_SERVER_RECEIPT}')
test -r /share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif"

echo '[stage] immutable task bundle'
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
remote_protocol=${task_root}/MemNavData/$(basename "${protocol}")
remote_manifest=${task_root}/sealed_inputs/paper_power_expansion_manifest.json
echo '[gate] remote structural imports, consumed-manifest hashes, Slurm test-only'
remote "PYTHONDONTWRITEBYTECODE=1 PYTHONPATH='${task_root}:${task_root}/MemNavData' /scratch/lg154/conda-envs/memnav/bin/python - '${remote_protocol}' <<'PY'
import hashlib,json,sys
p=json.load(open(sys.argv[1]))
for spec in p['consumed_query_exclusion']['old_query_manifests']:
 path=spec['path']; got=hashlib.sha256(open(path,'rb').read()).hexdigest()
 if got != spec['sha256']: raise SystemExit('consumed query manifest changed: '+path)
print('CONSUMED_QUERY_MANIFESTS_OK')
PY"
remote "PYTHONDONTWRITEBYTECODE=1 PYTHONPATH='${task_root}:${task_root}/MemNavData' /scratch/lg154/conda-envs/memnav/bin/python - '${remote_protocol}' <<'PY'
import json,sys
from freeze_mp3d_table1_fullmono_expanded_source_ledger import read_consumed_query_manifests
p=json.load(open(sys.argv[1]))
rows,receipts=read_consumed_query_manifests(p['consumed_query_exclusion']['old_query_manifests'])
assert rows and len(receipts)==4 and sum(x['query_identities'] for x in receipts)>0
print('CONSUMED_QUERY_IMAGES_OK',len(rows),sum(len(x) for x in rows.values()))
PY"
remote "singularity exec --nv -B /scratch/lg154 -B /scratch/yz11502 /share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH='${task_root}:${task_root}/MemNavData:${NAVDP_SERVER_SOURCE_ROOT}:${NAVDP_SERVER_SOURCE_ROOT}/MemNavData:${NAVDP_BASE_SOURCE_ROOT}:${NAVDP_BASE_SOURCE_ROOT}/MemNavData' /scratch/lg154/conda-envs/habitat/bin/python -c 'import collect_mp3d_table1_fullmono_goal_a,freeze_mp3d_table1_fullmono_expanded_source_ledger,build_mp3d_table1_new_query_scene'"
remote "singularity exec --nv -B /scratch/lg154 -B /scratch/yz11502 /share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH='${NAVDP_SERVER_SOURCE_ROOT}/NavDP/baselines/navdp:${NAVDP_BASE_SOURCE_ROOT}/NavDP/baselines/navdp:${NAVDP_SERVER_SOURCE_ROOT}/NavDP/baselines/memnav:${NAVDP_BASE_SOURCE_ROOT}/NavDP/baselines/memnav' /scratch/lg154/conda-envs/memnav/bin/python -c 'import policy_agent; assert hasattr(policy_agent,\"NavDP_Agent\")'"

common="ALL,TASK_ROOT=${task_root},TASK_RECEIPT=${task_receipt},EXPECTED_TASK_RECEIPT_SHA=${task_receipt_sha},SERVER_SOURCE_ROOT=${NAVDP_SERVER_SOURCE_ROOT},SERVER_SOURCE_RECEIPT=${NAVDP_SERVER_RECEIPT},EXPECTED_SERVER_SOURCE_RECEIPT_SHA=${EXPECTED_NAVDP_SERVER_RECEIPT_SHA},BASE_SOURCE_ROOT=${NAVDP_BASE_SOURCE_ROOT},BASE_RECEIPT=${NAVDP_BASE_RECEIPT},EXPECTED_BASE_RECEIPT_SHA=${EXPECTED_NAVDP_BASE_RECEIPT_SHA},PARENT_MANIFEST=${remote_manifest},PROTOCOL=${remote_protocol}"
source_script=${task_root}/MemNavData/slurm_mp3d_table1_fullmono_source.sbatch
deferred_script=${task_root}/MemNavData/slurm_mp3d_table1_fullmono_source_deferred.sbatch
remote "source '${task_root}/MemNavData/slurm_safe_submit.sh'; safe_sbatch --lint-fatal --test-only --qos=gpu48 --array=0 --export='${common},RUN_ROOT=${smoke_root},MP3D_SOURCE_SMOKE=1,MAX_STEPS=80' '${source_script}' >/dev/null; safe_sbatch --lint-fatal --test-only --qos=gpu48 --array=0-15%${SOURCE_CONCURRENCY} --export='${common},RUN_ROOT=${run_root},MP3D_SOURCE_SMOKE=0' '${source_script}' >/dev/null; safe_sbatch --lint-fatal --test-only --partition=cpu_short --export='ALL,TASK_ROOT=${task_root},TASK_RECEIPT=${task_receipt},EXPECTED_TASK_RECEIPT_SHA=${task_receipt_sha},RUN_ROOT=${run_root},BASE_SOURCE_LEDGER=${BASE_SOURCE_LEDGER},PROTOCOL=${remote_protocol},EXPANSION_MANIFEST=${remote_manifest},CONSTRUCT_CONCURRENCY=${CONSTRUCT_CONCURRENCY}' '${deferred_script}' >/dev/null"

echo '[submit] source smoke -> 16-scene full-mono collection -> construction verifier'
smoke_raw=$(remote "source '${task_root}/MemNavData/slurm_safe_submit.sh'; safe_sbatch --lint-fatal --parsable --qos=gpu48 --array=0 --export='${common},RUN_ROOT=${smoke_root},MP3D_SOURCE_SMOKE=1,MAX_STEPS=80' '${source_script}'" | tr -d '\r')
smoke_id=${smoke_raw%%;*}
formal_raw=$(remote "source '${task_root}/MemNavData/slurm_safe_submit.sh'; safe_sbatch --lint-fatal --parsable --qos=gpu48 --dependency=afterok:${smoke_id} --kill-on-invalid-dep=yes --array=0-15%${SOURCE_CONCURRENCY} --export='${common},RUN_ROOT=${run_root},MP3D_SOURCE_SMOKE=0' '${source_script}'" | tr -d '\r')
formal_id=${formal_raw%%;*}
deferred_env="ALL,TASK_ROOT=${task_root},TASK_RECEIPT=${task_receipt},EXPECTED_TASK_RECEIPT_SHA=${task_receipt_sha},RUN_ROOT=${run_root},BASE_SOURCE_LEDGER=${BASE_SOURCE_LEDGER},PROTOCOL=${remote_protocol},EXPANSION_MANIFEST=${remote_manifest},CONSTRUCT_CONCURRENCY=${CONSTRUCT_CONCURRENCY}"
deferred_raw=$(remote "source '${task_root}/MemNavData/slurm_safe_submit.sh'; safe_sbatch --lint-fatal --parsable --partition=cpu_short --dependency=afterok:${formal_id} --kill-on-invalid-dep=yes --export='${deferred_env}' '${deferred_script}'" | tr -d '\r')
deferred_id=${deferred_raw%%;*}
for id in "${smoke_id}" "${formal_id}" "${deferred_id}"; do
  [[ "${id}" =~ ^[0-9]+$ ]] || fail "invalid job id ${id}"
done

receipt=MemNavData/MP3D_TABLE1_FULLMONO_SOURCE_EXPANSION_SUBMISSION_20260829.json
[[ ! -e "${receipt}" ]] || fail "local submission receipt already exists"
"${LOCAL_MEMNAV_PY}" - "${receipt}" "${run_root}" "${smoke_root}" \
  "${task_root}" "${task_receipt_sha}" "${smoke_id}" "${formal_id}" \
  "${deferred_id}" <<'PY'
import json,sys
path,run,smoke,bundle,sha,smoke_job,formal,deferred=sys.argv[1:]
payload={
 'schema_version':'mp3d_table1_fullmono_source_expansion_submission_v1_20260829',
 'scope':'fullmono_source_collection_then_outcome_blind_query_construction',
 'run_root':run,'smoke_root':smoke,'task_bundle':bundle,
 'task_receipt_sha256':sha,
 'source_scenes':16,'source_episodes':64,
 'old_query_policy_outcomes_read_at_submission':False,
 'new_query_policy_outcomes_read_at_submission':False,
 'formal_controller_rollout_submitted':False,
 'jobs':{'source_smoke':int(smoke_job),
         'source_collection_array':int(formal),
         'ledger_and_construction_deferred':int(deferred)},
}
open(path,'x').write(json.dumps(payload,indent=2,sort_keys=True)+'\n')
print(json.dumps(payload,indent=2,sort_keys=True))
PY
scp -q -o BatchMode=yes -o ControlMaster=no -o ControlPath="${SSH_CONTROL_PATH}" \
  "${ROOT}/${receipt}" "${SSH_ALIAS}:${run_root}/submission.json"
remote "sha256sum '${run_root}/submission.json' >'${run_root}/submission.json.sha256' && chmod a-w '${run_root}/submission.json' '${run_root}/submission.json.sha256'"
printf 'RUN_ROOT=%s\nTASK_ROOT=%s\nSMOKE=%s\nSOURCE_COLLECTION=%s\nDEFERRED=%s\n' \
  "${run_root}" "${task_root}" "${smoke_id}" "${formal_id}" "${deferred_id}"
