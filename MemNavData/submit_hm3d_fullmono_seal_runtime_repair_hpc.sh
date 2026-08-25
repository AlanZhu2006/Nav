#!/usr/bin/env bash
# Rebuild the downstream DAG after the CPU seal selected the wrong Python env.
set -euo pipefail
umask 0022

ROOT=${ROOT:-/home/asus/Research/Nav-graph-blind}
SSH_ALIAS=${SSH_ALIAS:-alantorch}
LOCAL_PY=${LOCAL_PY:-/home/asus/miniconda3/envs/memnav/bin/python}
RUN_ROOT=/scratch/yz11502/Research/Nav-axis-uturn-results/hm3d_fullmono_lifelong_20260824/formal_20260824T041000Z_cbef63fd
SMOKE_ROOT=${RUN_ROOT}_smoke
TASK_ROOT=/scratch/yz11502/Research/Nav-axis-uturn-source-bundles/hm3d_fullmono_lifelong_cbef63fd46d88451
TASK_RECEIPT=${TASK_ROOT}/SOURCE_BUNDLE.sha256
EXPECTED_TASK_RECEIPT_SHA=cbef63fd46d88451296fbfcb88ee605861497795c916c28deffbac2f1fdee909
REPAIR_ROOT=/scratch/yz11502/Research/Nav-axis-uturn-source-bundles/hm3d_fullmono_abfix_171ccb30ff2c17f8
REPAIR_RECEIPT=${REPAIR_ROOT}/SOURCE_BUNDLE.sha256
EXPECTED_REPAIR_RECEIPT_SHA=171ccb30ff2c17f8523b7d533ded2705a80ffcf3b8a9e5990223e36995f6ad64
BASE_SOURCE_ROOT=/scratch/yz11502/Research/Nav-axis-uturn-source-bundles/final14_mono_factorial_5690569a4373f2d2
BASE_RECEIPT=${BASE_SOURCE_ROOT}/source_inputs.sha256
EXPECTED_BASE_RECEIPT_SHA=5690569a4373f2d2768671418f0c604c4a03aa4b0ffe01baf70b288af03ba216
PARENT_ROOT=/scratch/yz11502/Research/Nav-axis-uturn-results/hm3d_fresh_fullmono_mixed_role_20260820/formal_20260820T143609Z_e6dd44c6
PROTOCOL=${TASK_ROOT}/MemNavData/hm3d_fullmono_lifelong_protocol_20260824.json
DEPENDENCY_RECEIPT=${RUN_ROOT}/sealed_inputs/dependency_receipt.json
EXPECTED_DEPENDENCY_RECEIPT_SHA=4eb0ca6479a26f8e04f85a31d906cee4e68b1785f66cfd3ac23bf65424d36e5e
ORIGINAL_BUILD_JOB=16265026
ZERO_HISTORY_REPAIR_JOB=16266646
FAILED_SEAL_JOB=16266719
SSH_CONTROL_PATH=${SSH_CONTROL_PATH:-$(ssh -G "${SSH_ALIAS}" 2>/dev/null | awk '$1=="controlpath"{value=$2} END{print value}')}

cd "${ROOT}"
fail() { echo "ABORT: $*" >&2; exit 2; }
remote() {
  timeout 180 ssh -n -tt -o BatchMode=yes -o ControlMaster=no \
    -S "${SSH_CONTROL_PATH}" "${SSH_ALIAS}" "$@"
}
job_id() {
  tr -d '\r' | awk -F';' '/^[0-9]+(;|$)/ {print $1; exit}'
}
upload_bundle() {
  local source=$1 destination=$2 attempt
  for attempt in 1 2 3; do
    if timeout 240 rsync -a --partial \
      --chmod=Du=rwx,Dgo=rx,Fu=rw,Fgo=r \
      -e "ssh -o BatchMode=yes -o ControlMaster=no -S ${SSH_CONTROL_PATH}" \
      "${source}/" "${SSH_ALIAS}:${destination}/"; then
      return 0
    fi
  done
  return 1
}

[[ -S "${SSH_CONTROL_PATH}" ]] || fail "authoritative shared SSH socket missing"
timeout 15 ssh -O check -S "${SSH_CONTROL_PATH}" "${SSH_ALIAS}" >/dev/null 2>&1 \
  || fail "shared SSH master is not responsive"
[[ -x "${LOCAL_PY}" ]] || fail "local Python missing"
bash -n MemNavData/slurm_hm3d_fullmono_lifelong_finalize_ab.sbatch

staging=$(mktemp -d /tmp/h3life_sealfix_bundle.XXXXXX)
cleanup() { rm -rf -- "${staging}"; }
trap cleanup EXIT
mkdir -p "${staging}/root/MemNavData"
cp -p MemNavData/slurm_hm3d_fullmono_lifelong_finalize_ab.sbatch \
  "${staging}/root/MemNavData/"
cp -p MemNavData/submit_hm3d_fullmono_seal_runtime_repair_hpc.sh \
  "${staging}/root/MemNavData/"
(
  cd "${staging}/root"
  find . -type f ! -name SOURCE_BUNDLE.sha256 -print0 | sort -z | \
    xargs -0 sha256sum >SOURCE_BUNDLE.sha256
  sha256sum -c --quiet SOURCE_BUNDLE.sha256
)
wrapper_receipt_sha=$(sha256sum "${staging}/root/SOURCE_BUNDLE.sha256" | awk '{print $1}')
bundle_key=${wrapper_receipt_sha:0:16}
wrapper_root=/scratch/yz11502/Research/Nav-axis-uturn-source-bundles/hm3d_fullmono_sealfix_${bundle_key}
wrapper_stage=${wrapper_root}.partial.$$

remote "set -euo pipefail
test \"\$(sha256sum '${TASK_RECEIPT}' | awk '{print \$1}')\" = '${EXPECTED_TASK_RECEIPT_SHA}'
cd '${TASK_ROOT}' && sha256sum -c --quiet '${TASK_RECEIPT}'
test \"\$(sha256sum '${REPAIR_RECEIPT}' | awk '{print \$1}')\" = '${EXPECTED_REPAIR_RECEIPT_SHA}'
cd '${REPAIR_ROOT}' && sha256sum -c --quiet '${REPAIR_RECEIPT}'
test \"\$(sha256sum '${BASE_RECEIPT}' | awk '{print \$1}')\" = '${EXPECTED_BASE_RECEIPT_SHA}'
cd '${BASE_SOURCE_ROOT}' && sha256sum -c --quiet '${BASE_RECEIPT}'
test \"\$(sha256sum '${DEPENDENCY_RECEIPT}' | awk '{print \$1}')\" = '${EXPECTED_DEPENDENCY_RECEIPT_SHA}'
test ! -e '${RUN_ROOT}/benchmarks_ab'
test \"\$(sacct -j '${FAILED_SEAL_JOB}' -X -n -o State | awk 'NF{print \$1; exit}')\" = FAILED
test \"\$(sacct -j '${ZERO_HISTORY_REPAIR_JOB}' -X -n -o State | awk 'NF{print \$1; exit}')\" = COMPLETED"

# This audit reads only result-blind construction receipts, never C/B2/C2.
remote "/scratch/lg154/conda-envs/memnav/bin/python - '${RUN_ROOT}' '${PARENT_ROOT}' '${PROTOCOL}' <<'PY'
import hashlib, json, sys
from pathlib import Path
run, parent, protocol = map(Path, sys.argv[1:])
manifest = json.loads((parent/'sealed_inputs/parent_manifest.json').read_text())
protocol_hash = hashlib.sha256(protocol.read_bytes()).hexdigest()
rows = []
for index, scene_raw in enumerate(manifest['scenes']):
    scene = str(scene_raw)
    root = run/'construct_ab/scenes'/f'{index:02d}_{scene}'
    path = root/'completion.json'
    sidecar = root/'completion.json.sha256'
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert sidecar.read_text().split() == [digest, 'completion.json']
    row = json.loads(path.read_text())
    assert row['status'] == 'complete'
    assert row['scene'] == scene and int(row['scene_index']) == index
    assert row['protocol_sha256'] == protocol_hash
    assert row['query_policy_outcomes_read'] is False
    rows.append(row)
assert len(rows) == 54
assert sum(int(r['materialized_A_histories']) for r in rows) == 130
assert [r['scene_index'] for r in rows if r.get('upstream_parent_completion_sha256')] == [11,15,34,40,44]
print(json.dumps({
  'verified': True,
  'fragments': len(rows),
  'materialized_A_histories': 130,
  'constructible_AB_C_histories': sum(int(r['constructible_AB_C_histories']) for r in rows),
  'constructible_scene_clusters': sum(int(r['constructible_AB_C_histories']) > 0 for r in rows),
  'query_policy_outcomes_read': False,
}, sort_keys=True))
PY"

if remote "test -d '${wrapper_root}'"; then
  remote "test \"\$(sha256sum '${wrapper_root}/SOURCE_BUNDLE.sha256' | awk '{print \$1}')\" = '${wrapper_receipt_sha}' && cd '${wrapper_root}' && sha256sum -c --quiet SOURCE_BUNDLE.sha256"
else
  remote "test ! -e '${wrapper_stage}' && mkdir -p '${wrapper_stage}'"
  upload_bundle "${staging}/root" "${wrapper_stage}" || fail "seal wrapper upload failed"
  remote "cd '${wrapper_stage}' && sha256sum -c --quiet SOURCE_BUNDLE.sha256 && chmod -R a-w '${wrapper_stage}' && mv '${wrapper_stage}' '${wrapper_root}'"
fi

seal_wrapper=${wrapper_root}/MemNavData/slurm_hm3d_fullmono_lifelong_finalize_ab.sbatch
wrapper_receipt=${wrapper_root}/SOURCE_BUNDLE.sha256
vendor=/scratch/lg154/conda-envs/habitat/lib/python3.9/site-packages/pip/_vendor
remote "singularity exec -B /scratch/lg154 -B /scratch/yz11502 /share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH='${TASK_ROOT}:${TASK_ROOT}/MemNavData:${BASE_SOURCE_ROOT}:${BASE_SOURCE_ROOT}/MemNavData:${vendor}' /scratch/lg154/conda-envs/habitat/bin/python '${TASK_ROOT}/MemNavData/finalize_hm3d_fullmono_lifelong_ab.py' --help >/dev/null"

common="ALL,TASK_ROOT=${TASK_ROOT},BASE_SOURCE_ROOT=${BASE_SOURCE_ROOT},RUN_ROOT=${RUN_ROOT},PARENT_ROOT=${PARENT_ROOT},PROTOCOL=${PROTOCOL},TASK_RECEIPT=${TASK_RECEIPT},EXPECTED_TASK_RECEIPT_SHA=${EXPECTED_TASK_RECEIPT_SHA},BASE_RECEIPT=${BASE_RECEIPT},EXPECTED_BASE_RECEIPT_SHA=${EXPECTED_BASE_RECEIPT_SHA},DEPENDENCY_RECEIPT=${DEPENDENCY_RECEIPT},EXPECTED_DEPENDENCY_RECEIPT_SHA=${EXPECTED_DEPENDENCY_RECEIPT_SHA},EXPECTED_HAB_REQUESTS_VERSION=2.32.4,EXPECTED_HAB_REQUESTS_INIT_BYTES=5057,EXPECTED_HAB_REQUESTS_INIT_SHA=1e507f1f386bcc6b5f0ff69a614c14875cd65cb67be7f6022f28adef9774573f,EXPECTED_HAB_REQUESTS_VERSION_BYTES=435,EXPECTED_HAB_REQUESTS_VERSION_SHA=143abaf3563712f063743a7952aa65319dbcb934d894cfc989bd2c015f8da577"
seal_common="${common},WRAPPER_ROOT=${wrapper_root},WRAPPER_RECEIPT=${wrapper_receipt},EXPECTED_WRAPPER_RECEIPT_SHA=${wrapper_receipt_sha}"
collect_b=${TASK_ROOT}/MemNavData/slurm_hm3d_fullmono_lifelong_collect_b.sbatch
prefix=${TASK_ROOT}/MemNavData/slurm_hm3d_fullmono_lifelong_construct_prefix.sbatch
seal_population=${TASK_ROOT}/MemNavData/slurm_hm3d_fullmono_lifelong_finalize_population.sbatch
evaluate=${TASK_ROOT}/MemNavData/slurm_hm3d_fullmono_lifelong_eval.sbatch
analysis=${TASK_ROOT}/MemNavData/slurm_hm3d_fullmono_lifelong_analysis.sbatch

remote "sbatch --test-only --export='${seal_common}' '${seal_wrapper}' >/dev/null"
seal=$(remote "sbatch --parsable --export='${seal_common}' '${seal_wrapper}'" | job_id)
[[ "${seal}" =~ ^[0-9]+$ ]] || fail "bad Habitat seal job id"
collect=$(remote "sbatch --parsable --qos=gpu48 --array=0-53%4 --dependency=afterok:${seal} --kill-on-invalid-dep=yes --export='${common},MAX_STEPS=600' '${collect_b}'" | job_id)
[[ "${collect}" =~ ^[0-9]+$ ]] || fail "bad factual-B job id"
prefix_job=$(remote "sbatch --parsable --qos=gpu48 --array=0-129%4 --dependency=afterok:${collect} --kill-on-invalid-dep=yes --export='${common}' '${prefix}'" | job_id)
[[ "${prefix_job}" =~ ^[0-9]+$ ]] || fail "bad prefix job id"
population=$(remote "sbatch --parsable --dependency=afterok:${prefix_job} --kill-on-invalid-dep=yes --export='${common}' '${seal_population}'" | job_id)
[[ "${population}" =~ ^[0-9]+$ ]] || fail "bad population seal job id"
smoke=$(remote "sbatch --parsable --qos=gpu48 --array=0 --dependency=afterok:${population} --kill-on-invalid-dep=yes --export='${common},OUTPUT_ROOT=${SMOKE_ROOT},MAX_STEPS=80' '${evaluate}'" | job_id)
[[ "${smoke}" =~ ^[0-9]+$ ]] || fail "bad smoke job id"
evaluation=$(remote "sbatch --parsable --qos=gpu48 --array=0-129%4 --dependency=afterok:${smoke} --kill-on-invalid-dep=yes --export='${common},OUTPUT_ROOT=${RUN_ROOT},MAX_STEPS=600' '${evaluate}'" | job_id)
[[ "${evaluation}" =~ ^[0-9]+$ ]] || fail "bad evaluation job id"
aggregate=$(remote "sbatch --parsable --dependency=afterok:${evaluation} --kill-on-invalid-dep=yes --export='${common},MODE=aggregate' '${analysis}'" | job_id)
[[ "${aggregate}" =~ ^[0-9]+$ ]] || fail "bad aggregate job id"
verify=$(remote "sbatch --parsable --dependency=afterok:${aggregate} --kill-on-invalid-dep=yes --export='${common},MODE=verify' '${analysis}'" | job_id)
[[ "${verify}" =~ ^[0-9]+$ ]] || fail "bad verifier job id"

jobs=${seal},${collect},${prefix_job},${population},${smoke},${evaluation},${aggregate},${verify}
receipt=MemNavData/HM3D_FULLMONO_LIFELONG_SEAL_RUNTIME_REPAIR_RECEIPT_20260824.json
"${LOCAL_PY}" - "${receipt}" "${wrapper_root}" "${wrapper_receipt_sha}" "${jobs}" <<'PY'
import json, sys
path, wrapper, wrapper_sha, jobs = sys.argv[1:]
names = ['seal_AB_population','collect_factual_B_array','construct_actual_prefix_array','seal_query_population','remote_true_stack_smoke','formal_three_arm_evaluation','aggregate','independent_verification']
ids = list(map(int, jobs.split(',')))
payload = {
  'schema_version': 'hm3d_fullmono_lifelong_seal_runtime_repair_v1_20260824',
  'failure': {'job_id': 16266719, 'stage': 'import', 'error': "ModuleNotFoundError: No module named 'quaternion'", 'scientific_output_created': False},
  'wrapper_bundle': wrapper,
  'wrapper_bundle_receipt_sha256': wrapper_sha,
  'seal_runtime': 'Habitat Python inside frozen CUDA Singularity image on cpu_short',
  'preseal_raw_receipts_verified': True,
  'selection_reads_navigation_outcomes': False,
  'scientific_thresholds_changed': False,
  'replacement_jobs': dict(zip(names, ids)),
}
open(path,'x').write(json.dumps(payload,indent=2,sort_keys=True)+'\n')
print(json.dumps(payload,indent=2,sort_keys=True))
PY
timeout 120 scp -q -o BatchMode=yes -o ControlMaster=no \
  -o ControlPath="${SSH_CONTROL_PATH}" "${receipt}" \
  "${SSH_ALIAS}:${RUN_ROOT}/seal_runtime_repair_submission.json" \
  || fail "seal repair receipt upload failed"
remote "sha256sum '${RUN_ROOT}/seal_runtime_repair_submission.json' >'${RUN_ROOT}/seal_runtime_repair_submission.json.sha256'; squeue -j '${jobs}' -o '%.18i %.22j %.2t %.12M %.40R'"
printf 'WRAPPER_ROOT=%s\nWRAPPER_SHA=%s\nREPLACEMENT_JOBS=%s\n' \
  "${wrapper_root}" "${wrapper_receipt_sha}" "${jobs}"
