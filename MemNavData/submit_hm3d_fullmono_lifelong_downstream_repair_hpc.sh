#!/usr/bin/env bash
# Replace only the dependency-invalid downstream DAG after the exact AB repair.
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
REPAIR_MANIFEST=${REPAIR_ROOT}/MemNavData/hm3d_fullmono_zero_history_repair_manifest_20260824.json
BASE_SOURCE_ROOT=/scratch/yz11502/Research/Nav-axis-uturn-source-bundles/final14_mono_factorial_5690569a4373f2d2
BASE_RECEIPT=${BASE_SOURCE_ROOT}/source_inputs.sha256
EXPECTED_BASE_RECEIPT_SHA=5690569a4373f2d2768671418f0c604c4a03aa4b0ffe01baf70b288af03ba216
PARENT_ROOT=/scratch/yz11502/Research/Nav-axis-uturn-results/hm3d_fresh_fullmono_mixed_role_20260820/formal_20260820T143609Z_e6dd44c6
PROTOCOL=${TASK_ROOT}/MemNavData/hm3d_fullmono_lifelong_protocol_20260824.json
DEPENDENCY_RECEIPT=${RUN_ROOT}/sealed_inputs/dependency_receipt.json
EXPECTED_DEPENDENCY_RECEIPT_SHA=4eb0ca6479a26f8e04f85a31d906cee4e68b1785f66cfd3ac23bf65424d36e5e
ORIGINAL_BUILD_JOB=16265026
REPAIR_JOB=16266646
OLD_DOWNSTREAM_JOBS="16265034 16265042 16265051 16265056 16265066 16265073 16265078 16265088"
REPLACEMENT_SEAL_JOB=${REPLACEMENT_SEAL_JOB:-}
REPLACEMENT_B_JOB=${REPLACEMENT_B_JOB:-}
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

[[ -S "${SSH_CONTROL_PATH}" ]] || fail "authoritative shared SSH socket missing"
timeout 15 ssh -O check -S "${SSH_CONTROL_PATH}" "${SSH_ALIAS}" >/dev/null 2>&1 \
  || fail "shared SSH master is not responsive"
[[ -x "${LOCAL_PY}" ]] || fail "local Python missing"

remote "set -euo pipefail
test \"\$(sha256sum '${TASK_RECEIPT}' | awk '{print \$1}')\" = '${EXPECTED_TASK_RECEIPT_SHA}'
cd '${TASK_ROOT}' && sha256sum -c --quiet '${TASK_RECEIPT}'
test \"\$(sha256sum '${REPAIR_RECEIPT}' | awk '{print \$1}')\" = '${EXPECTED_REPAIR_RECEIPT_SHA}'
cd '${REPAIR_ROOT}' && sha256sum -c --quiet '${REPAIR_RECEIPT}'
test \"\$(sha256sum '${BASE_RECEIPT}' | awk '{print \$1}')\" = '${EXPECTED_BASE_RECEIPT_SHA}'
cd '${BASE_SOURCE_ROOT}' && sha256sum -c --quiet '${BASE_RECEIPT}'
test \"\$(sha256sum '${DEPENDENCY_RECEIPT}' | awk '{print \$1}')\" = '${EXPECTED_DEPENDENCY_RECEIPT_SHA}'
test \"\$(sha256sum '${PROTOCOL}' | awk '{print \$1}')\" = 759fa88464791f175e0985a6a82469ca40a35119dfb827bb8f48f7d31ca3493e
test \"\$(sacct -j '${REPAIR_JOB}' -X -n -o State | awk 'NF{print \$1; exit}')\" = COMPLETED"

# Re-read every raw repair completion before it may satisfy a dependency.
remote "/scratch/lg154/conda-envs/memnav/bin/python - '${RUN_ROOT}' '${REPAIR_MANIFEST}' <<'PY'
import hashlib, json, sys
from pathlib import Path
run = Path(sys.argv[1])
manifest = json.loads(Path(sys.argv[2]).read_text())
assert manifest['selection_reads_navigation_outcomes'] is False
assert manifest['scientific_thresholds_changed'] is False
assert manifest['population_rule_changed'] is False
assert manifest['indices'] == [11, 15, 34, 40, 44]
for item in manifest['fragments']:
    index = int(item['scene_index'])
    scene = str(item['scene'])
    root = run / 'construct_ab/scenes' / f'{index:02d}_{scene}'
    completion = root / 'completion.json'
    sidecar = root / 'completion.json.sha256'
    digest = hashlib.sha256(completion.read_bytes()).hexdigest()
    assert sidecar.read_text().split() == [digest, 'completion.json']
    row = json.loads(completion.read_text())
    assert row['status'] == 'complete'
    assert row['query_policy_outcomes_read'] is False
    assert row['materialized_A_histories'] == 0
    assert row['constructible_AB_C_histories'] == 0
    assert row['upstream_parent_completion_sha256'] == item['parent_construction_completion_sha256']
print('zero-history repair raw receipts verified')
PY"

common="ALL,TASK_ROOT=${TASK_ROOT},BASE_SOURCE_ROOT=${BASE_SOURCE_ROOT},RUN_ROOT=${RUN_ROOT},PARENT_ROOT=${PARENT_ROOT},PROTOCOL=${PROTOCOL},TASK_RECEIPT=${TASK_RECEIPT},EXPECTED_TASK_RECEIPT_SHA=${EXPECTED_TASK_RECEIPT_SHA},BASE_RECEIPT=${BASE_RECEIPT},EXPECTED_BASE_RECEIPT_SHA=${EXPECTED_BASE_RECEIPT_SHA},DEPENDENCY_RECEIPT=${DEPENDENCY_RECEIPT},EXPECTED_DEPENDENCY_RECEIPT_SHA=${EXPECTED_DEPENDENCY_RECEIPT_SHA},EXPECTED_HAB_REQUESTS_VERSION=2.32.4,EXPECTED_HAB_REQUESTS_INIT_BYTES=5057,EXPECTED_HAB_REQUESTS_INIT_SHA=1e507f1f386bcc6b5f0ff69a614c14875cd65cb67be7f6022f28adef9774573f,EXPECTED_HAB_REQUESTS_VERSION_BYTES=435,EXPECTED_HAB_REQUESTS_VERSION_SHA=143abaf3563712f063743a7952aa65319dbcb934d894cfc989bd2c015f8da577"
seal_ab=${TASK_ROOT}/MemNavData/slurm_hm3d_fullmono_lifelong_finalize_ab.sbatch
collect_b=${TASK_ROOT}/MemNavData/slurm_hm3d_fullmono_lifelong_collect_b.sbatch
prefix=${TASK_ROOT}/MemNavData/slurm_hm3d_fullmono_lifelong_construct_prefix.sbatch
seal_population=${TASK_ROOT}/MemNavData/slurm_hm3d_fullmono_lifelong_finalize_population.sbatch
evaluate=${TASK_ROOT}/MemNavData/slurm_hm3d_fullmono_lifelong_eval.sbatch
analysis=${TASK_ROOT}/MemNavData/slurm_hm3d_fullmono_lifelong_analysis.sbatch

remote "sbatch --test-only --dependency=afterany:${ORIGINAL_BUILD_JOB} --export='${common}' '${seal_ab}' >/dev/null"
if [[ -n "${REPLACEMENT_SEAL_JOB}" ]]; then
  replacement_seal=${REPLACEMENT_SEAL_JOB}
  remote "scontrol show job '${replacement_seal}' | grep -q 'JobName=h3lifeSealAB'"
else
  replacement_seal=$(remote "sbatch --parsable --dependency=afterany:${ORIGINAL_BUILD_JOB} --kill-on-invalid-dep=yes --export='${common}' '${seal_ab}'" | job_id)
fi
[[ "${replacement_seal}" =~ ^[0-9]+$ ]] || fail "bad replacement seal job id"
if [[ -n "${REPLACEMENT_B_JOB}" ]]; then
  replacement_b=${REPLACEMENT_B_JOB}
  remote "scontrol show job '${replacement_b}' | grep -q 'JobName=h3lifeB'"
else
  replacement_b=$(remote "sbatch --parsable --qos=gpu48 --array=0-53%4 --dependency=afterok:${replacement_seal} --kill-on-invalid-dep=yes --export='${common},MAX_STEPS=600' '${collect_b}'" | job_id)
fi
[[ "${replacement_b}" =~ ^[0-9]+$ ]] || fail "bad replacement factual-B job id"

# Old dependency-pending arrays consume the per-user submitted-task quota.
# Cancel only those exact invalid downstream jobs after the replacement seal
# and factual-B nodes are known to exist; repeated invocation is harmless.
remote "scancel ${OLD_DOWNSTREAM_JOBS} 2>/dev/null || true"

replacement_prefix=$(remote "sbatch --parsable --qos=gpu48 --array=0-129%4 --dependency=afterok:${replacement_b} --kill-on-invalid-dep=yes --export='${common}' '${prefix}'" | job_id)
[[ "${replacement_prefix}" =~ ^[0-9]+$ ]] || fail "bad replacement prefix job id"
replacement_population=$(remote "sbatch --parsable --dependency=afterok:${replacement_prefix} --kill-on-invalid-dep=yes --export='${common}' '${seal_population}'" | job_id)
[[ "${replacement_population}" =~ ^[0-9]+$ ]] || fail "bad replacement population seal job id"
replacement_smoke=$(remote "sbatch --parsable --qos=gpu48 --array=0 --dependency=afterok:${replacement_population} --kill-on-invalid-dep=yes --export='${common},OUTPUT_ROOT=${SMOKE_ROOT},MAX_STEPS=80' '${evaluate}'" | job_id)
[[ "${replacement_smoke}" =~ ^[0-9]+$ ]] || fail "bad replacement smoke job id"
replacement_eval=$(remote "sbatch --parsable --qos=gpu48 --array=0-129%4 --dependency=afterok:${replacement_smoke} --kill-on-invalid-dep=yes --export='${common},OUTPUT_ROOT=${RUN_ROOT},MAX_STEPS=600' '${evaluate}'" | job_id)
[[ "${replacement_eval}" =~ ^[0-9]+$ ]] || fail "bad replacement eval job id"
replacement_aggregate=$(remote "sbatch --parsable --dependency=afterok:${replacement_eval} --kill-on-invalid-dep=yes --export='${common},MODE=aggregate' '${analysis}'" | job_id)
[[ "${replacement_aggregate}" =~ ^[0-9]+$ ]] || fail "bad replacement aggregate job id"
replacement_verify=$(remote "sbatch --parsable --dependency=afterok:${replacement_aggregate} --kill-on-invalid-dep=yes --export='${common},MODE=verify' '${analysis}'" | job_id)
[[ "${replacement_verify}" =~ ^[0-9]+$ ]] || fail "bad replacement verifier job id"

new_jobs=${replacement_seal},${replacement_b},${replacement_prefix},${replacement_population},${replacement_smoke},${replacement_eval},${replacement_aggregate},${replacement_verify}
remote "squeue -j '${new_jobs}' -h -o '%i|%T|%R' | grep -q ."

receipt=MemNavData/HM3D_FULLMONO_LIFELONG_DOWNSTREAM_REPAIR_RECEIPT_20260824.json
"${LOCAL_PY}" - "${receipt}" "${new_jobs}" <<'PY'
import json, sys
path, jobs = sys.argv[1:]
names = [
    'seal_AB_population', 'collect_factual_B_array',
    'construct_actual_prefix_array', 'seal_query_population',
    'remote_true_stack_smoke', 'formal_three_arm_evaluation',
    'aggregate', 'independent_verification',
]
ids = [int(value) for value in jobs.split(',')]
payload = {
    'schema_version': 'hm3d_fullmono_lifelong_downstream_repair_v1_20260824',
    'original_build_job_id': 16265026,
    'zero_history_repair_job_id': 16266646,
    'zero_history_repair_bundle_receipt_sha256': '171ccb30ff2c17f8523b7d533ded2705a80ffcf3b8a9e5990223e36995f6ad64',
    'replacement_seal_dependency': 'afterany:16265026',
    'zero_history_repair_prevalidated_before_submission': True,
    'selection_reads_navigation_outcomes': False,
    'scientific_thresholds_changed': False,
    'old_pending_downstream_jobs_cancelled': [16265034,16265042,16265051,16265056,16265066,16265073,16265078,16265088],
    'replacement_jobs': dict(zip(names, ids)),
}
open(path, 'x').write(json.dumps(payload, indent=2, sort_keys=True) + '\n')
print(json.dumps(payload, indent=2, sort_keys=True))
PY
timeout 120 scp -q -o BatchMode=yes -o ControlMaster=no \
  -o ControlPath="${SSH_CONTROL_PATH}" "${receipt}" \
  "${SSH_ALIAS}:${RUN_ROOT}/downstream_repair_submission.json" \
  || fail "downstream repair receipt upload failed"
remote "sha256sum '${RUN_ROOT}/downstream_repair_submission.json' >'${RUN_ROOT}/downstream_repair_submission.json.sha256'; squeue -j '${ORIGINAL_BUILD_JOB},${new_jobs}' -o '%.18i %.22j %.2t %.12M %.40R'"
printf 'REPLACEMENT_JOBS=%s\n' "${new_jobs}"
