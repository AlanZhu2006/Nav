#!/usr/bin/env bash
# Submit only the result-blind additional Natural-B constructibility audit.
# This launcher cannot run factual B, factual C, B2, C2, or any policy arm.
set -euo pipefail
umask 0022

ROOT=${ROOT:-$(git rev-parse --show-toplevel)}
SSH_ALIAS=${SSH_ALIAS:-alantorch}
DRY_RUN=${DRY_RUN:-0}
AUDIT_CONCURRENCY=${AUDIT_CONCURRENCY:-4}
LOCAL_HAB_PY=${LOCAL_HAB_PY:-/home/asus/miniconda3/envs/habitat/bin/python}
REMOTE_BUNDLES=/scratch/yz11502/Research/Nav-axis-uturn-source-bundles
REMOTE_RESULTS=/scratch/yz11502/Research/Nav-axis-uturn-results/hm3d_fullmono_lifelong_natural_b_expansion_20260828
PARENT_ROOT=/scratch/yz11502/Research/Nav-axis-uturn-results/hm3d_fresh_fullmono_mixed_role_20260820/formal_20260820T143609Z_e6dd44c6
SOURCE_RUN_ROOT=/scratch/yz11502/Research/Nav-axis-uturn-results/hm3d_fullmono_lifelong_power_v3_20260826/formal_20260826T141733Z_375f0b68
SOURCE_TASK_ROOT=/scratch/yz11502/Research/Nav-axis-uturn-source-bundles/hm3d_fullmono_lifelong_375f0b6879b2ff87
SOURCE_TASK_RECEIPT=${SOURCE_TASK_ROOT}/SOURCE_BUNDLE.sha256
EXPECTED_SOURCE_TASK_RECEIPT_SHA=375f0b6879b2ff87b7019dae4727880d1b03fd3185a1862e6239942a76b5bcc8
SOURCE_PROTOCOL_SHA=127a6796c64eeafd4b48906baad09c48c41edb925fefe0fa964ccb584d4af228
BASE_SOURCE_ROOT=/scratch/yz11502/Research/Nav-axis-uturn-source-bundles/final14_mono_factorial_5690569a4373f2d2
BASE_RECEIPT=${BASE_SOURCE_ROOT}/source_inputs.sha256
EXPECTED_BASE_RECEIPT_SHA=5690569a4373f2d2768671418f0c604c4a03aa4b0ffe01baf70b288af03ba216
ORIGINAL_V4_ROOT=/scratch/yz11502/Research/Nav-axis-uturn-source-bundles/hm3d_lifelong_natural_v4_d85fc50df19b1384
ORIGINAL_V4_RECEIPT=${ORIGINAL_V4_ROOT}/SOURCE_BUNDLE.sha256
EXPECTED_ORIGINAL_V4_RECEIPT_SHA=d85fc50df19b138499e07bb9555d9f9a0088da0f5040e32f78f24b74a975c59a
CURRENT_V4_RUN_ROOT=/scratch/yz11502/Research/Nav-axis-uturn-results/hm3d_fullmono_lifelong_natural_v4_20260827/formal_materialize_20260827T133704Z_d85fc50d
ORIGINAL_MANIFEST=${CURRENT_V4_RUN_ROOT}/ab_population/role_pairs/manifest.json
ORIGINAL_MANIFEST_SHA=e5c97dad42b26c67032c5b42a8467d857ce57f4b627a3f4851616a159ecef978
ORIGINAL_POPULATION=${CURRENT_V4_RUN_ROOT}/population/population.json
ORIGINAL_POPULATION_SHA=ec11c0dbc43a4abe585330c1ce52a8c14ad1d4b1da6fd8397e1d15592707a6d5
ORIGINAL_POPULATION_VERIFY=${CURRENT_V4_RUN_ROOT}/independent_natural_v4_population_verification.json
ORIGINAL_POPULATION_VERIFY_SHA=d9ce97df4b0687969090e710ef719f6da56fc5d39a0535a7e8afd6c5d852499b
BASE_SIF=/share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif
REMOTE_HAB_PY=/scratch/lg154/conda-envs/habitat/bin/python
HAB_REQUESTS_VENDOR=/scratch/lg154/conda-envs/habitat/lib/python3.9/site-packages/pip/_vendor
SSH_CONTROL_PATH=${SSH_CONTROL_PATH:-$(
  ssh -G "${SSH_ALIAS}" 2>/dev/null |
    awk '$1=="controlpath"{value=$2} END{print value}'
)}

cd "${ROOT}"
fail() { echo "ABORT: $*" >&2; exit 2; }
remote() {
  timeout 300 ssh -n -T -o BatchMode=yes -o ControlMaster=no \
    -S "${SSH_CONTROL_PATH}" "${SSH_ALIAS}" "$@" | tr -d '\r'
}
job_id() {
  awk -F';' '/^[0-9]+(;|$)/ {print $1; exit}'
}
upload_bundle() {
  local source=$1 destination=$2
  timeout 300 rsync -a --partial \
    --chmod=Du=rwx,Dgo=rx,Fu=rw,Fgo=r \
    -e "ssh -o BatchMode=yes -o ControlMaster=no -S ${SSH_CONTROL_PATH}" \
    "${source}/" "${SSH_ALIAS}:${destination}/"
}

[[ "${DRY_RUN}" =~ ^[01]$ ]] || fail "DRY_RUN must be 0 or 1"
[[ "${AUDIT_CONCURRENCY}" =~ ^[1-4]$ ]] || \
  fail "AUDIT_CONCURRENCY must be in [1,4]"
[[ -x "${LOCAL_HAB_PY}" ]] || fail "local Habitat interpreter missing"
[[ -n "${SSH_CONTROL_PATH}" && -S "${SSH_CONTROL_PATH}" ]] || \
  fail "authoritative shared SSH socket missing"
timeout 15 ssh -O check -S "${SSH_CONTROL_PATH}" "${SSH_ALIAS}" \
  >/dev/null 2>&1 || fail "shared SSH master is not responsive"

files=(
  MemNavData/audit_hm3d_fullmono_lifelong_constructibility.py
  MemNavData/audit_hm3d_fullmono_lifelong_natural_b_expansion.py
  MemNavData/build_final14_role_pair_scene.py
  MemNavData/independent_verify_hm3d_fullmono_lifelong_natural_b_expansion.py
  MemNavData/test_audit_hm3d_fullmono_lifelong_natural_b_expansion.py
  MemNavData/hm3d_fullmono_lifelong_natural_b_expansion_audit_protocol_20260828.json
  MemNavData/slurm_hm3d_fullmono_lifelong_natural_b_expansion_audit.sbatch
  MemNavData/slurm_hm3d_fullmono_lifelong_natural_b_expansion_finalize.sbatch
  MemNavData/slurm_independent_verify_hm3d_fullmono_lifelong_natural_b_expansion.sbatch
  MemNavData/slurm_safe_submit.sh
  MemNavData/bundle_selftest.sh
  MemNavData/submit_hm3d_fullmono_lifelong_natural_b_expansion_audit_hpc.sh
)
for path in "${files[@]}"; do
  [[ -f "${path}" && ! -L "${path}" ]] || fail "missing input ${path}"
done
[[ "$(sha256sum MemNavData/audit_hm3d_fullmono_lifelong_constructibility.py | awk '{print $1}')" == \
   2aed5a7efd96de521265f4e6b4561abdfe4cf65ae0b242ee7680c19d9fc9ef3f ]] || \
  fail "frozen constructibility auditor changed"
[[ "$(sha256sum MemNavData/build_final14_role_pair_scene.py | awk '{print $1}')" == \
   5919d8bf58dd984879ce91634372ac7be0837596455cc864143054db0af397c1 ]] || \
  fail "frozen Natural-B constructor changed"

python -m json.tool \
  MemNavData/hm3d_fullmono_lifelong_natural_b_expansion_audit_protocol_20260828.json \
  >/dev/null
bash -n \
  MemNavData/slurm_hm3d_fullmono_lifelong_natural_b_expansion_audit.sbatch \
  MemNavData/slurm_hm3d_fullmono_lifelong_natural_b_expansion_finalize.sbatch \
  MemNavData/slurm_independent_verify_hm3d_fullmono_lifelong_natural_b_expansion.sbatch \
  MemNavData/slurm_safe_submit.sh MemNavData/bundle_selftest.sh \
  MemNavData/submit_hm3d_fullmono_lifelong_natural_b_expansion_audit_hpc.sh
(
  source MemNavData/slurm_safe_submit.sh
  lint_sbatch_template \
    MemNavData/slurm_hm3d_fullmono_lifelong_natural_b_expansion_audit.sbatch
  lint_sbatch_template \
    MemNavData/slurm_hm3d_fullmono_lifelong_natural_b_expansion_finalize.sbatch
  lint_sbatch_template \
    MemNavData/slurm_independent_verify_hm3d_fullmono_lifelong_natural_b_expansion.sbatch
)

scratch=$(mktemp -d /tmp/h3life_natbx_submit.XXXXXX)
cleanup() { rm -rf -- "${scratch}"; }
trap cleanup EXIT
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPYCACHEPREFIX=${scratch}/pycache
PYTHONPATH="${ROOT}:${ROOT}/MemNavData" "${LOCAL_HAB_PY}" -m unittest -q \
  MemNavData.test_audit_hm3d_fullmono_lifelong_natural_b_expansion

mkdir -p "${scratch}/root/MemNavData"
for path in "${files[@]}"; do
  cp -p -- "${path}" "${scratch}/root/MemNavData/$(basename "${path}")"
done
(
  cd "${scratch}/root"
  find . -type f ! -name SOURCE_BUNDLE.sha256 -print0 | sort -z | \
    xargs -0 sha256sum >SOURCE_BUNDLE.sha256
  sha256sum -c --quiet SOURCE_BUNDLE.sha256
)
audit_receipt_sha=$(sha256sum "${scratch}/root/SOURCE_BUNDLE.sha256" | awk '{print $1}')
bundle_key=${audit_receipt_sha:0:16}
audit_root=${REMOTE_BUNDLES}/hm3d_lifelong_natural_b_expansion_${bundle_key}
audit_stage=${audit_root}.partial.$$
run_tag=natural_b_expansion_audit_$(date -u +%Y%m%dT%H%M%SZ)_${bundle_key:0:8}
run_root=${REMOTE_RESULTS}/${run_tag}

if [[ "${DRY_RUN}" == 1 ]]; then
  printf 'DRY_RUN_AUDIT_ROOT=%s\nDRY_RUN_RUN_ROOT=%s\nDRY_RUN_RECEIPT_SHA=%s\n' \
    "${audit_root}" "${run_root}" "${audit_receipt_sha}"
  exit 0
fi

preflight=$(remote "set -euo pipefail
test \"\$(id -un)\" = yz11502
test -x '${REMOTE_HAB_PY}'
test -r '${BASE_SIF}'
test \"\$(sha256sum '${SOURCE_TASK_RECEIPT}' | awk '{print \$1}')\" = '${EXPECTED_SOURCE_TASK_RECEIPT_SHA}'
test \"\$(sha256sum '${BASE_RECEIPT}' | awk '{print \$1}')\" = '${EXPECTED_BASE_RECEIPT_SHA}'
test \"\$(sha256sum '${ORIGINAL_V4_RECEIPT}' | awk '{print \$1}')\" = '${EXPECTED_ORIGINAL_V4_RECEIPT_SHA}'
test \"\$(sha256sum '${SOURCE_TASK_ROOT}/MemNavData/hm3d_fullmono_lifelong_power_expansion_protocol_20260826.json' | awk '{print \$1}')\" = '${SOURCE_PROTOCOL_SHA}'
test \"\$(sha256sum '${ORIGINAL_MANIFEST}' | awk '{print \$1}')\" = '${ORIGINAL_MANIFEST_SHA}'
test \"\$(sha256sum '${ORIGINAL_POPULATION}' | awk '{print \$1}')\" = '${ORIGINAL_POPULATION_SHA}'
test \"\$(sha256sum '${ORIGINAL_POPULATION_VERIFY}' | awk '{print \$1}')\" = '${ORIGINAL_POPULATION_VERIFY_SHA}'
cd '${SOURCE_TASK_ROOT}' && sha256sum -c --quiet '${SOURCE_TASK_RECEIPT}'
cd '${BASE_SOURCE_ROOT}' && sha256sum -c --quiet '${BASE_RECEIPT}'
cd '${ORIGINAL_V4_ROOT}' && sha256sum -c --quiet '${ORIGINAL_V4_RECEIPT}'
'${REMOTE_HAB_PY}' - '${ORIGINAL_POPULATION_VERIFY}' <<'PY'
import json,sys
p=json.load(open(sys.argv[1]))
assert p['verified'] is True
assert p['supported_population'] == 22 and p['scene_clusters'] == 15
assert p['target_met'] is False
assert p['factual_C_B2_C2_executed'] is False
assert p['query_navigation_outcomes_read'] is False
PY
echo PREFLIGHT_OK")
[[ "${preflight}" == *PREFLIGHT_OK* ]] || fail "remote preflight incomplete"

if remote "test -d '${audit_root}'" >/dev/null 2>&1; then
  remote "test \"\$(sha256sum '${audit_root}/SOURCE_BUNDLE.sha256' | awk '{print \$1}')\" = '${audit_receipt_sha}'; cd '${audit_root}'; sha256sum -c --quiet SOURCE_BUNDLE.sha256" >/dev/null
else
  remote "test ! -e '${audit_stage}'; mkdir -p '${audit_stage}'" >/dev/null
  upload_bundle "${scratch}/root" "${audit_stage}" || fail "bundle upload failed"
  remote "cd '${audit_stage}'; sha256sum -c --quiet SOURCE_BUNDLE.sha256; chmod -R a-w '${audit_stage}'; mv '${audit_stage}' '${audit_root}'" >/dev/null
fi

audit_receipt=${audit_root}/SOURCE_BUNDLE.sha256
scene_script=${audit_root}/MemNavData/slurm_hm3d_fullmono_lifelong_natural_b_expansion_audit.sbatch
seal_script=${audit_root}/MemNavData/slurm_hm3d_fullmono_lifelong_natural_b_expansion_finalize.sbatch
verify_script=${audit_root}/MemNavData/slurm_independent_verify_hm3d_fullmono_lifelong_natural_b_expansion.sbatch
common="ALL,AUDIT_ROOT=${audit_root},SOURCE_TASK_ROOT=${SOURCE_TASK_ROOT},BASE_SOURCE_ROOT=${BASE_SOURCE_ROOT},RUN_ROOT=${run_root},PARENT_ROOT=${PARENT_ROOT},SOURCE_RUN_ROOT=${SOURCE_RUN_ROOT},CURRENT_V4_RUN_ROOT=${CURRENT_V4_RUN_ROOT},AUDIT_RECEIPT=${audit_receipt},EXPECTED_AUDIT_RECEIPT_SHA=${audit_receipt_sha},SOURCE_TASK_RECEIPT=${SOURCE_TASK_RECEIPT},EXPECTED_SOURCE_TASK_RECEIPT_SHA=${EXPECTED_SOURCE_TASK_RECEIPT_SHA},BASE_RECEIPT=${BASE_RECEIPT},EXPECTED_BASE_RECEIPT_SHA=${EXPECTED_BASE_RECEIPT_SHA},ORIGINAL_MANIFEST_SHA=${ORIGINAL_MANIFEST_SHA},SOURCE_PROTOCOL_SHA=${SOURCE_PROTOCOL_SHA}"

remote "tmp=\$(mktemp -d /tmp/h3life_natbx_preflight.XXXXXX); singularity exec -B /scratch/lg154 -B /scratch/yz11502 '${BASE_SIF}' env PYTHONDONTWRITEBYTECODE=1 PYTHONPYCACHEPREFIX=\${tmp} PYTHONPATH='${audit_root}/MemNavData:${SOURCE_TASK_ROOT}:${SOURCE_TASK_ROOT}/MemNavData:${BASE_SOURCE_ROOT}:${BASE_SOURCE_ROOT}/MemNavData:${HAB_REQUESTS_VENDOR}' '${REMOTE_HAB_PY}' -m unittest -q test_audit_hm3d_fullmono_lifelong_natural_b_expansion; rc=\$?; rm -rf -- \${tmp}; exit \${rc}" >/dev/null

remote "source '${audit_root}/MemNavData/slurm_safe_submit.sh'; safe_sbatch --lint-fatal --test-only --qos=gpu48 --array=0 --export='${common}' '${scene_script}' >/dev/null; safe_sbatch --lint-fatal --test-only --partition=cpu_short --export='${common}' '${seal_script}' >/dev/null; safe_sbatch --lint-fatal --test-only --partition=cpu_short --export='${common}' '${verify_script}' >/dev/null; echo TEST_ONLY_OK" >/dev/null

remote "test ! -e '${run_root}'; mkdir -p '${run_root}/expansion_audit/scenes'" >/dev/null
scene_id=$(remote "source '${audit_root}/MemNavData/slurm_safe_submit.sh'; safe_sbatch --lint-fatal --parsable --qos=gpu48 --array=0-53%${AUDIT_CONCURRENCY} --export='${common}' '${scene_script}'" | job_id)
[[ "${scene_id}" =~ ^[0-9]+$ ]] || fail "bad scene-array job id"
seal_id=$(remote "source '${audit_root}/MemNavData/slurm_safe_submit.sh'; safe_sbatch --lint-fatal --parsable --partition=cpu_short --dependency=afterany:${scene_id} --export='${common}' '${seal_script}'" | job_id)
[[ "${seal_id}" =~ ^[0-9]+$ ]] || fail "bad seal job id"
verify_id=$(remote "source '${audit_root}/MemNavData/slurm_safe_submit.sh'; safe_sbatch --lint-fatal --parsable --partition=cpu_short --dependency=afterok:${seal_id} --kill-on-invalid-dep=yes --export='${common}' '${verify_script}'" | job_id)
[[ "${verify_id}" =~ ^[0-9]+$ ]] || fail "bad verifier job id"

receipt=${scratch}/submission.json
python - "${receipt}" "${audit_root}" "${audit_receipt_sha}" "${run_root}" \
  "${scene_id}" "${seal_id}" "${verify_id}" "${AUDIT_CONCURRENCY}" <<'PY'
import json,sys
out,audit,sha,run,scene,seal,verify,concurrency=sys.argv[1:]
payload={
 "schema_version":"hm3d_fullmono_lifelong_natural_b_expansion_audit_submission_v1_20260828",
 "scope":"result-blind additional Natural-B constructibility audit only",
 "audit_bundle":audit,"audit_bundle_receipt_sha256":sha,"run_root":run,
 "scene_array_job":int(scene),"scene_array":f"0-53%{concurrency}",
 "aggregate_job":int(seal),"independent_verifier_job":int(verify),
 "query_policy_outcomes_read":False,"navigation_outcomes_read":False,
 "factual_B_authorized":False,"C_B2_C2_authorized":False,
 "evaluation_authorized":False,
}
open(out,'w').write(json.dumps(payload,indent=2,sort_keys=True)+'\n')
PY
timeout 120 rsync -a \
  -e "ssh -o BatchMode=yes -o ControlMaster=no -S ${SSH_CONTROL_PATH}" \
  "${receipt}" "${SSH_ALIAS}:${run_root}/submission.json"
remote "sha256sum '${run_root}/submission.json' >'${run_root}/submission.json.sha256'; chmod a-w '${run_root}/submission.json' '${run_root}/submission.json.sha256'" >/dev/null

printf 'RUN_ROOT=%s\nAUDIT_ROOT=%s\nSCENE_ARRAY_JOB=%s\nAGGREGATE_JOB=%s\nVERIFY_JOB=%s\n' \
  "${run_root}" "${audit_root}" "${scene_id}" "${seal_id}" "${verify_id}"
