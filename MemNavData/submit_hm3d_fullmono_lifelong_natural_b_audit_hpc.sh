#!/usr/bin/env bash
# Submit the result-blind 54-scene direct-natural-B constructibility audit.
# This script cannot launch factual-B, factual-C, B2, or any navigation arm.
set -euo pipefail
umask 0022

ROOT=${ROOT:-/home/asus/Research/Nav-graph-blind}
SSH_ALIAS=${SSH_ALIAS:-alantorch}
DRY_RUN=${DRY_RUN:-0}
AUDIT_CONCURRENCY=${AUDIT_CONCURRENCY:-4}
LOCAL_HAB_PY=${LOCAL_HAB_PY:-/home/asus/miniconda3/envs/habitat/bin/python}
REMOTE_BUNDLES=/scratch/yz11502/Research/Nav-axis-uturn-source-bundles
REMOTE_RESULTS=/scratch/yz11502/Research/Nav-axis-uturn-results/hm3d_fullmono_lifelong_power_v3_20260826
SOURCE_RUN_ROOT=${REMOTE_RESULTS}/formal_20260826T141733Z_375f0b68
PARENT_ROOT=/scratch/yz11502/Research/Nav-axis-uturn-results/hm3d_fresh_fullmono_mixed_role_20260820/formal_20260820T143609Z_e6dd44c6
SOURCE_TASK_ROOT=/scratch/yz11502/Research/Nav-axis-uturn-source-bundles/hm3d_fullmono_lifelong_375f0b6879b2ff87
SOURCE_TASK_RECEIPT=${SOURCE_TASK_ROOT}/SOURCE_BUNDLE.sha256
EXPECTED_SOURCE_TASK_RECEIPT_SHA=375f0b6879b2ff87b7019dae4727880d1b03fd3185a1862e6239942a76b5bcc8
BASE_SOURCE_ROOT=/scratch/yz11502/Research/Nav-axis-uturn-source-bundles/final14_mono_factorial_5690569a4373f2d2
BASE_RECEIPT=${BASE_SOURCE_ROOT}/source_inputs.sha256
EXPECTED_BASE_RECEIPT_SHA=5690569a4373f2d2768671418f0c604c4a03aa4b0ffe01baf70b288af03ba216
PROTOCOL=${SOURCE_TASK_ROOT}/MemNavData/hm3d_fullmono_lifelong_power_expansion_protocol_20260826.json
EXPECTED_PROTOCOL_SHA=127a6796c64eeafd4b48906baad09c48c41edb925fefe0fa964ccb584d4af228
PARENT_MANIFEST=${SOURCE_RUN_ROOT}/sealed_inputs/parent_manifest.json
EXPECTED_PARENT_MANIFEST_SHA=a96a0b96fab7b7b47709b36cb8eeb9410b42b09f095f87ef01304a68de716dd5
HAB_REQUESTS_VENDOR=/scratch/lg154/conda-envs/habitat/lib/python3.9/site-packages/pip/_vendor
BASE_SIF=/share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif
REMOTE_HAB_PY=/scratch/lg154/conda-envs/habitat/bin/python
SSH_CONTROL_PATH=${SSH_CONTROL_PATH:-$(
  ssh -G "${SSH_ALIAS}" 2>/dev/null |
    awk '$1=="controlpath"{value=$2} END{print value}'
)}

cd "${ROOT}"
fail() { echo "ABORT: $*" >&2; exit 2; }
remote() {
  timeout 180 ssh -n -tt -o BatchMode=yes -o ControlMaster=no \
    -S "${SSH_CONTROL_PATH}" "${SSH_ALIAS}" "$@"
}
remote_query() {
  timeout 180 ssh -n -T -o BatchMode=yes -o ControlMaster=no \
    -S "${SSH_CONTROL_PATH}" "${SSH_ALIAS}" "$@"
}
remote_query_readonly() {
  local label=$1 command=$2 attempt
  for attempt in 1 2 3; do
    if remote_query "${command}"; then
      return 0
    fi
    echo "read-only remote check ${label} channel attempt ${attempt} failed" >&2
  done
  return 1
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
    echo "rsync attempt ${attempt} failed; retrying immutable stage" >&2
  done
  return 1
}

[[ "${DRY_RUN}" =~ ^[01]$ ]] || fail "DRY_RUN must be 0 or 1"
[[ "${AUDIT_CONCURRENCY}" =~ ^[1-4]$ ]] || \
  fail "AUDIT_CONCURRENCY must be in [1,4]"
[[ -x "${LOCAL_HAB_PY}" ]] || fail "local Habitat interpreter missing"
[[ -n "${SSH_CONTROL_PATH}" && -S "${SSH_CONTROL_PATH}" ]] || \
  fail "authoritative shared SSH socket missing: ${SSH_CONTROL_PATH}"
timeout 15 ssh -O check -S "${SSH_CONTROL_PATH}" "${SSH_ALIAS}" \
  >/dev/null 2>&1 || fail "shared SSH master is not responsive"

files=(
  MemNavData/audit_hm3d_fullmono_lifelong_constructibility.py
  MemNavData/audit_hm3d_fullmono_lifelong_natural_b.py
  MemNavData/build_final14_role_pair_scene.py
  MemNavData/test_audit_hm3d_fullmono_lifelong_natural_b.py
  MemNavData/slurm_hm3d_fullmono_lifelong_natural_b_audit.sbatch
  MemNavData/slurm_hm3d_fullmono_lifelong_natural_b_audit_finalize.sbatch
  MemNavData/slurm_safe_submit.sh
  MemNavData/bundle_selftest.sh
  MemNavData/submit_hm3d_fullmono_lifelong_natural_b_audit_hpc.sh
)
for path in "${files[@]}"; do
  [[ -f "${path}" && ! -L "${path}" ]] || fail "missing physical input ${path}"
done

# Every Python input is byte-identical to the already successful GPU smoke.
[[ "$(sha256sum MemNavData/audit_hm3d_fullmono_lifelong_constructibility.py | awk '{print $1}')" == \
   2aed5a7efd96de521265f4e6b4561abdfe4cf65ae0b242ee7680c19d9fc9ef3f ]] || fail "constructibility auditor changed"
[[ "$(sha256sum MemNavData/audit_hm3d_fullmono_lifelong_natural_b.py | awk '{print $1}')" == \
   58025ce1b26ec1246199d2b8a8105d0cf9e63bf12d870e977e044f9ce96a63e9 ]] || fail "natural-B auditor changed"
[[ "$(sha256sum MemNavData/build_final14_role_pair_scene.py | awk '{print $1}')" == \
   5919d8bf58dd984879ce91634372ac7be0837596455cc864143054db0af397c1 ]] || fail "natural-goal constructor changed"
[[ "$(sha256sum MemNavData/test_audit_hm3d_fullmono_lifelong_natural_b.py | awk '{print $1}')" == \
   bf02059a1df9adf57f6d8f285f268108dc1bc693755e839b4c96925f4cf9f947 ]] || fail "natural-B tests changed"

bash -n \
  MemNavData/slurm_hm3d_fullmono_lifelong_natural_b_audit.sbatch \
  MemNavData/slurm_hm3d_fullmono_lifelong_natural_b_audit_finalize.sbatch \
  MemNavData/slurm_safe_submit.sh \
  MemNavData/bundle_selftest.sh \
  MemNavData/submit_hm3d_fullmono_lifelong_natural_b_audit_hpc.sh
(
  source MemNavData/slurm_safe_submit.sh
  lint_sbatch_template \
    MemNavData/slurm_hm3d_fullmono_lifelong_natural_b_audit.sbatch
  lint_sbatch_template \
    MemNavData/slurm_hm3d_fullmono_lifelong_natural_b_audit_finalize.sbatch
)

scratch=$(mktemp -d /tmp/h3life_natb_submit.XXXXXX)
cleanup() { rm -rf -- "${scratch}"; }
trap cleanup EXIT
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPYCACHEPREFIX=${scratch}/pycache
PYTHONPATH="${ROOT}:${ROOT}/MemNavData" "${LOCAL_HAB_PY}" -m unittest -q \
  MemNavData.test_audit_hm3d_fullmono_lifelong_natural_b

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
audit_root=${REMOTE_BUNDLES}/hm3d_lifelong_natural_b_audit_${bundle_key}
audit_stage=${audit_root}.partial.$$
run_tag=natural_b_audit_formal_$(date -u +%Y%m%dT%H%M%SZ)_${bundle_key:0:8}
run_root=${REMOTE_RESULTS}/${run_tag}

if [[ "${DRY_RUN}" == 1 ]]; then
  printf 'DRY_RUN_AUDIT_ROOT=%s\nDRY_RUN_RUN_ROOT=%s\nDRY_RUN_RECEIPT_SHA=%s\n' \
    "${audit_root}" "${run_root}" "${audit_receipt_sha}"
  exit 0
fi

remote_identity=$(remote 'id -un' | tr -d '\r')
[[ "${remote_identity}" == yz11502 ]] || fail "wrong remote identity: ${remote_identity}"
preflight_report=$(remote_query_readonly all_preflight "
  test -x '${REMOTE_HAB_PY}' && test -r '${BASE_SIF}' &&
  test -d '${SOURCE_RUN_ROOT}/construct_ab/scenes' &&
  echo SOURCE_RECEIPT=\$(sha256sum '${SOURCE_TASK_RECEIPT}' | awk '{print \$1}') &&
  echo BASE_RECEIPT=\$(sha256sum '${BASE_RECEIPT}' | awk '{print \$1}') &&
  echo PROTOCOL=\$(sha256sum '${PROTOCOL}' | awk '{print \$1}') &&
  echo PARENT_MANIFEST=\$(sha256sum '${PARENT_MANIFEST}' | awk '{print \$1}') &&
  (cd '${SOURCE_TASK_ROOT}' && sha256sum -c --quiet '${SOURCE_TASK_RECEIPT}') &&
  (cd '${BASE_SOURCE_ROOT}' && sha256sum -c --quiet '${BASE_RECEIPT}') &&
  '${REMOTE_HAB_PY}' -c \"import json; print('SCENE_COUNT=' + str(len(json.load(open('${PARENT_MANIFEST}'))['scenes'])))\" &&
  echo PREFLIGHT_COMPLETE")
value_for() {
  local key=$1
  printf '%s\n' "${preflight_report}" | awk -F= -v key="${key}" \
    '$1 == key {gsub(/[[:space:]]/, "", $2); print $2; exit}'
}
source_receipt_actual=$(value_for SOURCE_RECEIPT)
base_receipt_actual=$(value_for BASE_RECEIPT)
protocol_actual=$(value_for PROTOCOL)
parent_actual=$(value_for PARENT_MANIFEST)
[[ "${source_receipt_actual}" == "${EXPECTED_SOURCE_TASK_RECEIPT_SHA}" ]] || fail "source receipt changed: ${source_receipt_actual}"
[[ "${base_receipt_actual}" == "${EXPECTED_BASE_RECEIPT_SHA}" ]] || fail "base receipt changed: ${base_receipt_actual}"
[[ "${protocol_actual}" == "${EXPECTED_PROTOCOL_SHA}" ]] || fail "protocol changed: ${protocol_actual}"
[[ "${parent_actual}" == "${EXPECTED_PARENT_MANIFEST_SHA}" ]] || fail "parent manifest changed: ${parent_actual}"
scene_count=$(value_for SCENE_COUNT)
[[ "${scene_count}" == 54 ]] || fail "sealed scene count changed: ${scene_count}"
printf '%s\n' "${preflight_report}" | grep -qx PREFLIGHT_COMPLETE || fail "remote preflight incomplete"

if remote "test -d '${audit_root}'"; then
  remote "test \"\$(sha256sum '${audit_root}/SOURCE_BUNDLE.sha256' | awk '{print \$1}')\" = '${audit_receipt_sha}' && cd '${audit_root}' && sha256sum -c --quiet SOURCE_BUNDLE.sha256"
else
  remote "test ! -e '${audit_stage}' && mkdir -p '${audit_stage}'"
  upload_bundle "${scratch}/root" "${audit_stage}" || fail "bundle upload failed"
  remote "cd '${audit_stage}' && sha256sum -c --quiet SOURCE_BUNDLE.sha256 && chmod -R a-w '${audit_stage}' && mv '${audit_stage}' '${audit_root}'"
fi

audit_receipt=${audit_root}/SOURCE_BUNDLE.sha256
scene_script=${audit_root}/MemNavData/slurm_hm3d_fullmono_lifelong_natural_b_audit.sbatch
seal_script=${audit_root}/MemNavData/slurm_hm3d_fullmono_lifelong_natural_b_audit_finalize.sbatch
common="ALL,AUDIT_ROOT=${audit_root},SOURCE_TASK_ROOT=${SOURCE_TASK_ROOT},BASE_SOURCE_ROOT=${BASE_SOURCE_ROOT},RUN_ROOT=${run_root},PARENT_ROOT=${PARENT_ROOT},PROTOCOL=${PROTOCOL},AUDIT_RECEIPT=${audit_receipt},EXPECTED_AUDIT_RECEIPT_SHA=${audit_receipt_sha},SOURCE_TASK_RECEIPT=${SOURCE_TASK_RECEIPT},EXPECTED_SOURCE_TASK_RECEIPT_SHA=${EXPECTED_SOURCE_TASK_RECEIPT_SHA},BASE_RECEIPT=${BASE_RECEIPT},EXPECTED_BASE_RECEIPT_SHA=${EXPECTED_BASE_RECEIPT_SHA}"

# Exact overlay import/test under the production container.  Python bytecode
# goes to /tmp, never into an immutable bundle.
remote "tmp=\$(mktemp -d /tmp/h3life_natb_preflight.XXXXXX) && \
  singularity exec -B /scratch/lg154 -B /scratch/yz11502 '${BASE_SIF}' \
  env PYTHONDONTWRITEBYTECODE=1 PYTHONPYCACHEPREFIX=\${tmp} \
  PYTHONPATH='${audit_root}/MemNavData:${SOURCE_TASK_ROOT}:${SOURCE_TASK_ROOT}/MemNavData:${BASE_SOURCE_ROOT}:${BASE_SOURCE_ROOT}/MemNavData:${HAB_REQUESTS_VENDOR}' \
  '${REMOTE_HAB_PY}' -m unittest -q test_audit_hm3d_fullmono_lifelong_natural_b; \
  rc=\$?; rm -rf -- \${tmp}; exit \${rc}"

# Test-only uses the same safe wrapper as formal submission.
remote "source '${audit_root}/MemNavData/slurm_safe_submit.sh'; \
  safe_sbatch --lint-fatal --test-only --qos=gpu48 --array=0 \
  --export='${common}' '${scene_script}' >/dev/null; \
  safe_sbatch --lint-fatal --test-only --partition=cpu_short \
  --export='${common}' '${seal_script}' >/dev/null"

remote "test ! -e '${run_root}' && mkdir -p '${run_root}/natural_b_audit/scenes' && \
  ln -s '${SOURCE_RUN_ROOT}/sealed_inputs' '${run_root}/sealed_inputs' && \
  ln -s '${SOURCE_RUN_ROOT}/construct_ab' '${run_root}/construct_ab' && \
  test \"\$(readlink -f '${run_root}/sealed_inputs')\" = '${SOURCE_RUN_ROOT}/sealed_inputs' && \
  test \"\$(readlink -f '${run_root}/construct_ab')\" = '${SOURCE_RUN_ROOT}/construct_ab'"

scene_id=$(remote "source '${audit_root}/MemNavData/slurm_safe_submit.sh'; \
  safe_sbatch --lint-fatal --parsable --qos=gpu48 \
  --array=0-53%${AUDIT_CONCURRENCY} --export='${common}' \
  '${scene_script}'" | job_id)
[[ "${scene_id}" =~ ^[0-9]+$ ]] || fail "bad scene-array job id"
seal_id=$(remote "source '${audit_root}/MemNavData/slurm_safe_submit.sh'; \
  safe_sbatch --lint-fatal --parsable --partition=cpu_short \
  --dependency=afterany:${scene_id} --export='${common}' \
  '${seal_script}'" | job_id)
[[ "${seal_id}" =~ ^[0-9]+$ ]] || fail "bad seal job id"

receipt=${scratch}/submission.json
printf '{\n  "schema_version": "hm3d_fullmono_lifelong_natural_b_audit_submission_v2_20260827",\n  "scope": "result-blind 54-scene multi-candidate constructibility audit",\n  "source_run_root": "%s",\n  "audit_bundle": "%s",\n  "audit_bundle_receipt_sha256": "%s",\n  "run_root": "%s",\n  "scene_array_job": %s,\n  "scene_array": "0-53%%%s",\n  "scene_partitions": ["h100_tandon", "a100_tandon"],\n  "time_per_scene": "00:45:00",\n  "aggregate_job": %s,\n  "aggregate_dependency": "afterany:%s",\n  "query_policy_outcomes_read": false,\n  "navigation_outcomes_read": false,\n  "navigation_rollouts_authorized": false,\n  "evaluation_authorized": false\n}\n' \
  "${SOURCE_RUN_ROOT}" "${audit_root}" "${audit_receipt_sha}" \
  "${run_root}" "${scene_id}" "${AUDIT_CONCURRENCY}" "${seal_id}" \
  "${scene_id}" >"${receipt}"
timeout 120 rsync -a \
  -e "ssh -o BatchMode=yes -o ControlMaster=no -S ${SSH_CONTROL_PATH}" \
  "${receipt}" "${SSH_ALIAS}:${run_root}/submission.json"
remote "sha256sum '${run_root}/submission.json' >'${run_root}/submission.json.sha256' && chmod a-w '${run_root}/submission.json' '${run_root}/submission.json.sha256'"

printf 'RUN_ROOT=%s\nAUDIT_ROOT=%s\nSCENE_ARRAY_JOB=%s\nAGGREGATE_JOB=%s\n' \
  "${run_root}" "${audit_root}" "${scene_id}" "${seal_id}"
