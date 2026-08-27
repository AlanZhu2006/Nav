#!/usr/bin/env bash
# Materialize only the sealed v4 A/Natural-B/Revisit-C query assets.
# This script cannot launch factual B or any C/B2/C2 navigation rollout.
set -euo pipefail
umask 0022

ROOT=${ROOT:-/home/asus/Research/Nav-graph-blind}
SSH_ALIAS=${SSH_ALIAS:-alantorch}
MODE=${MODE:-smoke}
SMOKE_SCENE_INDEX=${SMOKE_SCENE_INDEX:-0}
CONCURRENCY=${CONCURRENCY:-4}
LOCAL_HAB_PY=${LOCAL_HAB_PY:-/home/asus/miniconda3/envs/habitat/bin/python}
REMOTE_BUNDLES=/scratch/yz11502/Research/Nav-axis-uturn-source-bundles
REMOTE_RESULTS=/scratch/yz11502/Research/Nav-axis-uturn-results/hm3d_fullmono_lifelong_natural_v4_20260827
PARENT_ROOT=/scratch/yz11502/Research/Nav-axis-uturn-results/hm3d_fresh_fullmono_mixed_role_20260820/formal_20260820T143609Z_e6dd44c6
PARENT_MANIFEST=${PARENT_ROOT}/sealed_inputs/parent_manifest.json
EXPECTED_PARENT_MANIFEST_SHA=a96a0b96fab7b7b47709b36cb8eeb9410b42b09f095f87ef01304a68de716dd5
AUDIT_RUN_ROOT=/scratch/yz11502/Research/Nav-axis-uturn-results/hm3d_fullmono_lifelong_power_v3_20260826/natural_b_audit_formal_20260827T125748Z_e2832e17
AUDIT_SUMMARY=${AUDIT_RUN_ROOT}/natural_b_audit/summary.json
EXPECTED_AUDIT_SUMMARY_SHA=4edbb3f076360063f3dd267a62d90c09d0cd1425973404e47b41bb4cc04ad60f
AUDIT_VERIFY=${AUDIT_RUN_ROOT}/independent_natural_b_verification.json
EXPECTED_AUDIT_VERIFY_SHA=52b15b5e05f21e5ab3bc460f351bdc2068d2bdf121a1eae08646a2d3dd591ab7
SOURCE_CONSTRUCTION_ROOT=/scratch/yz11502/Research/Nav-axis-uturn-results/hm3d_fullmono_lifelong_power_v3_20260826/formal_20260826T141733Z_375f0b68/construct_ab/scenes
AUDIT_SOURCE_ROOT=/scratch/yz11502/Research/Nav-axis-uturn-source-bundles/hm3d_lifelong_natural_b_audit_e2832e17231534e3
AUDIT_SOURCE_RECEIPT=${AUDIT_SOURCE_ROOT}/SOURCE_BUNDLE.sha256
EXPECTED_AUDIT_SOURCE_RECEIPT_SHA=e2832e17231534e38db1b3b507ddf68881ee8bd56548c3151235e52a397f3121
SOURCE_TASK_ROOT=/scratch/yz11502/Research/Nav-axis-uturn-source-bundles/hm3d_fullmono_lifelong_375f0b6879b2ff87
SOURCE_TASK_RECEIPT=${SOURCE_TASK_ROOT}/SOURCE_BUNDLE.sha256
EXPECTED_SOURCE_TASK_RECEIPT_SHA=375f0b6879b2ff87b7019dae4727880d1b03fd3185a1862e6239942a76b5bcc8
BASE_SOURCE_ROOT=/scratch/yz11502/Research/Nav-axis-uturn-source-bundles/final14_mono_factorial_5690569a4373f2d2
BASE_RECEIPT=${BASE_SOURCE_ROOT}/source_inputs.sha256
EXPECTED_BASE_RECEIPT_SHA=5690569a4373f2d2768671418f0c604c4a03aa4b0ffe01baf70b288af03ba216
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
  timeout 180 ssh -n -tt -o BatchMode=yes -o ControlMaster=no \
    -S "${SSH_CONTROL_PATH}" "${SSH_ALIAS}" "$@"
}
remote_query() {
  timeout 180 ssh -n -T -o BatchMode=yes -o ControlMaster=no \
    -S "${SSH_CONTROL_PATH}" "${SSH_ALIAS}" "$@"
}
job_id() {
  tr -d '\r' | awk -F';' '/^[0-9]+(;|$)/ {print $1; exit}'
}

[[ "${MODE}" == smoke || "${MODE}" == formal ]] || \
  fail "MODE must be smoke or formal"
[[ "${SMOKE_SCENE_INDEX}" =~ ^[0-9]+$ && "${SMOKE_SCENE_INDEX}" -lt 54 ]] || \
  fail "invalid smoke scene index"
[[ "${CONCURRENCY}" =~ ^[1-4]$ ]] || fail "CONCURRENCY must be in [1,4]"
[[ -x "${LOCAL_HAB_PY}" ]] || fail "local Habitat Python missing"
[[ -n "${SSH_CONTROL_PATH}" && -S "${SSH_CONTROL_PATH}" ]] || \
  fail "authoritative shared SSH socket missing"
timeout 15 ssh -O check -S "${SSH_CONTROL_PATH}" "${SSH_ALIAS}" \
  >/dev/null 2>&1 || fail "shared SSH master is not responsive"

files=(
  MemNavData/hm3d_fullmono_lifelong.py
  MemNavData/hm3d_fullmono_lifelong_direct_natural_protocol_20260827.json
  MemNavData/materialize_hm3d_fullmono_lifelong_natural_ab.py
  MemNavData/test_materialize_hm3d_fullmono_lifelong_natural_ab.py
  MemNavData/finalize_hm3d_fullmono_lifelong_ab.py
  MemNavData/slurm_hm3d_fullmono_lifelong_natural_v4_materialize.sbatch
  MemNavData/slurm_hm3d_fullmono_lifelong_natural_v4_finalize.sbatch
  MemNavData/slurm_safe_submit.sh
  MemNavData/submit_hm3d_fullmono_lifelong_natural_v4_materialize_hpc.sh
)
for path in "${files[@]}"; do
  [[ -f "${path}" && ! -L "${path}" ]] || fail "missing physical input ${path}"
done
bash -n \
  MemNavData/slurm_hm3d_fullmono_lifelong_natural_v4_materialize.sbatch \
  MemNavData/slurm_hm3d_fullmono_lifelong_natural_v4_finalize.sbatch \
  MemNavData/slurm_safe_submit.sh \
  MemNavData/submit_hm3d_fullmono_lifelong_natural_v4_materialize_hpc.sh
(
  source MemNavData/slurm_safe_submit.sh
  lint_sbatch_template \
    MemNavData/slurm_hm3d_fullmono_lifelong_natural_v4_materialize.sbatch
  lint_sbatch_template \
    MemNavData/slurm_hm3d_fullmono_lifelong_natural_v4_finalize.sbatch
)
scratch=$(mktemp -d /tmp/h3life_v4_submit.XXXXXX)
cleanup() { rm -rf -- "${scratch}"; }
trap cleanup EXIT
PYTHONDONTWRITEBYTECODE=1 PYTHONPYCACHEPREFIX=${scratch}/pycache \
  PYTHONPATH="${ROOT}/MemNavData" "${LOCAL_HAB_PY}" -m unittest -q \
  test_materialize_hm3d_fullmono_lifelong_natural_ab

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
v4_receipt_sha=$(sha256sum "${scratch}/root/SOURCE_BUNDLE.sha256" | awk '{print $1}')
bundle_key=${v4_receipt_sha:0:16}
v4_root=${REMOTE_BUNDLES}/hm3d_lifelong_natural_v4_${bundle_key}
v4_stage=${v4_root}.partial.$$
run_tag=${MODE}_materialize_$(date -u +%Y%m%dT%H%M%SZ)_${bundle_key:0:8}
run_root=${REMOTE_RESULTS}/${run_tag}

identity=$(remote 'id -un' | tr -d '\r')
[[ "${identity}" == yz11502 ]] || fail "wrong remote identity: ${identity}"
preflight=$(remote_query "set -e; \
  echo PARENT=\$(sha256sum '${PARENT_MANIFEST}' | awk '{print \$1}'); \
  echo AUDIT_SUMMARY=\$(sha256sum '${AUDIT_SUMMARY}' | awk '{print \$1}'); \
  echo AUDIT_VERIFY=\$(sha256sum '${AUDIT_VERIFY}' | awk '{print \$1}'); \
  echo AUDIT_RECEIPT=\$(sha256sum '${AUDIT_SOURCE_RECEIPT}' | awk '{print \$1}'); \
  echo SOURCE_RECEIPT=\$(sha256sum '${SOURCE_TASK_RECEIPT}' | awk '{print \$1}'); \
  echo BASE_RECEIPT=\$(sha256sum '${BASE_RECEIPT}' | awk '{print \$1}'); \
  test -d '${SOURCE_CONSTRUCTION_ROOT}'; test -x '${REMOTE_HAB_PY}'; \
  test -r '${BASE_SIF}'; echo COMPLETE")
value_for() {
  local key=$1
  printf '%s\n' "${preflight}" | tr -d '\r' | awk -F= -v key="${key}" \
    '$1==key {print $2; exit}'
}
[[ "$(value_for PARENT)" == "${EXPECTED_PARENT_MANIFEST_SHA}" ]] || fail "parent changed"
[[ "$(value_for AUDIT_SUMMARY)" == "${EXPECTED_AUDIT_SUMMARY_SHA}" ]] || fail "audit summary changed"
[[ "$(value_for AUDIT_VERIFY)" == "${EXPECTED_AUDIT_VERIFY_SHA}" ]] || fail "audit verifier changed"
[[ "$(value_for AUDIT_RECEIPT)" == "${EXPECTED_AUDIT_SOURCE_RECEIPT_SHA}" ]] || fail "audit source changed"
[[ "$(value_for SOURCE_RECEIPT)" == "${EXPECTED_SOURCE_TASK_RECEIPT_SHA}" ]] || fail "source task changed"
[[ "$(value_for BASE_RECEIPT)" == "${EXPECTED_BASE_RECEIPT_SHA}" ]] || fail "base source changed"
printf '%s\n' "${preflight}" | tr -d '\r' | grep -qx COMPLETE || fail "preflight incomplete"

if remote "test -d '${v4_root}'"; then
  remote "test \"\$(sha256sum '${v4_root}/SOURCE_BUNDLE.sha256' | awk '{print \$1}')\" = '${v4_receipt_sha}' && cd '${v4_root}' && sha256sum -c --quiet SOURCE_BUNDLE.sha256"
else
  remote "test ! -e '${v4_stage}' && mkdir -p '${v4_stage}'"
  timeout 240 rsync -a --partial \
    --chmod=Du=rwx,Dgo=rx,Fu=rw,Fgo=r \
    -e "ssh -o BatchMode=yes -o ControlMaster=no -S ${SSH_CONTROL_PATH}" \
    "${scratch}/root/" "${SSH_ALIAS}:${v4_stage}/"
  remote "cd '${v4_stage}' && sha256sum -c --quiet SOURCE_BUNDLE.sha256 && chmod -R a-w '${v4_stage}' && mv '${v4_stage}' '${v4_root}'"
fi

v4_receipt=${v4_root}/SOURCE_BUNDLE.sha256
protocol=${v4_root}/MemNavData/hm3d_fullmono_lifelong_direct_natural_protocol_20260827.json
materialize_script=${v4_root}/MemNavData/slurm_hm3d_fullmono_lifelong_natural_v4_materialize.sbatch
finalize_script=${v4_root}/MemNavData/slurm_hm3d_fullmono_lifelong_natural_v4_finalize.sbatch
common="ALL,V4_ROOT=${v4_root},AUDIT_SOURCE_ROOT=${AUDIT_SOURCE_ROOT},SOURCE_TASK_ROOT=${SOURCE_TASK_ROOT},BASE_SOURCE_ROOT=${BASE_SOURCE_ROOT},RUN_ROOT=${run_root},PARENT_ROOT=${PARENT_ROOT},PROTOCOL=${protocol},AUDIT_RUN_ROOT=${AUDIT_RUN_ROOT},SOURCE_CONSTRUCTION_ROOT=${SOURCE_CONSTRUCTION_ROOT},V4_RECEIPT=${v4_receipt},EXPECTED_V4_RECEIPT_SHA=${v4_receipt_sha},AUDIT_SOURCE_RECEIPT=${AUDIT_SOURCE_RECEIPT},EXPECTED_AUDIT_SOURCE_RECEIPT_SHA=${EXPECTED_AUDIT_SOURCE_RECEIPT_SHA},SOURCE_TASK_RECEIPT=${SOURCE_TASK_RECEIPT},EXPECTED_SOURCE_TASK_RECEIPT_SHA=${EXPECTED_SOURCE_TASK_RECEIPT_SHA},BASE_RECEIPT=${BASE_RECEIPT},EXPECTED_BASE_RECEIPT_SHA=${EXPECTED_BASE_RECEIPT_SHA}"

remote "tmp=\$(mktemp -d /tmp/h3life_v4_preflight.XXXXXX); \
  singularity exec -B /scratch/lg154 -B /scratch/yz11502 '${BASE_SIF}' \
  env PYTHONDONTWRITEBYTECODE=1 PYTHONPYCACHEPREFIX=\${tmp} \
  PYTHONPATH='${v4_root}/MemNavData:${AUDIT_SOURCE_ROOT}/MemNavData:${SOURCE_TASK_ROOT}:${SOURCE_TASK_ROOT}/MemNavData:${BASE_SOURCE_ROOT}:${BASE_SOURCE_ROOT}/MemNavData:${HAB_REQUESTS_VENDOR}' \
  '${REMOTE_HAB_PY}' -m unittest -q test_materialize_hm3d_fullmono_lifelong_natural_ab; \
  rc=\$?; rm -rf -- \${tmp}; exit \${rc}"
remote "source '${v4_root}/MemNavData/slurm_safe_submit.sh'; \
  safe_sbatch --lint-fatal --test-only --qos=gpu48 --array=${SMOKE_SCENE_INDEX} \
  --export='${common}' '${materialize_script}' >/dev/null; \
  safe_sbatch --lint-fatal --test-only --partition=cpu_short \
  --export='${common}' '${finalize_script}' >/dev/null"
remote "test ! -e '${run_root}' && mkdir -p '${run_root}/construct_ab/scenes' && \
  ln -s '${PARENT_ROOT}/sealed_inputs' '${run_root}/sealed_inputs'"

if [[ "${MODE}" == smoke ]]; then
  array_spec=${SMOKE_SCENE_INDEX}
else
  array_spec=0-53%${CONCURRENCY}
fi
materialize_id=$(remote "source '${v4_root}/MemNavData/slurm_safe_submit.sh'; \
  safe_sbatch --lint-fatal --parsable --qos=gpu48 --array=${array_spec} \
  --export='${common}' '${materialize_script}'" | job_id)
[[ "${materialize_id}" =~ ^[0-9]+$ ]] || fail "bad materialize job id"
finalize_id=null
if [[ "${MODE}" == formal ]]; then
  finalize_id=$(remote "source '${v4_root}/MemNavData/slurm_safe_submit.sh'; \
    safe_sbatch --lint-fatal --parsable --partition=cpu_short \
    --dependency=afterany:${materialize_id} --export='${common}' \
    '${finalize_script}'" | job_id)
  [[ "${finalize_id}" =~ ^[0-9]+$ ]] || fail "bad finalize job id"
fi

receipt=MemNavData/HM3D_FULLMONO_LIFELONG_NATURAL_V4_${MODE^^}_SUBMISSION_20260827.json
printf '{\n  "schema_version": "hm3d_fullmono_lifelong_natural_v4_materialization_submission_v1_20260827",\n  "mode": "%s",\n  "scope": "result-blind query-asset materialization only",\n  "run_root": "%s",\n  "source_bundle": "%s",\n  "source_bundle_receipt_sha256": "%s",\n  "array": "%s",\n  "materialize_job": %s,\n  "finalize_job": %s,\n  "factual_B_authorized_by_this_script": false,\n  "navigation_rollouts_created": false\n}\n' \
  "${MODE}" "${run_root}" "${v4_root}" "${v4_receipt_sha}" \
  "${array_spec}" "${materialize_id}" "${finalize_id}" >"${receipt}"
printf 'RUN_ROOT=%s\nV4_ROOT=%s\nMATERIALIZE_JOB=%s\nFINALIZE_JOB=%s\n' \
  "${run_root}" "${v4_root}" "${materialize_id}" "${finalize_id}"
