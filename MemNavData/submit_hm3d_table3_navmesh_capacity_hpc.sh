#!/usr/bin/env bash
# Submit the result-blind geometry-capacity gate for conference Table III.
# This chain cannot render queries, run a controller, or authorize evaluation.
set -euo pipefail
umask 0022

ROOT=${ROOT:-$(git rev-parse --show-toplevel)}
SSH_ALIAS=${SSH_ALIAS:-alantorch}
DRY_RUN=${DRY_RUN:-0}
CONCURRENCY=${CONCURRENCY:-12}
REMOTE_BUNDLES=/scratch/yz11502/Research/Nav-axis-uturn-source-bundles
REMOTE_RESULTS=/scratch/yz11502/Research/Nav-axis-uturn-results/hm3d_table3_navmesh_capacity_20260830
PARENT_ROOT=/scratch/yz11502/Research/Nav-axis-uturn-results/hm3d_fresh_fullmono_mixed_role_20260820/formal_20260820T143609Z_e6dd44c6
PARENT_MANIFEST=${PARENT_ROOT}/sealed_inputs/parent_manifest.json
EXPECTED_PARENT_MANIFEST_SHA=a96a0b96fab7b7b47709b36cb8eeb9410b42b09f095f87ef01304a68de716dd5
PROTOCOL_REL=MemNavData/hm3d_table3_navmesh_capacity_protocol_20260830.json
SSH_CONTROL_PATH=${SSH_CONTROL_PATH:-$(
  ssh -G "${SSH_ALIAS}" 2>/dev/null |
    awk '$1=="controlpath"{value=$2} END{print value}'
)}

cd "${ROOT}"
fail() { echo "ABORT: $*" >&2; exit 2; }
lock_file=/tmp/yz11502_hm3d_table3_capacity_submit.lock
exec 9>"${lock_file}"
flock -n 9 || fail "another Table-3 capacity submission is active"
remote() {
  timeout 300 ssh -n -T -o BatchMode=yes -o ControlMaster=no \
    -S "${SSH_CONTROL_PATH}" "${SSH_ALIAS}" "$@" | tr -d '\r'
}
job_id() { awk -F';' '/^[0-9]+(;|$)/ {print $1; exit}'; }
upload_bundle() {
  local source=$1 destination=$2
  timeout 300 rsync -a --partial \
    --chmod=Du=rwx,Dgo=rx,Fu=rw,Fgo=r \
    -e "ssh -o BatchMode=yes -o ControlMaster=no -S ${SSH_CONTROL_PATH}" \
    "${source}/" "${SSH_ALIAS}:${destination}/"
}

[[ "${DRY_RUN}" =~ ^[01]$ ]] || fail "DRY_RUN must be 0 or 1"
[[ "${CONCURRENCY}" =~ ^[1-9][0-9]*$ ]] || fail "invalid concurrency"
[[ -n "${SSH_CONTROL_PATH}" && -S "${SSH_CONTROL_PATH}" ]] || \
  fail "authoritative shared SSH socket missing"
timeout 15 ssh -O check -S "${SSH_CONTROL_PATH}" "${SSH_ALIAS}" \
  >/dev/null 2>&1 || fail "shared SSH master is not responsive"

files=(
  MemNavData/audit_hm3d_table3_navmesh_capacity.py
  MemNavData/independent_verify_hm3d_table3_navmesh_capacity.py
  MemNavData/test_hm3d_table3_navmesh_capacity.py
  "${PROTOCOL_REL}"
  MemNavData/slurm_hm3d_table3_navmesh_capacity_scene.sbatch
  MemNavData/slurm_hm3d_table3_navmesh_capacity_finalize.sbatch
  MemNavData/slurm_safe_submit.sh
  MemNavData/bundle_selftest.sh
  MemNavData/submit_hm3d_table3_navmesh_capacity_hpc.sh
)
for path in "${files[@]}"; do
  [[ -f "${path}" && ! -L "${path}" ]] || fail "missing input ${path}"
done
python -m json.tool "${PROTOCOL_REL}" >/dev/null
bash -n MemNavData/slurm_hm3d_table3_navmesh_capacity_scene.sbatch \
  MemNavData/slurm_hm3d_table3_navmesh_capacity_finalize.sbatch \
  MemNavData/submit_hm3d_table3_navmesh_capacity_hpc.sh
(
  source MemNavData/slurm_safe_submit.sh
  lint_sbatch_template MemNavData/slurm_hm3d_table3_navmesh_capacity_scene.sbatch
  lint_sbatch_template MemNavData/slurm_hm3d_table3_navmesh_capacity_finalize.sbatch
)
PYTHONPATH=MemNavData /home/asus/miniconda3/envs/memnav/bin/python -m pytest -q \
  MemNavData/test_hm3d_table3_navmesh_capacity.py

scratch=$(mktemp -d /tmp/h3_t3_capacity_submit.XXXXXX)
cleanup() { rm -rf -- "${scratch}"; }
trap cleanup EXIT
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
receipt_sha=$(sha256sum "${scratch}/root/SOURCE_BUNDLE.sha256" | awk '{print $1}')
bundle=${REMOTE_BUNDLES}/hm3d_table3_capacity_${receipt_sha:0:16}
stage=${bundle}.partial.$$
run=${REMOTE_RESULTS}/capacity_$(date -u +%Y%m%dT%H%M%SZ)_${receipt_sha:0:8}
protocol_sha=$(sha256sum "${PROTOCOL_REL}" | awk '{print $1}')

if [[ "${DRY_RUN}" == 1 ]]; then
  printf 'DRY_RUN_BUNDLE=%s\nDRY_RUN_RUN_ROOT=%s\nRECEIPT_SHA=%s\n' \
    "${bundle}" "${run}" "${receipt_sha}"
  exit 0
fi

preflight=$(remote "set -euo pipefail
test \"\$(id -un)\" = yz11502
test -x /scratch/lg154/conda-envs/habitat/bin/python
test -x /scratch/lg154/conda-envs/memnav/bin/python
test \"\$(sha256sum '${PARENT_MANIFEST}' | awk '{print \$1}')\" = '${EXPECTED_PARENT_MANIFEST_SHA}'
echo PREFLIGHT_OK")
[[ "${preflight}" == *PREFLIGHT_OK* ]] || fail "remote preflight incomplete"

if remote "test -d '${bundle}'" >/dev/null 2>&1; then
  remote "test \"\$(sha256sum '${bundle}/SOURCE_BUNDLE.sha256' | awk '{print \$1}')\" = '${receipt_sha}'; cd '${bundle}'; sha256sum -c --quiet SOURCE_BUNDLE.sha256" >/dev/null
else
  remote "test ! -e '${stage}'; mkdir -p '${stage}'" >/dev/null
  upload_bundle "${scratch}/root" "${stage}"
  remote "cd '${stage}'; sha256sum -c --quiet SOURCE_BUNDLE.sha256; chmod -R a-w '${stage}'; mv '${stage}' '${bundle}'" >/dev/null
fi

receipt=${bundle}/SOURCE_BUNDLE.sha256
protocol=${bundle}/${PROTOCOL_REL}
scene_script=${bundle}/MemNavData/slurm_hm3d_table3_navmesh_capacity_scene.sbatch
post_script=${bundle}/MemNavData/slurm_hm3d_table3_navmesh_capacity_finalize.sbatch
common_base="ALL,TASK_ROOT=${bundle},TASK_RECEIPT=${receipt},EXPECTED_TASK_RECEIPT_SHA=${receipt_sha},PARENT_ROOT=${PARENT_ROOT},PROTOCOL=${protocol},EXPECTED_PROTOCOL_SHA=${protocol_sha},EXPECTED_PARENT_MANIFEST_SHA=${EXPECTED_PARENT_MANIFEST_SHA}"
formal_root=${run}/formal
smoke_root=${run}/smoke
formal_common="${common_base},RUN_ROOT=${formal_root}"
smoke_common="${common_base},RUN_ROOT=${smoke_root}"
remote "mkdir -p '${formal_root}/scenes' '${smoke_root}/scenes'; source '${bundle}/MemNavData/slurm_safe_submit.sh'; safe_sbatch --lint-fatal --test-only --partition=cpu_short --array=0 --export='${smoke_common}' '${scene_script}' >/dev/null; safe_sbatch --lint-fatal --test-only --partition=cpu_short --array=0 --export='${formal_common}' '${scene_script}' >/dev/null; safe_sbatch --lint-fatal --test-only --partition=cpu_short --export='${formal_common},MODE=finalize' '${post_script}' >/dev/null" >/dev/null

smoke=$(remote "source '${bundle}/MemNavData/slurm_safe_submit.sh'; safe_sbatch --lint-fatal --parsable --partition=cpu_short --array=0 --export='${smoke_common}' '${scene_script}'" | job_id)
[[ "${smoke}" =~ ^[0-9]+$ ]] || fail "invalid capacity smoke job"
array=$(remote "source '${bundle}/MemNavData/slurm_safe_submit.sh'; safe_sbatch --lint-fatal --parsable --partition=cpu_short --array=0-53%${CONCURRENCY} --dependency=afterok:${smoke} --kill-on-invalid-dep=yes --export='${formal_common}' '${scene_script}'" | job_id)
[[ "${array}" =~ ^[0-9]+$ ]] || fail "invalid capacity array job"
finalize=$(remote "source '${bundle}/MemNavData/slurm_safe_submit.sh'; safe_sbatch --lint-fatal --parsable --partition=cpu_short --dependency=afterok:${array} --kill-on-invalid-dep=yes --export='${formal_common},MODE=finalize' '${post_script}'" | job_id)
[[ "${finalize}" =~ ^[0-9]+$ ]] || fail "invalid capacity finalizer job"
verify=$(remote "source '${bundle}/MemNavData/slurm_safe_submit.sh'; safe_sbatch --lint-fatal --parsable --partition=cpu_short --dependency=afterok:${finalize} --kill-on-invalid-dep=yes --export='${formal_common},MODE=verify' '${post_script}'" | job_id)
[[ "${verify}" =~ ^[0-9]+$ ]] || fail "invalid capacity verifier job"

submission=${scratch}/submission.json
python - "${submission}" "${bundle}" "${receipt_sha}" "${run}" \
  "${smoke}" "${array}" "${finalize}" "${verify}" <<'PY'
import json,sys
out,bundle,sha,run,smoke,array,finalize,verify=sys.argv[1:]
p={
 'schema_version':'hm3d_table3_navmesh_capacity_submission_v1_20260830',
 'scope':'result-blind geometry capacity only',
 'bundle':bundle,'bundle_receipt_sha256':sha,'run_root':run,
 'smoke_job':int(smoke),'smoke_scene_index':0,
 'scene_array_job':int(array),'scene_array':'0-53',
 'finalize_job':int(finalize),'independent_verify_job':int(verify),
 'query_policy_outcomes_read':False,'navigation_outcomes_read':False,
 'rendered_support_verified':False,'policy_evaluation_authorized':False,
}
open(out,'w').write(json.dumps(p,indent=2,sort_keys=True)+'\n')
PY
timeout 120 rsync -a \
  -e "ssh -o BatchMode=yes -o ControlMaster=no -S ${SSH_CONTROL_PATH}" \
  "${submission}" "${SSH_ALIAS}:${run}/submission.json"
remote "sha256sum '${run}/submission.json' >'${run}/submission.json.sha256'; chmod a-w '${run}/submission.json' '${run}/submission.json.sha256'" >/dev/null
cp -p "${submission}" MemNavData/HM3D_TABLE3_NAVMESH_CAPACITY_SUBMISSION_20260830.json
printf 'RUN_ROOT=%s\nBUNDLE=%s\nSMOKE_JOB=%s\nARRAY_JOB=%s\nFINALIZE_JOB=%s\nVERIFY_JOB=%s\n' \
  "${run}" "${bundle}" "${smoke}" "${array}" "${finalize}" "${verify}"
