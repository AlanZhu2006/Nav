#!/usr/bin/env bash
set -euo pipefail
umask 0022

ROOT=${ROOT:-$(git rev-parse --show-toplevel)}
SSH_ALIAS=${SSH_ALIAS:-alantorch}
REMOTE_BUNDLES=/scratch/yz11502/Research/Nav-axis-uturn-source-bundles
REMOTE_RESULTS=/scratch/yz11502/Research/Nav-axis-uturn-results/hm3d_table3_actual_mono_20260830
CAPACITY_ROOT=/scratch/yz11502/Research/Nav-axis-uturn-results/hm3d_table3_navmesh_capacity_dense100_20260830/capacity_dense100_20260830T060412Z_5a3c38e0/formal
PROTOCOL_REL=MemNavData/hm3d_table3_actual_mono_protocol_20260830.json
PARENT_REL=MemNavData/hm3d_table3_combined_assets_20260830.json
LOCAL_RECEIPT=MemNavData/HM3D_TABLE3_ACTUAL_MONO_PLAN_SUBMISSION_20260830.json
SSH_CONTROL_PATH=${SSH_CONTROL_PATH:-$(ssh -G "${SSH_ALIAS}" 2>/dev/null | awk '$1=="controlpath"{x=$2} END{print x}')}

cd "${ROOT}"
fail() { echo "ABORT: $*" >&2; exit 2; }
remote() {
  timeout 300 ssh -n -T -o BatchMode=yes -o ControlMaster=no \
    -S "${SSH_CONTROL_PATH}" "${SSH_ALIAS}" "$@" | tr -d '\r'
}
job_id() { awk -F';' '/^[0-9]+(;|$)/ {print $1; exit}'; }
[[ -n "${SSH_CONTROL_PATH}" && -S "${SSH_CONTROL_PATH}" ]] || fail "shared SSH socket missing"
timeout 15 ssh -O check -S "${SSH_CONTROL_PATH}" "${SSH_ALIAS}" >/dev/null 2>&1 || fail "shared SSH master unavailable"

files=(
  "${PROTOCOL_REL}" "${PARENT_REL}"
  MemNavData/freeze_hm3d_table3_actual_mono_plan.py
  MemNavData/independent_verify_hm3d_table3_actual_mono_plan.py
  MemNavData/test_freeze_hm3d_table3_actual_mono_plan.py
  MemNavData/slurm_hm3d_table3_actual_mono_plan.sbatch
  MemNavData/slurm_safe_submit.sh MemNavData/bundle_selftest.sh
)
for path in "${files[@]}"; do [[ -f "${path}" && ! -L "${path}" ]] || fail "missing ${path}"; done
python -m json.tool "${PROTOCOL_REL}" >/dev/null
bash -n MemNavData/slurm_hm3d_table3_actual_mono_plan.sbatch "$0"
PYTHONPATH=MemNavData /home/asus/miniconda3/envs/memnav/bin/python -m pytest -q \
  MemNavData/test_freeze_hm3d_table3_actual_mono_plan.py

scratch=$(mktemp -d /tmp/h3_table3_plan.XXXXXX)
cleanup() { rm -rf -- "${scratch}"; }
trap cleanup EXIT
mkdir -p "${scratch}/root/MemNavData"
for path in "${files[@]}"; do cp -p -- "${path}" "${scratch}/root/MemNavData/$(basename "${path}")"; done
(
  cd "${scratch}/root"
  find . -type f ! -name SOURCE_BUNDLE.sha256 -print0 | sort -z | xargs -0 sha256sum >SOURCE_BUNDLE.sha256
  sha256sum -c --quiet SOURCE_BUNDLE.sha256
)
bundle_receipt_sha=$(sha256sum "${scratch}/root/SOURCE_BUNDLE.sha256" | awk '{print $1}')
bundle=${REMOTE_BUNDLES}/hm3d_table3_actual_mono_plan_${bundle_receipt_sha:0:16}
stage=${bundle}.partial.$$
run=${REMOTE_RESULTS}/plan_$(date -u +%Y%m%dT%H%M%SZ)_${bundle_receipt_sha:0:8}
if remote "test -d '${bundle}'" >/dev/null 2>&1; then
  remote "cd '${bundle}'; test \"\$(sha256sum SOURCE_BUNDLE.sha256 | awk '{print \$1}')\" = '${bundle_receipt_sha}'; sha256sum -c --quiet SOURCE_BUNDLE.sha256" >/dev/null
else
  remote "test ! -e '${stage}'; mkdir -p '${stage}'" >/dev/null
  timeout 300 rsync -a --partial --chmod=Du=rwx,Dgo=rx,Fu=rw,Fgo=r \
    -e "ssh -o BatchMode=yes -o ControlMaster=no -S ${SSH_CONTROL_PATH}" \
    "${scratch}/root/" "${SSH_ALIAS}:${stage}/"
  remote "cd '${stage}'; sha256sum -c --quiet SOURCE_BUNDLE.sha256; chmod -R a-w '${stage}'; mv '${stage}' '${bundle}'" >/dev/null
fi

protocol=${bundle}/${PROTOCOL_REL}
parent=${bundle}/${PARENT_REL}
receipt=${bundle}/SOURCE_BUNDLE.sha256
protocol_sha=$(sha256sum "${PROTOCOL_REL}" | awk '{print $1}')
parent_sha=$(sha256sum "${PARENT_REL}" | awk '{print $1}')
common="ALL,TASK_ROOT=${bundle},TASK_RECEIPT=${receipt},EXPECTED_TASK_RECEIPT_SHA=${bundle_receipt_sha},RUN_ROOT=${run},PROTOCOL=${protocol},EXPECTED_PROTOCOL_SHA=${protocol_sha},PARENT_MANIFEST=${parent},EXPECTED_PARENT_MANIFEST_SHA=${parent_sha},CAPACITY_ROOT=${CAPACITY_ROOT}"
script=${bundle}/MemNavData/slurm_hm3d_table3_actual_mono_plan.sbatch
remote "mkdir -p '${run}'; source '${bundle}/MemNavData/slurm_safe_submit.sh'; safe_sbatch --lint-fatal --test-only --partition=cpu_short --export='${common},MODE=freeze' '${script}' >/dev/null; safe_sbatch --lint-fatal --test-only --partition=cpu_short --export='${common},MODE=verify' '${script}' >/dev/null" >/dev/null
freeze_job=$(remote "source '${bundle}/MemNavData/slurm_safe_submit.sh'; safe_sbatch --lint-fatal --parsable --partition=cpu_short --export='${common},MODE=freeze' '${script}'" | job_id)
[[ "${freeze_job}" =~ ^[0-9]+$ ]] || fail "invalid freeze job"
verify_job=$(remote "source '${bundle}/MemNavData/slurm_safe_submit.sh'; safe_sbatch --lint-fatal --parsable --partition=cpu_short --dependency=afterok:${freeze_job} --kill-on-invalid-dep=yes --export='${common},MODE=verify' '${script}'" | job_id)
[[ "${verify_job}" =~ ^[0-9]+$ ]] || fail "invalid verify job"
python - "${LOCAL_RECEIPT}" "${bundle}" "${bundle_receipt_sha}" "${run}" "${freeze_job}" "${verify_job}" <<'PY'
import json,sys
out,bundle,sha,run,freeze,verify=sys.argv[1:]
payload={
  'schema_version':'hm3d_table3_actual_mono_plan_submission_v1_20260830',
  'bundle':bundle,'bundle_receipt_sha256':sha,'run_root':run,
  'freeze_job':int(freeze),'independent_verify_job':int(verify),
  'factual_A_outcomes_read':False,'query_policy_outcomes_read':False,
  'query_policy_evaluation_authorized':False,
}
open(out,'w').write(json.dumps(payload,indent=2,sort_keys=True)+'\n')
PY
rsync -a -e "ssh -o BatchMode=yes -o ControlMaster=no -S ${SSH_CONTROL_PATH}" \
  "${LOCAL_RECEIPT}" "${SSH_ALIAS}:${run}/submission.json"
remote "sha256sum '${run}/submission.json' >'${run}/submission.json.sha256'; chmod a-w '${run}/submission.json' '${run}/submission.json.sha256'" >/dev/null
printf 'RUN_ROOT=%s\nFREEZE_JOB=%s\nVERIFY_JOB=%s\n' "${run}" "${freeze_job}" "${verify_job}"
