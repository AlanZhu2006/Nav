#!/usr/bin/env bash
set -euo pipefail
umask 0022

ROOT=${ROOT:-/home/asus/Research/Nav-graph-blind}
SSH_ALIAS=${SSH_ALIAS:-alantorch}
SEAL_JOB_ID=${SEAL_JOB_ID:-16465110}
RUN_ROOT=${RUN_ROOT:-/scratch/yz11502/Research/Nav-axis-uturn-results/hm3d_fullmono_lifelong_natural_v4_20260827/formal_materialize_20260827T133704Z_d85fc50d}
AUDIT_RUN_ROOT=/scratch/yz11502/Research/Nav-axis-uturn-results/hm3d_fullmono_lifelong_power_v3_20260826/natural_b_audit_formal_20260827T125748Z_e2832e17
V4_ROOT=/scratch/yz11502/Research/Nav-axis-uturn-source-bundles/hm3d_lifelong_natural_v4_d85fc50df19b1384
PROTOCOL=${V4_ROOT}/MemNavData/hm3d_fullmono_lifelong_direct_natural_protocol_20260827.json
EXPECTED_PROTOCOL_SHA=2bfc62c08cbee1dffd3c5a3f627b1cb58a7c5076a9c9b2e2554e28843564492f
REMOTE_BUNDLES=/scratch/yz11502/Research/Nav-axis-uturn-source-bundles
LOCAL_PY=${LOCAL_PY:-/home/asus/miniconda3/bin/python}
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
job_id() {
  tr -d '\r' | awk -F';' '/^[0-9]+(;|$)/ {print $1; exit}'
}
[[ "${SEAL_JOB_ID}" =~ ^[0-9]+$ ]] || fail "invalid seal job id"
[[ -x "${LOCAL_PY}" ]] || fail "local Python missing"
[[ -n "${SSH_CONTROL_PATH}" && -S "${SSH_CONTROL_PATH}" ]] || \
  fail "authoritative shared SSH socket missing"
timeout 15 ssh -O check -S "${SSH_CONTROL_PATH}" "${SSH_ALIAS}" \
  >/dev/null 2>&1 || fail "shared SSH master is not responsive"

files=(
  MemNavData/independent_verify_hm3d_fullmono_lifelong_natural_v4_materialization.py
  MemNavData/test_independent_verify_hm3d_fullmono_lifelong_natural_v4_materialization.py
  MemNavData/slurm_independent_verify_hm3d_fullmono_lifelong_natural_v4_materialization.sbatch
  MemNavData/slurm_safe_submit.sh
)
for path in "${files[@]}"; do
  [[ -f "${path}" && ! -L "${path}" ]] || fail "missing physical input ${path}"
done
bash -n \
  MemNavData/slurm_independent_verify_hm3d_fullmono_lifelong_natural_v4_materialization.sbatch \
  MemNavData/slurm_safe_submit.sh \
  MemNavData/submit_independent_verify_hm3d_fullmono_lifelong_natural_v4_materialization_hpc.sh
(
  source MemNavData/slurm_safe_submit.sh
  lint_sbatch_template \
    MemNavData/slurm_independent_verify_hm3d_fullmono_lifelong_natural_v4_materialization.sbatch
)
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="${ROOT}/MemNavData" \
  "${LOCAL_PY}" -m unittest -q \
  test_independent_verify_hm3d_fullmono_lifelong_natural_v4_materialization

scratch=$(mktemp -d /tmp/h3life_v4_verify_submit.XXXXXX)
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
bundle=${REMOTE_BUNDLES}/hm3d_lifelong_natural_v4_verifier_${receipt_sha:0:16}
stage=${bundle}.partial.$$
identity=$(remote 'id -un' | tr -d '\r')
[[ "${identity}" == yz11502 ]] || fail "wrong remote identity: ${identity}"
if remote "test -d '${bundle}'"; then
  remote "test \"\$(sha256sum '${bundle}/SOURCE_BUNDLE.sha256' | awk '{print \$1}')\" = '${receipt_sha}' && cd '${bundle}' && sha256sum -c --quiet SOURCE_BUNDLE.sha256"
else
  remote "test ! -e '${stage}' && mkdir -p '${stage}'"
  timeout 240 rsync -a --partial \
    --chmod=Du=rwx,Dgo=rx,Fu=rw,Fgo=r \
    -e "ssh -o BatchMode=yes -o ControlMaster=no -S ${SSH_CONTROL_PATH}" \
    "${scratch}/root/" "${SSH_ALIAS}:${stage}/"
  remote "cd '${stage}' && sha256sum -c --quiet SOURCE_BUNDLE.sha256 && chmod -R a-w '${stage}' && mv '${stage}' '${bundle}'"
fi

source_receipt=${bundle}/SOURCE_BUNDLE.sha256
script=${bundle}/MemNavData/slurm_independent_verify_hm3d_fullmono_lifelong_natural_v4_materialization.sbatch
common="ALL,SOURCE_ROOT=${bundle},SOURCE_RECEIPT=${source_receipt},EXPECTED_SOURCE_RECEIPT_SHA=${receipt_sha},RUN_ROOT=${RUN_ROOT},AUDIT_RUN_ROOT=${AUDIT_RUN_ROOT},PROTOCOL=${PROTOCOL},EXPECTED_PROTOCOL_SHA=${EXPECTED_PROTOCOL_SHA}"
remote "test -d '${RUN_ROOT}' && test \"\$(sha256sum '${PROTOCOL}' | awk '{print \$1}')\" = '${EXPECTED_PROTOCOL_SHA}' && scontrol show job '${SEAL_JOB_ID}' >/dev/null"
remote "source '${bundle}/MemNavData/slurm_safe_submit.sh'; safe_sbatch --lint-fatal --test-only --partition=cpu_short --export='${common}' '${script}' >/dev/null"
verify_id=$(remote "source '${bundle}/MemNavData/slurm_safe_submit.sh'; safe_sbatch --lint-fatal --parsable --partition=cpu_short --dependency=afterok:${SEAL_JOB_ID} --export='${common}' '${script}'" | job_id)
[[ "${verify_id}" =~ ^[0-9]+$ ]] || fail "bad verifier job id"

printf '{\n  "schema_version": "hm3d_fullmono_lifelong_natural_v4_materialization_verifier_submission_v1_20260827",\n  "scope": "read-only independent asset-ledger verification",\n  "run_root": "%s",\n  "source_bundle": "%s",\n  "source_bundle_receipt_sha256": "%s",\n  "seal_job": %s,\n  "verifier_job": %s,\n  "dependency": "afterok:%s",\n  "navigation_rollouts_authorized": false\n}\n' \
  "${RUN_ROOT}" "${bundle}" "${receipt_sha}" "${SEAL_JOB_ID}" \
  "${verify_id}" "${SEAL_JOB_ID}" \
  >MemNavData/HM3D_FULLMONO_LIFELONG_NATURAL_V4_VERIFIER_SUBMISSION_20260827.json
printf 'VERIFY_BUNDLE=%s\nVERIFY_JOB=%s\n' "${bundle}" "${verify_id}"
