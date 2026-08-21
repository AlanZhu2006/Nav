#!/usr/bin/env bash
# Build an immutable minimal bundle and submit the frozen GOAT/NavDP pilot.

set -euo pipefail
umask 0022

LOCAL_ROOT=${LOCAL_ROOT:-$(git rev-parse --show-toplevel)}
REMOTE_HOST=${REMOTE_HOST:-alantorch}
EXPECTED_REMOTE_USER=${EXPECTED_REMOTE_USER:-yz11502}
SSH_CONTROL_PATH=${SSH_CONTROL_PATH:-/home/asus/.ssh/cm-e3ce4155ccb925413d599e13706baddf79fddff3}
REMOTE_BUNDLE_BASE=${REMOTE_BUNDLE_BASE:-/scratch/yz11502/Research/source_bundles}
REMOTE_RESULT_BASE=${REMOTE_RESULT_BASE:-/scratch/yz11502/Research/Nav-axis-uturn-results/goat_benchmark_20260814}
REMOTE_PY=${REMOTE_PY:-/scratch/lg154/conda-envs/memnav/bin/python}
LOCAL_PY=${LOCAL_PY:-/home/asus/miniconda3/envs/memnav/bin/python}
BASE_SOURCE_ROOT=${BASE_SOURCE_ROOT:-/scratch/yz11502/Research/Nav-axis-uturn-source-bundles/certified_relocalization_closed_loop_d3bd281fc374cc80}
EXPECTED_BASE_SOURCE_RECEIPT_SHA=74001a9e0150c38c599a206fa0f4dd5e1279b9bed5d167119f4d14cb77995e98
EXPECTED_MANIFEST_SHA=652cbe0f731c3b817e9c1e0f5e516ae4f386d74380a7ed06c4910651357b5db5
DRY_RUN=${DRY_RUN:-0}

fail() { echo "ABORT: $*" >&2; exit 2; }
[[ "${DRY_RUN}" =~ ^[01]$ ]] || fail "DRY_RUN must be 0 or 1"
[[ -S "${SSH_CONTROL_PATH}" ]] || fail "shared SSH control socket is missing"
SSH_ARGS=(-o BatchMode=yes -S "${SSH_CONTROL_PATH}" -o ControlMaster=no)
RSYNC_RSH="ssh -o BatchMode=yes -S ${SSH_CONTROL_PATH} -o ControlMaster=no"
remote() { ssh "${SSH_ARGS[@]}" "${REMOTE_HOST}" "$@"; }

files=(
  MemNavData/goat_navdp_runtime_pilot.py
  MemNavData/goat_navdp_runtime_pilot_manifest.json
  MemNavData/goat_navdp_discrete_adapter.py
  MemNavData/goat_contract_smoke.py
  MemNavData/slurm_goat_navdp_runtime_pilot.sbatch
  MemNavData/test_goat_navdp_runtime_pilot.py
  MemNavData/test_goat_navdp_discrete_adapter.py
  MemNavData/test_goat_contract_smoke.py
  MemNavData/GOAT_NAVDP_RUNTIME_PILOT_PROTOCOL_20260814.md
)
for relative in "${files[@]}"; do
  path=${LOCAL_ROOT}/${relative}
  [[ -f "${path}" && ! -L "${path}" ]] || fail "missing physical input: ${path}"
done
manifest=${LOCAL_ROOT}/MemNavData/goat_navdp_runtime_pilot_manifest.json
[[ "$(sha256sum "${manifest}" | awk '{print $1}')" == \
   "${EXPECTED_MANIFEST_SHA}" ]] || fail "frozen manifest changed"

export PYTHONPATH="${LOCAL_ROOT}"
"${LOCAL_PY}" -m py_compile \
  "${LOCAL_ROOT}/MemNavData/goat_navdp_runtime_pilot.py" \
  "${LOCAL_ROOT}/MemNavData/goat_navdp_discrete_adapter.py" \
  "${LOCAL_ROOT}/MemNavData/goat_contract_smoke.py"
"${LOCAL_PY}" -m unittest -v \
  MemNavData.test_goat_navdp_runtime_pilot \
  MemNavData.test_goat_navdp_discrete_adapter \
  MemNavData.test_goat_contract_smoke
bash -n "${LOCAL_ROOT}/MemNavData/slurm_goat_navdp_runtime_pilot.sbatch"

[[ "$(remote 'id -un')" == "${EXPECTED_REMOTE_USER}" ]] || \
  fail "remote identity mismatch"
remote "test \"\$(sha256sum '${BASE_SOURCE_ROOT}/SOURCE_BUNDLE.sha256' | awk '{print \$1}')\" = '${EXPECTED_BASE_SOURCE_RECEIPT_SHA}' && cd '${BASE_SOURCE_ROOT}' && sha256sum -c SOURCE_BUNDLE.sha256 >/dev/null"
remote "test -f /scratch/yz11502/Research/datasets/goat_bench_20260814/data/scene_datasets/hm3d/val/SEALED"

if [[ "${DRY_RUN}" == 1 ]]; then
  echo "DRY_RUN_MANIFEST_SHA=${EXPECTED_MANIFEST_SHA}"
  echo "DRY_RUN_BASE_SOURCE=${BASE_SOURCE_ROOT}"
  exit 0
fi

stage=$(mktemp -d /tmp/goat_navdp_runtime_pilot.XXXXXX)
remote_partial=${REMOTE_BUNDLE_BASE}/goat_navdp_runtime_pilot.partial.$$
cleanup() { rm -rf -- "${stage}"; }
trap cleanup EXIT
for relative in "${files[@]}"; do
  mkdir -p "${stage}/$(dirname "${relative}")"
  cp "${LOCAL_ROOT}/${relative}" "${stage}/${relative}"
done
"${LOCAL_PY}" - "${stage}/source_bundle_manifest.json" \
  "${EXPECTED_MANIFEST_SHA}" "${EXPECTED_BASE_SOURCE_RECEIPT_SHA}" <<'PY'
import datetime
import hashlib
import json
import pathlib
import sys

out = pathlib.Path(sys.argv[1])
root = out.parent
payload = {
    "schema_version": "goat_navdp_runtime_pilot_bundle_v1_20260814",
    "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "scope": "outcome-blind ten-scene first-ImageGoal runtime gate",
    "is_goat_navigation_score": False,
    "method_or_threshold_selection_allowed": False,
    "manifest_sha256": sys.argv[2],
    "base_navdp_source_receipt_sha256": sys.argv[3],
    "files": {},
}
for path in sorted(root.rglob("*")):
    if path.is_file() and path != out:
        relative = path.relative_to(root).as_posix()
        payload["files"][relative] = hashlib.sha256(path.read_bytes()).hexdigest()
out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY

remote "test ! -e '${remote_partial}' && mkdir -p '${remote_partial}'"
rsync -e "${RSYNC_RSH}" -a --chmod=Fu=rw,Fgo=r \
  "${stage}/" "${REMOTE_HOST}:${remote_partial}/"
remote "cd '${remote_partial}' && find . -type f ! -name SOURCE_BUNDLE.sha256 -print0 | sort -z | xargs -0 sha256sum > SOURCE_BUNDLE.sha256 && sha256sum -c SOURCE_BUNDLE.sha256 >/dev/null"
source_receipt_sha=$(remote "sha256sum '${remote_partial}/SOURCE_BUNDLE.sha256' | awk '{print \$1}'")
[[ "${source_receipt_sha}" =~ ^[0-9a-f]{64}$ ]] || fail "bad source receipt SHA"
remote_bundle=${REMOTE_BUNDLE_BASE}/goat_navdp_runtime_pilot_${source_receipt_sha:0:16}
remote "test ! -e '${remote_bundle}' && chmod -R a-w '${remote_partial}' && mv '${remote_partial}' '${remote_bundle}'"
remote "cd '${remote_bundle}' && sha256sum -c SOURCE_BUNDLE.sha256 >/dev/null"

run_tag=$(date -u +%Y%m%dT%H%M%SZ)
run_root=${REMOTE_RESULT_BASE}/navdp_runtime_pilot_${run_tag}
exports="ALL,SOURCE_ROOT=${remote_bundle},SOURCE_RECEIPT=${remote_bundle}/SOURCE_BUNDLE.sha256,EXPECTED_SOURCE_RECEIPT_SHA=${source_receipt_sha},BASE_SOURCE_ROOT=${BASE_SOURCE_ROOT},EXPECTED_BASE_SOURCE_RECEIPT_SHA=${EXPECTED_BASE_SOURCE_RECEIPT_SHA},MANIFEST=${remote_bundle}/MemNavData/goat_navdp_runtime_pilot_manifest.json,EXPECTED_MANIFEST_SHA=${EXPECTED_MANIFEST_SHA},RUN_ROOT=${run_root}"
launcher=${remote_bundle}/MemNavData/slurm_goat_navdp_runtime_pilot.sbatch
remote "test ! -e '${run_root}'"
remote "sbatch --test-only --export='${exports}' '${launcher}' >/dev/null"
# A short BeginTime delay closes the job-id/receipt race without relying on
# `scontrol release`, which this cluster rejects for user-submitted --hold jobs.
job_raw=$(remote "sbatch --begin=now+5minutes --parsable --export='${exports}' '${launcher}'")
job_id=${job_raw%%;*}
[[ "${job_id}" =~ ^[0-9]+$ ]] || fail "bad Slurm job id"

remote "mkdir -p '${run_root}' && '${REMOTE_PY}' - '${run_root}/submission.json' '${job_id}' '${remote_bundle}' '${source_receipt_sha}' '${EXPECTED_MANIFEST_SHA}' '${BASE_SOURCE_ROOT}' '${EXPECTED_BASE_SOURCE_RECEIPT_SHA}'" <<'PY'
import json
import sys
import time
path, job, bundle, receipt, manifest, base, base_receipt = sys.argv[1:]
payload = {
    "schema_version": "goat_navdp_runtime_pilot_submission_v1_20260814",
    "job_id": int(job),
    "source_bundle": bundle,
    "source_receipt_sha256": receipt,
    "manifest_sha256": manifest,
    "base_navdp_source": base,
    "base_navdp_source_receipt_sha256": base_receipt,
    "is_goat_navigation_score": False,
    "method_or_threshold_selection_allowed": False,
    "submission_unix_time": time.time(),
}
open(path, "x").write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY
remote "chmod a-w '${run_root}/submission.json'"

echo "job_id=${job_id}"
echo "run_root=${run_root}"
echo "source_bundle=${remote_bundle}"
echo "source_receipt_sha256=${source_receipt_sha}"
echo "manifest_sha256=${EXPECTED_MANIFEST_SHA}"
