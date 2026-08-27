#!/usr/bin/env bash
# Freeze and submit the six-episode suffix repair for failed formal job 15753043.

set -euo pipefail
umask 0022

LOCAL_ROOT=${LOCAL_ROOT:-$(git rev-parse --show-toplevel)}
REMOTE_HOST=${REMOTE_HOST:-alantorch}
EXPECTED_REMOTE_USER=${EXPECTED_REMOTE_USER:-yz11502}
REMOTE_BUNDLE_BASE=${REMOTE_BUNDLE_BASE:-/scratch/yz11502/Research/source_bundles}
REMOTE_RESULT_BASE=${REMOTE_RESULT_BASE:-/scratch/yz11502/Research/Nav-axis-uturn-results/navdp_arrival_consensus_20260815}
REMOTE_PY=${REMOTE_PY:-/scratch/lg154/conda-envs/memnav/bin/python}
LOCAL_PY=${LOCAL_PY:-/home/asus/miniconda3/envs/memnav/bin/python}
BASE_SOURCE_ROOT=${BASE_SOURCE_ROOT:-/scratch/yz11502/Research/Nav-axis-uturn-source-bundles/certified_relocalization_closed_loop_d3bd281fc374cc80}
EXPECTED_BASE_SOURCE_RECEIPT_SHA=74001a9e0150c38c599a206fa0f4dd5e1279b9bed5d167119f4d14cb77995e98
PREFIX_SAMPLES=/scratch/yz11502/Research/Nav-axis-uturn-results/navdp_arrival_consensus_20260815/formal_20260814T171428Z/collection/samples.partial.csv
EXPECTED_PREFIX_SAMPLES_SHA=24315d919863f11497715a0dc5d64e66461b99027eb011337325618c4d571949
RAW_GAP_AUDIT=/scratch/yz11502/Research/Nav-axis-uturn-results/nlsr_gapfill_20260807/nlsr_routes_41c1773_20260807_v1/raw_audit.json
EXPECTED_RAW_GAP_AUDIT_SHA=d5a9e7548aa897f04be4e75cf27ad0634b4573f8c8b2ad79a7dbbf79997f771d
DRY_RUN=${DRY_RUN:-0}

fail() { echo "ABORT: $*" >&2; exit 2; }
[[ "${DRY_RUN}" =~ ^[01]$ ]] || fail "DRY_RUN must be 0 or 1"
SSH_ARGS=(-o BatchMode=yes -o ConnectTimeout=20)
RSYNC_RSH="ssh -o BatchMode=yes -o ConnectTimeout=20"
remote() { ssh "${SSH_ARGS[@]}" "${REMOTE_HOST}" "$@"; }

files=(
  MemNavData/audit_navdp_arrival_consensus.py
  MemNavData/merge_navdp_arrival_consensus_repair.py
  MemNavData/verify_navdp_arrival_consensus.py
  MemNavData/test_audit_navdp_arrival_consensus.py
  MemNavData/slurm_navdp_arrival_consensus.sbatch
  MemNavData/NAVDP_ARRIVAL_CONSENSUS_PROTOCOL_20260815.md
  MemNavData/router_multiscene_split_20260805.json
  .diagnostics/certificate_distilled_compass_20260813/static_top8_480_lightglue_open_set_rows.csv
)
for relative in "${files[@]}"; do
  path=${LOCAL_ROOT}/${relative}
  [[ -f "${path}" && ! -L "${path}" ]] || fail "missing physical input: ${path}"
done

export PYTHONPATH="${LOCAL_ROOT}"
"${LOCAL_PY}" -m py_compile \
  "${LOCAL_ROOT}/MemNavData/audit_navdp_arrival_consensus.py" \
  "${LOCAL_ROOT}/MemNavData/merge_navdp_arrival_consensus_repair.py" \
  "${LOCAL_ROOT}/MemNavData/verify_navdp_arrival_consensus.py"
"${LOCAL_PY}" -m unittest -v MemNavData.test_audit_navdp_arrival_consensus
bash -n "${LOCAL_ROOT}/MemNavData/slurm_navdp_arrival_consensus.sbatch"

[[ "$(remote 'id -un')" == "${EXPECTED_REMOTE_USER}" ]] || \
  fail "remote identity mismatch"
remote "test \"\$(sha256sum '${BASE_SOURCE_ROOT}/SOURCE_BUNDLE.sha256' | awk '{print \$1}')\" = '${EXPECTED_BASE_SOURCE_RECEIPT_SHA}' && cd '${BASE_SOURCE_ROOT}' && sha256sum -c SOURCE_BUNDLE.sha256 >/dev/null"
remote "test \"\$(sha256sum '${PREFIX_SAMPLES}' | awk '{print \$1}')\" = '${EXPECTED_PREFIX_SAMPLES_SHA}'"
remote "test \"\$(sha256sum '${RAW_GAP_AUDIT}' | awk '{print \$1}')\" = '${EXPECTED_RAW_GAP_AUDIT_SHA}'"

if [[ "${DRY_RUN}" == 1 ]]; then
  echo "DRY_RUN_PREFIX=${PREFIX_SAMPLES}"
  echo "DRY_RUN_PREFIX_SHA=${EXPECTED_PREFIX_SAMPLES_SHA}"
  echo "DRY_RUN_FILES=${#files[@]}"
  exit 0
fi

stage=$(mktemp -d /tmp/navdp_arrival_repair.XXXXXX)
remote_partial=${REMOTE_BUNDLE_BASE}/navdp_arrival_repair.partial.$$
cleanup() { rm -rf -- "${stage}"; }
trap cleanup EXIT
for relative in "${files[@]}"; do
  mkdir -p "${stage}/$(dirname "${relative}")"
  cp "${LOCAL_ROOT}/${relative}" "${stage}/${relative}"
done
"${LOCAL_PY}" - "${stage}/source_bundle_manifest.json" \
  "${EXPECTED_BASE_SOURCE_RECEIPT_SHA}" "${EXPECTED_PREFIX_SAMPLES_SHA}" \
  "${EXPECTED_RAW_GAP_AUDIT_SHA}" <<'PY'
import datetime
import hashlib
import json
import pathlib
import sys
out = pathlib.Path(sys.argv[1])
root = out.parent
payload = {
    "schema_version": "navdp_arrival_consensus_repair_bundle_v1_20260815",
    "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "scope": "six-episode suffix repair and deterministic prefix merge",
    "method_or_threshold_authorized": False,
    "goat_validation_read": False,
    "base_navdp_source_receipt_sha256": sys.argv[2],
    "prefix_samples_sha256": sys.argv[3],
    "raw_gap_audit_sha256": sys.argv[4],
    "failed_parent_job_id": 15753043,
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
remote_bundle=${REMOTE_BUNDLE_BASE}/navdp_arrival_repair_${source_receipt_sha:0:16}
remote "test ! -e '${remote_bundle}' && chmod -R a-w '${remote_partial}' && mv '${remote_partial}' '${remote_bundle}'"
remote "cd '${remote_bundle}' && sha256sum -c SOURCE_BUNDLE.sha256 >/dev/null"

run_tag=$(date -u +%Y%m%dT%H%M%SZ)
run_root=${REMOTE_RESULT_BASE}/repair_${run_tag}
launcher=${remote_bundle}/MemNavData/slurm_navdp_arrival_consensus.sbatch
exports="ALL,SOURCE_ROOT=${remote_bundle},SOURCE_RECEIPT=${remote_bundle}/SOURCE_BUNDLE.sha256,EXPECTED_SOURCE_RECEIPT_SHA=${source_receipt_sha},BASE_SOURCE_ROOT=${BASE_SOURCE_ROOT},EXPECTED_BASE_SOURCE_RECEIPT_SHA=${EXPECTED_BASE_SOURCE_RECEIPT_SHA},RUN_ROOT=${run_root},MODE=repair,MAX_EPISODES=0,EPISODE_START_INDEX=74,PREFIX_SAMPLES=${PREFIX_SAMPLES},EXPECTED_PREFIX_SAMPLES_SHA=${EXPECTED_PREFIX_SAMPLES_SHA}"
remote "test ! -e '${run_root}' && mkdir -p '${run_root}'"
remote "sbatch --test-only --export='${exports}' '${launcher}' >/dev/null"
raw=$(remote "sbatch --begin=now+2minutes --parsable --export='${exports}' '${launcher}'")
job=${raw%%;*}
[[ "${job}" =~ ^[0-9]+$ ]] || fail "bad Slurm job id: ${raw}"
remote "'${REMOTE_PY}' - '${run_root}/submission.json' '${job}' '${remote_bundle}' '${source_receipt_sha}' '${BASE_SOURCE_ROOT}' '${EXPECTED_BASE_SOURCE_RECEIPT_SHA}' '${PREFIX_SAMPLES}' '${EXPECTED_PREFIX_SAMPLES_SHA}' '${RAW_GAP_AUDIT}' '${EXPECTED_RAW_GAP_AUDIT_SHA}'" <<'PY'
import json
import sys
import time
(path, job, bundle, receipt, base, base_receipt, prefix, prefix_sha,
 raw_audit, raw_audit_sha) = sys.argv[1:]
payload = {
    "schema_version": "navdp_arrival_consensus_repair_submission_v1_20260815",
    "job_id": int(job),
    "mode": "repair",
    "episode_start_index": 74,
    "expected_repair_episode_count": 6,
    "failed_parent_job_id": 15753043,
    "source_bundle": bundle,
    "source_receipt_sha256": receipt,
    "base_navdp_source": base,
    "base_navdp_source_receipt_sha256": base_receipt,
    "prefix_samples": prefix,
    "prefix_samples_sha256": prefix_sha,
    "raw_gap_audit": raw_audit,
    "raw_gap_audit_sha256": raw_audit_sha,
    "method_or_threshold_authorized": False,
    "goat_validation_read": False,
    "submission_unix_time": time.time(),
}
open(path, "x").write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY
remote "chmod a-w '${run_root}/submission.json'"

echo "repair_job_id=${job}"
echo "repair_root=${run_root}"
echo "source_bundle=${remote_bundle}"
echo "source_receipt_sha256=${source_receipt_sha}"
