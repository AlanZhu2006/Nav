#!/usr/bin/env bash
# Freeze, upload, and submit smoke -> 20-scene formal -> summary -> verifier.

set -euo pipefail
umask 0022

LOCAL_ROOT=${LOCAL_ROOT:-$(git rev-parse --show-toplevel)}
REMOTE_HOST=${REMOTE_HOST:-alantorch}
EXPECTED_REMOTE_USER=${EXPECTED_REMOTE_USER:-yz11502}
SSH_CONTROL_PATH=${SSH_CONTROL_PATH:-/home/asus/.ssh/cm-e3ce4155ccb925413d599e13706baddf79fddff3}
REMOTE_BUNDLE_BASE=${REMOTE_BUNDLE_BASE:-/scratch/yz11502/Research/source_bundles}
REMOTE_RESULT_BASE=${REMOTE_RESULT_BASE:-/scratch/yz11502/Research/Nav-axis-uturn-results/goat_certified_arrival_20260815}
BASE_SOURCE_ROOT=${BASE_SOURCE_ROOT:-/scratch/yz11502/Research/Nav-axis-uturn-source-bundles/certified_relocalization_closed_loop_d3bd281fc374cc80}
EXPECTED_BASE_SOURCE_RECEIPT_SHA=74001a9e0150c38c599a206fa0f4dd5e1279b9bed5d167119f4d14cb77995e98
FORMAL_MANIFEST_SHA=3120625b8b6e86d9d517f08dd4d3b366c0d417cfdb738b50dc01834186458b79
SMOKE_MANIFEST_SHA=324d4947a95830ddfd86ccee2a63e8f29f223c77dae0d39e33b6e54e5141a862
MEMNAV_CKPT=/scratch/yz11502/Research/Nav-axis-uturn/.diagnostics/unseen_scene_eval_20260803/checkpoints/gatecurr600.memnav.ckpt
NAVDP_CKPT=/scratch/yz11502/Research/Nav-axis-uturn/.diagnostics/unseen_scene_eval_20260803/checkpoints/navdp_checkpoint.ckpt
LINGBOT_WEIGHTS=/scratch/lg154/Research/Nav/NavDP/baselines/memnav/lingbot-map/weights/lingbot-map-long.pt
EXPECTED_MEMNAV_CKPT_SHA=9b7a5811ff0aea212503f58b45258ba4f66b06420f87c350946aead39db6fdb7
EXPECTED_NAVDP_CKPT_SHA=3bb3ad4ab241e857bb57a4021cc6aab76d5263e81fbf80298d579053ef011947
EXPECTED_LINGBOT_WEIGHTS_SHA=832bc82cbae0bc9bbe946ef5ee1f7226abd8c0e183ccf8beddbb3d133576f409
LOCAL_PY=${LOCAL_PY:-python}
REMOTE_PY=${REMOTE_PY:-/scratch/lg154/conda-envs/memnav/bin/python}
FORMAL_CONCURRENCY=${FORMAL_CONCURRENCY:-4}
DRY_RUN=${DRY_RUN:-0}

fail() { echo "ABORT: $*" >&2; exit 2; }
[[ "${DRY_RUN}" =~ ^[01]$ ]] || fail "DRY_RUN must be 0 or 1"
[[ "${FORMAL_CONCURRENCY}" =~ ^[1-9][0-9]*$ ]] || \
  fail "FORMAL_CONCURRENCY must be positive"
[[ -S "${SSH_CONTROL_PATH}" ]] || fail "shared SSH control socket is missing"
SSH_ARGS=(-o BatchMode=yes -S "${SSH_CONTROL_PATH}" -o ControlMaster=no)
RSYNC_RSH="ssh -o BatchMode=yes -S ${SSH_CONTROL_PATH} -o ControlMaster=no"
remote() { ssh "${SSH_ARGS[@]}" "${REMOTE_HOST}" "$@"; }

files=(
  MemNavData/goat_certified_arrival_confirmation.py
  MemNavData/goat_certified_arrival_contract.py
  MemNavData/goat_certified_arrival_manifest.json
  MemNavData/goat_certified_arrival_smoke_manifest.json
  MemNavData/goat_navdp_discrete_adapter.py
  MemNavData/goat_contract_smoke.py
  MemNavData/goat_navdp_runtime_pilot.py
  MemNavData/certified_relocalization_runtime.py
  MemNavData/lingbot_pnp_localization.py
  MemNavData/summarize_goat_certified_arrival.py
  MemNavData/verify_goat_certified_arrival.py
  MemNavData/test_goat_certified_arrival_contract.py
  MemNavData/test_goat_certified_arrival_confirmation.py
  MemNavData/test_goat_navdp_discrete_adapter.py
  MemNavData/test_goat_contract_smoke.py
  MemNavData/test_summarize_goat_certified_arrival.py
  MemNavData/GOAT_CERTIFIED_ARRIVAL_CONFIRMATION_PROTOCOL_20260815.md
  MemNavData/GOAT_CERTIFIED_ARRIVAL_RESET_SEED_REPAIR_20260815.md
  MemNavData/slurm_goat_certified_arrival.sbatch
  MemNavData/slurm_summarize_goat_certified_arrival.sbatch
  MemNavData/slurm_verify_goat_certified_arrival.sbatch
  NavDP/baselines/memnav/memnav_server.py
  NavDP/baselines/memnav/policy_agent.py
  NavDP/baselines/memnav/pose_alignment.py
  NavDP/baselines/memnav/reverse_memory_graph.py
  NavDP/baselines/memnav/router_candidates.py
)
for relative in "${files[@]}"; do
  path=${LOCAL_ROOT}/${relative}
  [[ -f "${path}" && ! -L "${path}" ]] || fail "missing physical input ${path}"
done
formal_manifest=${LOCAL_ROOT}/MemNavData/goat_certified_arrival_manifest.json
smoke_manifest=${LOCAL_ROOT}/MemNavData/goat_certified_arrival_smoke_manifest.json
[[ "$(sha256sum "${formal_manifest}" | awk '{print $1}')" == \
   "${FORMAL_MANIFEST_SHA}" ]] || fail "formal manifest changed"
[[ "$(sha256sum "${smoke_manifest}" | awk '{print $1}')" == \
   "${SMOKE_MANIFEST_SHA}" ]] || fail "smoke manifest changed"

export PYTHONPATH="${LOCAL_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
"${LOCAL_PY}" -m py_compile \
  "${LOCAL_ROOT}/MemNavData/goat_certified_arrival_confirmation.py" \
  "${LOCAL_ROOT}/MemNavData/goat_certified_arrival_contract.py" \
  "${LOCAL_ROOT}/MemNavData/goat_navdp_discrete_adapter.py" \
  "${LOCAL_ROOT}/MemNavData/summarize_goat_certified_arrival.py" \
  "${LOCAL_ROOT}/MemNavData/verify_goat_certified_arrival.py" \
  "${LOCAL_ROOT}/NavDP/baselines/memnav/policy_agent.py" \
  "${LOCAL_ROOT}/NavDP/baselines/memnav/memnav_server.py"
"${LOCAL_PY}" -m unittest \
  MemNavData.test_goat_certified_arrival_contract \
  MemNavData.test_goat_certified_arrival_confirmation \
  MemNavData.test_goat_navdp_discrete_adapter \
  MemNavData.test_goat_contract_smoke \
  MemNavData.test_summarize_goat_certified_arrival
for script in \
  MemNavData/slurm_goat_certified_arrival.sbatch \
  MemNavData/slurm_summarize_goat_certified_arrival.sbatch \
  MemNavData/slurm_verify_goat_certified_arrival.sbatch; do
  bash -n "${LOCAL_ROOT}/${script}"
done

[[ "$(remote 'id -un')" == "${EXPECTED_REMOTE_USER}" ]] || \
  fail "remote identity mismatch"
remote "test \"\$(sha256sum '${BASE_SOURCE_ROOT}/SOURCE_BUNDLE.sha256' | awk '{print \$1}')\" = '${EXPECTED_BASE_SOURCE_RECEIPT_SHA}' && cd '${BASE_SOURCE_ROOT}' && sha256sum -c SOURCE_BUNDLE.sha256 >/dev/null"
remote "test -x /scratch/yz11502/conda_envs/goat-bench-habitat023-20260814/bin/python && test -f /scratch/yz11502/Research/datasets/goat_bench_20260814/data/scene_datasets/hm3d/val/SEALED"

# Hash each large dependency once at submission. Per-episode tasks validate the
# sealed receipt and file size, avoiding 20 redundant multi-GB reads.
dependency_lines=$(remote "for p in '${MEMNAV_CKPT}' '${NAVDP_CKPT}' '${LINGBOT_WEIGHTS}'; do stat -c '%s' \"\$p\"; sha256sum \"\$p\" | awk '{print \$1}'; done")
mapfile -t dependency_values <<<"${dependency_lines}"
[[ "${#dependency_values[@]}" -eq 6 ]] || fail "dependency audit output malformed"
[[ "${dependency_values[1]}" == "${EXPECTED_MEMNAV_CKPT_SHA}" ]] || \
  fail "remote MemNav checkpoint changed"
[[ "${dependency_values[3]}" == "${EXPECTED_NAVDP_CKPT_SHA}" ]] || \
  fail "remote NavDP checkpoint changed"
[[ "${dependency_values[5]}" == "${EXPECTED_LINGBOT_WEIGHTS_SHA}" ]] || \
  fail "remote LingBot weights changed"

if [[ "${DRY_RUN}" == 1 ]]; then
  echo "DRY_RUN_FORMAL_MANIFEST_SHA=${FORMAL_MANIFEST_SHA}"
  echo "DRY_RUN_SMOKE_MANIFEST_SHA=${SMOKE_MANIFEST_SHA}"
  echo "DRY_RUN_BASE_SOURCE=${BASE_SOURCE_ROOT}"
  exit 0
fi

stage=$(mktemp -d /tmp/goat_certified_arrival.XXXXXX)
remote_partial=${REMOTE_BUNDLE_BASE}/goat_certified_arrival.partial.$$
cleanup() { find "${stage}" -depth -delete 2>/dev/null || true; }
trap cleanup EXIT
for relative in "${files[@]}"; do
  mkdir -p "${stage}/$(dirname "${relative}")"
  cp --preserve=mode,timestamps "${LOCAL_ROOT}/${relative}" "${stage}/${relative}"
done
"${LOCAL_PY}" - "${stage}/dependency_receipt.json" \
  "${MEMNAV_CKPT}" "${dependency_values[0]}" "${EXPECTED_MEMNAV_CKPT_SHA}" \
  "${NAVDP_CKPT}" "${dependency_values[2]}" "${EXPECTED_NAVDP_CKPT_SHA}" \
  "${LINGBOT_WEIGHTS}" "${dependency_values[4]}" "${EXPECTED_LINGBOT_WEIGHTS_SHA}" <<'PY'
import json
import sys
out = sys.argv[1]
raw = sys.argv[2:]
names = ("gatecurr600", "navdp_checkpoint", "lingbot_map_long")
dependencies = {}
for index, name in enumerate(names):
    path, size, digest = raw[index * 3:(index + 1) * 3]
    dependencies[name] = {
        "path": path, "bytes": int(size), "sha256": digest}
payload = {
    "schema_version": "goat_certified_arrival_dependencies_v1_20260815",
    "dependencies": dependencies,
}
open(out, "x").write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY
"${LOCAL_PY}" - "${stage}/source_bundle_manifest.json" \
  "${FORMAL_MANIFEST_SHA}" "${SMOKE_MANIFEST_SHA}" \
  "${EXPECTED_BASE_SOURCE_RECEIPT_SHA}" <<'PY'
import datetime
import hashlib
import json
import pathlib
import sys
out = pathlib.Path(sys.argv[1])
root = out.parent
payload = {
    "schema_version": "goat_certified_arrival_bundle_v1_20260815",
    "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "scope": "frozen disjoint first-ImageGoal semantic-arrival confirmation",
    "is_full_goat_benchmark_score": False,
    "method_or_threshold_selection_allowed": False,
    "formal_manifest_sha256": sys.argv[2],
    "smoke_manifest_sha256": sys.argv[3],
    "base_source_receipt_sha256": sys.argv[4],
    "files": {},
}
for path in sorted(root.rglob("*")):
    if path.is_file() and path != out:
        payload["files"][path.relative_to(root).as_posix()] = hashlib.sha256(
            path.read_bytes()).hexdigest()
out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY

remote "test ! -e '${remote_partial}' && mkdir -p '${remote_partial}'"
rsync -e "${RSYNC_RSH}" -a --chmod=Du=rwx,Dgo=rx,Fu=rw,Fgo=r \
  "${stage}/" "${REMOTE_HOST}:${remote_partial}/"
remote "cd '${remote_partial}' && find . -type f ! -name SOURCE_BUNDLE.sha256 -print0 | sort -z | xargs -0 sha256sum > SOURCE_BUNDLE.sha256 && sha256sum -c SOURCE_BUNDLE.sha256 >/dev/null"
source_receipt_sha=$(remote "sha256sum '${remote_partial}/SOURCE_BUNDLE.sha256' | awk '{print \$1}'")
[[ "${source_receipt_sha}" =~ ^[0-9a-f]{64}$ ]] || fail "bad source receipt SHA"
remote_bundle=${REMOTE_BUNDLE_BASE}/goat_certified_arrival_${source_receipt_sha:0:16}
remote "test ! -e '${remote_bundle}' && chmod -R a-w '${remote_partial}' && mv '${remote_partial}' '${remote_bundle}' && cd '${remote_bundle}' && sha256sum -c SOURCE_BUNDLE.sha256 >/dev/null"

tag=$(date -u +%Y%m%dT%H%M%SZ)
smoke_root=${REMOTE_RESULT_BASE}/smoke_${tag}
formal_root=${REMOTE_RESULT_BASE}/formal_${tag}
eval_launcher=${remote_bundle}/MemNavData/slurm_goat_certified_arrival.sbatch
summary_launcher=${remote_bundle}/MemNavData/slurm_summarize_goat_certified_arrival.sbatch
verify_launcher=${remote_bundle}/MemNavData/slurm_verify_goat_certified_arrival.sbatch
common="SOURCE_ROOT=${remote_bundle},SOURCE_RECEIPT=${remote_bundle}/SOURCE_BUNDLE.sha256,EXPECTED_SOURCE_RECEIPT_SHA=${source_receipt_sha},BASE_SOURCE_ROOT=${BASE_SOURCE_ROOT},EXPECTED_BASE_SOURCE_RECEIPT_SHA=${EXPECTED_BASE_SOURCE_RECEIPT_SHA}"
smoke_exports="ALL,${common},MANIFEST=${remote_bundle}/MemNavData/goat_certified_arrival_smoke_manifest.json,EXPECTED_MANIFEST_SHA=${SMOKE_MANIFEST_SHA},RUN_ROOT=${smoke_root},RUN_MODE=smoke"
formal_exports="ALL,${common},MANIFEST=${remote_bundle}/MemNavData/goat_certified_arrival_manifest.json,EXPECTED_MANIFEST_SHA=${FORMAL_MANIFEST_SHA},RUN_ROOT=${formal_root},RUN_MODE=formal"
post_exports="ALL,SOURCE_ROOT=${remote_bundle},SOURCE_RECEIPT=${remote_bundle}/SOURCE_BUNDLE.sha256,EXPECTED_SOURCE_RECEIPT_SHA=${source_receipt_sha},MANIFEST=${remote_bundle}/MemNavData/goat_certified_arrival_manifest.json,EXPECTED_MANIFEST_SHA=${FORMAL_MANIFEST_SHA},RUN_ROOT=${formal_root}"

remote "sbatch --test-only --array=0-0 --export='${smoke_exports}' '${eval_launcher}' >/dev/null"
remote "sbatch --test-only --array=0-19%${FORMAL_CONCURRENCY} --export='${formal_exports}' '${eval_launcher}' >/dev/null"
smoke_raw=$(remote "sbatch --begin=now+5minutes --array=0-0 --parsable --export='${smoke_exports}' '${eval_launcher}'")
smoke_job=${smoke_raw%%;*}
[[ "${smoke_job}" =~ ^[0-9]+$ ]] || fail "bad smoke job id"
formal_raw=$(remote "sbatch --array=0-19%${FORMAL_CONCURRENCY} --parsable --dependency=afterok:${smoke_job} --export='${formal_exports}' '${eval_launcher}'")
formal_job=${formal_raw%%;*}
[[ "${formal_job}" =~ ^[0-9]+$ ]] || fail "bad formal job id"
summary_raw=$(remote "sbatch --parsable --dependency=afterok:${formal_job} --export='${post_exports}' '${summary_launcher}'")
summary_job=${summary_raw%%;*}
[[ "${summary_job}" =~ ^[0-9]+$ ]] || fail "bad summary job id"
verify_raw=$(remote "sbatch --parsable --dependency=afterok:${summary_job} --export='${post_exports}' '${verify_launcher}'")
verify_job=${verify_raw%%;*}
[[ "${verify_job}" =~ ^[0-9]+$ ]] || fail "bad verifier job id"

remote "mkdir -p '${smoke_root}' '${formal_root}' && '${REMOTE_PY}' - '${smoke_root}/submission.json' '${formal_root}/submission.json' '${smoke_job}' '${formal_job}' '${summary_job}' '${verify_job}' '${remote_bundle}' '${source_receipt_sha}' '${SMOKE_MANIFEST_SHA}' '${FORMAL_MANIFEST_SHA}'" <<'PY'
import json
import sys
import time
smoke_path, formal_path = sys.argv[1:3]
smoke_job, formal_job, summary_job, verify_job = map(int, sys.argv[3:7])
bundle, receipt, smoke_manifest, formal_manifest = sys.argv[7:]
common = {
    "schema_version": "goat_certified_arrival_submission_v1_20260815",
    "source_bundle": bundle,
    "source_receipt_sha256": receipt,
    "is_full_goat_benchmark_score": False,
    "method_or_threshold_selection_allowed": False,
    "submission_unix_time": time.time(),
}
smoke = dict(common, mode="engineering_smoke", job_id=smoke_job,
             manifest_sha256=smoke_manifest)
formal = dict(common, mode="formal_disjoint_confirmation",
              eval_job_id=formal_job, smoke_dependency_job_id=smoke_job,
              summary_job_id=summary_job, verifier_job_id=verify_job,
              manifest_sha256=formal_manifest)
open(smoke_path, "x").write(json.dumps(smoke, indent=2, sort_keys=True) + "\n")
open(formal_path, "x").write(json.dumps(formal, indent=2, sort_keys=True) + "\n")
PY
remote "chmod a-w '${smoke_root}/submission.json' '${formal_root}/submission.json'"

echo "GOAT_ARRIVAL_SMOKE_JOB=${smoke_job}"
echo "GOAT_ARRIVAL_FORMAL_JOB=${formal_job}"
echo "GOAT_ARRIVAL_SUMMARY_JOB=${summary_job}"
echo "GOAT_ARRIVAL_VERIFY_JOB=${verify_job}"
echo "GOAT_ARRIVAL_BUNDLE=${remote_bundle}"
echo "GOAT_ARRIVAL_SOURCE_RECEIPT_SHA=${source_receipt_sha}"
echo "GOAT_ARRIVAL_SMOKE_ROOT=${smoke_root}"
echo "GOAT_ARRIVAL_FORMAL_ROOT=${formal_root}"
