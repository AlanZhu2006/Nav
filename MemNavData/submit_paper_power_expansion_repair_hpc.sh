#!/usr/bin/env bash
# Resume the frozen phase-2 run after the manifest-count summarizer incident.

set -euo pipefail
umask 0022

LOCAL_ROOT=${LOCAL_ROOT:-$(git rev-parse --show-toplevel)}
REMOTE_HOST=${REMOTE_HOST:-alantorch}
EXPECTED_REMOTE_USER=${EXPECTED_REMOTE_USER:-yz11502}
SSH_CONTROL_PATH=${SSH_CONTROL_PATH:-/home/asus/.ssh/cm-e3ce4155ccb925413d599e13706baddf79fddff3}
PARENT_BUNDLE=${PARENT_BUNDLE:-/scratch/yz11502/Research/source_bundles/paper_power_expansion_915f6c6a30837ee5}
PARENT_RECEIPT_SHA=e29c996fbcf4ce9dabf1bae6db8bf61e2b8d7e924ea967307b8cd6772012f10b
PARENT_MANIFEST_SHA=915f6c6a30837ee5e86f4b4334b7322de9469f4f84b5ed74e98c34a7c539b1e0
REMOTE_BUNDLE_BASE=${REMOTE_BUNDLE_BASE:-/scratch/yz11502/Research/source_bundles}
RUN_ROOT=${RUN_ROOT:-/scratch/yz11502/Research/Nav-axis-uturn-results/paper_certified_compass_20260814/paper_power_expansion_20260814_pre_result}
MANIFEST_SHA=c148c9695d0a03f877cd860b1c1810ace36e4750da9a7ed5ec385bb29336a598
FAILED_SUMMARY_JOB=15729702
BASE_SOURCE_ROOT=${BASE_SOURCE_ROOT:-/scratch/yz11502/Research/Nav-axis-uturn-source-bundles/certified_relocalization_closed_loop_d3bd281fc374cc80}
BASE_SOURCE_RECEIPT_SHA=74001a9e0150c38c599a206fa0f4dd5e1279b9bed5d167119f4d14cb77995e98
DEPENDENCY_RECEIPT=${DEPENDENCY_RECEIPT:-/scratch/yz11502/Research/Nav-axis-uturn-results/shared_online_double_revisit_fresh_20260813/double_revisit_fresh40_20260813T200121Z/dependency_receipt.json}
EXPECTED_DEPENDENCY_RECEIPT_SHA=4eb0ca6479a26f8e04f85a31d906cee4e68b1785f66cfd3ac23bf65424d36e5e
REMOTE_PY=${REMOTE_PY:-/scratch/lg154/conda-envs/memnav/bin/python}
MEMNAV_PY=${MEMNAV_PY:-/home/asus/miniconda3/envs/memnav/bin/python}
EVAL_CONCURRENCY=${EVAL_CONCURRENCY:-4}
POPULATION_CAPACITY=64
EVAL_ARRAY_MAX=127
DRY_RUN=${DRY_RUN:-0}

fail() { echo "ABORT: $*" >&2; exit 2; }
[[ "${EVAL_CONCURRENCY}" =~ ^[1-9][0-9]*$ ]] || fail "bad eval concurrency"
[[ "${DRY_RUN}" =~ ^[01]$ ]] || fail "DRY_RUN must be 0 or 1"
[[ -S "${SSH_CONTROL_PATH}" ]] || fail "shared SSH control socket is missing"
SSH_ARGS=(-o BatchMode=yes -S "${SSH_CONTROL_PATH}" -o ControlMaster=no)
RSYNC_RSH="ssh -o BatchMode=yes -S ${SSH_CONTROL_PATH} -o ControlMaster=no"
remote() { ssh "${SSH_ARGS[@]}" "${REMOTE_HOST}" "$@"; }

SUMMARY_PY=${LOCAL_ROOT}/MemNavData/summarize_paper_online_a.py
SUMMARY_SBATCH=${LOCAL_ROOT}/MemNavData/slurm_paper_online_a_summary.sbatch
REPAIR_PROTOCOL=${LOCAL_ROOT}/MemNavData/PAPER_POWER_EXPANSION_REPAIR_PROTOCOL_20260814.md
TEST_SUMMARY=${LOCAL_ROOT}/MemNavData/test_summarize_paper_online_a.py
for path in "${SUMMARY_PY}" "${SUMMARY_SBATCH}" "${REPAIR_PROTOCOL}" "${TEST_SUMMARY}"; do
  [[ -f "${path}" && ! -L "${path}" ]] || fail "missing physical repair input: ${path}"
done

export PYTHONPATH=${LOCAL_ROOT}:${LOCAL_ROOT}/MemNavData${PYTHONPATH:+:${PYTHONPATH}}
"${MEMNAV_PY}" -m py_compile "${SUMMARY_PY}" "${TEST_SUMMARY}"
"${MEMNAV_PY}" "${TEST_SUMMARY}" -q
bash -n "${SUMMARY_SBATCH}"

[[ "$(remote 'id -un')" == "${EXPECTED_REMOTE_USER}" ]] || \
  fail "remote identity mismatch"
remote "test -d '${PARENT_BUNDLE}'"
remote "test \"\$(sha256sum '${PARENT_BUNDLE}/SOURCE_BUNDLE.sha256' | awk '{print \$1}')\" = '${PARENT_RECEIPT_SHA}'"
remote "test \"\$(sha256sum '${PARENT_BUNDLE}/source_bundle_manifest.json' | awk '{print \$1}')\" = '${PARENT_MANIFEST_SHA}'"
remote "cd '${PARENT_BUNDLE}' && sha256sum -c SOURCE_BUNDLE.sha256 >/dev/null"
remote "test \"\$(sha256sum '${BASE_SOURCE_ROOT}/SOURCE_BUNDLE.sha256' | awk '{print \$1}')\" = '${BASE_SOURCE_RECEIPT_SHA}'"
remote "test \"\$(sha256sum '${DEPENDENCY_RECEIPT}' | awk '{print \$1}')\" = '${EXPECTED_DEPENDENCY_RECEIPT_SHA}'"

# The completed collection is immutable input.  No official downstream output
# may pre-exist, otherwise the repair would risk mixing two attempts.
remote "test -f '${RUN_ROOT}/submission.json' && test ! -e '${RUN_ROOT}/online_a_inventory.json' && test ! -e '${RUN_ROOT}/benchmarks' && test ! -e '${RUN_ROOT}/evaluation' && test ! -e '${RUN_ROOT}/paper_role_pair_summary.json' && test ! -e '${RUN_ROOT}/paper_role_pair_independent_verification.json' && test ! -e '${RUN_ROOT}/repair_submission.json'"
remote "test \"\$(find '${RUN_ROOT}/traces' -mindepth 1 -maxdepth 1 -type d | wc -l)\" -eq 16"
remote "test \"\$(find '${RUN_ROOT}/traces' -mindepth 2 -maxdepth 2 -name receipt.json -type f | wc -l)\" -eq 16"

if [[ "${DRY_RUN}" == 1 ]]; then
  echo "DRY_RUN_PARENT_BUNDLE=${PARENT_BUNDLE}"
  echo "DRY_RUN_RUN_ROOT=${RUN_ROOT}"
  echo "DRY_RUN_ALLOWED_EXECUTABLE_DIFFS=2"
  exit 0
fi

PATCH_STAGE=$(mktemp -d /tmp/paper_power_repair_patch.XXXXXX)
REMOTE_PARTIAL=${REMOTE_BUNDLE_BASE}/paper_power_expansion_repair.partial.$$
cleanup() {
  rm -rf -- "${PATCH_STAGE}"
}
trap cleanup EXIT
mkdir -p "${PATCH_STAGE}/MemNavData"
cp "${SUMMARY_PY}" "${PATCH_STAGE}/MemNavData/summarize_paper_online_a.py"
cp "${SUMMARY_SBATCH}" "${PATCH_STAGE}/MemNavData/slurm_paper_online_a_summary.sbatch"
cp "${REPAIR_PROTOCOL}" "${PATCH_STAGE}/MemNavData/PAPER_POWER_EXPANSION_REPAIR_PROTOCOL_20260814.md"

remote "test ! -e '${REMOTE_PARTIAL}' && cp -a '${PARENT_BUNDLE}' '${REMOTE_PARTIAL}' && chmod -R u+w '${REMOTE_PARTIAL}'"
rsync -e "${RSYNC_RSH}" -a --chmod=Fu=rw,Fgo=r \
  "${PATCH_STAGE}/" "${REMOTE_HOST}:${REMOTE_PARTIAL}/"

remote "'${REMOTE_PY}' - '${PARENT_BUNDLE}' '${REMOTE_PARTIAL}' '${PARENT_RECEIPT_SHA}' '${PARENT_MANIFEST_SHA}'" <<'PY'
import hashlib,json,sys
from pathlib import Path

parent=Path(sys.argv[1]); child=Path(sys.argv[2])
parent_receipt_sha=sys.argv[3]; parent_manifest_sha=sys.argv[4]
ignored={"SOURCE_BUNDLE.sha256","source_bundle_manifest.json"}
allowed_changed={
    "MemNavData/summarize_paper_online_a.py",
    "MemNavData/slurm_paper_online_a_summary.sbatch",
}
allowed_added={
    "MemNavData/PAPER_POWER_EXPANSION_REPAIR_PROTOCOL_20260814.md",
}

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def inventory(root):
    return {
        p.relative_to(root).as_posix(): digest(p)
        for p in sorted(root.rglob("*"))
        if p.is_file() and p.relative_to(root).as_posix() not in ignored
    }

before=inventory(parent); after=inventory(child)
missing=set(before)-set(after)
added=set(after)-set(before)
changed={name for name in set(before)&set(after) if before[name]!=after[name]}
if missing:
    raise SystemExit(f"parent files missing from repair bundle: {sorted(missing)}")
if added != allowed_added:
    raise SystemExit(f"unexpected repair additions: {sorted(added)}")
if changed != allowed_changed:
    raise SystemExit(f"unexpected repair changes: {sorted(changed)}")

payload={
    "schema_version":"paper_power_expansion_repair_bundle_v1_20260814",
    "scope":"outcome-blind infrastructure repair of frozen phase-2 run",
    "parent_bundle":str(parent),
    "parent_source_receipt_sha256":parent_receipt_sha,
    "parent_bundle_manifest_sha256":parent_manifest_sha,
    "expansion_manifest_sha256":
        "c148c9695d0a03f877cd860b1c1810ace36e4750da9a7ed5ec385bb29336a598",
    "failed_summary_job":15729702,
    "query_outcomes_read_before_repair":False,
    "method_adaptation":"none",
    "policy_or_controller_files_changed":False,
    "allowed_executable_diffs":sorted(allowed_changed),
    "allowed_added_files":sorted(allowed_added),
    "files":after,
}
(child/"source_bundle_manifest.json").write_text(
    json.dumps(payload,indent=2,sort_keys=True)+"\n")
PY
remote "cd '${REMOTE_PARTIAL}' && find . -type f ! -name SOURCE_BUNDLE.sha256 -print0 | sort -z | xargs -0 sha256sum > SOURCE_BUNDLE.sha256 && sha256sum -c SOURCE_BUNDLE.sha256 >/dev/null"
SOURCE_RECEIPT_SHA=$(remote "sha256sum '${REMOTE_PARTIAL}/SOURCE_BUNDLE.sha256' | awk '{print \$1}'")
BUNDLE_MANIFEST_SHA=$(remote "sha256sum '${REMOTE_PARTIAL}/source_bundle_manifest.json' | awk '{print \$1}'")
[[ "${SOURCE_RECEIPT_SHA}" =~ ^[0-9a-f]{64}$ ]] || fail "bad child receipt SHA"
[[ "${BUNDLE_MANIFEST_SHA}" =~ ^[0-9a-f]{64}$ ]] || fail "bad child manifest SHA"
REMOTE_BUNDLE=${REMOTE_BUNDLE_BASE}/paper_power_expansion_repair_${BUNDLE_MANIFEST_SHA:0:16}
remote "test ! -e '${REMOTE_BUNDLE}' && chmod -R a-w '${REMOTE_PARTIAL}' && mv '${REMOTE_PARTIAL}' '${REMOTE_BUNDLE}'"
remote "test \"\$(sha256sum '${REMOTE_BUNDLE}/SOURCE_BUNDLE.sha256' | awk '{print \$1}')\" = '${SOURCE_RECEIPT_SHA}' && cd '${REMOTE_BUNDLE}' && sha256sum -c SOURCE_BUNDLE.sha256 >/dev/null"

SOURCE_RECEIPT=${REMOTE_BUNDLE}/SOURCE_BUNDLE.sha256
MANIFEST=${REMOTE_BUNDLE}/MemNavData/paper_source_manifest.json
exports="ALL,SOURCE_ROOT=${REMOTE_BUNDLE},SOURCE_RECEIPT=${SOURCE_RECEIPT},EXPECTED_SOURCE_RECEIPT_SHA=${SOURCE_RECEIPT_SHA},RUN_ROOT=${RUN_ROOT},MANIFEST=${MANIFEST},EXPECTED_MANIFEST_SHA=${MANIFEST_SHA},BASE_SOURCE_ROOT=${BASE_SOURCE_ROOT},BASE_SOURCE_RECEIPT_SHA=${BASE_SOURCE_RECEIPT_SHA},DEPENDENCY_RECEIPT=${DEPENDENCY_RECEIPT},EXPECTED_DEPENDENCY_RECEIPT_SHA=${EXPECTED_DEPENDENCY_RECEIPT_SHA},MAX_POPULATION_PER_PROTOCOL=${POPULATION_CAPACITY}"
SUMMARY=${REMOTE_BUNDLE}/MemNavData/slurm_paper_online_a_summary.sbatch
EVAL=${REMOTE_BUNDLE}/MemNavData/slurm_paper_role_pair_eval.sbatch
PAIR_SUMMARY=${REMOTE_BUNDLE}/MemNavData/slurm_paper_role_pair_summary.sbatch
VERIFY=${REMOTE_BUNDLE}/MemNavData/slurm_paper_role_pair_verify.sbatch

remote "sbatch --test-only --export='${exports}' '${SUMMARY}' >/dev/null"
summary_raw=$(remote "sbatch --parsable --export='${exports}' '${SUMMARY}'")
summary_id=${summary_raw%%;*}; [[ "${summary_id}" =~ ^[0-9]+$ ]] || fail "bad summary id"
remote "sbatch --test-only --array=0 --dependency=afterok:${summary_id} --kill-on-invalid-dep=yes --export='${exports}' '${EVAL}' >/dev/null"
eval_raw=$(remote "sbatch --parsable --array=0-${EVAL_ARRAY_MAX}%${EVAL_CONCURRENCY} --dependency=afterok:${summary_id} --kill-on-invalid-dep=yes --export='${exports}' '${EVAL}'")
eval_id=${eval_raw%%;*}; [[ "${eval_id}" =~ ^[0-9]+$ ]] || fail "bad eval id"
pair_summary_raw=$(remote "sbatch --parsable --dependency=afterok:${eval_id} --kill-on-invalid-dep=yes --export='${exports}' '${PAIR_SUMMARY}'")
pair_summary_id=${pair_summary_raw%%;*}; [[ "${pair_summary_id}" =~ ^[0-9]+$ ]] || fail "bad policy summary id"
verify_raw=$(remote "sbatch --parsable --dependency=afterok:${pair_summary_id} --kill-on-invalid-dep=yes --export='${exports}' '${VERIFY}'")
verify_id=${verify_raw%%;*}; [[ "${verify_id}" =~ ^[0-9]+$ ]] || fail "bad verify id"

remote "'${REMOTE_PY}' - '${RUN_ROOT}/repair_submission.json' '${REMOTE_BUNDLE}' '${SOURCE_RECEIPT_SHA}' '${BUNDLE_MANIFEST_SHA}' '${summary_id}' '${eval_id}' '${pair_summary_id}' '${verify_id}'" <<'PY'
import json,sys,time
(path,bundle,receipt,manifest,summary,evaluation,policy_summary,verification)=sys.argv[1:]
payload={
 "schema_version":"paper_power_expansion_repair_submission_v1_20260814",
 "scope":"resume frozen phase-2 after manifest-count infrastructure repair",
 "source_bundle":bundle,
 "source_receipt_sha256":receipt,
 "bundle_manifest_sha256":manifest,
 "parent_summary_job":15729702,
 "query_outcomes_read_before_repair":False,
 "method_adaptation":"none",
 "collection_reused":True,
 "collection_rerun":False,
 "allowed_executable_diffs":[
   "MemNavData/summarize_paper_online_a.py",
   "MemNavData/slurm_paper_online_a_summary.sbatch",
 ],
 "jobs":{"construction_summary":int(summary),
         "evaluation_array":int(evaluation),
         "policy_summary":int(policy_summary),
         "independent_verification":int(verification)},
 "submission_unix_time":time.time(),
}
open(path,"x").write(json.dumps(payload,indent=2,sort_keys=True)+"\n")
PY

echo "RUN_ROOT=${RUN_ROOT}"
echo "SOURCE_BUNDLE=${REMOTE_BUNDLE}"
echo "SOURCE_RECEIPT_SHA=${SOURCE_RECEIPT_SHA}"
echo "BUNDLE_MANIFEST_SHA=${BUNDLE_MANIFEST_SHA}"
echo "construction_summary=${summary_id} eval=${eval_id} policy_summary=${pair_summary_id} verify=${verify_id}"
