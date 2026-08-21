#!/usr/bin/env bash
# Submit the pre-result MP3D phase-2 power expansion from attempt-7 code.

set -euo pipefail
umask 0022

LOCAL_ROOT=${LOCAL_ROOT:-$(git rev-parse --show-toplevel)}
BASE_BUNDLE_LOCAL=${BASE_BUNDLE_LOCAL:-${LOCAL_ROOT}/.diagnostics/source_bundles/paper_online_a_340e6f1c61ae5052_attempt7_exact}
BASE_BUNDLE_RECEIPT_SHA=0487558de263b473590e67fbe7df464698f565b066d68f540f312dbf611df9fe
EXPANSION_MANIFEST=${EXPANSION_MANIFEST:-${LOCAL_ROOT}/.diagnostics/paper_power_expansion_freeze_20260814_pre_result/paper_power_expansion_manifest.json}
EXPANSION_MANIFEST_SHA=c148c9695d0a03f877cd860b1c1810ace36e4750da9a7ed5ec385bb29336a598
TRIGGER_RECEIPT=${TRIGGER_RECEIPT:-${LOCAL_ROOT}/.diagnostics/paper_power_expansion_freeze_20260814_pre_result/attempt7_population_receipt.json}
TRIGGER_RECEIPT_SHA=2ecb102f137f0ec25abd615ec544f342cb4d259a9d945fa069041a8a5bb611bc
UPSTREAM_VERIFY_JOB=${UPSTREAM_VERIFY_JOB:-15727268}
REMOTE_HOST=${REMOTE_HOST:-torch-login-a-2}
EXPECTED_REMOTE_USER=${EXPECTED_REMOTE_USER:-yz11502}
SSH_CONTROL_PATH=${SSH_CONTROL_PATH:-/home/asus/.ssh/cm-navpaper-20260814}
REMOTE_BUNDLE_BASE=${REMOTE_BUNDLE_BASE:-/scratch/yz11502/Research/source_bundles}
REMOTE_RESULT_BASE=${REMOTE_RESULT_BASE:-/scratch/yz11502/Research/Nav-axis-uturn-results/paper_certified_compass_20260814}
RUN_TAG=${RUN_TAG:-paper_power_expansion_20260814_pre_result}
RUN_ROOT=${RUN_ROOT:-${REMOTE_RESULT_BASE}/${RUN_TAG}}
CONCURRENCY=${CONCURRENCY:-8}
EVAL_CONCURRENCY=${EVAL_CONCURRENCY:-4}
DRY_RUN=${DRY_RUN:-0}
POPULATION_CAPACITY=64
EVAL_ARRAY_MAX=127
BASE_SOURCE_ROOT=${BASE_SOURCE_ROOT:-/scratch/yz11502/Research/Nav-axis-uturn-source-bundles/certified_relocalization_closed_loop_d3bd281fc374cc80}
BASE_SOURCE_RECEIPT_SHA=74001a9e0150c38c599a206fa0f4dd5e1279b9bed5d167119f4d14cb77995e98
DEPENDENCY_RECEIPT=${DEPENDENCY_RECEIPT:-/scratch/yz11502/Research/Nav-axis-uturn-results/shared_online_double_revisit_fresh_20260813/double_revisit_fresh40_20260813T200121Z/dependency_receipt.json}
EXPECTED_DEPENDENCY_RECEIPT_SHA=4eb0ca6479a26f8e04f85a31d906cee4e68b1785f66cfd3ac23bf65424d36e5e
MEMNAV_PY=${MEMNAV_PY:-/home/asus/miniconda3/envs/memnav/bin/python}
HAB_PY=${HAB_PY:-/home/asus/miniconda3/envs/habitat/bin/python}

fail() { echo "ABORT: $*" >&2; exit 2; }
[[ "${RUN_TAG}" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]] || fail "bad RUN_TAG"
[[ "${UPSTREAM_VERIFY_JOB}" =~ ^[0-9]+$ ]] || fail "bad upstream job id"
[[ "${CONCURRENCY}" =~ ^[1-9][0-9]*$ \
   && "${EVAL_CONCURRENCY}" =~ ^[1-9][0-9]*$ ]] || fail "bad concurrency"
[[ "${DRY_RUN}" =~ ^[01]$ ]] || fail "DRY_RUN must be 0 or 1"
[[ -S "${SSH_CONTROL_PATH}" ]] || fail "SSH control socket is missing"
SSH_ARGS=(-o BatchMode=yes -S "${SSH_CONTROL_PATH}" -o ControlMaster=no)
RSYNC_RSH="ssh -o BatchMode=yes -S ${SSH_CONTROL_PATH} -o ControlMaster=no"
remote() { ssh "${SSH_ARGS[@]}" "${REMOTE_HOST}" "$@"; }

[[ -r "${BASE_BUNDLE_LOCAL}/SOURCE_BUNDLE.sha256" ]] || \
  fail "attempt-7 exact bundle is missing"
[[ "$(sha256sum "${BASE_BUNDLE_LOCAL}/SOURCE_BUNDLE.sha256" | awk '{print $1}')" == \
  "${BASE_BUNDLE_RECEIPT_SHA}" ]] || fail "attempt-7 bundle receipt changed"
(cd "${BASE_BUNDLE_LOCAL}" && sha256sum -c SOURCE_BUNDLE.sha256) >/dev/null || \
  fail "attempt-7 exact bundle content changed"
[[ "$(sha256sum "${EXPANSION_MANIFEST}" | awk '{print $1}')" == \
  "${EXPANSION_MANIFEST_SHA}" ]] || fail "expansion manifest changed"
[[ "$(sha256sum "${TRIGGER_RECEIPT}" | awk '{print $1}')" == \
  "${TRIGGER_RECEIPT_SHA}" ]] || fail "trigger population receipt changed"
"${MEMNAV_PY}" - "${EXPANSION_MANIFEST}" "${TRIGGER_RECEIPT}" <<'PY'
import json,sys
m=json.load(open(sys.argv[1])); t=json.load(open(sys.argv[2]))
assert m["power_expansion"]["query_outcomes_read_before_freeze"] is False
assert m["power_expansion"]["scene_replacement"] is False
assert m["evaluation"]["episodes_per_scene"] == 4
assert len(m["episodes"]) == 16
assert sum(map(len,m["episodes"].values())) == 64
assert t["policy_outcomes_read"] is False and t["target_met"] is False
assert t["role_pair_constructible_histories"] == 9
assert t["role_pair_scene_count"] == 9
PY
"${HAB_PY}" -m py_compile \
  "${LOCAL_ROOT}/MemNavData/validate_paper_online_a_scene.py"
"${MEMNAV_PY}" -m py_compile \
  "${LOCAL_ROOT}/MemNavData/summarize_paper_online_a.py"
bash -n \
  "${LOCAL_ROOT}/MemNavData/slurm_paper_online_a_summary.sbatch" \
  "${LOCAL_ROOT}/MemNavData/slurm_paper_role_pair_eval.sbatch"

staging=$(mktemp -d /tmp/paper_power_expansion_bundle.XXXXXX)
cleanup() {
  if [[ -d "${staging}" ]]; then rm -rf -- "${staging}"; fi
}
trap cleanup EXIT
cp -a "${BASE_BUNDLE_LOCAL}/." "${staging}/"
chmod -R u+w "${staging}"
mkdir -p "${staging}/receipts"
cp "${staging}/source_bundle_manifest.json" \
  "${staging}/receipts/attempt7_source_bundle_manifest.json"
cp "${staging}/SOURCE_BUNDLE.sha256" \
  "${staging}/receipts/attempt7_SOURCE_BUNDLE.sha256"
rm -f "${staging}/source_bundle_manifest.json" \
  "${staging}/SOURCE_BUNDLE.sha256"
cp "${EXPANSION_MANIFEST}" \
  "${staging}/MemNavData/paper_source_manifest.json"
cp "${TRIGGER_RECEIPT}" \
  "${staging}/receipts/attempt7_population_receipt.json"
cp "${LOCAL_ROOT}/MemNavData/PAPER_POWER_EXPANSION_PROTOCOL_20260814.md" \
  "${staging}/MemNavData/PAPER_POWER_EXPANSION_PROTOCOL_20260814.md"
cp "${LOCAL_ROOT}/MemNavData/freeze_paper_power_expansion_manifest.py" \
  "${staging}/MemNavData/freeze_paper_power_expansion_manifest.py"
# These are orchestration-only generalizations.  All policy, method, builder,
# arm and metric files remain byte-identical to the verified attempt-7 bundle.
cp "${LOCAL_ROOT}/MemNavData/validate_paper_online_a_scene.py" \
  "${staging}/MemNavData/validate_paper_online_a_scene.py"
cp "${LOCAL_ROOT}/MemNavData/summarize_paper_online_a.py" \
  "${staging}/MemNavData/summarize_paper_online_a.py"
cp "${LOCAL_ROOT}/MemNavData/slurm_paper_online_a_summary.sbatch" \
  "${staging}/MemNavData/slurm_paper_online_a_summary.sbatch"
cp "${LOCAL_ROOT}/MemNavData/slurm_paper_role_pair_eval.sbatch" \
  "${staging}/MemNavData/slurm_paper_role_pair_eval.sbatch"

"${MEMNAV_PY}" - "${staging}" <<'PY'
import hashlib,json,sys
from pathlib import Path
root=Path(sys.argv[1]); files={}
for path in sorted(root.rglob("*")):
    if path.is_symlink(): raise SystemExit(f"bundle symlink: {path}")
    if path.is_file() and path.name not in {
        "source_bundle_manifest.json", "SOURCE_BUNDLE.sha256"}:
        files[path.relative_to(root).as_posix()]=hashlib.sha256(
            path.read_bytes()).hexdigest()
payload={
 "schema_version":"paper_power_expansion_bundle_v1_20260814",
 "scope":"pre-result phase-2 MP3D power replication",
 "parent_attempt7_source_receipt_sha256":
   "0487558de263b473590e67fbe7df464698f565b066d68f540f312dbf611df9fe",
 "expansion_manifest_sha256":
   "c148c9695d0a03f877cd860b1c1810ace36e4750da9a7ed5ec385bb29336a598",
 "trigger_population_receipt_sha256":
   "2ecb102f137f0ec25abd615ec544f342cb4d259a9d945fa069041a8a5bb611bc",
 "query_outcomes_read_before_freeze":False,
 "method_adaptation":"none",
 "orchestration_overlays":[
   "MemNavData/validate_paper_online_a_scene.py: frozen episodes_per_scene",
   "MemNavData/summarize_paper_online_a.py: manifest-driven episodes_per_scene",
   "MemNavData/slurm_paper_online_a_summary.sbatch: explicit frozen manifest",
   "MemNavData/slurm_paper_role_pair_eval.sbatch: array capacity 64/protocol",
 ],
 "files":files,
}
(root/"source_bundle_manifest.json").write_text(
    json.dumps(payload,indent=2,sort_keys=True)+"\n")
PY
(
  cd "${staging}"
  find . -type f ! -name SOURCE_BUNDLE.sha256 -print0 | sort -z | \
    xargs -0 sha256sum > SOURCE_BUNDLE.sha256
  sha256sum -c SOURCE_BUNDLE.sha256 >/dev/null
)
SOURCE_RECEIPT_SHA=$(sha256sum "${staging}/SOURCE_BUNDLE.sha256" | awk '{print $1}')
BUNDLE_MANIFEST_SHA=$(sha256sum "${staging}/source_bundle_manifest.json" | awk '{print $1}')
REMOTE_BUNDLE=${REMOTE_BUNDLE_BASE}/paper_power_expansion_${BUNDLE_MANIFEST_SHA:0:16}
REMOTE_STAGING=${REMOTE_BUNDLE}.partial-$$

if [[ "${DRY_RUN}" == 1 ]]; then
  echo "DRY_RUN_RUN_ROOT=${RUN_ROOT}"
  echo "DRY_RUN_REMOTE_BUNDLE=${REMOTE_BUNDLE}"
  echo "DRY_RUN_SOURCE_RECEIPT_SHA=${SOURCE_RECEIPT_SHA}"
  echo "DRY_RUN_MANIFEST_SHA=${EXPANSION_MANIFEST_SHA}"
  exit 0
fi

actual_remote_user=$(remote "id -un")
[[ "${actual_remote_user}" == "${EXPECTED_REMOTE_USER}" ]] || \
  fail "remote identity mismatch"
remote "scontrol show job '${UPSTREAM_VERIFY_JOB}' >/dev/null"
if remote "test -d '${REMOTE_BUNDLE}' && test \"\$(sha256sum '${REMOTE_BUNDLE}/SOURCE_BUNDLE.sha256' | awk '{print \$1}')\" = '${SOURCE_RECEIPT_SHA}' && cd '${REMOTE_BUNDLE}' && sha256sum -c SOURCE_BUNDLE.sha256 >/dev/null"; then
  echo "Reusing verified bundle ${REMOTE_BUNDLE}"
else
  remote "test ! -e '${REMOTE_BUNDLE}' && mkdir -p '${REMOTE_STAGING}'"
  rsync -e "${RSYNC_RSH}" -a --chmod=Du=rwx,Dgo=rx,Fu=rw,Fgo=r \
    "${staging}/" "${REMOTE_HOST}:${REMOTE_STAGING}/"
  remote "test ! -e '${REMOTE_BUNDLE}' && cd '${REMOTE_STAGING}' && sha256sum -c SOURCE_BUNDLE.sha256 >/dev/null && chmod -R a-w '${REMOTE_STAGING}' && mv '${REMOTE_STAGING}' '${REMOTE_BUNDLE}'"
fi
remote "test ! -e '${RUN_ROOT}' && mkdir -p '${RUN_ROOT}/logs'"
remote "test \"\$(sha256sum '${BASE_SOURCE_ROOT}/SOURCE_BUNDLE.sha256' | awk '{print \$1}')\" = '${BASE_SOURCE_RECEIPT_SHA}'"
remote "test \"\$(sha256sum '${DEPENDENCY_RECEIPT}' | awk '{print \$1}')\" = '${EXPECTED_DEPENDENCY_RECEIPT_SHA}'"

SOURCE_RECEIPT=${REMOTE_BUNDLE}/SOURCE_BUNDLE.sha256
MANIFEST=${REMOTE_BUNDLE}/MemNavData/paper_source_manifest.json
exports="ALL,SOURCE_ROOT=${REMOTE_BUNDLE},SOURCE_RECEIPT=${SOURCE_RECEIPT},EXPECTED_SOURCE_RECEIPT_SHA=${SOURCE_RECEIPT_SHA},RUN_ROOT=${RUN_ROOT},MANIFEST=${MANIFEST},EXPECTED_MANIFEST_SHA=${EXPANSION_MANIFEST_SHA},BASE_SOURCE_ROOT=${BASE_SOURCE_ROOT},BASE_SOURCE_RECEIPT_SHA=${BASE_SOURCE_RECEIPT_SHA},DEPENDENCY_RECEIPT=${DEPENDENCY_RECEIPT},EXPECTED_DEPENDENCY_RECEIPT_SHA=${EXPECTED_DEPENDENCY_RECEIPT_SHA},MAX_POPULATION_PER_PROTOCOL=${POPULATION_CAPACITY}"
COLLECT=${REMOTE_BUNDLE}/MemNavData/slurm_paper_online_a_collect.sbatch
SUMMARY=${REMOTE_BUNDLE}/MemNavData/slurm_paper_online_a_summary.sbatch
EVAL=${REMOTE_BUNDLE}/MemNavData/slurm_paper_role_pair_eval.sbatch
PAIR_SUMMARY=${REMOTE_BUNDLE}/MemNavData/slurm_paper_role_pair_summary.sbatch
VERIFY=${REMOTE_BUNDLE}/MemNavData/slurm_paper_role_pair_verify.sbatch

remote "sbatch --test-only --array=0 --dependency=afterok:${UPSTREAM_VERIFY_JOB} --kill-on-invalid-dep=yes --export='${exports}' '${COLLECT}' >/dev/null"
collect_raw=$(remote "sbatch --parsable --array=0-15%${CONCURRENCY} --dependency=afterok:${UPSTREAM_VERIFY_JOB} --kill-on-invalid-dep=yes --export='${exports}' '${COLLECT}'")
collect_id=${collect_raw%%;*}; [[ "${collect_id}" =~ ^[0-9]+$ ]] || fail "bad collect id"
summary_raw=$(remote "sbatch --parsable --dependency=afterok:${collect_id} --kill-on-invalid-dep=yes --export='${exports}' '${SUMMARY}'")
summary_id=${summary_raw%%;*}; [[ "${summary_id}" =~ ^[0-9]+$ ]] || fail "bad summary id"
remote "sbatch --test-only --array=0 --dependency=afterok:${summary_id} --kill-on-invalid-dep=yes --export='${exports}' '${EVAL}' >/dev/null"
eval_raw=$(remote "sbatch --parsable --array=0-${EVAL_ARRAY_MAX}%${EVAL_CONCURRENCY} --dependency=afterok:${summary_id} --kill-on-invalid-dep=yes --export='${exports}' '${EVAL}'")
eval_id=${eval_raw%%;*}; [[ "${eval_id}" =~ ^[0-9]+$ ]] || fail "bad eval id"
pair_summary_raw=$(remote "sbatch --parsable --dependency=afterok:${eval_id} --kill-on-invalid-dep=yes --export='${exports}' '${PAIR_SUMMARY}'")
pair_summary_id=${pair_summary_raw%%;*}; [[ "${pair_summary_id}" =~ ^[0-9]+$ ]] || fail "bad policy summary id"
verify_raw=$(remote "sbatch --parsable --dependency=afterok:${pair_summary_id} --kill-on-invalid-dep=yes --export='${exports}' '${VERIFY}'")
verify_id=${verify_raw%%;*}; [[ "${verify_id}" =~ ^[0-9]+$ ]] || fail "bad verify id"

remote "'/scratch/lg154/conda-envs/memnav/bin/python' - '${RUN_ROOT}/submission.json' '${REMOTE_BUNDLE}' '${SOURCE_RECEIPT_SHA}' '${EXPANSION_MANIFEST_SHA}' '${UPSTREAM_VERIFY_JOB}' '${collect_id}' '${summary_id}' '${eval_id}' '${pair_summary_id}' '${verify_id}'" <<'PY'
import json,sys,time
(path,bundle,receipt,manifest,upstream,collect,construction,evaluation,
 policy_summary,verification)=sys.argv[1:]
payload={
 "schema_version":"paper_power_expansion_submission_v1_20260814",
 "scope":"pre-result phase-2 MP3D power replication",
 "source_bundle":bundle,"source_receipt_sha256":receipt,
 "expansion_manifest_sha256":manifest,
 "query_outcomes_read_before_freeze":False,
 "method_adaptation":"none",
 "upstream_attempt7_independent_verification_job":int(upstream),
 "jobs":{"collect_array":int(collect),"construction_summary":int(construction),
         "evaluation_array":int(evaluation),"policy_summary":int(policy_summary),
         "independent_verification":int(verification)},
 "arrays":{"source_scenes":16,"source_episodes":64,
           "per_protocol_capacity":64,"evaluation_tasks":128},
 "submission_unix_time":time.time(),
}
open(path,"x").write(json.dumps(payload,indent=2,sort_keys=True)+"\n")
PY

echo "RUN_ROOT=${RUN_ROOT}"
echo "SOURCE_BUNDLE=${REMOTE_BUNDLE}"
echo "SOURCE_RECEIPT_SHA=${SOURCE_RECEIPT_SHA}"
echo "MANIFEST_SHA=${EXPANSION_MANIFEST_SHA}"
echo "upstream_verify=${UPSTREAM_VERIFY_JOB} collect=${collect_id} construction_summary=${summary_id} eval=${eval_id} policy_summary=${pair_summary_id} verify=${verify_id}"
