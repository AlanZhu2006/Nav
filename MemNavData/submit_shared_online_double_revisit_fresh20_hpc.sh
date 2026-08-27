#!/usr/bin/env bash
# Amend the failed fresh40 preparation into a clearly lower-power fresh20 gate,
# then submit eval[20] -> independent audit.  The original submission and
# failed-job evidence remain untouched.
set -euo pipefail
umask 0022

LOCAL_ROOT=$(git rev-parse --show-toplevel)
REMOTE_HOST=${REMOTE_HOST:-alantorch}
REMOTE_BUNDLE_BASE=${REMOTE_BUNDLE_BASE:-/scratch/yz11502/Research/Nav-axis-uturn-source-bundles}
RUN_ROOT=${RUN_ROOT:-/scratch/yz11502/Research/Nav-axis-uturn-results/shared_online_double_revisit_fresh_20260813/double_revisit_fresh40_20260813T200121Z}
UPSTREAM_RUN_ROOT=${UPSTREAM_RUN_ROOT:-/scratch/yz11502/Research/Nav-axis-uturn-results/certified_relocalization_closed_loop_20260812/certrel_bearing_v1_20260812T1050}
EXPECTED_UPSTREAM_MANIFEST_SHA=8013fa2a768d84638a9f9ecc50df46dda67ebb79250d14ad0a8087ac52fd33e5
BASE_SOURCE_ROOT=${BASE_SOURCE_ROOT:-/scratch/yz11502/Research/Nav-axis-uturn-source-bundles/certified_relocalization_closed_loop_d3bd281fc374cc80}
BASE_SOURCE_RECEIPT_SHA=74001a9e0150c38c599a206fa0f4dd5e1279b9bed5d167119f4d14cb77995e98
FAILED_PREP_JOB_ID=${FAILED_PREP_JOB_ID:-15663165}
MEMNAV_PY=${MEMNAV_PY:-/home/asus/miniconda3/envs/memnav/bin/python}
HAB_PY=${HAB_PY:-/home/asus/miniconda3/envs/habitat/bin/python}
REMOTE_MEMNAV_PY=${REMOTE_MEMNAV_PY:-/scratch/lg154/conda-envs/memnav/bin/python}
ARRAY_CONCURRENCY=${ARRAY_CONCURRENCY:-4}
DRY_RUN=${DRY_RUN:-0}
EVAL_ONLY_RETRY=${EVAL_ONLY_RETRY:-0}
EVAL_ATTEMPT_TAG=${EVAL_ATTEMPT_TAG:-retry1}
PRIOR_FAILED_EVAL_JOB_ID=${PRIOR_FAILED_EVAL_JOB_ID:-15669808}
PRIOR_FAILURE_REASON=${PRIOR_FAILURE_REASON:-startup-only failure: Habitat process resolved host libstdc++ instead of the container runtime}
PRIOR_ATTEMPT_ARCHIVE=${PRIOR_ATTEMPT_ARCHIVE:-${RUN_ROOT}/failed_attempts/retry1_eval_15669808}
RETRY_SUBMISSION_PATH=${RUN_ROOT}/submission_fresh20_${EVAL_ATTEMPT_TAG}.json
RETRY_SUBMISSION_LOCK=${RETRY_SUBMISSION_PATH}.lock
remote() {
  ssh -o BatchMode=yes "${REMOTE_HOST}" "$@"
}

[[ "${FAILED_PREP_JOB_ID}" =~ ^[0-9]+$ ]] || {
  echo "ABORT: invalid failed preparation job id" >&2; exit 2; }
[[ "${ARRAY_CONCURRENCY}" =~ ^[1-9][0-9]*$ ]] || {
  echo "ABORT: ARRAY_CONCURRENCY must be positive" >&2; exit 2; }
[[ "${EVAL_ONLY_RETRY}" == 0 || "${EVAL_ONLY_RETRY}" == 1 ]] || {
  echo "ABORT: EVAL_ONLY_RETRY must be 0 or 1" >&2; exit 2; }
[[ "${EVAL_ATTEMPT_TAG}" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]] || {
  echo "ABORT: invalid EVAL_ATTEMPT_TAG" >&2; exit 2; }
[[ "${PRIOR_FAILED_EVAL_JOB_ID}" =~ ^[0-9]+$ ]] || {
  echo "ABORT: invalid prior failed evaluation job id" >&2; exit 2; }

# Serialize retries before any sbatch call.  A stale lock is intentionally
# fail-closed: it must be audited against squeue/sacct before another array can
# be submitted.  This prevents duplicate arrays if the client loses the final
# stdout after Slurm has already accepted the jobs.
if [[ "${DRY_RUN}" == 0 && "${EVAL_ONLY_RETRY}" == 1 ]]; then
  remote \
    "test ! -e '${RETRY_SUBMISSION_PATH}' && mkdir '${RETRY_SUBMISSION_LOCK}'"
fi

required=(
  MemNavData/finalize_shared_online_double_revisit_fresh20.py
  MemNavData/prepare_shared_online_double_revisit_fresh.py
  MemNavData/audit_shared_online_double_revisit_fresh.py
  MemNavData/run_shared_online_double_revisit_fresh_episode.sh
  MemNavData/slurm_shared_online_double_revisit_finalize20.sbatch
  MemNavData/slurm_shared_online_double_revisit_eval.sbatch
  MemNavData/slurm_shared_online_double_revisit_summary.sbatch
  MemNavData/SHARED_ONLINE_DOUBLE_REVISIT_FRESH_PROTOCOL_20260813.md
  NavDP/baselines/memnav/memnav_server.py
  NavDP/baselines/memnav/policy_agent.py
  NavDP/baselines/navdp/navdp_server.py
  NavDP/baselines/navdp/policy_agent.py
)
for relative in "${required[@]}"; do
  [[ -f "${LOCAL_ROOT}/${relative}" && ! -L "${LOCAL_ROOT}/${relative}" ]] || {
    echo "ABORT: missing physical input ${relative}" >&2; exit 2; }
done

export PYTHONPATH=${LOCAL_ROOT}:${LOCAL_ROOT}/MemNavData${PYTHONPATH:+:${PYTHONPATH}}
"${HAB_PY}" -m py_compile \
  "${LOCAL_ROOT}/MemNavData/finalize_shared_online_double_revisit_fresh20.py" \
  "${LOCAL_ROOT}/MemNavData/audit_shared_online_double_revisit_fresh.py" \
  "${LOCAL_ROOT}/MemNavData/eval_shared_online_double_revisit.py"
"${MEMNAV_PY}" -m py_compile \
  "${LOCAL_ROOT}/NavDP/baselines/memnav/memnav_server.py" \
  "${LOCAL_ROOT}/NavDP/baselines/memnav/policy_agent.py" \
  "${LOCAL_ROOT}/NavDP/baselines/navdp/navdp_server.py" \
  "${LOCAL_ROOT}/NavDP/baselines/navdp/policy_agent.py"
"${HAB_PY}" -m unittest \
  MemNavData.test_prepare_shared_online_double_revisit_fresh \
  MemNavData.test_audit_shared_online_double_revisit_fresh \
  MemNavData.test_multigoal_policy_contract \
  MemNavData.test_shared_online_double_revisit_runtime \
  MemNavData.test_navdp_goal_switch
bash -n \
  "${LOCAL_ROOT}/MemNavData/run_shared_online_double_revisit_fresh_episode.sh" \
  "${LOCAL_ROOT}/MemNavData/slurm_shared_online_double_revisit_finalize20.sbatch" \
  "${LOCAL_ROOT}/MemNavData/slurm_shared_online_double_revisit_eval.sbatch" \
  "${LOCAL_ROOT}/MemNavData/slurm_shared_online_double_revisit_summary.sbatch"

STAGING=$(mktemp -d)
trap 'rm -rf -- "${STAGING}"' EXIT
mkdir -p "${STAGING}/MemNavData" \
  "${STAGING}/NavDP/baselines/memnav" \
  "${STAGING}/NavDP/baselines/navdp"
while IFS= read -r -d '' path; do
  cp --preserve=mode,timestamps "${path}" \
    "${STAGING}/MemNavData/$(basename "${path}")"
done < <(find "${LOCAL_ROOT}/MemNavData" -maxdepth 1 -type f -name '*.py' -print0)
for relative in \
  MemNavData/run_shared_online_double_revisit_fresh_episode.sh \
  MemNavData/slurm_shared_online_double_revisit_finalize20.sbatch \
  MemNavData/slurm_shared_online_double_revisit_eval.sbatch \
  MemNavData/slurm_shared_online_double_revisit_summary.sbatch \
  MemNavData/SHARED_ONLINE_DOUBLE_REVISIT_FRESH_PROTOCOL_20260813.md; do
  cp --preserve=mode,timestamps "${LOCAL_ROOT}/${relative}" "${STAGING}/${relative}"
done
for component in memnav navdp; do
  while IFS= read -r -d '' path; do
    cp --preserve=mode,timestamps "${path}" \
      "${STAGING}/NavDP/baselines/${component}/$(basename "${path}")"
  done < <(find "${LOCAL_ROOT}/NavDP/baselines/${component}" \
    -maxdepth 1 -type f -name '*.py' -print0)
done

LOCAL_HEAD=$(git -C "${LOCAL_ROOT}" rev-parse HEAD)
"${MEMNAV_PY}" - "${STAGING}" "${LOCAL_HEAD}" \
  "${BASE_SOURCE_ROOT}" "${BASE_SOURCE_RECEIPT_SHA}" <<'PY'
import hashlib,json,sys
from pathlib import Path
root=Path(sys.argv[1]); files={}
for path in sorted(root.rglob("*")):
    if path.is_symlink(): raise SystemExit(f"task bundle contains symlink: {path}")
    if path.is_file() and path.name not in {"source_bundle_manifest.json","SOURCE_BUNDLE.sha256"}:
        files[path.relative_to(root).as_posix()]=hashlib.sha256(path.read_bytes()).hexdigest()
payload={
 "schema_version":"shared_online_double_revisit_fresh20_task_bundle_v1",
 "local_git_head_context":sys.argv[2],
 "base_source_root":sys.argv[3],
 "base_source_receipt_sha256":sys.argv[4],
 "preregistered_power_target":40,
 "strict_constructible_population":20,
 "formal_power_target_met":False,
 "navigation_arms":["full_memory","memory_b_native_c","certified","native"],
 "files":files,
}
(root/"source_bundle_manifest.json").write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n")
PY
(
  cd "${STAGING}"
  find . -type f ! -name SOURCE_BUNDLE.sha256 -print0 | sort -z | \
    xargs -0 sha256sum > SOURCE_BUNDLE.sha256
  sha256sum -c SOURCE_BUNDLE.sha256 >/dev/null
)

SOURCE_RECEIPT_SHA=$(sha256sum "${STAGING}/SOURCE_BUNDLE.sha256" | awk '{print $1}')
BUNDLE_MANIFEST_SHA=$(sha256sum "${STAGING}/source_bundle_manifest.json" | awk '{print $1}')
REMOTE_BUNDLE=${REMOTE_BUNDLE_BASE}/shared_online_double_revisit_fresh20_${BUNDLE_MANIFEST_SHA:0:16}
REMOTE_STAGING=${REMOTE_BUNDLE}.partial-${FAILED_PREP_JOB_ID}

if [[ "${DRY_RUN}" == 1 ]]; then
  echo "DRY_RUN_SOURCE_RECEIPT_SHA=${SOURCE_RECEIPT_SHA}"
  echo "DRY_RUN_BUNDLE_MANIFEST_SHA=${BUNDLE_MANIFEST_SHA}"
  echo "DRY_RUN_REMOTE_BUNDLE=${REMOTE_BUNDLE}"
  echo "DRY_RUN_RUN_ROOT=${RUN_ROOT}"
  exit 0
fi

remote \
  "test \"\$(sha256sum '${BASE_SOURCE_ROOT}/SOURCE_BUNDLE.sha256' | awk '{print \$1}')\" = '${BASE_SOURCE_RECEIPT_SHA}'"
if [[ "${EVAL_ONLY_RETRY}" == 1 ]]; then
  remote \
    "test \"\$(sha256sum '${UPSTREAM_RUN_ROOT}/data_manifest.json' | awk '{print \$1}')\" = '${EXPECTED_UPSTREAM_MANIFEST_SHA}' && test -f '${RUN_ROOT}/prepared/SEALED' && test ! -e '${RUN_ROOT}/prepared/INCOMPLETE' && test ! -e '${RUN_ROOT}/scenes' && test -f '${PRIOR_ATTEMPT_ARCHIVE}/failure_receipt.json' && test ! -e '${RUN_ROOT}/report.json' && test ! -e '${RETRY_SUBMISSION_PATH}' && test -d '${RETRY_SUBMISSION_LOCK}'"
else
  remote \
    "test \"\$(sha256sum '${UPSTREAM_RUN_ROOT}/data_manifest.json' | awk '{print \$1}')\" = '${EXPECTED_UPSTREAM_MANIFEST_SHA}' && test -f '${RUN_ROOT}/prepared/INCOMPLETE' && test ! -e '${RUN_ROOT}/prepared/SEALED' && test ! -e '${RUN_ROOT}/scenes' && test ! -e '${RUN_ROOT}/submission_fresh20.json'"
fi

if remote \
    "test -d '${REMOTE_BUNDLE}' && test \"\$(sha256sum '${REMOTE_BUNDLE}/SOURCE_BUNDLE.sha256' | awk '{print \$1}')\" = '${SOURCE_RECEIPT_SHA}' && cd '${REMOTE_BUNDLE}' && sha256sum -c SOURCE_BUNDLE.sha256 >/dev/null"; then
  echo "Reusing verified task bundle ${REMOTE_BUNDLE}"
else
  remote \
    "test ! -e '${REMOTE_BUNDLE}' && mkdir -p '${REMOTE_STAGING}'"
  rsync -a --chmod=Du=rwx,Dgo=rx,Fu=rw,Fgo=r \
    "${STAGING}/" "${REMOTE_HOST}:${REMOTE_STAGING}/"
  remote \
    "test ! -e '${REMOTE_BUNDLE}' && cd '${REMOTE_STAGING}' && sha256sum -c SOURCE_BUNDLE.sha256 >/dev/null && chmod -R a-w '${REMOTE_STAGING}' && mv '${REMOTE_STAGING}' '${REMOTE_BUNDLE}'"
fi

REMOTE_SOURCE_RECEIPT=${REMOTE_BUNDLE}/SOURCE_BUNDLE.sha256
exports="ALL,SOURCE_ROOT=${REMOTE_BUNDLE},SOURCE_RECEIPT=${REMOTE_SOURCE_RECEIPT},EXPECTED_SOURCE_RECEIPT_SHA=${SOURCE_RECEIPT_SHA},BASE_SOURCE_ROOT=${BASE_SOURCE_ROOT},BASE_SOURCE_RECEIPT_SHA=${BASE_SOURCE_RECEIPT_SHA},RUN_ROOT=${RUN_ROOT},UPSTREAM_RUN_ROOT=${UPSTREAM_RUN_ROOT},EXPECTED_UPSTREAM_MANIFEST_SHA=${EXPECTED_UPSTREAM_MANIFEST_SHA},FAILED_PREP_JOB_ID=${FAILED_PREP_JOB_ID},EVAL_ATTEMPT_TAG=${EVAL_ATTEMPT_TAG}"
FINALIZE_SBATCH=${REMOTE_BUNDLE}/MemNavData/slurm_shared_online_double_revisit_finalize20.sbatch
EVAL_SBATCH=${REMOTE_BUNDLE}/MemNavData/slurm_shared_online_double_revisit_eval.sbatch
SUMMARY_SBATCH=${REMOTE_BUNDLE}/MemNavData/slurm_shared_online_double_revisit_summary.sbatch

if [[ "${EVAL_ONLY_RETRY}" == 1 ]]; then
  remote \
    "cd '${RUN_ROOT}' && sha256sum -c dependency_receipt.json.sha256 >/dev/null && cd prepared/benchmark && sha256sum -c manifest.json.sha256 >/dev/null && sbatch --test-only --array=0 --export='${exports}' '${EVAL_SBATCH}' >/dev/null"
  eval_raw=$(remote \
    "sbatch --parsable --array=0-19%${ARRAY_CONCURRENCY} --export='${exports}' '${EVAL_SBATCH}'")
  eval_id=${eval_raw%%;*}
  [[ "${eval_id}" =~ ^[0-9]+$ ]] || { echo "ABORT: bad retry eval job" >&2; exit 2; }
  remote \
    "sbatch --test-only --dependency=afterok:${eval_id} --kill-on-invalid-dep=yes --export='${exports}' '${SUMMARY_SBATCH}' >/dev/null"
  summary_raw=$(remote \
    "sbatch --parsable --dependency=afterok:${eval_id} --kill-on-invalid-dep=yes --export='${exports}' '${SUMMARY_SBATCH}'")
  summary_id=${summary_raw%%;*}
  [[ "${summary_id}" =~ ^[0-9]+$ ]] || { echo "ABORT: bad retry summary job" >&2; exit 2; }
  remote \
    "'${REMOTE_MEMNAV_PY}' - '${RUN_ROOT}/submission_fresh20_${EVAL_ATTEMPT_TAG}.json' '${REMOTE_BUNDLE}' '${SOURCE_RECEIPT_SHA}' '${eval_id}' '${summary_id}' '${ARRAY_CONCURRENCY}' '${EVAL_ATTEMPT_TAG}' '${PRIOR_FAILED_EVAL_JOB_ID}' '${PRIOR_FAILURE_REASON}' '${PRIOR_ATTEMPT_ARCHIVE}'" <<'PY'
import hashlib,json,sys
from pathlib import Path
path,bundle,receipt,evaluation,summary,concurrency,attempt,failed_job,reason,archive=sys.argv[1:]
failed_job=int(failed_job)
logs=[]
for suffix in ("out","err"):
    for index in range(20):
        candidate=Path(f"/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/drev_eval_{failed_job}_{index}.{suffix}")
        if candidate.is_file():
            logs.append({"path":str(candidate),"sha256":hashlib.sha256(candidate.read_bytes()).hexdigest()})
with open(path,"x",encoding="utf-8") as handle:
    json.dump({
      "schema_version":"shared_online_double_revisit_fresh20_eval_retry_v1",
      "attempt":attempt,
      "reason":reason,
      "prior_failed_evaluation_array":failed_job,
      "prior_failure_logs":logs,
      "prior_navigation_outcomes_created":False,
      "prior_attempt_archive":archive,
      "source_bundle":bundle,"source_receipt_sha256":receipt,
      "array_concurrency":int(concurrency),
      "jobs":{"evaluation_array":int(evaluation),"summary":int(summary)},
    },handle,indent=2,sort_keys=True); handle.write("\n")
PY
  echo "RUN_ROOT=${RUN_ROOT}"
  echo "SOURCE_BUNDLE=${REMOTE_BUNDLE}"
  echo "SOURCE_RECEIPT_SHA=${SOURCE_RECEIPT_SHA}"
  echo "retry_evaluation=${eval_id} retry_summary=${summary_id}"
  exit 0
fi

remote \
  "cd '${RUN_ROOT}' && sha256sum -c dependency_receipt.json.sha256 >/dev/null && sbatch --test-only --export='${exports}' '${FINALIZE_SBATCH}' >/dev/null"
finalize_raw=$(remote \
  "sbatch --parsable --export='${exports}' '${FINALIZE_SBATCH}'")
finalize_id=${finalize_raw%%;*}
[[ "${finalize_id}" =~ ^[0-9]+$ ]] || { echo "ABORT: bad finalizer job" >&2; exit 2; }

remote \
  "sbatch --test-only --array=0 --dependency=afterok:${finalize_id} --kill-on-invalid-dep=yes --export='${exports}' '${EVAL_SBATCH}' >/dev/null"
eval_raw=$(remote \
  "sbatch --parsable --array=0-19%${ARRAY_CONCURRENCY} --dependency=afterok:${finalize_id} --kill-on-invalid-dep=yes --export='${exports}' '${EVAL_SBATCH}'")
eval_id=${eval_raw%%;*}
[[ "${eval_id}" =~ ^[0-9]+$ ]] || { echo "ABORT: bad eval job" >&2; exit 2; }

remote \
  "sbatch --test-only --dependency=afterok:${eval_id} --kill-on-invalid-dep=yes --export='${exports}' '${SUMMARY_SBATCH}' >/dev/null"
summary_raw=$(remote \
  "sbatch --parsable --dependency=afterok:${eval_id} --kill-on-invalid-dep=yes --export='${exports}' '${SUMMARY_SBATCH}'")
summary_id=${summary_raw%%;*}
[[ "${summary_id}" =~ ^[0-9]+$ ]] || { echo "ABORT: bad summary job" >&2; exit 2; }

remote \
  "'${REMOTE_MEMNAV_PY}' - '${RUN_ROOT}/submission_fresh20.json' '${REMOTE_BUNDLE}' '${SOURCE_RECEIPT_SHA}' '${finalize_id}' '${eval_id}' '${summary_id}' '${ARRAY_CONCURRENCY}' '${FAILED_PREP_JOB_ID}'" <<'PY'
import json,sys
path,bundle,receipt,finalizer,evaluation,summary,concurrency,failed=sys.argv[1:]
with open(path,"x",encoding="utf-8") as handle:
    json.dump({
      "schema_version":"shared_online_double_revisit_fresh20_submission_v1",
      "scope":"feasibility-limited internal gate; formal fresh40 power target failed",
      "source_bundle":bundle,
      "source_receipt_sha256":receipt,
      "preregistered_power_target":40,
      "strict_constructible_population":20,
      "formal_power_target_met":False,
      "array_concurrency":int(concurrency),
      "failed_preparation_job":int(failed),
      "jobs":{"finalization":int(finalizer),"evaluation_array":int(evaluation),"summary":int(summary)},
    },handle,indent=2,sort_keys=True)
    handle.write("\n")
PY

echo "RUN_ROOT=${RUN_ROOT}"
echo "SOURCE_BUNDLE=${REMOTE_BUNDLE}"
echo "SOURCE_RECEIPT_SHA=${SOURCE_RECEIPT_SHA}"
echo "finalization=${finalize_id} evaluation=${eval_id} summary=${summary_id}"
