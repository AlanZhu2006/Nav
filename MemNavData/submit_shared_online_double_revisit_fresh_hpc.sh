#!/usr/bin/env bash
# Stage an immutable task overlay and submit prepare -> eval[40] -> audit.
set -euo pipefail
umask 0022

LOCAL_ROOT=$(git rev-parse --show-toplevel)
REMOTE_HOST=${REMOTE_HOST:-alantorch}
REMOTE_BUNDLE_BASE=${REMOTE_BUNDLE_BASE:-/scratch/yz11502/Research/Nav-axis-uturn-source-bundles}
RESULT_BASE=${RESULT_BASE:-/scratch/yz11502/Research/Nav-axis-uturn-results/shared_online_double_revisit_fresh_20260813}
UPSTREAM_RUN_ROOT=${UPSTREAM_RUN_ROOT:-/scratch/yz11502/Research/Nav-axis-uturn-results/certified_relocalization_closed_loop_20260812/certrel_bearing_v1_20260812T1050}
EXPECTED_UPSTREAM_MANIFEST_SHA=8013fa2a768d84638a9f9ecc50df46dda67ebb79250d14ad0a8087ac52fd33e5
BASE_SOURCE_ROOT=${BASE_SOURCE_ROOT:-/scratch/yz11502/Research/Nav-axis-uturn-source-bundles/certified_relocalization_closed_loop_d3bd281fc374cc80}
BASE_SOURCE_RECEIPT_SHA=74001a9e0150c38c599a206fa0f4dd5e1279b9bed5d167119f4d14cb77995e98
PRIOR_DEPENDENCY_ROOT=${PRIOR_DEPENDENCY_ROOT:-/scratch/yz11502/Research/Nav-axis-uturn-results/certified_relocalization_closed_loop_20260812/certrel_bearing_v1_20260812T1050}
MEMNAV_PY=${MEMNAV_PY:-/home/asus/miniconda3/envs/memnav/bin/python}
HAB_PY=${HAB_PY:-/home/asus/miniconda3/envs/habitat/bin/python}
REMOTE_MEMNAV_PY=${REMOTE_MEMNAV_PY:-/scratch/lg154/conda-envs/memnav/bin/python}
RUN_TAG=${RUN_TAG:-double_revisit_fresh40_$(date -u +%Y%m%dT%H%M%SZ)}
ARRAY_CONCURRENCY=${ARRAY_CONCURRENCY:-4}
DRY_RUN=${DRY_RUN:-0}

[[ "${RUN_TAG}" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]] || {
  echo "ABORT: invalid RUN_TAG" >&2; exit 2; }
[[ "${ARRAY_CONCURRENCY}" =~ ^[1-9][0-9]*$ ]] || {
  echo "ABORT: ARRAY_CONCURRENCY must be positive" >&2; exit 2; }

required=(
  MemNavData/prepare_shared_online_double_revisit_fresh.py
  MemNavData/audit_shared_online_double_revisit_fresh.py
  MemNavData/run_shared_online_double_revisit_fresh_episode.sh
  MemNavData/slurm_shared_online_double_revisit_prepare.sbatch
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
  "${LOCAL_ROOT}/MemNavData/prepare_shared_online_double_revisit_fresh.py" \
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
  "${LOCAL_ROOT}/MemNavData/slurm_shared_online_double_revisit_prepare.sbatch" \
  "${LOCAL_ROOT}/MemNavData/slurm_shared_online_double_revisit_eval.sbatch" \
  "${LOCAL_ROOT}/MemNavData/slurm_shared_online_double_revisit_summary.sbatch"

STAGING=$(mktemp -d)
trap 'rm -rf -- "${STAGING}"' EXIT
mkdir -p "${STAGING}/MemNavData" \
  "${STAGING}/NavDP/baselines/memnav" \
  "${STAGING}/NavDP/baselines/navdp"

# The task overlay is intentionally small.  InternNav, pinned LightGlue/kornia
# and torch assets remain in the separately immutable, already audited base.
while IFS= read -r -d '' path; do
  cp --preserve=mode,timestamps "${path}" \
    "${STAGING}/MemNavData/$(basename "${path}")"
done < <(find "${LOCAL_ROOT}/MemNavData" -maxdepth 1 -type f -name '*.py' -print0)
for relative in \
  MemNavData/run_shared_online_double_revisit_fresh_episode.sh \
  MemNavData/slurm_shared_online_double_revisit_prepare.sbatch \
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
root=Path(sys.argv[1])
files={}
for path in sorted(root.rglob("*")):
    if path.is_symlink():
        raise SystemExit(f"task bundle contains symlink: {path}")
    if path.is_file() and path.name not in {
        "source_bundle_manifest.json","SOURCE_BUNDLE.sha256"}:
        files[path.relative_to(root).as_posix()]=hashlib.sha256(
            path.read_bytes()).hexdigest()
payload={
    "schema_version":"shared_online_double_revisit_fresh_task_bundle_v1",
    "local_git_head_context":sys.argv[2],
    "base_source_root":sys.argv[3],
    "base_source_receipt_sha256":sys.argv[4],
    "benchmark_target_episodes":40,
    "navigation_arms":["full_memory","memory_b_native_c","certified","native"],
    "files":files,
}
(root/"source_bundle_manifest.json").write_text(
    json.dumps(payload,indent=2,sort_keys=True)+"\n")
PY
(
  cd "${STAGING}"
  find . -type f ! -name SOURCE_BUNDLE.sha256 -print0 | sort -z | \
    xargs -0 sha256sum > SOURCE_BUNDLE.sha256
  sha256sum -c SOURCE_BUNDLE.sha256 >/dev/null
)

SOURCE_RECEIPT_SHA=$(sha256sum "${STAGING}/SOURCE_BUNDLE.sha256" | awk '{print $1}')
BUNDLE_MANIFEST_SHA=$(sha256sum "${STAGING}/source_bundle_manifest.json" | awk '{print $1}')
BUNDLE_TAG=${BUNDLE_MANIFEST_SHA:0:16}
REMOTE_BUNDLE=${REMOTE_BUNDLE_BASE}/shared_online_double_revisit_fresh_${BUNDLE_TAG}
REMOTE_STAGING=${REMOTE_BUNDLE}.partial-${RUN_TAG}
RUN_ROOT=${RESULT_BASE}/${RUN_TAG}

if [[ "${DRY_RUN}" == 1 ]]; then
  echo "DRY_RUN_SOURCE_RECEIPT_SHA=${SOURCE_RECEIPT_SHA}"
  echo "DRY_RUN_BUNDLE_MANIFEST_SHA=${BUNDLE_MANIFEST_SHA}"
  echo "DRY_RUN_BUNDLE_FILES=$(find "${STAGING}" -type f | wc -l)"
  echo "DRY_RUN_BUNDLE_BYTES=$(du -sb "${STAGING}" | awk '{print $1}')"
  echo "DRY_RUN_REMOTE_BUNDLE=${REMOTE_BUNDLE}"
  echo "DRY_RUN_RUN_ROOT=${RUN_ROOT}"
  exit 0
fi

# Verify the immutable base once at submission; tasks subsequently bind its
# receipt SHA and avoid re-reading the full 295 MB payload forty times.
ssh -o BatchMode=yes "${REMOTE_HOST}" \
  "test \"\$(sha256sum '${BASE_SOURCE_ROOT}/SOURCE_BUNDLE.sha256' | awk '{print \$1}')\" = '${BASE_SOURCE_RECEIPT_SHA}' && cd '${BASE_SOURCE_ROOT}' && sha256sum -c SOURCE_BUNDLE.sha256 >/dev/null"
ssh -o BatchMode=yes "${REMOTE_HOST}" \
  "test \"\$(sha256sum '${UPSTREAM_RUN_ROOT}/data_manifest.json' | awk '{print \$1}')\" = '${EXPECTED_UPSTREAM_MANIFEST_SHA}'"

if ssh -o BatchMode=yes "${REMOTE_HOST}" \
    "test -d '${REMOTE_BUNDLE}' && test \"\$(sha256sum '${REMOTE_BUNDLE}/SOURCE_BUNDLE.sha256' | awk '{print \$1}')\" = '${SOURCE_RECEIPT_SHA}' && cd '${REMOTE_BUNDLE}' && sha256sum -c SOURCE_BUNDLE.sha256 >/dev/null"; then
  echo "Reusing verified task bundle ${REMOTE_BUNDLE}"
else
  ssh -o BatchMode=yes "${REMOTE_HOST}" \
    "test ! -e '${REMOTE_BUNDLE}' && mkdir -p '${REMOTE_STAGING}'"
  rsync -a --chmod=Du=rwx,Dgo=rx,Fu=rw,Fgo=r \
    "${STAGING}/" "${REMOTE_HOST}:${REMOTE_STAGING}/"
  ssh -o BatchMode=yes "${REMOTE_HOST}" \
    "test ! -e '${REMOTE_BUNDLE}' && cd '${REMOTE_STAGING}' && sha256sum -c SOURCE_BUNDLE.sha256 >/dev/null && chmod -R a-w '${REMOTE_STAGING}' && mv '${REMOTE_STAGING}' '${REMOTE_BUNDLE}'"
fi

REMOTE_SOURCE_RECEIPT=${REMOTE_BUNDLE}/SOURCE_BUNDLE.sha256
ssh -o BatchMode=yes "${REMOTE_HOST}" \
  "test ! -e '${RUN_ROOT}' && mkdir -p '${RUN_ROOT}' && cp '${PRIOR_DEPENDENCY_ROOT}/dependency_receipt.json' '${RUN_ROOT}/dependency_receipt.json' && cp '${PRIOR_DEPENDENCY_ROOT}/dependency_receipt.json.sha256' '${RUN_ROOT}/dependency_receipt.json.sha256' && cd '${RUN_ROOT}' && sha256sum -c dependency_receipt.json.sha256 >/dev/null"

exports="ALL,SOURCE_ROOT=${REMOTE_BUNDLE},SOURCE_RECEIPT=${REMOTE_SOURCE_RECEIPT},EXPECTED_SOURCE_RECEIPT_SHA=${SOURCE_RECEIPT_SHA},BASE_SOURCE_ROOT=${BASE_SOURCE_ROOT},BASE_SOURCE_RECEIPT_SHA=${BASE_SOURCE_RECEIPT_SHA},RUN_ROOT=${RUN_ROOT},UPSTREAM_RUN_ROOT=${UPSTREAM_RUN_ROOT},EXPECTED_UPSTREAM_MANIFEST_SHA=${EXPECTED_UPSTREAM_MANIFEST_SHA}"
PREP_SBATCH=${REMOTE_BUNDLE}/MemNavData/slurm_shared_online_double_revisit_prepare.sbatch
EVAL_SBATCH=${REMOTE_BUNDLE}/MemNavData/slurm_shared_online_double_revisit_eval.sbatch
SUMMARY_SBATCH=${REMOTE_BUNDLE}/MemNavData/slurm_shared_online_double_revisit_summary.sbatch
ssh -o BatchMode=yes "${REMOTE_HOST}" \
  "mkdir -p /scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs && sbatch --test-only --export='${exports}' '${PREP_SBATCH}' >/dev/null"
prep_raw=$(ssh -o BatchMode=yes "${REMOTE_HOST}" \
  "sbatch --parsable --export='${exports}' '${PREP_SBATCH}'")
prep_id=${prep_raw%%;*}
[[ "${prep_id}" =~ ^[0-9]+$ ]] || { echo "ABORT: bad prep job ${prep_raw}" >&2; exit 2; }

ssh -o BatchMode=yes "${REMOTE_HOST}" \
  "sbatch --test-only --array=0 --dependency=afterok:${prep_id} --kill-on-invalid-dep=yes --export='${exports}' '${EVAL_SBATCH}' >/dev/null"
eval_raw=$(ssh -o BatchMode=yes "${REMOTE_HOST}" \
  "sbatch --parsable --array=0-39%${ARRAY_CONCURRENCY} --dependency=afterok:${prep_id} --kill-on-invalid-dep=yes --export='${exports}' '${EVAL_SBATCH}'")
eval_id=${eval_raw%%;*}
[[ "${eval_id}" =~ ^[0-9]+$ ]] || { echo "ABORT: bad eval job ${eval_raw}" >&2; exit 2; }

ssh -o BatchMode=yes "${REMOTE_HOST}" \
  "sbatch --test-only --dependency=afterok:${eval_id} --kill-on-invalid-dep=yes --export='${exports}' '${SUMMARY_SBATCH}' >/dev/null"
summary_raw=$(ssh -o BatchMode=yes "${REMOTE_HOST}" \
  "sbatch --parsable --dependency=afterok:${eval_id} --kill-on-invalid-dep=yes --export='${exports}' '${SUMMARY_SBATCH}'")
summary_id=${summary_raw%%;*}
[[ "${summary_id}" =~ ^[0-9]+$ ]] || { echo "ABORT: bad summary job ${summary_raw}" >&2; exit 2; }

ssh -o BatchMode=yes "${REMOTE_HOST}" \
  "'${REMOTE_MEMNAV_PY}' - '${RUN_ROOT}/submission.json' '${RUN_TAG}' '${REMOTE_BUNDLE}' '${SOURCE_RECEIPT_SHA}' '${BASE_SOURCE_ROOT}' '${BASE_SOURCE_RECEIPT_SHA}' '${prep_id}' '${eval_id}' '${summary_id}' '${ARRAY_CONCURRENCY}'" <<'PY'
import json,sys
(path,tag,bundle,receipt,base,base_receipt,prep,evaluation,summary,concurrency)=sys.argv[1:]
with open(path,"x",encoding="utf-8") as handle:
    json.dump({
        "schema_version":"shared_online_double_revisit_fresh_submission_v1",
        "run_tag":tag,
        "source_bundle":bundle,
        "source_receipt_sha256":receipt,
        "base_source_bundle":base,
        "base_source_receipt_sha256":base_receipt,
        "array_concurrency":int(concurrency),
        "jobs":{
            "preparation":int(prep),
            "evaluation_array":int(evaluation),
            "summary":int(summary),
        },
    },handle,indent=2,sort_keys=True)
    handle.write("\n")
PY

echo "RUN_ROOT=${RUN_ROOT}"
echo "SOURCE_BUNDLE=${REMOTE_BUNDLE}"
echo "SOURCE_RECEIPT_SHA=${SOURCE_RECEIPT_SHA}"
echo "preparation=${prep_id} evaluation=${eval_id} summary=${summary_id}"
