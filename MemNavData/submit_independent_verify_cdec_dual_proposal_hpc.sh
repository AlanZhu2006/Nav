#!/usr/bin/env bash
# Submit a frozen independent verifier behind the authoritative CDEC summary.
set -euo pipefail
umask 0022

LOCAL_ROOT=$(git rev-parse --show-toplevel)
REMOTE_HOST=${REMOTE_HOST:-alantorch}
REMOTE_BUNDLE_BASE=${REMOTE_BUNDLE_BASE:-/scratch/yz11502/Research/Nav-axis-uturn-source-bundles}
RUN_ROOT=${RUN_ROOT:-/scratch/yz11502/Research/Nav-axis-uturn-results/cdec_dual_proposal_certificate_20260813/cdec_dual_sameprocess_nonarrayfix_20260813}
UPSTREAM_SUMMARY_JOB=${UPSTREAM_SUMMARY_JOB:-15672692}
OFFICIAL_REPORT_NAME=${OFFICIAL_REPORT_NAME:-report_repeatability_v2.json}
ROLE_SPLIT=${ROLE_SPLIT:-/scratch/yz11502/Research/Nav-axis-uturn/MemNavData/router_multiscene_split_20260805.json}
EXPECTED_ROLE_SPLIT_SHA=97309c183e25cb3dd65472908748d55a94798a636db6157ab6fe120fca05cf7a
MEMNAV_PY=${MEMNAV_PY:-/home/asus/miniconda3/envs/memnav/bin/python}
DRY_RUN=${DRY_RUN:-0}
SUBMISSION_PATH=${RUN_ROOT}/submission_independent_verifier_v1.json
SUBMISSION_LOCK=${SUBMISSION_PATH}.lock

remote() { ssh -o BatchMode=yes "${REMOTE_HOST}" "$@"; }
[[ "${UPSTREAM_SUMMARY_JOB}" =~ ^[1-9][0-9]*$ ]] || {
  echo "ABORT: invalid upstream summary job" >&2; exit 2; }
[[ "${OFFICIAL_REPORT_NAME}" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*\.json$ ]] || {
  echo "ABORT: invalid official report name" >&2; exit 2; }

files=(
  MemNavData/independent_verify_cdec_dual_proposal_certificate.py
  MemNavData/test_independent_verify_cdec_dual_proposal_certificate.py
  MemNavData/slurm_independent_verify_cdec_dual_proposal_certificate.sbatch
)
for relative in "${files[@]}"; do
  [[ -f "${LOCAL_ROOT}/${relative}" && ! -L "${LOCAL_ROOT}/${relative}" ]] || {
    echo "ABORT: missing source ${relative}" >&2; exit 2; }
done

export PYTHONPATH=${LOCAL_ROOT}${PYTHONPATH:+:${PYTHONPATH}}
"${MEMNAV_PY}" -m unittest \
  MemNavData.test_independent_verify_cdec_dual_proposal_certificate
bash -n \
  "${LOCAL_ROOT}/MemNavData/slurm_independent_verify_cdec_dual_proposal_certificate.sbatch" \
  "${LOCAL_ROOT}/MemNavData/submit_independent_verify_cdec_dual_proposal_hpc.sh"

staging=$(mktemp -d)
trap 'test ! -d "${staging}" || find "${staging}" -depth -delete' EXIT
mkdir -p "${staging}/MemNavData"
for relative in "${files[@]}"; do
  cp --preserve=mode,timestamps "${LOCAL_ROOT}/${relative}" "${staging}/${relative}"
done
local_head=$(git -C "${LOCAL_ROOT}" rev-parse HEAD)
"${MEMNAV_PY}" - "${staging}" "${local_head}" "${UPSTREAM_SUMMARY_JOB}" <<'PY'
import hashlib,json,sys
from pathlib import Path
root=Path(sys.argv[1]); files={}
for path in sorted(root.rglob("*")):
    if path.is_symlink(): raise SystemExit(f"bundle symlink: {path}")
    if path.is_file() and path.name not in {"source_bundle_manifest.json","SOURCE_BUNDLE.sha256"}:
        files[path.relative_to(root).as_posix()]=hashlib.sha256(path.read_bytes()).hexdigest()
payload={
 "schema_version":"independent_cdec_dual_verifier_bundle_v1",
 "local_git_head_context":sys.argv[2],
 "upstream_summary_job":int(sys.argv[3]),
 "independent_of_primary_summarizer":True,
 "files":files,
}
(root/"source_bundle_manifest.json").write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n")
PY
(
  cd "${staging}"
  find . -type f ! -name SOURCE_BUNDLE.sha256 -print0 | sort -z | \
    xargs -0 sha256sum > SOURCE_BUNDLE.sha256
  sha256sum -c SOURCE_BUNDLE.sha256 >/dev/null
)
receipt_sha=$(sha256sum "${staging}/SOURCE_BUNDLE.sha256" | awk '{print $1}')
manifest_sha=$(sha256sum "${staging}/source_bundle_manifest.json" | awk '{print $1}')
remote_bundle=${REMOTE_BUNDLE_BASE}/independent_cdec_dual_verifier_${manifest_sha:0:16}
remote_staging=${remote_bundle}.partial-$$

if [[ "${DRY_RUN}" == 1 ]]; then
  echo "DRY_RUN_REMOTE_BUNDLE=${remote_bundle}"
  echo "DRY_RUN_SOURCE_RECEIPT_SHA=${receipt_sha}"
  exit 0
fi

remote "test ! -e '${SUBMISSION_PATH}' && mkdir '${SUBMISSION_LOCK}'"
if remote \
    "test -d '${remote_bundle}' && test \"\$(sha256sum '${remote_bundle}/SOURCE_BUNDLE.sha256' | awk '{print \$1}')\" = '${receipt_sha}' && cd '${remote_bundle}' && sha256sum -c SOURCE_BUNDLE.sha256 >/dev/null"; then
  echo "Reusing verified bundle ${remote_bundle}"
else
  remote "test ! -e '${remote_bundle}' && mkdir -p '${remote_staging}'"
  rsync -a --chmod=Du=rwx,Dgo=rx,Fu=rw,Fgo=r \
    "${staging}/" "${REMOTE_HOST}:${remote_staging}/"
  remote \
    "test ! -e '${remote_bundle}' && cd '${remote_staging}' && sha256sum -c SOURCE_BUNDLE.sha256 >/dev/null && chmod -R a-w '${remote_staging}' && mv '${remote_staging}' '${remote_bundle}'"
fi

source_receipt=${remote_bundle}/SOURCE_BUNDLE.sha256
sbatch_path=${remote_bundle}/MemNavData/slurm_independent_verify_cdec_dual_proposal_certificate.sbatch
exports="ALL,SOURCE_ROOT=${remote_bundle},SOURCE_RECEIPT=${source_receipt},EXPECTED_SOURCE_RECEIPT_SHA=${receipt_sha},RUN_ROOT=${RUN_ROOT},OFFICIAL_REPORT_NAME=${OFFICIAL_REPORT_NAME},ROLE_SPLIT=${ROLE_SPLIT},EXPECTED_ROLE_SPLIT_SHA=${EXPECTED_ROLE_SPLIT_SHA}"
remote \
  "sbatch --test-only --dependency=afterok:${UPSTREAM_SUMMARY_JOB} --kill-on-invalid-dep=yes --export='${exports}' '${sbatch_path}' >/dev/null"
raw=$(remote \
  "sbatch --parsable --dependency=afterok:${UPSTREAM_SUMMARY_JOB} --kill-on-invalid-dep=yes --export='${exports}' '${sbatch_path}'")
job=${raw%%;*}
[[ "${job}" =~ ^[0-9]+$ ]] || { echo "ABORT: bad verifier job" >&2; exit 2; }
remote \
  "/scratch/lg154/conda-envs/memnav/bin/python - '${SUBMISSION_PATH}' '${remote_bundle}' '${receipt_sha}' '${UPSTREAM_SUMMARY_JOB}' '${job}' '${OFFICIAL_REPORT_NAME}' '${ROLE_SPLIT}' '${EXPECTED_ROLE_SPLIT_SHA}'" <<'PY'
import json,sys
path,bundle,receipt,upstream,job,report,split,split_sha=sys.argv[1:]
with open(path,"x",encoding="utf-8") as handle:
    json.dump({
      "schema_version":"independent_cdec_dual_verifier_submission_v1",
      "source_bundle":bundle,"source_receipt_sha256":receipt,
      "upstream_summary_job":int(upstream),"verifier_job":int(job),
      "official_report_name":report,"role_split":split,
      "role_split_sha256":split_sha,
    },handle,indent=2,sort_keys=True); handle.write("\n")
PY
remote "rmdir '${SUBMISSION_LOCK}'"
echo "independent_verifier=${job}"
echo "SOURCE_BUNDLE=${remote_bundle}"
echo "SOURCE_RECEIPT_SHA=${receipt_sha}"
