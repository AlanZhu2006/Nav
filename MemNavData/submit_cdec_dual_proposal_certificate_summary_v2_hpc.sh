#!/usr/bin/env bash
# Submit the safety-first geometry->CDEC reject-only audit against an existing
# immutable same-process dual collector.  This does not rerun the GPU collector.
set -euo pipefail
umask 0022

LOCAL_ROOT=$(git rev-parse --show-toplevel)
REMOTE_HOST=${REMOTE_HOST:-alantorch}
REMOTE_BUNDLE_BASE=${REMOTE_BUNDLE_BASE:-/scratch/yz11502/Research/Nav-axis-uturn-source-bundles}
RUN_ROOT=${RUN_ROOT:-/scratch/yz11502/Research/Nav-axis-uturn-results/cdec_dual_proposal_certificate_20260813/cdec_dual_pnp_pycachefix_20260813}
UPSTREAM_COLLECTOR_JOB=${UPSTREAM_COLLECTOR_JOB:-15670080}
REPORT_NAME=${REPORT_NAME:-report_safety_first_v2.json}
MEMNAV_PY=${MEMNAV_PY:-/home/asus/miniconda3/envs/memnav/bin/python}
DRY_RUN=${DRY_RUN:-0}
SUBMISSION_PATH=${RUN_ROOT}/submission_summary_v2.json
SUBMISSION_LOCK=${SUBMISSION_PATH}.lock
remote() {
  ssh -o BatchMode=yes "${REMOTE_HOST}" "$@"
}

[[ "${UPSTREAM_COLLECTOR_JOB}" =~ ^[0-9]+$ ]] || {
  echo "ABORT: invalid upstream collector job" >&2; exit 2; }
[[ "${REPORT_NAME}" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*\.json$ ]] || {
  echo "ABORT: invalid report name" >&2; exit 2; }
if [[ "${DRY_RUN}" == 0 ]]; then
  remote \
    "test ! -e '${SUBMISSION_PATH}' && mkdir '${SUBMISSION_LOCK}'"
fi

files=(
  MemNavData/summarize_cdec_dual_proposal_certificate.py
  MemNavData/summarize_lingbot_lightglue_localization.py
  MemNavData/test_summarize_cdec_dual_proposal_certificate.py
  MemNavData/test_summarize_lingbot_lightglue_localization.py
  MemNavData/slurm_cdec_dual_proposal_certificate_summary.sbatch
)
for relative in "${files[@]}"; do
  [[ -f "${LOCAL_ROOT}/${relative}" && ! -L "${LOCAL_ROOT}/${relative}" ]] || {
    echo "ABORT: missing source ${relative}" >&2; exit 2; }
done

export PYTHONPATH=${LOCAL_ROOT}:${LOCAL_ROOT}/MemNavData${PYTHONPATH:+:${PYTHONPATH}}
"${MEMNAV_PY}" -m pytest -q \
  MemNavData/test_summarize_cdec_dual_proposal_certificate.py \
  MemNavData/test_summarize_lingbot_lightglue_localization.py
bash -n "${LOCAL_ROOT}/MemNavData/slurm_cdec_dual_proposal_certificate_summary.sbatch"

staging=$(mktemp -d)
trap 'test ! -d "${staging}" || find "${staging}" -depth -delete' EXIT
mkdir -p "${staging}/MemNavData"
for relative in "${files[@]}"; do
  cp --preserve=mode,timestamps "${LOCAL_ROOT}/${relative}" "${staging}/${relative}"
done
local_head=$(git -C "${LOCAL_ROOT}" rev-parse HEAD)
"${MEMNAV_PY}" - "${staging}" "${local_head}" "${UPSTREAM_COLLECTOR_JOB}" <<'PY'
import hashlib,json,sys
from pathlib import Path
root=Path(sys.argv[1]); files={}
for path in sorted(root.rglob("*")):
    if path.is_symlink(): raise SystemExit(f"bundle symlink: {path}")
    if path.is_file() and path.name not in {"source_bundle_manifest.json","SOURCE_BUNDLE.sha256"}:
        files[path.relative_to(root).as_posix()]=hashlib.sha256(path.read_bytes()).hexdigest()
payload={
 "schema_version":"cdec_dual_proposal_safety_first_summary_bundle_v1",
 "local_git_head_context":sys.argv[2],
 "upstream_collector_job":int(sys.argv[3]),
 "deployment_order":"geometry_first_then_cdec_on_reject",
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
remote_bundle=${REMOTE_BUNDLE_BASE}/cdec_dual_proposal_summary_v2_${manifest_sha:0:16}
remote_staging=${remote_bundle}.partial-${UPSTREAM_COLLECTOR_JOB}

if [[ "${DRY_RUN}" == 1 ]]; then
  echo "DRY_RUN_SOURCE_RECEIPT_SHA=${receipt_sha}"
  echo "DRY_RUN_REMOTE_BUNDLE=${remote_bundle}"
  echo "DRY_RUN_RUN_ROOT=${RUN_ROOT}"
  exit 0
fi

remote \
  "test -f '${RUN_ROOT}/submission.json' && test ! -e '${RUN_ROOT}/${REPORT_NAME}' && test ! -e '${SUBMISSION_PATH}' && test -d '${SUBMISSION_LOCK}'"
if remote \
    "test -d '${remote_bundle}' && test \"\$(sha256sum '${remote_bundle}/SOURCE_BUNDLE.sha256' | awk '{print \$1}')\" = '${receipt_sha}' && cd '${remote_bundle}' && sha256sum -c SOURCE_BUNDLE.sha256 >/dev/null"; then
  echo "Reusing verified bundle ${remote_bundle}"
else
  remote \
    "test ! -e '${remote_bundle}' && mkdir -p '${remote_staging}'"
  rsync -a --chmod=Du=rwx,Dgo=rx,Fu=rw,Fgo=r \
    "${staging}/" "${REMOTE_HOST}:${remote_staging}/"
  remote \
    "test ! -e '${remote_bundle}' && cd '${remote_staging}' && sha256sum -c SOURCE_BUNDLE.sha256 >/dev/null && chmod -R a-w '${remote_staging}' && mv '${remote_staging}' '${remote_bundle}'"
fi

source_receipt=${remote_bundle}/SOURCE_BUNDLE.sha256
summary_sbatch=${remote_bundle}/MemNavData/slurm_cdec_dual_proposal_certificate_summary.sbatch
exports="ALL,SOURCE_ROOT=${remote_bundle},SOURCE_RECEIPT=${source_receipt},EXPECTED_SOURCE_RECEIPT_SHA=${receipt_sha},RUN_ROOT=${RUN_ROOT},REPORT_NAME=${REPORT_NAME}"
remote \
  "sbatch --test-only --dependency=afterok:${UPSTREAM_COLLECTOR_JOB} --kill-on-invalid-dep=yes --export='${exports}' '${summary_sbatch}' >/dev/null"
raw=$(remote \
  "sbatch --parsable --dependency=afterok:${UPSTREAM_COLLECTOR_JOB} --kill-on-invalid-dep=yes --export='${exports}' '${summary_sbatch}'")
job=${raw%%;*}
[[ "${job}" =~ ^[0-9]+$ ]] || { echo "ABORT: bad summary job" >&2; exit 2; }
remote \
  "/scratch/lg154/conda-envs/memnav/bin/python - '${RUN_ROOT}/submission_summary_v2.json' '${remote_bundle}' '${receipt_sha}' '${UPSTREAM_COLLECTOR_JOB}' '${job}' '${REPORT_NAME}'" <<'PY'
import json,sys
path,bundle,receipt,collector,summary,report=sys.argv[1:]
with open(path,"x",encoding="utf-8") as handle:
    json.dump({
      "schema_version":"cdec_dual_proposal_safety_first_summary_submission_v1",
      "source_bundle":bundle,"source_receipt_sha256":receipt,
      "deployment_order":"geometry_first_then_cdec_on_reject",
      "upstream_collector_job":int(collector),"summary_job":int(summary),
      "report_name":report,
    },handle,indent=2,sort_keys=True); handle.write("\n")
PY
echo "summary_v2=${job}"
echo "SOURCE_BUNDLE=${remote_bundle}"
echo "SOURCE_RECEIPT_SHA=${receipt_sha}"
