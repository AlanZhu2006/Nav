#!/usr/bin/env bash
# Stage the factorized CDEC code/artifacts and run the paired PnP teacher audit.
set -euo pipefail
umask 0022

LOCAL_ROOT=$(git rev-parse --show-toplevel)
REMOTE_HOST=${REMOTE_HOST:-alantorch}
REMOTE_BUNDLE_BASE=${REMOTE_BUNDLE_BASE:-/scratch/yz11502/Research/Nav-axis-uturn-source-bundles}
RESULT_BASE=${RESULT_BASE:-/scratch/yz11502/Research/Nav-axis-uturn-results/cdec_dual_proposal_certificate_20260813}
MEMNAV_PY=${MEMNAV_PY:-/home/asus/miniconda3/envs/memnav/bin/python}
RUN_TAG=${RUN_TAG:-cdec_dual_pnp_$(date -u +%Y%m%dT%H%M%SZ)}
DRY_RUN=${DRY_RUN:-0}
CDEC_LOCAL=${LOCAL_ROOT}/.diagnostics/certificate_distilled_compass_20260813/factorized_pairwise_oof_v1
remote() {
  ssh -o BatchMode=yes "${REMOTE_HOST}" "$@"
}

[[ "${RUN_TAG}" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]] || {
  echo "ABORT: invalid RUN_TAG" >&2; exit 2; }

FILES=(
  MemNavData/diag_lingbot_goal_loop_closure.py
  MemNavData/lingbot_colored_registration.py
  MemNavData/lingbot_pnp_localization.py
  MemNavData/external_causal_scale_contract.py
  MemNavData/phase_b_feature_schema.py
  MemNavData/flow_cache_routing.py
  MemNavData/phase_b_upstream_receipts.py
  MemNavData/summarize_lingbot_lightglue_localization.py
  MemNavData/summarize_cdec_dual_proposal_certificate.py
  MemNavData/test_lingbot_goal_loop_closure.py
  MemNavData/test_lingbot_pnp_localization.py
  MemNavData/test_summarize_lingbot_lightglue_localization.py
  MemNavData/test_summarize_cdec_dual_proposal_certificate.py
  MemNavData/slurm_lingbot_native_localizer.sbatch
  MemNavData/slurm_cdec_dual_proposal_certificate_collect.sbatch
  MemNavData/slurm_cdec_dual_proposal_certificate_summary.sbatch
)
for relative in "${FILES[@]}"; do
  [[ -f "${LOCAL_ROOT}/${relative}" && ! -L "${LOCAL_ROOT}/${relative}" ]] || {
    echo "ABORT: missing source ${relative}" >&2; exit 2; }
done
for artifact in "${CDEC_LOCAL}/cdec_oof_selection_rows.csv" \
                "${CDEC_LOCAL}/report.json"; do
  [[ -f "${artifact}" && ! -L "${artifact}" ]] || {
    echo "ABORT: missing CDEC artifact ${artifact}" >&2; exit 2; }
done
[[ "$(sha256sum "${CDEC_LOCAL}/cdec_oof_selection_rows.csv" | awk '{print $1}')" == \
   65a3f7ccc876e7fbf6857b5b4a13f8e9061e092ecc76c4273c0ffec6bf0133c3 ]] || {
  echo "ABORT: CDEC selection artifact changed" >&2; exit 2; }
[[ "$(sha256sum "${CDEC_LOCAL}/report.json" | awk '{print $1}')" == \
   88417e3b02dfd16bf94c4dbd0353372a505700bce969a4cf787bb5c3897211e6 ]] || {
  echo "ABORT: CDEC report changed" >&2; exit 2; }

export PYTHONPATH=${LOCAL_ROOT}:${LOCAL_ROOT}/MemNavData${PYTHONPATH:+:${PYTHONPATH}}
"${MEMNAV_PY}" -m py_compile \
  "${LOCAL_ROOT}/MemNavData/diag_lingbot_goal_loop_closure.py" \
  "${LOCAL_ROOT}/MemNavData/summarize_cdec_dual_proposal_certificate.py"
"${MEMNAV_PY}" -m pytest -q \
  MemNavData/test_lingbot_goal_loop_closure.py \
  MemNavData/test_lingbot_pnp_localization.py \
  MemNavData/test_summarize_lingbot_lightglue_localization.py \
  MemNavData/test_summarize_cdec_dual_proposal_certificate.py
bash -n \
  "${LOCAL_ROOT}/MemNavData/slurm_cdec_dual_proposal_certificate_collect.sbatch" \
  "${LOCAL_ROOT}/MemNavData/slurm_cdec_dual_proposal_certificate_summary.sbatch"
if grep -Eq 'SLURM_ARRAY_TASK_ID|%A|%a' \
    "${LOCAL_ROOT}/MemNavData/slurm_cdec_dual_proposal_certificate_collect.sbatch"; then
  echo "ABORT: the same-process collector must not depend on Slurm array state" >&2
  exit 2
fi

STAGING=$(mktemp -d)
trap 'rm -rf -- "${STAGING}"' EXIT
mkdir -p "${STAGING}/MemNavData" "${STAGING}/artifacts"
for relative in "${FILES[@]}"; do
  cp --preserve=mode,timestamps "${LOCAL_ROOT}/${relative}" "${STAGING}/${relative}"
done
cp --preserve=mode,timestamps \
  "${CDEC_LOCAL}/cdec_oof_selection_rows.csv" \
  "${STAGING}/artifacts/cdec_oof_selection_rows.csv"
cp --preserve=mode,timestamps \
  "${CDEC_LOCAL}/report.json" \
  "${STAGING}/artifacts/cdec_pairwise_oof_report.json"

# Test the exact staged import closure, not the full working tree.  The latter
# can hide a missing module that will only fail inside the immutable bundle.
PYTHONPATH=${STAGING} "${MEMNAV_PY}" -c \
  'from MemNavData.diag_lingbot_goal_loop_closure import select_cdec_oof_ranked_seeds'
PYTHONPATH=${STAGING} "${MEMNAV_PY}" -m pytest -q -p no:cacheprovider \
  "${STAGING}/MemNavData/test_lingbot_goal_loop_closure.py" \
  "${STAGING}/MemNavData/test_lingbot_pnp_localization.py" \
  "${STAGING}/MemNavData/test_summarize_cdec_dual_proposal_certificate.py"

LOCAL_HEAD=$(git -C "${LOCAL_ROOT}" rev-parse HEAD)
"${MEMNAV_PY}" - "${STAGING}" "${LOCAL_HEAD}" <<'PY'
import hashlib,json,sys
from pathlib import Path
root=Path(sys.argv[1]); files={}
for path in sorted(root.rglob("*")):
    if path.is_symlink(): raise SystemExit(f"bundle symlink: {path}")
    if path.is_file() and path.name not in {"source_bundle_manifest.json","SOURCE_BUNDLE.sha256"}:
        files[path.relative_to(root).as_posix()]=hashlib.sha256(path.read_bytes()).hexdigest()
payload={
 "schema_version":"cdec_dual_proposal_certificate_task_bundle_v1",
 "local_git_head_context":sys.argv[2],
 "scope":"train-only scene-OOF proposal comparison; no development/blind",
 "policies":["lightglue_fundamental_rank_v1","cdec_scene_oof_pairwise_rank_v1"],
 "execution_contract":"same_gpu_same_lingbot_process_geometry_then_cdec",
 "activation_authority":"unchanged atomic LingBot-depth PnP certificate",
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
RECEIPT_SHA=$(sha256sum "${STAGING}/SOURCE_BUNDLE.sha256" | awk '{print $1}')
MANIFEST_SHA=$(sha256sum "${STAGING}/source_bundle_manifest.json" | awk '{print $1}')
REMOTE_BUNDLE=${REMOTE_BUNDLE_BASE}/cdec_dual_proposal_certificate_${MANIFEST_SHA:0:16}
REMOTE_STAGING=${REMOTE_BUNDLE}.partial-${RUN_TAG}
RUN_ROOT=${RESULT_BASE}/${RUN_TAG}

if [[ "${DRY_RUN}" == 1 ]]; then
  echo "DRY_RUN_SOURCE_RECEIPT_SHA=${RECEIPT_SHA}"
  echo "DRY_RUN_BUNDLE_MANIFEST_SHA=${MANIFEST_SHA}"
  echo "DRY_RUN_REMOTE_BUNDLE=${REMOTE_BUNDLE}"
  echo "DRY_RUN_RUN_ROOT=${RUN_ROOT}"
  exit 0
fi

if remote \
    "test -d '${REMOTE_BUNDLE}' && test \"\$(sha256sum '${REMOTE_BUNDLE}/SOURCE_BUNDLE.sha256' | awk '{print \$1}')\" = '${RECEIPT_SHA}' && cd '${REMOTE_BUNDLE}' && sha256sum -c SOURCE_BUNDLE.sha256 >/dev/null"; then
  echo "Reusing verified bundle ${REMOTE_BUNDLE}"
else
  remote \
    "test ! -e '${REMOTE_BUNDLE}' && mkdir -p '${REMOTE_STAGING}'"
  rsync -a --chmod=Du=rwx,Dgo=rx,Fu=rw,Fgo=r \
    "${STAGING}/" "${REMOTE_HOST}:${REMOTE_STAGING}/"
  remote \
    "test ! -e '${REMOTE_BUNDLE}' && cd '${REMOTE_STAGING}' && sha256sum -c SOURCE_BUNDLE.sha256 >/dev/null && chmod -R a-w '${REMOTE_STAGING}' && mv '${REMOTE_STAGING}' '${REMOTE_BUNDLE}'"
fi

remote \
  "test ! -e '${RUN_ROOT}' && mkdir -p '${RUN_ROOT}'"
SOURCE_RECEIPT=${REMOTE_BUNDLE}/SOURCE_BUNDLE.sha256
exports="ALL,SOURCE_ROOT=${REMOTE_BUNDLE},SOURCE_RECEIPT=${SOURCE_RECEIPT},EXPECTED_SOURCE_RECEIPT_SHA=${RECEIPT_SHA},RUN_ROOT=${RUN_ROOT}"
COLLECT=${REMOTE_BUNDLE}/MemNavData/slurm_cdec_dual_proposal_certificate_collect.sbatch
SUMMARY=${REMOTE_BUNDLE}/MemNavData/slurm_cdec_dual_proposal_certificate_summary.sbatch
remote \
  "sbatch --test-only --export='${exports}' '${COLLECT}' >/dev/null"
collect_raw=$(remote \
  "sbatch --parsable --export='${exports}' '${COLLECT}'")
collect_id=${collect_raw%%;*}
[[ "${collect_id}" =~ ^[0-9]+$ ]] || { echo "ABORT: bad collector job" >&2; exit 2; }
remote \
  "sbatch --test-only --dependency=afterok:${collect_id} --kill-on-invalid-dep=yes --export='${exports}' '${SUMMARY}' >/dev/null"
summary_raw=$(remote \
  "sbatch --parsable --dependency=afterok:${collect_id} --kill-on-invalid-dep=yes --export='${exports}' '${SUMMARY}'")
summary_id=${summary_raw%%;*}
[[ "${summary_id}" =~ ^[0-9]+$ ]] || { echo "ABORT: bad summary job" >&2; exit 2; }

remote \
  "/scratch/lg154/conda-envs/memnav/bin/python - '${RUN_ROOT}/submission.json' '${REMOTE_BUNDLE}' '${RECEIPT_SHA}' '${collect_id}' '${summary_id}'" <<'PY'
import json,sys
path,bundle,receipt,collector,summary=sys.argv[1:]
with open(path,"x",encoding="utf-8") as handle:
    json.dump({
      "schema_version":"cdec_dual_proposal_certificate_submission_v2",
      "source_bundle":bundle,"source_receipt_sha256":receipt,
      "scope":"train-only paired scene-OOF proposal PnP audit",
      "execution_contract":"same_gpu_same_lingbot_process",
      "navigation_closed_loop":False,"development_or_blind_read":False,
      "jobs":{"collector":int(collector),"summary":int(summary)},
    },handle,indent=2,sort_keys=True); handle.write("\n")
PY

echo "RUN_ROOT=${RUN_ROOT}"
echo "SOURCE_BUNDLE=${REMOTE_BUNDLE}"
echo "SOURCE_RECEIPT_SHA=${RECEIPT_SHA}"
echo "collector=${collect_id} summary=${summary_id}"
