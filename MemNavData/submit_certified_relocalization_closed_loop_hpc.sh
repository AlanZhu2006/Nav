#!/usr/bin/env bash
# Build an immutable source/dependency bundle and submit the 20-scene array.
set -euo pipefail
umask 0022

LOCAL_ROOT=$(git rev-parse --show-toplevel)
REMOTE_HOST=${REMOTE_HOST:-alantorch}
REMOTE_BUNDLE_BASE=${REMOTE_BUNDLE_BASE:-/scratch/yz11502/Research/Nav-axis-uturn-source-bundles}
RESULT_BASE=${RESULT_BASE:-/scratch/yz11502/Research/Nav-axis-uturn-results/certified_relocalization_closed_loop_20260812}
UPSTREAM_RUN_ROOT=${UPSTREAM_RUN_ROOT:-/scratch/yz11502/Research/Nav-axis-uturn-results/revisit_fresh_confirmation_20260811/fresh160_v3_attempt600_20260811T2000}
EXPECTED_UPSTREAM_MANIFEST_SHA=8013fa2a768d84638a9f9ecc50df46dda67ebb79250d14ad0a8087ac52fd33e5
MEMNAV_PY=${MEMNAV_PY:-/home/asus/miniconda3/envs/memnav/bin/python}
REMOTE_MEMNAV_PY=${REMOTE_MEMNAV_PY:-/scratch/lg154/conda-envs/memnav/bin/python}
LIGHTGLUE_SOURCE=${LIGHTGLUE_SOURCE:-${LOCAL_ROOT}/.diagnostics/dependencies/LightGlue}
DEPENDENCY_SOURCE=${DEPENDENCY_SOURCE:-${LOCAL_ROOT}/.diagnostics/dependencies/python}
TORCH_CHECKPOINT_SOURCE=${TORCH_CHECKPOINT_SOURCE:-/home/asus/.cache/torch/hub/checkpoints}
RUN_TAG=${RUN_TAG:-certrel_bearing_v1_$(date -u +%Y%m%dT%H%M%SZ)}
ARRAY_CONCURRENCY=${ARRAY_CONCURRENCY:-4}
DRY_RUN=${DRY_RUN:-0}

[[ "${RUN_TAG}" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]] || {
  echo "ABORT: invalid RUN_TAG" >&2; exit 2; }
[[ "${ARRAY_CONCURRENCY}" =~ ^[1-9][0-9]*$ ]] || {
  echo "ABORT: ARRAY_CONCURRENCY must be positive" >&2; exit 2; }

required_local=(
  "${LIGHTGLUE_SOURCE}/lightglue"
  "${LIGHTGLUE_SOURCE}/LICENSE"
  "${DEPENDENCY_SOURCE}/kornia"
  "${DEPENDENCY_SOURCE}/kornia_rs"
  "${TORCH_CHECKPOINT_SOURCE}/superpoint_v1.pth"
  "${TORCH_CHECKPOINT_SOURCE}/superpoint_lightglue_v0-1_arxiv.pth"
  "${LOCAL_ROOT}/MemNavData/run_certified_relocalization_closed_loop_scene.sh"
  "${LOCAL_ROOT}/MemNavData/slurm_certified_relocalization_closed_loop.sbatch"
  "${LOCAL_ROOT}/MemNavData/slurm_certified_relocalization_closed_loop_summary.sbatch"
  "${LOCAL_ROOT}/MemNavData/summarize_certified_relocalization_closed_loop.py"
  "${LOCAL_ROOT}/MemNavData/prepare_certified_relocalization_closed_loop.py"
)
for path in "${required_local[@]}"; do
  [[ -e "${path}" && ! -L "${path}" ]] || {
    echo "ABORT: missing physical local input ${path}" >&2; exit 2; }
done

"${MEMNAV_PY}" -m py_compile \
  "${LOCAL_ROOT}/MemNavData/certified_relocalization_runtime.py" \
  "${LOCAL_ROOT}/MemNavData/revisit_bearing_adapter.py" \
  "${LOCAL_ROOT}/MemNavData/lingbot_pnp_localization.py" \
  "${LOCAL_ROOT}/MemNavData/eval_2leg_habitat.py" \
  "${LOCAL_ROOT}/MemNavData/summarize_certified_relocalization_closed_loop.py" \
  "${LOCAL_ROOT}/MemNavData/prepare_certified_relocalization_closed_loop.py" \
  "${LOCAL_ROOT}/NavDP/baselines/memnav/memnav_server.py" \
  "${LOCAL_ROOT}/NavDP/baselines/memnav/policy_agent.py"
(
  cd "${LOCAL_ROOT}"
  "${MEMNAV_PY}" -m pytest -q \
    MemNavData/test_certified_relocalization_runtime.py \
    MemNavData/test_revisit_bearing_adapter.py \
    MemNavData/test_lingbot_pnp_localization.py \
    MemNavData/test_lingbot_goal_loop_closure.py \
    MemNavData/test_prepare_certified_relocalization_closed_loop.py \
    MemNavData/test_summarize_certified_relocalization_closed_loop.py \
    MemNavData/test_policy_agent_graph.py \
    MemNavData/test_router_candidates.py
)
bash -n \
  "${LOCAL_ROOT}/MemNavData/run_certified_relocalization_closed_loop_scene.sh" \
  "${LOCAL_ROOT}/MemNavData/slurm_certified_relocalization_closed_loop.sbatch" \
  "${LOCAL_ROOT}/MemNavData/slurm_certified_relocalization_closed_loop_summary.sbatch"

STAGING=$(mktemp -d)
trap 'rm -rf -- "${STAGING}"' EXIT

# Copy the current tracked working tree, including intentional dirty changes.
# The one broken historical LongCLIP symlink is unused by these entrypoints and
# is excluded so the immutable bundle contains only physical files.
while IFS= read -r -d '' relative; do
  [[ "${relative}" == "InternNav/internnav/model/basemodel/LongCLIP" ]] && continue
  source_path=${LOCAL_ROOT}/${relative}
  [[ -f "${source_path}" && ! -L "${source_path}" ]] || continue
  mkdir -p "${STAGING}/$(dirname "${relative}")"
  cp --preserve=mode,timestamps "${source_path}" "${STAGING}/${relative}"
done < <(git -C "${LOCAL_ROOT}" ls-files -z)

# Several audited research modules are intentionally untracked in this dirty
# workspace. Include every source/protocol file under MemNavData, but no logs,
# checkpoints, images, caches, or generated rollouts.
while IFS= read -r -d '' relative; do
  [[ "${relative}" == \
    "MemNavData/CERTIFIED_RELOCALIZATION_CLOSED_LOOP_RUN_20260812.md" ]] && \
    continue
  case "${relative}" in
    *.py|*.sh|*.sbatch|*.json|*.md|*.txt) ;;
    *) continue ;;
  esac
  source_path=${LOCAL_ROOT}/${relative}
  [[ -f "${source_path}" && ! -L "${source_path}" ]] || continue
  mkdir -p "${STAGING}/$(dirname "${relative}")"
  cp --preserve=mode,timestamps "${source_path}" "${STAGING}/${relative}"
done < <(git -C "${LOCAL_ROOT}" ls-files --others --exclude-standard \
  -z -- MemNavData)

mkdir -p "${STAGING}/third_party/LightGlue" \
  "${STAGING}/third_party/python" \
  "${STAGING}/torch_home/hub/checkpoints"
cp -a "${LIGHTGLUE_SOURCE}/lightglue" \
  "${STAGING}/third_party/LightGlue/"
cp --preserve=mode,timestamps "${LIGHTGLUE_SOURCE}/LICENSE" \
  "${STAGING}/third_party/LightGlue/LICENSE"
for dependency in kornia kornia-0.8.1.dist-info \
                  kornia_rs kornia_rs-0.1.9.dist-info; do
  cp -a "${DEPENDENCY_SOURCE}/${dependency}" \
    "${STAGING}/third_party/python/"
done
cp --preserve=mode,timestamps \
  "${TORCH_CHECKPOINT_SOURCE}/superpoint_v1.pth" \
  "${TORCH_CHECKPOINT_SOURCE}/superpoint_lightglue_v0-1_arxiv.pth" \
  "${STAGING}/torch_home/hub/checkpoints/"

LOCAL_HEAD=$(git -C "${LOCAL_ROOT}" rev-parse HEAD)
"${MEMNAV_PY}" - "${STAGING}" "${LOCAL_HEAD}" <<'PY'
import hashlib, json, sys
from pathlib import Path
root=Path(sys.argv[1])
files={}
for path in sorted(root.rglob("*")):
    if path.is_symlink():
        raise SystemExit(f"bundle contains symlink: {path}")
    if path.is_file() and path.name not in {
        "source_bundle_manifest.json", "SOURCE_BUNDLE.sha256"}:
        files[path.relative_to(root).as_posix()]=hashlib.sha256(
            path.read_bytes()).hexdigest()
payload={
    "schema_version":"certified_relocalization_closed_loop_bundle_v1",
    "local_git_head_context":sys.argv[2],
    "runtime_schema_version":3,
    "geometry_certificate_version":2,
    "controller_adapter":"verified_bearing_v1_fixed_2.5m",
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
REMOTE_BUNDLE=${REMOTE_BUNDLE_BASE}/certified_relocalization_closed_loop_${BUNDLE_TAG}
REMOTE_BUNDLE_STAGING=${REMOTE_BUNDLE}.partial-${RUN_TAG}
RUN_ROOT=${RESULT_BASE}/${RUN_TAG}

if [[ "${DRY_RUN}" == "1" ]]; then
  echo "DRY_RUN_SOURCE_RECEIPT_SHA=${SOURCE_RECEIPT_SHA}"
  echo "DRY_RUN_BUNDLE_MANIFEST_SHA=${BUNDLE_MANIFEST_SHA}"
  echo "DRY_RUN_BUNDLE_FILES=$(find "${STAGING}" -type f | wc -l)"
  echo "DRY_RUN_BUNDLE_BYTES=$(du -sb "${STAGING}" | awk '{print $1}')"
  echo "DRY_RUN_REMOTE_BUNDLE=${REMOTE_BUNDLE}"
  echo "DRY_RUN_RUN_ROOT=${RUN_ROOT}"
  exit 0
fi

# A dropped 310 MB transfer must be resumable, while an already frozen bundle
# may only be reused after both its receipt identity and every payload hash pass.
if ssh -o BatchMode=yes "${REMOTE_HOST}" \
    "test -d '${REMOTE_BUNDLE}' && \
     test \"\$(sha256sum '${REMOTE_BUNDLE}/SOURCE_BUNDLE.sha256' | awk '{print \$1}')\" = '${SOURCE_RECEIPT_SHA}' && \
     cd '${REMOTE_BUNDLE}' && sha256sum -c SOURCE_BUNDLE.sha256 >/dev/null"; then
  echo "Reusing verified immutable source bundle ${REMOTE_BUNDLE}"
else
  ssh -o BatchMode=yes "${REMOTE_HOST}" \
    "test ! -e '${REMOTE_BUNDLE}' && mkdir -p '${REMOTE_BUNDLE_STAGING}'"
  rsync -a --chmod=Du=rwx,Dgo=rx,Fu=rw,Fgo=r \
    "${STAGING}/" "${REMOTE_HOST}:${REMOTE_BUNDLE_STAGING}/"
  ssh -o BatchMode=yes "${REMOTE_HOST}" \
    "test ! -e '${REMOTE_BUNDLE}' && \
     test \"\$(sha256sum '${REMOTE_BUNDLE_STAGING}/SOURCE_BUNDLE.sha256' | awk '{print \$1}')\" = '${SOURCE_RECEIPT_SHA}' && \
     cd '${REMOTE_BUNDLE_STAGING}' && \
     sha256sum -c SOURCE_BUNDLE.sha256 >/dev/null && \
     chmod -R a-w '${REMOTE_BUNDLE_STAGING}' && \
     mv '${REMOTE_BUNDLE_STAGING}' '${REMOTE_BUNDLE}'"
fi

REMOTE_SOURCE_RECEIPT=${REMOTE_BUNDLE}/SOURCE_BUNDLE.sha256
UPSTREAM_MANIFEST=${UPSTREAM_RUN_ROOT}/data_manifest.json
ssh -o BatchMode=yes "${REMOTE_HOST}" \
  "'${REMOTE_MEMNAV_PY}' '${REMOTE_BUNDLE}/MemNavData/prepare_certified_relocalization_closed_loop.py' \
    --source-manifest '${UPSTREAM_MANIFEST}' \
    --expected-manifest-sha '${EXPECTED_UPSTREAM_MANIFEST_SHA}' \
    --source-receipt '${REMOTE_SOURCE_RECEIPT}' \
    --expected-source-receipt-sha '${SOURCE_RECEIPT_SHA}' \
    --run-root '${RUN_ROOT}'"

exports="ALL,SOURCE_ROOT=${REMOTE_BUNDLE},RUN_ROOT=${RUN_ROOT},SOURCE_RECEIPT=${REMOTE_SOURCE_RECEIPT},EXPECTED_SOURCE_RECEIPT_SHA=${SOURCE_RECEIPT_SHA}"
EVAL_SBATCH=${REMOTE_BUNDLE}/MemNavData/slurm_certified_relocalization_closed_loop.sbatch
SUMMARY_SBATCH=${REMOTE_BUNDLE}/MemNavData/slurm_certified_relocalization_closed_loop_summary.sbatch
ssh -o BatchMode=yes "${REMOTE_HOST}" \
  "mkdir -p /scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs && \
   sbatch --test-only --array=0 --export='${exports}' '${EVAL_SBATCH}' >/dev/null"
evaluation_job=$(ssh -o BatchMode=yes "${REMOTE_HOST}" \
  "sbatch --parsable --array=0-19%${ARRAY_CONCURRENCY} --export='${exports}' '${EVAL_SBATCH}'")
evaluation_id=${evaluation_job%%;*}
[[ "${evaluation_id}" =~ ^[0-9]+$ ]] || {
  echo "ABORT: malformed evaluation job id ${evaluation_job}" >&2; exit 2; }
ssh -o BatchMode=yes "${REMOTE_HOST}" \
  "sbatch --test-only --dependency=afterok:${evaluation_id} \
     --kill-on-invalid-dep=yes --export='${exports}' '${SUMMARY_SBATCH}' >/dev/null"
summary_job=$(ssh -o BatchMode=yes "${REMOTE_HOST}" \
  "sbatch --parsable --dependency=afterok:${evaluation_id} \
     --kill-on-invalid-dep=yes --export='${exports}' '${SUMMARY_SBATCH}'")
summary_id=${summary_job%%;*}
[[ "${summary_id}" =~ ^[0-9]+$ ]] || {
  echo "ABORT: malformed summary job id ${summary_job}" >&2; exit 2; }

ssh -o BatchMode=yes "${REMOTE_HOST}" \
  "'${REMOTE_MEMNAV_PY}' - '${RUN_ROOT}/submission.json' '${RUN_TAG}' \
    '${REMOTE_BUNDLE}' '${SOURCE_RECEIPT_SHA}' '${evaluation_id}' '${summary_id}'" <<'PY'
import json,sys
path,tag,bundle,receipt,evaluation,summary=sys.argv[1:]
with open(path,"x",encoding="utf-8") as handle:
    json.dump({
        "schema_version":"certified_relocalization_closed_loop_submission_v1",
        "run_tag":tag,
        "source_bundle":bundle,
        "source_receipt_sha256":receipt,
        "jobs":{
            "evaluation_array":int(evaluation),
            "summary":int(summary),
        },
    },handle,indent=2,sort_keys=True)
    handle.write("\n")
PY

echo "RUN_ROOT=${RUN_ROOT}"
echo "SOURCE_BUNDLE=${REMOTE_BUNDLE}"
echo "SOURCE_RECEIPT_SHA=${SOURCE_RECEIPT_SHA}"
echo "evaluation=${evaluation_id} summary=${summary_id}"
