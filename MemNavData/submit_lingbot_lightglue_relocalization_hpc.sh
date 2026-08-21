#!/usr/bin/env bash
# Build an immutable LightGlue+LingBot relocalization bundle and submit it.
set -euo pipefail
umask 0022

LOCAL_ROOT="$(git rev-parse --show-toplevel)"
REMOTE_HOST="${REMOTE_HOST:-alantorch}"
REMOTE_BUNDLE_BASE="${REMOTE_BUNDLE_BASE:-/scratch/yz11502/Research/Nav-axis-uturn-source-bundles}"
MEMNAV_PY="${MEMNAV_PY:-/home/asus/miniconda3/envs/memnav/bin/python}"
DRY_RUN="${DRY_RUN:-0}"
RUN_SCOPE="${RUN_SCOPE:-confirm24}"
[[ "${RUN_SCOPE}" == "confirm24" || "${RUN_SCOPE}" == "train40_all480" ]] || {
  echo "ABORT: unsupported RUN_SCOPE=${RUN_SCOPE}" >&2
  exit 2
}
LIGHTGLUE_SOURCE="${LIGHTGLUE_SOURCE:-${LOCAL_ROOT}/.diagnostics/dependencies/LightGlue}"
DEPENDENCY_SOURCE="${DEPENDENCY_SOURCE:-${LOCAL_ROOT}/.diagnostics/dependencies/python}"
TORCH_CHECKPOINT_SOURCE="${TORCH_CHECKPOINT_SOURCE:-/home/asus/.cache/torch/hub/checkpoints}"

FILES=(
  MemNavData/diag_lingbot_goal_loop_closure.py
  MemNavData/lingbot_colored_registration.py
  MemNavData/lingbot_pnp_localization.py
  MemNavData/audit_lightglue_open_set_localization.py
  MemNavData/summarize_lingbot_lightglue_localization.py
  MemNavData/test_lingbot_goal_loop_closure.py
  MemNavData/slurm_lingbot_native_localizer.sbatch
  MemNavData/test_lingbot_pnp_localization.py
  MemNavData/test_audit_lightglue_open_set_localization.py
  MemNavData/test_summarize_lingbot_lightglue_localization.py
  MemNavData/external_causal_scale_contract.py
  MemNavData/phase_b_feature_schema.py
  MemNavData/phase_b_upstream_receipts.py
  MemNavData/test_phase_b_upstream_receipts.py
  MemNavData/flow_cache_routing.py
  MemNavData/audit_unknown_goal_mrc_smoke.py
  MemNavData/test_audit_unknown_goal_mrc_smoke.py
  MemNavData/build_train40_certificate_challenge_manifest.py
  MemNavData/test_build_train40_certificate_challenge_manifest.py
  MemNavData/unknown_goal_relocalization_v2_confirm_sessions_20260812.json
  MemNavData/train40_certificate_challenge_manifest_20260814.json
  MemNavData/lightglue_bundled_commit.txt
  MemNavData/OPEN_SET_RELOCALIZATION_PROTOCOL_20260812.md
  MemNavData/slurm_lingbot_lightglue_relocalization.sbatch
)

for relative in "${FILES[@]}"; do
  [[ -f "${LOCAL_ROOT}/${relative}" && ! -L "${LOCAL_ROOT}/${relative}" ]] || {
    echo "ABORT: missing physical source file ${relative}" >&2
    exit 2
  }
done
for required in \
  "${LIGHTGLUE_SOURCE}/lightglue" \
  "${LIGHTGLUE_SOURCE}/LICENSE" \
  "${DEPENDENCY_SOURCE}/kornia" \
  "${DEPENDENCY_SOURCE}/kornia_rs" \
  "${TORCH_CHECKPOINT_SOURCE}/superpoint_v1.pth" \
  "${TORCH_CHECKPOINT_SOURCE}/superpoint_lightglue_v0-1_arxiv.pth"; do
  [[ -e "${required}" && ! -L "${required}" ]] || {
    echo "ABORT: missing physical dependency ${required}" >&2
    exit 2
  }
done

"${MEMNAV_PY}" -m py_compile \
  "${LOCAL_ROOT}/MemNavData/diag_lingbot_goal_loop_closure.py" \
  "${LOCAL_ROOT}/MemNavData/lingbot_pnp_localization.py" \
  "${LOCAL_ROOT}/MemNavData/audit_lightglue_open_set_localization.py" \
  "${LOCAL_ROOT}/MemNavData/summarize_lingbot_lightglue_localization.py" \
  "${LOCAL_ROOT}/MemNavData/audit_unknown_goal_mrc_smoke.py" \
  "${LOCAL_ROOT}/MemNavData/build_train40_certificate_challenge_manifest.py"
(
  cd "${LOCAL_ROOT}"
  "${MEMNAV_PY}" -m pytest -q \
    MemNavData/test_lingbot_goal_loop_closure.py \
    MemNavData/test_lingbot_pnp_localization.py \
    MemNavData/test_audit_lightglue_open_set_localization.py \
    MemNavData/test_audit_unknown_goal_mrc_smoke.py \
    MemNavData/test_build_train40_certificate_challenge_manifest.py \
    MemNavData/test_summarize_lingbot_lightglue_localization.py \
    MemNavData/test_phase_b_upstream_receipts.py
)

STAGING="$(mktemp -d)"
trap 'rm -rf -- "${STAGING}"' EXIT
mkdir -p \
  "${STAGING}/MemNavData" \
  "${STAGING}/third_party/LightGlue" \
  "${STAGING}/third_party/python" \
  "${STAGING}/torch_home/hub/checkpoints"
for relative in "${FILES[@]}"; do
  cp --preserve=mode,timestamps "${LOCAL_ROOT}/${relative}" \
    "${STAGING}/${relative}"
done
cp -a "${LIGHTGLUE_SOURCE}/lightglue" \
  "${STAGING}/third_party/LightGlue/"
cp --preserve=mode,timestamps "${LIGHTGLUE_SOURCE}/LICENSE" \
  "${STAGING}/third_party/LightGlue/LICENSE"
cp --preserve=mode,timestamps \
  "${LOCAL_ROOT}/MemNavData/lightglue_bundled_commit.txt" \
  "${STAGING}/third_party/LightGlue/BUNDLED_COMMIT"
for dependency in kornia kornia-0.8.1.dist-info \
                  kornia_rs kornia_rs-0.1.9.dist-info; do
  cp -a "${DEPENDENCY_SOURCE}/${dependency}" \
    "${STAGING}/third_party/python/"
done
cp --preserve=mode,timestamps \
  "${TORCH_CHECKPOINT_SOURCE}/superpoint_v1.pth" \
  "${TORCH_CHECKPOINT_SOURCE}/superpoint_lightglue_v0-1_arxiv.pth" \
  "${STAGING}/torch_home/hub/checkpoints/"

LOCAL_HEAD="$(git -C "${LOCAL_ROOT}" rev-parse HEAD)"
python3 - "${STAGING}" "${LOCAL_HEAD}" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

root = Path(sys.argv[1])
files = {}
for path in sorted(root.rglob("*")):
    if path.is_symlink():
        raise SystemExit(f"bundle contains a symlink: {path}")
    if path.is_file() and path.name != "source_bundle_manifest.json":
        files[path.relative_to(root).as_posix()] = hashlib.sha256(
            path.read_bytes()).hexdigest()
manifest = {
    "schema_version": "lingbot_lightglue_relocalization_bundle_v1",
    "local_git_head_context": sys.argv[2],
    "files": files,
}
(root / "source_bundle_manifest.json").write_text(
    json.dumps(manifest, indent=2, sort_keys=True) + "\n")
PY

MANIFEST_SHA="$(sha256sum "${STAGING}/source_bundle_manifest.json" | awk '{print $1}')"
BUNDLE_TAG="${MANIFEST_SHA:0:16}"
REMOTE_BUNDLE="${REMOTE_BUNDLE_BASE}/lingbot_lightglue_relocalization_${BUNDLE_TAG}"
if [[ "${DRY_RUN}" == "1" ]]; then
  printf 'DRY_RUN_BUNDLE_MANIFEST_SHA=%s\n' "${MANIFEST_SHA}"
  printf 'DRY_RUN_BUNDLE_FILES=%s\n' \
    "$(find "${STAGING}" -type f | wc -l)"
  exit 0
fi
ssh -o BatchMode=yes "${REMOTE_HOST}" "mkdir -p '${REMOTE_BUNDLE}'"
rsync -a --chmod=Du=rwx,Dgo=rx,Fu=rw,Fgo=r \
  "${STAGING}/" "${REMOTE_HOST}:${REMOTE_BUNDLE}/"
REMOTE_MANIFEST_SHA="$(ssh -o BatchMode=yes "${REMOTE_HOST}" \
  "sha256sum '${REMOTE_BUNDLE}/source_bundle_manifest.json' | awk '{print \$1}'")"
[[ "${REMOTE_MANIFEST_SHA}" == "${MANIFEST_SHA}" ]] || {
  echo "ABORT: staged bundle manifest differs" >&2
  exit 2
}

if [[ "${RUN_SCOPE}" == "train40_all480" ]]; then
  REMOTE_SBATCH="sbatch --parsable --job-name=lg_rel_t40 --time=12:00:00 --export=ALL,RUN_SCOPE='${RUN_SCOPE}',SOURCE_BUNDLE='${REMOTE_BUNDLE}',EXPECTED_BUNDLE_MANIFEST_SHA='${MANIFEST_SHA}' '${REMOTE_BUNDLE}/MemNavData/slurm_lingbot_lightglue_relocalization.sbatch'"
else
  REMOTE_SBATCH="sbatch --parsable --export=ALL,RUN_SCOPE='${RUN_SCOPE}',SOURCE_BUNDLE='${REMOTE_BUNDLE}',EXPECTED_BUNDLE_MANIFEST_SHA='${MANIFEST_SHA}' '${REMOTE_BUNDLE}/MemNavData/slurm_lingbot_lightglue_relocalization.sbatch'"
fi
JOB_ID="$(ssh -o BatchMode=yes "${REMOTE_HOST}" "${REMOTE_SBATCH}")"
printf 'LINGBOT_LIGHTGLUE_RELOCALIZATION_JOB_ID=%s\n' "${JOB_ID}"
printf 'LINGBOT_LIGHTGLUE_RELOCALIZATION_BUNDLE=%s\n' "${REMOTE_BUNDLE}"
printf 'LINGBOT_LIGHTGLUE_RELOCALIZATION_MANIFEST_SHA=%s\n' "${MANIFEST_SHA}"
printf 'LINGBOT_LIGHTGLUE_RELOCALIZATION_SCOPE=%s\n' "${RUN_SCOPE}"
