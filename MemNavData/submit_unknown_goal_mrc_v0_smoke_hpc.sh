#!/usr/bin/env bash
# Build an immutable task-only source bundle and submit the MRC-v0 smoke.
set -euo pipefail
umask 0022

LOCAL_ROOT="$(git rev-parse --show-toplevel)"
REMOTE_HOST="${REMOTE_HOST:-alantorch}"
REMOTE_BUNDLE_BASE="${REMOTE_BUNDLE_BASE:-/scratch/yz11502/Research/Nav-axis-uturn-source-bundles}"

FILES=(
  MemNavData/diag_lingbot_goal_loop_closure.py
  MemNavData/test_lingbot_goal_loop_closure.py
  MemNavData/slurm_lingbot_native_localizer.sbatch
  MemNavData/external_causal_scale_contract.py
  MemNavData/phase_b_feature_schema.py
  MemNavData/phase_b_upstream_receipts.py
  MemNavData/test_phase_b_upstream_receipts.py
  MemNavData/flow_cache_routing.py
  MemNavData/audit_unknown_goal_mrc_smoke.py
  MemNavData/test_audit_unknown_goal_mrc_smoke.py
  MemNavData/unknown_goal_mrc_v0_smoke_sessions_20260812.json
  MemNavData/UNKNOWN_GOAL_MRC_V0_PROTOCOL_20260812.md
  MemNavData/slurm_unknown_goal_mrc_v0_smoke.sbatch
)

for relative in "${FILES[@]}"; do
  [[ -f "${LOCAL_ROOT}/${relative}" && ! -L "${LOCAL_ROOT}/${relative}" ]] || {
    echo "ABORT: missing physical source file ${relative}" >&2
    exit 2
  }
done

MEMNAV_PY="${MEMNAV_PY:-/home/asus/miniconda3/envs/memnav/bin/python}"
"${MEMNAV_PY}" -m py_compile \
  "${LOCAL_ROOT}/MemNavData/diag_lingbot_goal_loop_closure.py" \
  "${LOCAL_ROOT}/MemNavData/audit_unknown_goal_mrc_smoke.py"
(
  cd "${LOCAL_ROOT}"
  "${MEMNAV_PY}" -m unittest -v \
    MemNavData.test_lingbot_goal_loop_closure \
    MemNavData.test_audit_unknown_goal_mrc_smoke \
    MemNavData.test_phase_b_upstream_receipts
)

STAGING="$(mktemp -d)"
trap 'rm -rf -- "${STAGING}"' EXIT
mkdir -p "${STAGING}/MemNavData"
for relative in "${FILES[@]}"; do
  cp --preserve=mode,timestamps "${LOCAL_ROOT}/${relative}" \
    "${STAGING}/${relative}"
done

LOCAL_HEAD="$(git -C "${LOCAL_ROOT}" rev-parse HEAD)"
python3 - "${STAGING}" "${LOCAL_HEAD}" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

root = Path(sys.argv[1])
files = {}
for path in sorted((root / "MemNavData").iterdir()):
    if path.is_file():
        files[path.relative_to(root).as_posix()] = hashlib.sha256(
            path.read_bytes()).hexdigest()
manifest = {
    "schema_version": "unknown_goal_mrc_v0_source_bundle_v1",
    "local_git_head_context": sys.argv[2],
    "files": files,
}
(root / "source_bundle_manifest.json").write_text(
    json.dumps(manifest, indent=2, sort_keys=True) + "\n")
PY

MANIFEST_SHA="$(sha256sum "${STAGING}/source_bundle_manifest.json" | awk '{print $1}')"
BUNDLE_TAG="${MANIFEST_SHA:0:16}"
REMOTE_BUNDLE="${REMOTE_BUNDLE_BASE}/unknown_goal_mrc_v0_${BUNDLE_TAG}"
ssh -o BatchMode=yes "${REMOTE_HOST}" \
  "mkdir -p '${REMOTE_BUNDLE}'"
rsync -a --chmod=Du=rwx,Dgo=rx,Fu=rw,Fgo=r \
  "${STAGING}/" "${REMOTE_HOST}:${REMOTE_BUNDLE}/"
REMOTE_MANIFEST_SHA="$(ssh -o BatchMode=yes "${REMOTE_HOST}" \
  "sha256sum '${REMOTE_BUNDLE}/source_bundle_manifest.json' | awk '{print \$1}'")"
[[ "${REMOTE_MANIFEST_SHA}" == "${MANIFEST_SHA}" ]] || {
  echo "ABORT: staged bundle manifest differs" >&2
  exit 2
}

JOB_ID="$(ssh -o BatchMode=yes "${REMOTE_HOST}" \
  "sbatch --parsable --export=ALL,SOURCE_BUNDLE='${REMOTE_BUNDLE}',EXPECTED_BUNDLE_MANIFEST_SHA='${MANIFEST_SHA}' '${REMOTE_BUNDLE}/MemNavData/slurm_unknown_goal_mrc_v0_smoke.sbatch'")"
printf 'MRC_V0_SMOKE_JOB_ID=%s\n' "${JOB_ID}"
printf 'MRC_V0_SOURCE_BUNDLE=%s\n' "${REMOTE_BUNDLE}"
printf 'MRC_V0_BUNDLE_MANIFEST_SHA=%s\n' "${MANIFEST_SHA}"
