#!/usr/bin/env bash
# Build the H24 frozen geometry identity map from the independently audited
# Habitat-0.3.1 NavMesh bake.  CPU-only; no geometry is recomputed here.
#SBATCH --job-name=nlsr_geom_map
#SBATCH --account=torch_pr_769_tandon_advanced
#SBATCH --partition=cpu_short
#SBATCH --qos=cpu48
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/%x_%j.out
#SBATCH --error=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/%x_%j.err

set -euo pipefail
umask 0022

: "${REPO_ROOT:?export REPO_ROOT to an exact clean producer checkout}"
: "${EXPECTED_COMMIT:?export its full commit SHA}"
: "${EXPECTED_LAUNCHER_SHA:?export the committed launcher SHA256}"
: "${RUN_TAG:?export the audited NLSR run tag}"

RESULT_BASE="${RESULT_BASE:-/scratch/yz11502/Research/Nav-axis-uturn-results/nlsr_gapfill_20260807}"
RUN_ROOT="${RESULT_BASE}/${RUN_TAG}"
NAVMESH_BAKE_REV="f8b275152dd1"
BAKE_ROOT="${RUN_ROOT}/navmesh_bake_habitat031_agent030_${NAVMESH_BAKE_REV}"
DERIVED_MANIFEST="${BAKE_ROOT}/published/derived_geometry_manifest.json"
BAKE_AUDIT_RECEIPT="${RUN_ROOT}/navmesh_bake_launcher_receipt.json"
SETTINGS_REL="MemNavData/navmesh_settings_habitat_0_3_1_agent030.json"
BUILDER_REL="MemNavData/build_frozen_geometry_map.py"
LAUNCHER_REL="MemNavData/slurm_nlsr_frozen_geometry_map.sbatch"
OUTPUT_ROOT="${RUN_ROOT}/frozen_geometry_map_${EXPECTED_COMMIT:0:12}_8a7e71e9514b"
PYTHON_BIN="/scratch/lg154/conda-envs/habitat/bin/python"
EXPECTED_PYTHON_TARGET="/scratch/lg154/conda-envs/habitat/bin/python3.9"

EXPECTED_DERIVED_MANIFEST_SHA="8a7e71e9514b5cf135b87d628ddabf4db805e95f6699baedfd8a0370f00e20db"
EXPECTED_DERIVED_SIDECAR_SHA="8d7ee80c35ef95169d1d66aa8a3392086e3ecccb8dd8165821dcb2517c5d10c6"
EXPECTED_BAKE_RECEIPT_SHA="2ffe075efd06e62d2b606c6678b9cad8be7d3c7fc25ced6e9252dbafde02dd8b"
EXPECTED_BAKE_RECEIPT_SIDECAR_SHA="c531feb64fdb79ee2a1d2489d5b1654bcc1cfca0f465c63e7d94cad3c844c7aa"
EXPECTED_PARENT_MANIFEST_SHA="bc6bf58536f6c159d1898ac03abe365eadba65c22cd246e39401821962abb34c"
EXPECTED_SETTINGS_SHA="62cf1f7a569939e0f8d92045483f76cc008ff71ab7a97e1a447a921722a7dcf9"
EXPECTED_BUILDER_SHA="c10e8f22530f08bb9547256ba7e8af3ba02479843f704e5d9466baf384d1f5fd"
EXPECTED_PYTHON_SHA="8e9776d5b77f7d22be61bfebc82a544313837abb7d6ca6b110f4547f73259d89"

fail() {
  echo "ABORT: $*" >&2
  exit 2
}

full_sha_is() {
  local path="$1" expected="$2" label="$3" actual
  [[ -f "${path}" && ! -L "${path}" ]] || fail "missing physical ${label}: ${path}"
  actual="$(sha256sum "${path}" | awk '{print $1}')"
  [[ "${actual}" == "${expected}" ]] || \
    fail "${label} SHA mismatch: ${actual} != ${expected}"
}

pinned_python_is() {
  local resolved actual
  [[ -L "${PYTHON_BIN}" ]] || fail "Python entry point is not the pinned symlink"
  [[ "$(readlink "${PYTHON_BIN}")" == "python3.9" ]] || \
    fail "Python symlink target changed"
  resolved="$(readlink -f "${PYTHON_BIN}")"
  [[ "${resolved}" == "${EXPECTED_PYTHON_TARGET}" ]] || \
    fail "resolved Python path changed: ${resolved}"
  [[ -f "${resolved}" && ! -L "${resolved}" ]] || \
    fail "resolved Python runtime is not a physical file"
  actual="$(sha256sum "${resolved}" | awk '{print $1}')"
  [[ "${actual}" == "${EXPECTED_PYTHON_SHA}" ]] || \
    fail "Python runtime SHA mismatch: ${actual}"
}

[[ "${RUN_TAG}" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]] || fail "invalid RUN_TAG"
[[ "${EXPECTED_COMMIT}" =~ ^[0-9a-f]{40}$ ]] || fail "invalid commit pin"
[[ "${EXPECTED_LAUNCHER_SHA}" =~ ^[0-9a-f]{64}$ ]] || fail "invalid launcher pin"
command -v git >/dev/null || fail "git is unavailable"
command -v sha256sum >/dev/null || fail "sha256sum is unavailable"
[[ -d "${REPO_ROOT}/.git" || -f "${REPO_ROOT}/.git" ]] || fail "not a checkout"
[[ "$(git -C "${REPO_ROOT}" rev-parse HEAD)" == "${EXPECTED_COMMIT}" ]] || \
  fail "producer commit mismatch"
if [[ -n "$(git -C "${REPO_ROOT}" status --porcelain)" ]]; then
  git -C "${REPO_ROOT}" status --short >&2
  fail "producer checkout is not clean"
fi
launcher_sha="$(sha256sum "$0" | awk '{print $1}')"
repository_launcher_sha="$(sha256sum \
  "${REPO_ROOT}/${LAUNCHER_REL}" | awk '{print $1}')"
[[ "${launcher_sha}" == "${EXPECTED_LAUNCHER_SHA}" ]] || fail "spooled launcher changed"
[[ "${repository_launcher_sha}" == "${EXPECTED_LAUNCHER_SHA}" ]] || \
  fail "repository launcher changed"
full_sha_is "${REPO_ROOT}/${BUILDER_REL}" "${EXPECTED_BUILDER_SHA}" "geometry-map builder"
full_sha_is "${REPO_ROOT}/${SETTINGS_REL}" "${EXPECTED_SETTINGS_SHA}" "NavMesh settings"
full_sha_is "${DERIVED_MANIFEST}" "${EXPECTED_DERIVED_MANIFEST_SHA}" \
  "derived geometry manifest"
full_sha_is "${DERIVED_MANIFEST}.sha256" "${EXPECTED_DERIVED_SIDECAR_SHA}" \
  "derived geometry manifest sidecar"
full_sha_is "${BAKE_AUDIT_RECEIPT}" "${EXPECTED_BAKE_RECEIPT_SHA}" \
  "NavMesh cross-audit receipt"
full_sha_is "${BAKE_AUDIT_RECEIPT}.sha256" \
  "${EXPECTED_BAKE_RECEIPT_SIDECAR_SHA}" "NavMesh cross-audit receipt sidecar"
pinned_python_is

"${PYTHON_BIN}" - "${DERIVED_MANIFEST}" "${EXPECTED_PARENT_MANIFEST_SHA}" \
  "${BAKE_AUDIT_RECEIPT}" "${EXPECTED_DERIVED_MANIFEST_SHA}" <<'PY'
import json
from pathlib import Path
import sys

manifest = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
derivation = manifest.get("geometry_bake_derivation", {})
receipt = json.loads(Path(sys.argv[3]).read_text(encoding="utf-8"))
if manifest.get("summary", {}).get("scene_count") != 50:
    raise SystemExit("derived geometry scene count changed")
if derivation.get("parent_manifest_sha256") != sys.argv[2]:
    raise SystemExit("derived geometry parent manifest changed")
if derivation.get("status") != "fresh_double_bake_roundtrip_verified":
    raise SystemExit("derived geometry bake status changed")
if receipt.get("status") != "cross_audited":
    raise SystemExit("derived geometry bake lacks cross-audit approval")
if receipt.get("outputs", {}).get("derived_geometry_manifest", {}).get(
        "content_sha256") != sys.argv[4]:
    raise SystemExit("cross-audit receipt binds another derived manifest")
print("frozen_geometry_preflight=passed")
PY

resume_args=()
if [[ -e "${OUTPUT_ROOT}" ]]; then
  resume_args+=(--resume)
fi
stdout_tmp="${RUN_ROOT}/.frozen_geometry_stdout.${SLURM_JOB_ID:-manual}.tmp"
rm -f "${stdout_tmp}"
trap 'rm -f "${stdout_tmp}"' EXIT
"${PYTHON_BIN}" "${REPO_ROOT}/${BUILDER_REL}" \
  --expert-manifest "${DERIVED_MANIFEST}" \
  --expected-manifest-sha256 "${EXPECTED_DERIVED_MANIFEST_SHA}" \
  --navmesh-settings-json "${REPO_ROOT}/${SETTINGS_REL}" \
  --expected-navmesh-settings-sha256 "${EXPECTED_SETTINGS_SHA}" \
  --habitat-sim-version 0.3.1 \
  --agent-radius-m 0.3 \
  --agent-height-m 1.5 \
  --output-directory "${OUTPUT_ROOT}" \
  "${resume_args[@]}" | tee "${stdout_tmp}"
mv -f "${stdout_tmp}" "${RUN_ROOT}/frozen_geometry_stdout.json"
trap - EXIT

"${PYTHON_BIN}" - "${OUTPUT_ROOT}" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

root = Path(sys.argv[1])
path = root / "frozen_geometry_map.json"
sidecar = root / "frozen_geometry_map.json.sha256"
record = json.loads(path.read_text(encoding="utf-8"))
digest = hashlib.sha256(path.read_bytes()).hexdigest()
if record.get("schema_version") != "frozen_geometry_map_v1":
    raise SystemExit("frozen geometry schema changed")
if len(record.get("scenes", {})) != 50:
    raise SystemExit("frozen geometry scene count changed")
if sidecar.read_text(encoding="ascii") != f"{digest}  frozen_geometry_map.json\n":
    raise SystemExit("frozen geometry exact sidecar changed")
print(json.dumps({"frozen_geometry_map": "passed", "sha256": digest,
                  "scene_count": 50}, sort_keys=True))
PY

echo "NLSR frozen geometry map passed: ${OUTPUT_ROOT}"
