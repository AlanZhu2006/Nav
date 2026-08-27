#!/usr/bin/env bash
# Stage 4: extend the audited Goal-B manifest with causal Goal-C decision pairs.
# Submit only with afterok:<corrected-manifest-job>; all upstream artifacts and
# the exact clean producer checkout are independently content-verified here.
#SBATCH --job-name=nlsr_multistage
#SBATCH --account=torch_pr_769_tandon_advanced
#SBATCH --partition=cpu_short
#SBATCH --qos=cpu48
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/%x_%j.out
#SBATCH --error=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/%x_%j.err

set -euo pipefail
umask 0022

: "${REPO_ROOT:?export REPO_ROOT to the exact clean producer checkout}"
: "${EXPECTED_COMMIT:?export EXPECTED_COMMIT to its full commit SHA}"
: "${RUN_TAG:?reuse the exact audited NLSR run tag}"
[[ "${RUN_TAG}" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]] || {
  echo "ABORT: invalid RUN_TAG=${RUN_TAG}" >&2
  exit 2
}

RESULT_BASE="${RESULT_BASE:-/scratch/yz11502/Research/Nav-axis-uturn-results/nlsr_gapfill_20260807}"
RUN_ROOT="${RESULT_BASE}/${RUN_TAG}"
SOURCE_ROOT="${RUN_ROOT}/manifest_corrected"
SOURCE_MANIFEST="${SOURCE_ROOT}/candidate_manifest.json"
SOURCE_AUDIT="${SOURCE_ROOT}/manifest_audit.json"
OUTPUT_ROOT="${RUN_ROOT}/manifest_multistage"
OUTPUT_MANIFEST="${OUTPUT_ROOT}/candidate_manifest_multistage.json"
BUILDER_REL="MemNavData/build_multistage_candidate_manifest.py"
MEMNAV_PY="${MEMNAV_PY:-/scratch/lg154/conda-envs/memnav/bin/python}"
BASE_SIF="${BASE_SIF:-/share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif}"
OVERLAY="${OVERLAY:-/scratch/lg154/Research/datasets/_overlays/mp3d_revisit_v0_pt1.sqf}"
RAW_SCENE="${RUN_ROOT}/raw/mp3d_3leg/YmJkqBEsHnH"
EPISODE_ROOT="/mp3d_revisit_v0/vln_n1/traj_data/mp3d_3leg"
LEGACY_CAMERA_HEIGHT_M="${LEGACY_CAMERA_HEIGHT_M:-0.5}"

EXPECTED_BUILDER_SHA="5e675505add359da88911ab98e18932cc3d77c232f843763b017403fe5ea3f10"
EXPECTED_MEMNAV_PY_SHA="cf18c572f59ed93caec89258837eb1cbc5b4a6893e272888c3abebb90ae5ff86"
EXPECTED_OVERLAY_BYTES="128854888448"
EXPECTED_OVERLAY_HEAD_SHA="43d803c2b39735e764f0c6df8f7c40a7035a1c23789e22fcc9831d29928c801b"
EXPECTED_SIF_BYTES="10035580928"
EXPECTED_SIF_HEAD_SHA="990c1377491fbf58c1d7d37bcf8043249826bc7a094eacc4c7a443b9905126c7"

fail() {
  echo "ABORT: $*" >&2
  exit 2
}

full_sha_is() {
  local path="$1" expected="$2" label="$3" actual
  [[ -f "${path}" ]] || fail "missing ${label}: ${path}"
  actual="$(sha256sum "${path}" | awk '{print $1}')"
  [[ "${actual}" == "${expected}" ]] || \
    fail "${label} SHA mismatch: ${actual} != ${expected}"
}

checked_sidecar_sha() {
  local path="$1" label="$2" actual declared
  [[ -f "${path}" && -f "${path}.sha256" ]] || \
    fail "${label} artifact pair is absent: ${path}"
  actual="$(sha256sum "${path}" | awk '{print $1}')"
  declared="$(awk 'NR==1 {print $1}' "${path}.sha256")"
  [[ "${actual}" == "${declared}" ]] || fail "${label} SHA sidecar mismatch"
  printf '%s' "${actual}"
}

command -v singularity >/dev/null || fail "singularity is unavailable"
command -v flock >/dev/null || fail "flock is unavailable"
[[ -d "${REPO_ROOT}/.git" || -f "${REPO_ROOT}/.git" ]] || \
  fail "producer is not a git checkout: ${REPO_ROOT}"
[[ "$(git -C "${REPO_ROOT}" rev-parse HEAD)" == "${EXPECTED_COMMIT}" ]] || \
  fail "producer commit mismatch"
if [[ -n "$(git -C "${REPO_ROOT}" status --porcelain)" ]]; then
  git -C "${REPO_ROOT}" status --short >&2
  fail "producer checkout is not clean"
fi
full_sha_is "${REPO_ROOT}/${BUILDER_REL}" "${EXPECTED_BUILDER_SHA}" \
  "multistage builder"
full_sha_is "${MEMNAV_PY}" "${EXPECTED_MEMNAV_PY_SHA}" "MemNav Python"
[[ -f "${BASE_SIF}" && "$(stat -c '%s' "${BASE_SIF}")" == "${EXPECTED_SIF_BYTES}" ]] || \
  fail "base SIF identity differs"
[[ "$(dd if="${BASE_SIF}" bs=1048576 count=1 status=none | sha256sum | awk '{print $1}')" == \
  "${EXPECTED_SIF_HEAD_SHA}" ]] || fail "base SIF header SHA mismatch"
[[ -f "${OVERLAY}" && "$(stat -c '%s' "${OVERLAY}")" == "${EXPECTED_OVERLAY_BYTES}" ]] || \
  fail "overlay identity differs"
[[ "$(dd if="${OVERLAY}" bs=1048576 count=1 status=none | sha256sum | awk '{print $1}')" == \
  "${EXPECTED_OVERLAY_HEAD_SHA}" ]] || fail "overlay header SHA mismatch"
[[ -d "${RAW_SCENE}" ]] || fail "audited YmJ raw scene is absent"
source_manifest_sha="$(checked_sidecar_sha "${SOURCE_MANIFEST}" "corrected manifest")"
source_audit_sha="$(checked_sidecar_sha "${SOURCE_AUDIT}" "corrected manifest audit")"
exec 9>"${RUN_ROOT}/.multistage_manifest.lock"
flock -n 9 || fail "another multistage manifest stage is active"
[[ ! -e "${OUTPUT_ROOT}" ]] || fail "output already exists: ${OUTPUT_ROOT}"

CONTAINER=(singularity exec
  --overlay "${OVERLAY}:ro"
  -B /scratch/lg154
  -B /scratch/yz11502
  -B "${RAW_SCENE}:${EPISODE_ROOT}/YmJkqBEsHnH:ro"
  "${BASE_SIF}")

# Dependency check precedes output creation.
"${CONTAINER[@]}" "${MEMNAV_PY}" - <<'PY'
import importlib.util
missing = [name for name in ("numpy", "pyarrow")
           if importlib.util.find_spec(name) is None]
if missing:
    raise SystemExit(f"missing multistage dependencies: {missing}")
print("multistage_dependency_preflight=passed")
PY

mkdir -p "${OUTPUT_ROOT}"
"${CONTAINER[@]}" "${MEMNAV_PY}" "${REPO_ROOT}/${BUILDER_REL}" \
  --routed-manifest "${SOURCE_MANIFEST}" \
  --expected-routed-manifest-sha "${source_manifest_sha}" \
  --routed-manifest-audit "${SOURCE_AUDIT}" \
  --expected-routed-manifest-audit-sha "${source_audit_sha}" \
  --legacy-camera-height-m "${LEGACY_CAMERA_HEIGHT_M}" \
  --out "${OUTPUT_MANIFEST}" \
  | tee "${OUTPUT_ROOT}/builder_stdout.json"

"${CONTAINER[@]}" "${MEMNAV_PY}" - \
  "${OUTPUT_MANIFEST}" "${source_manifest_sha}" "${source_audit_sha}" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

path = Path(sys.argv[1])
raw = path.read_bytes()
record = json.loads(raw)
canonical = (json.dumps(record, sort_keys=True, separators=(",", ":"),
                        ensure_ascii=False, allow_nan=False) + "\n").encode()
if raw != canonical:
    raise SystemExit("multistage manifest is not canonical")
digest = hashlib.sha256(raw).hexdigest()
sidecar = Path(f"{path}.sha256").read_text().split()[0]
if digest != sidecar:
    raise SystemExit("multistage manifest sidecar mismatch")
if record.get("schema_version") != "nlsr_v2_multistage_expert_candidate_manifest_v1":
    raise SystemExit("multistage schema changed")
source = record.get("source_routed_manifest", {})
audit = record.get("source_routed_manifest_audit", {})
if source.get("sha256") != sys.argv[2] or audit.get("sha256") != sys.argv[3]:
    raise SystemExit("multistage source binding changed")
summary = record.get("summary", {})
expected = {
    "scene_count": 50,
    "episode_count": 100,
    "goal_b_sample_count": 400,
    "goal_c_sample_count": 200,
    "sample_count": 600,
    "counterfactual_pair_group_count": 300,
}
for key, value in expected.items():
    if summary.get(key) != value:
        raise SystemExit(f"unexpected {key}: {summary.get(key)} != {value}")
print(json.dumps({"multistage_cross_audit": "passed", "sha256": digest,
                  **expected}, sort_keys=True))
PY

echo "NLSR multistage manifest passed: ${OUTPUT_MANIFEST}"
