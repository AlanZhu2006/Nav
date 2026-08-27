#!/usr/bin/env bash
# Stage 3: build and cross-audit the corrected 50-scene/100-episode manifest.
# Submit only with afterok:<flow4096-job>; artifact checks independently reject a
# missing, partial, mismatched, or differently tagged predecessor.
#SBATCH --job-name=nlsr_manifest_ok
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

: "${REPO_ROOT:?export REPO_ROOT to the clean committed wrapper worktree}"
: "${EXPECTED_COMMIT:?export EXPECTED_COMMIT to the full SHA of that worktree}"
: "${RUN_TAG:?reuse the exact RUN_TAG from both successful upstream jobs}"
[[ "${RUN_TAG}" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]] || {
  echo "ABORT: invalid RUN_TAG=${RUN_TAG}" >&2
  exit 2
}

RESULT_BASE="${RESULT_BASE:-/scratch/yz11502/Research/Nav-axis-uturn-results/nlsr_gapfill_20260807}"
RUN_ROOT="${RESULT_BASE}/${RUN_TAG}"
RAW_SCENE="${RUN_ROOT}/raw/mp3d_3leg/YmJkqBEsHnH"
RAW_AUDIT="${RUN_ROOT}/raw_audit.json"
FLOW_ROUTE_ROOT="${RUN_ROOT}/flow_routes"
FLOW_PROVENANCE="${FLOW_ROUTE_ROOT}/FLOW_ROUTE_PROVENANCE.json"
MANIFEST_ROOT="${RUN_ROOT}/manifest_corrected"
MANIFEST="${MANIFEST_ROOT}/candidate_manifest.json"
BUILDER_REL="MemNavData/build_novel_candidate_manifest.py"
AUDITOR_REL="MemNavData/audit_nlsr_corrected_manifest.py"
ROUTER_REL="MemNavData/flow_cache_routing.py"
SPLIT_REL="MemNavData/router_multiscene_split_20260805.json"
MEMNAV_PY="${MEMNAV_PY:-/scratch/lg154/conda-envs/memnav/bin/python}"
BASE_SIF="${BASE_SIF:-/share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif}"
OVERLAY="${OVERLAY:-/scratch/lg154/Research/datasets/_overlays/mp3d_revisit_v0_pt1.sqf}"
ASSET_ROOT="/scratch/lg154/Research/datasets/mp3d/mp3d"
EPISODE_ROOT="/mp3d_revisit_v0/vln_n1/traj_data/mp3d_3leg"

EXPECTED_BUILDER_SHA="00fe0e48a9fb342b48430d8a0f9175994d98db45a8ecd2c3d6e336701f8733d8"
EXPECTED_AUDITOR_SHA="5a90b779fa965255b8cef7608e3bc54f74895cff33215450b68ca24e86f5eb6a"
EXPECTED_ROUTER_SHA="4896a77fa35871588f4add5c83c10e48fbb9bf45a15845f0a5d4332bc63a0035"
EXPECTED_SPLIT_SHA="97309c183e25cb3dd65472908748d55a94798a636db6157ab6fe120fca05cf7a"
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
  [[ "${actual}" == "${declared}" ]] || \
    fail "${label} SHA sidecar mismatch"
  printf '%s' "${actual}"
}

command -v singularity >/dev/null || fail "singularity is unavailable"
command -v flock >/dev/null || fail "flock is unavailable"
[[ -d "${REPO_ROOT}/.git" || -f "${REPO_ROOT}/.git" ]] || \
  fail "wrapper is not a git worktree: ${REPO_ROOT}"
actual_commit="$(git -C "${REPO_ROOT}" rev-parse HEAD)"
[[ "${actual_commit}" == "${EXPECTED_COMMIT}" ]] || \
  fail "wrapper commit mismatch: ${actual_commit} != ${EXPECTED_COMMIT}"
if [[ -n "$(git -C "${REPO_ROOT}" status --porcelain)" ]]; then
  git -C "${REPO_ROOT}" status --short >&2
  fail "wrapper worktree is not clean"
fi

full_sha_is "${REPO_ROOT}/${BUILDER_REL}" "${EXPECTED_BUILDER_SHA}" \
  "manifest builder"
full_sha_is "${REPO_ROOT}/${AUDITOR_REL}" "${EXPECTED_AUDITOR_SHA}" \
  "manifest auditor"
full_sha_is "${REPO_ROOT}/${ROUTER_REL}" "${EXPECTED_ROUTER_SHA}" \
  "flow route resolver"
full_sha_is "${REPO_ROOT}/${SPLIT_REL}" "${EXPECTED_SPLIT_SHA}" "strict split"
full_sha_is "${MEMNAV_PY}" "${EXPECTED_MEMNAV_PY_SHA}" "MemNav Python"
[[ -f "${BASE_SIF}" ]] || fail "base SIF is absent: ${BASE_SIF}"
[[ "$(stat -c '%s' "${BASE_SIF}")" == "${EXPECTED_SIF_BYTES}" ]] || \
  fail "base SIF size mismatch"
sif_head_sha="$(dd if="${BASE_SIF}" bs=1048576 count=1 status=none | \
  sha256sum | awk '{print $1}')"
[[ "${sif_head_sha}" == "${EXPECTED_SIF_HEAD_SHA}" ]] || \
  fail "base SIF header SHA mismatch"
[[ -f "${OVERLAY}" ]] || fail "pt1 overlay is absent: ${OVERLAY}"
[[ "$(stat -c '%s' "${OVERLAY}")" == "${EXPECTED_OVERLAY_BYTES}" ]] || \
  fail "pt1 overlay size mismatch"
overlay_head_sha="$(dd if="${OVERLAY}" bs=1048576 count=1 status=none | \
  sha256sum | awk '{print $1}')"
[[ "${overlay_head_sha}" == "${EXPECTED_OVERLAY_HEAD_SHA}" ]] || \
  fail "pt1 overlay header SHA mismatch"

[[ -d "${RAW_SCENE}" ]] || fail "audited YmJ raw scene is absent: ${RAW_SCENE}"
[[ -d "${FLOW_ROUTE_ROOT}" ]] || fail "flow route root is absent: ${FLOW_ROUTE_ROOT}"
[[ -d "${ASSET_ROOT}" ]] || fail "MP3D asset root is absent: ${ASSET_ROOT}"
raw_audit_sha="$(checked_sidecar_sha "${RAW_AUDIT}" "raw audit")"
flow_provenance_sha="$(checked_sidecar_sha "${FLOW_PROVENANCE}" "flow provenance")"
exec 9>"${RUN_ROOT}/.manifest_stage.lock"
flock -n 9 || fail "another manifest stage is active for RUN_TAG=${RUN_TAG}"
[[ ! -e "${MANIFEST_ROOT}" ]] || \
  fail "manifest-stage output already exists: ${MANIFEST_ROOT}"

CONTAINER=(singularity exec
  --overlay "${OVERLAY}:ro"
  -B /scratch/lg154
  -B /scratch/yz11502
  -B "${RAW_SCENE}:${EPISODE_ROOT}/YmJkqBEsHnH:ro"
  "${BASE_SIF}")

# Dependency and exact-gate preflight runs before the output directory exists.
"${CONTAINER[@]}" "${MEMNAV_PY}" - \
  "${RAW_AUDIT}" "${FLOW_PROVENANCE}" \
  "${EPISODE_ROOT}" "${FLOW_ROUTE_ROOT}" <<'PY'
import importlib.util
import json
from pathlib import Path
import sys

required = ("numpy", "pyarrow")
missing = [name for name in required if importlib.util.find_spec(name) is None]
if missing:
    raise SystemExit(f"missing manifest dependencies: {missing}")
raw = json.loads(Path(sys.argv[1]).read_text())
flow = json.loads(Path(sys.argv[2]).read_text())
expected = {
    "episode_0000": {"n_frames": 871, "switches": [218, 548]},
    "episode_0001": {"n_frames": 591, "switches": [192, 296]},
}
if raw.get("status") != "audited_historical_summary_match":
    raise SystemExit("raw audit is not successful")
if raw.get("historical_three_leg_reference") != expected:
    raise SystemExit("raw exact historical gate differs")
if flow.get("status") != "flow_routes_audited":
    raise SystemExit("flow provenance is not successful")
if flow.get("counts") != {"scenes": 50, "pairs": 100, "official_base": 97, "flow4096_patch": 3}:
    raise SystemExit("flow provenance counts differ")
if Path(flow.get("route_root", "")).resolve() != Path(sys.argv[4]).resolve():
    raise SystemExit("flow provenance points at another route root")
for episode, summary in expected.items():
    meta = Path(sys.argv[3]) / "YmJkqBEsHnH" / episode / "meta/gen_meta.json"
    record = json.loads(meta.read_text())
    actual = {"n_frames": record.get("n_frames"), "switches": record.get("switches")}
    if actual != summary:
        raise SystemExit(f"bound raw summary mismatch for {episode}: {actual}")
print(json.dumps({"manifest_preflight": "passed", "raw_exact_gate": expected}, sort_keys=True))
PY

mkdir -p "${MANIFEST_ROOT}"
"${CONTAINER[@]}" "${MEMNAV_PY}" \
  "${REPO_ROOT}/${BUILDER_REL}" \
  --split-manifest "${REPO_ROOT}/${SPLIT_REL}" \
  --episode-root "${EPISODE_ROOT}" \
  --flow-route-provenance "${FLOW_PROVENANCE}" \
  --expected-flow-route-sha "${flow_provenance_sha}" \
  --environment-root "${ASSET_ROOT}" \
  --navmesh-root "${ASSET_ROOT}" \
  --role train \
  --role development \
  --out "${MANIFEST}" \
  | tee "${MANIFEST_ROOT}/builder_stdout.json"

"${CONTAINER[@]}" "${MEMNAV_PY}" \
  "${REPO_ROOT}/${AUDITOR_REL}" \
  --manifest "${MANIFEST}" \
  --raw-audit "${RAW_AUDIT}" \
  --flow-provenance "${FLOW_PROVENANCE}" \
  --expected-commit "${EXPECTED_COMMIT}" \
  --expected-episode-root "${EPISODE_ROOT}" \
  --expected-flow-route-root "${FLOW_ROUTE_ROOT}" \
  --out "${MANIFEST_ROOT}/manifest_audit.json" \
  | tee "${MANIFEST_ROOT}/audit_stdout.json"

echo "NLSR corrected manifest passed cross-artifact audit: ${MANIFEST_ROOT}"
echo "raw_audit_sha=${raw_audit_sha} flow_provenance_sha=${flow_provenance_sha}"
