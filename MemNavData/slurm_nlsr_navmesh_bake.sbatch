#!/usr/bin/env bash
# Stage 5b: CPU-only, receipt-backed Habitat-0.3.1 NavMesh bake.
# This may run in parallel with causal metric-scale estimation after the exact
# multistage manifest has completed.
#SBATCH --job-name=nlsr_navmesh
#SBATCH --account=torch_pr_769_tandon_advanced
#SBATCH --partition=cpu_short
#SBATCH --qos=cpu48
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/%x_%j.out
#SBATCH --error=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/%x_%j.err

set -euo pipefail
umask 0022

: "${REPO_ROOT:?export REPO_ROOT to the exact clean producer checkout}"
: "${EXPECTED_COMMIT:?export EXPECTED_COMMIT to its full commit SHA}"
: "${EXPECTED_LAUNCHER_SHA:?export the SHA256 of this committed sbatch file}"
: "${RUN_TAG:?reuse the audited NLSR run tag}"
[[ "${RUN_TAG}" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]] || {
  echo "ABORT: invalid RUN_TAG=${RUN_TAG}" >&2
  exit 2
}
[[ "${EXPECTED_LAUNCHER_SHA}" =~ ^[0-9a-f]{64}$ ]] || {
  echo "ABORT: EXPECTED_LAUNCHER_SHA is not a lowercase SHA256" >&2
  exit 2
}

RESULT_BASE="${RESULT_BASE:-/scratch/yz11502/Research/Nav-axis-uturn-results/nlsr_gapfill_20260807}"
RUN_ROOT="${RESULT_BASE}/${RUN_TAG}"
MANIFEST="${RUN_ROOT}/manifest_multistage/candidate_manifest_multistage.json"
# A source revision is part of the bake run contract.  Keep failed or completed
# attempts from another producer revision immutable instead of trying to
# resume them under new code.
OUTPUT_ROOT="${RUN_ROOT}/navmesh_bake_habitat031_agent030_${EXPECTED_COMMIT:0:12}"
PRODUCER_REL="MemNavData/bake_pinned_navmeshes.py"
AUDITOR_REL="MemNavData/audit_pinned_navmesh_bake.py"
LAUNCHER_REL="MemNavData/slurm_nlsr_navmesh_bake.sbatch"
SETTINGS_REL="MemNavData/navmesh_settings_habitat_0_3_1_agent030.json"
HABITAT_PY="${HABITAT_PY:-/scratch/lg154/conda-envs/habitat/bin/python}"
BASE_SIF="${BASE_SIF:-/share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif}"

EXPECTED_MANIFEST_SHA="bc6bf58536f6c159d1898ac03abe365eadba65c22cd246e39401821962abb34c"
EXPECTED_MANIFEST_SIDECAR_SHA="64de7deaba664264aaa74fb566ec9ae866b277b49930b404903f0ea737374d0c"
EXPECTED_SETTINGS_SHA="62cf1f7a569939e0f8d92045483f76cc008ff71ab7a97e1a447a921722a7dcf9"
EXPECTED_HABITAT_VERSION="0.3.1"
EXPECTED_BINDINGS_SHA="5f0a6a4036d10c6b396193001883cf08ddd9b20f5a90551ab5cfb5be705bd1a8"
EXPECTED_HABITAT_PY_SHA="8e9776d5b77f7d22be61bfebc82a544313837abb7d6ca6b110f4547f73259d89"
EXPECTED_PRODUCER_SHA="44c34381a2ef33f99b5bad0f7cee97e5e4aa3c39bc13dc06fb027d5ea7d2610b"
EXPECTED_AUDITOR_SHA="2ff9eef147e7742f4034386228491275eb90e6a18defe99ee091868ae3c8b477"
EXPECTED_SIF_BYTES="10035580928"
EXPECTED_SIF_SHA="d339d7c3da1c483a2c75e0fa97dd2c9f929a9aace79267e04f6402223942d1b8"

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

command -v singularity >/dev/null || fail "singularity is unavailable"
command -v sha256sum >/dev/null || fail "sha256sum is unavailable"
[[ -d "${REPO_ROOT}/.git" || -f "${REPO_ROOT}/.git" ]] || \
  fail "producer is not a git checkout"
[[ "$(git -C "${REPO_ROOT}" rev-parse HEAD)" == "${EXPECTED_COMMIT}" ]] || \
  fail "producer commit mismatch"
if [[ -n "$(git -C "${REPO_ROOT}" status --porcelain)" ]]; then
  git -C "${REPO_ROOT}" status --short >&2
  fail "producer checkout is not clean"
fi
launcher_sha="$(sha256sum "$0" | awk '{print $1}')"
repository_launcher_sha="$(sha256sum \
  "${REPO_ROOT}/${LAUNCHER_REL}" | awk '{print $1}')"
[[ "${launcher_sha}" == "${EXPECTED_LAUNCHER_SHA}" ]] || \
  fail "spooled launcher SHA mismatch"
[[ "${repository_launcher_sha}" == "${EXPECTED_LAUNCHER_SHA}" ]] || \
  fail "repository launcher SHA mismatch"
producer_sha="$(sha256sum "${REPO_ROOT}/${PRODUCER_REL}" | awk '{print $1}')"
[[ "${producer_sha}" == "${EXPECTED_PRODUCER_SHA}" ]] || \
  fail "navmesh baker source SHA mismatch"
full_sha_is "${REPO_ROOT}/${AUDITOR_REL}" "${EXPECTED_AUDITOR_SHA}" \
  "NavMesh bake auditor"
full_sha_is "${REPO_ROOT}/${SETTINGS_REL}" "${EXPECTED_SETTINGS_SHA}" \
  "canonical NavMesh settings"
full_sha_is "${MANIFEST}" "${EXPECTED_MANIFEST_SHA}" "multistage manifest"
full_sha_is "${MANIFEST}.sha256" "${EXPECTED_MANIFEST_SIDECAR_SHA}" \
  "multistage manifest exact sidecar"
full_sha_is "${HABITAT_PY}" "${EXPECTED_HABITAT_PY_SHA}" "Habitat Python"
[[ -f "${BASE_SIF}" && "$(stat -c '%s' "${BASE_SIF}")" == \
  "${EXPECTED_SIF_BYTES}" ]] || fail "base SIF identity differs"
full_sha_is "${BASE_SIF}" "${EXPECTED_SIF_SHA}" "base SIF"

CONTAINER=(singularity exec
  --cleanenv
  --no-home
  --env "PYTHONNOUSERSITE=1"
  --env "PYTHONPATH=${REPO_ROOT}"
  --env "MAGNUM_LOG=quiet"
  --env "HABITAT_SIM_LOG=quiet"
  -B /scratch/lg154
  -B /scratch/yz11502
  -B /share/apps/images:/share/apps/images:ro
  "${BASE_SIF}")

"${CONTAINER[@]}" "${HABITAT_PY}" - \
  "${EXPECTED_HABITAT_VERSION}" "${EXPECTED_BINDINGS_SHA}" \
  "${MANIFEST}" "${EXPECTED_MANIFEST_SHA}" <<'PY'
import hashlib
import json
from pathlib import Path
import sys
import habitat_sim

version, bindings_sha, manifest_path, manifest_sha = sys.argv[1:]
if habitat_sim.__version__ != version:
    raise SystemExit(f"Habitat version mismatch: {habitat_sim.__version__}")
bindings = Path(habitat_sim._ext.habitat_sim_bindings.__file__)
digest = hashlib.sha256()
with bindings.open("rb") as handle:
    for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
        digest.update(block)
if digest.hexdigest() != bindings_sha:
    raise SystemExit("Habitat bindings SHA mismatch")
path = Path(manifest_path)
if hashlib.sha256(path.read_bytes()).hexdigest() != manifest_sha:
    raise SystemExit("multistage manifest SHA mismatch")
manifest = json.loads(path.read_text())
if manifest.get("summary", {}).get("scene_count") != 50:
    raise SystemExit("multistage scene count changed")
print(json.dumps({"navmesh_bake_preflight": "passed",
                  "habitat_version": version,
                  "bindings_sha256": bindings_sha}, sort_keys=True))
PY

resume_args=()
if [[ -e "${OUTPUT_ROOT}" ]]; then
  resume_args+=(--resume)
fi
stdout_tmp="${RUN_ROOT}/.navmesh_bake_stdout.${SLURM_JOB_ID:-manual}.tmp"
rm -f "${stdout_tmp}"
trap 'rm -f "${stdout_tmp}"' EXIT
"${CONTAINER[@]}" "${HABITAT_PY}" "${REPO_ROOT}/${PRODUCER_REL}" \
  --expert-manifest "${MANIFEST}" \
  --expected-manifest-sha256 "${EXPECTED_MANIFEST_SHA}" \
  --settings-json "${REPO_ROOT}/${SETTINGS_REL}" \
  --expected-settings-sha256 "${EXPECTED_SETTINGS_SHA}" \
  --expected-habitat-version "${EXPECTED_HABITAT_VERSION}" \
  --expected-habitat-bindings-sha256 "${EXPECTED_BINDINGS_SHA}" \
  --output-root "${OUTPUT_ROOT}" \
  "${resume_args[@]}" \
  | tee "${stdout_tmp}"
mv -f "${stdout_tmp}" "${RUN_ROOT}/navmesh_bake_stdout.json"

"${CONTAINER[@]}" "${HABITAT_PY}" "${REPO_ROOT}/${AUDITOR_REL}" \
  --output-root "${OUTPUT_ROOT}" \
  --run-root "${RUN_ROOT}" \
  --expected-parent-manifest-sha256 "${EXPECTED_MANIFEST_SHA}" \
  --expected-settings-sha256 "${EXPECTED_SETTINGS_SHA}" \
  --expected-bindings-sha256 "${EXPECTED_BINDINGS_SHA}" \
  --expected-producer-sha256 "${EXPECTED_PRODUCER_SHA}" \
  --expected-auditor-sha256 "${EXPECTED_AUDITOR_SHA}" \
  --expected-habitat-version "${EXPECTED_HABITAT_VERSION}" \
  --expected-producer-commit "${EXPECTED_COMMIT}" \
  --expected-launcher-sha256 "${EXPECTED_LAUNCHER_SHA}" \
  --producer-path "${REPO_ROOT}/${PRODUCER_REL}" \
  --auditor-path "${REPO_ROOT}/${AUDITOR_REL}" \
  --launcher-path "${REPO_ROOT}/${LAUNCHER_REL}" \
  --habitat-python-path "${HABITAT_PY}" \
  --expected-habitat-python-sha256 "${EXPECTED_HABITAT_PY_SHA}" \
  --base-sif-path "${BASE_SIF}" \
  --expected-base-sif-sha256 "${EXPECTED_SIF_SHA}" \
  --expected-base-sif-bytes "${EXPECTED_SIF_BYTES}" \
  --expected-scene-count 50 \
  --expected-fresh-simulator-repetitions 2

echo "NLSR receipt-backed NavMesh bake passed: ${OUTPUT_ROOT}"
