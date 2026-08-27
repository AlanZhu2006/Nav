#!/usr/bin/env bash
# Reproduce the only missing strict-split scene with the historical generator.
# This is stage 1 of a three-job afterok DAG.  It fails unless the two retained
# historical summaries match exactly; therefore later jobs cannot consume a
# merely plausible replacement episode.
#SBATCH --job-name=nlsr_ymj_raw
#SBATCH --account=torch_pr_769_tandon_advanced
#SBATCH --partition=a100_tandon
#SBATCH --qos=gpu48
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --output=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/%x_%j.out
#SBATCH --error=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/%x_%j.err

set -euo pipefail
umask 0022

: "${REPO_ROOT:?export REPO_ROOT to the clean committed wrapper worktree}"
: "${EXPECTED_COMMIT:?export EXPECTED_COMMIT to the full SHA of that worktree}"
: "${RUN_TAG:?export one new RUN_TAG and reuse it for all three DAG jobs}"
[[ "${RUN_TAG}" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]] || {
  echo "ABORT: invalid RUN_TAG=${RUN_TAG}" >&2
  exit 2
}

RESULT_BASE="${RESULT_BASE:-/scratch/yz11502/Research/Nav-axis-uturn-results/nlsr_gapfill_20260807}"
RUN_ROOT="${RESULT_BASE}/${RUN_TAG}"
RAW_ROOT="${RUN_ROOT}/raw"
GENERATOR_ROOT="${GENERATOR_ROOT:-/scratch/yz11502/Research/Nav-gen-d6c8b56}"
GENERATOR_REL="MemNavData/generate_twoleg.py"
AUDITOR_REL="MemNavData/audit_nlsr_historical_gapfill.py"
SCENE="YmJkqBEsHnH"
ASSET_ROOT="/scratch/lg154/Research/datasets/mp3d/mp3d/${SCENE}"
SCENE_GLB="${ASSET_ROOT}/${SCENE}.glb"
NAVMESH="${ASSET_ROOT}/${SCENE}.navmesh"
HABITAT_PY="${HABITAT_PY:-/scratch/lg154/conda-envs/habitat/bin/python}"
BASE_SIF="${BASE_SIF:-/share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif}"

EXPECTED_GENERATOR_COMMIT="d6c8b56f6c14a677a1398e3600f2ee151222caec"
EXPECTED_GENERATOR_SHA="b2e05eb1f43780ef47696fa6d101515fe15154e159049e31d05f2b4996e1c3f9"
EXPECTED_AUDITOR_SHA="90809aa2aba3c6d130b0fbecc10f86c2f6a0517776e17360791b812fbc0d64e2"
EXPECTED_GLB_BYTES="23896396"
EXPECTED_GLB_SHA="a998d082cc6d05f090439f7afec62b7548cd24a54021c240f33bc04138c367bb"
EXPECTED_NAVMESH_BYTES="16460"
EXPECTED_NAVMESH_SHA="0ae236927f145b4586656491e932a913a4dc2f370a70d6ad999b7bc2733061bf"
EXPECTED_HABITAT_PY_SHA="8e9776d5b77f7d22be61bfebc82a544313837abb7d6ca6b110f4547f73259d89"
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

size_and_sha_is() {
  local path="$1" expected_bytes="$2" expected_sha="$3" label="$4"
  [[ -f "${path}" ]] || fail "missing ${label}: ${path}"
  [[ "$(stat -c '%s' "${path}")" == "${expected_bytes}" ]] || \
    fail "${label} size mismatch: ${path}"
  full_sha_is "${path}" "${expected_sha}" "${label}"
}

clean_commit_is() {
  local root="$1" expected="$2" label="$3" actual
  [[ -d "${root}/.git" || -f "${root}/.git" ]] || \
    fail "${label} is not a git worktree: ${root}"
  actual="$(git -C "${root}" rev-parse HEAD)"
  [[ "${actual}" == "${expected}" ]] || \
    fail "${label} commit mismatch: ${actual} != ${expected}"
  if [[ -n "$(git -C "${root}" status --porcelain)" ]]; then
    git -C "${root}" status --short >&2
    fail "${label} worktree is not clean"
  fi
}

command -v singularity >/dev/null || fail "singularity is unavailable"
command -v nvidia-smi >/dev/null || fail "nvidia-smi is unavailable"
clean_commit_is "${REPO_ROOT}" "${EXPECTED_COMMIT}" "wrapper"
clean_commit_is "${GENERATOR_ROOT}" "${EXPECTED_GENERATOR_COMMIT}" "generator"
full_sha_is "${GENERATOR_ROOT}/${GENERATOR_REL}" \
  "${EXPECTED_GENERATOR_SHA}" "historical generator"
full_sha_is "${REPO_ROOT}/${AUDITOR_REL}" \
  "${EXPECTED_AUDITOR_SHA}" "raw auditor"
size_and_sha_is "${SCENE_GLB}" "${EXPECTED_GLB_BYTES}" \
  "${EXPECTED_GLB_SHA}" "scene GLB"
size_and_sha_is "${NAVMESH}" "${EXPECTED_NAVMESH_BYTES}" \
  "${EXPECTED_NAVMESH_SHA}" "scene navmesh"
full_sha_is "${HABITAT_PY}" "${EXPECTED_HABITAT_PY_SHA}" "Habitat Python"
[[ -f "${BASE_SIF}" ]] || fail "base SIF is absent: ${BASE_SIF}"
[[ "$(stat -c '%s' "${BASE_SIF}")" == "${EXPECTED_SIF_BYTES}" ]] || \
  fail "base SIF size mismatch"
sif_head_sha="$(dd if="${BASE_SIF}" bs=1048576 count=1 status=none | \
  sha256sum | awk '{print $1}')"
[[ "${sif_head_sha}" == "${EXPECTED_SIF_HEAD_SHA}" ]] || \
  fail "base SIF header SHA mismatch: ${sif_head_sha}"

# Prove all Python dependencies before claiming the one-shot output root.
CONTAINER=(singularity exec --nv -B /scratch/lg154 -B /scratch/yz11502
  "${BASE_SIF}" "${HABITAT_PY}")
"${CONTAINER[@]}" - <<'PY'
import importlib.util
import json
import platform

required = ("habitat_sim", "numpy", "pandas", "PIL", "pyarrow", "quaternion", "scipy")
missing = [name for name in required if importlib.util.find_spec(name) is None]
if missing:
    raise SystemExit(f"missing generator/audit Python dependencies: {missing}")
print(json.dumps({"python": platform.python_version(), "dependencies": required}))
PY

mkdir -p "${RESULT_BASE}"
mkdir "${RUN_ROOT}" || \
  fail "RUN_TAG already exists or could not be claimed: ${RUN_ROOT}"

export PREFLIGHT_REPO_COMMIT="${EXPECTED_COMMIT}"
export PREFLIGHT_GENERATOR_COMMIT="${EXPECTED_GENERATOR_COMMIT}"
export PREFLIGHT_GENERATOR_SHA="${EXPECTED_GENERATOR_SHA}"
export PREFLIGHT_SIF_HEAD_SHA="${sif_head_sha}"
export PREFLIGHT_RUN_TAG="${RUN_TAG}"
"${CONTAINER[@]}" - "${RUN_ROOT}/dependency_preflight.json" <<'PY'
import json
import os
from pathlib import Path
import sys

record = {
    "status": "dependencies_pinned",
    "run_tag": os.environ["PREFLIGHT_RUN_TAG"],
    "wrapper_commit": os.environ["PREFLIGHT_REPO_COMMIT"],
    "generator_commit": os.environ["PREFLIGHT_GENERATOR_COMMIT"],
    "generator_sha256": os.environ["PREFLIGHT_GENERATOR_SHA"],
    "base_sif_head_sha256": os.environ["PREFLIGHT_SIF_HEAD_SHA"],
}
with Path(sys.argv[1]).open("x", encoding="utf-8") as handle:
    json.dump(record, handle, indent=2, sort_keys=True)
    handle.write("\n")
PY

echo "host=$(hostname) job=${SLURM_JOB_ID:-manual} run_tag=${RUN_TAG}"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

# n2=15 is intentional: it advances the single RNG through the exact historical
# dual-mode call order before either three-leg episode is sampled.
MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet \
"${CONTAINER[@]}" "${GENERATOR_ROOT}/${GENERATOR_REL}" \
  --scene "${SCENE_GLB}" \
  --navmesh "${NAVMESH}" \
  --out "${RAW_ROOT}" \
  --n2 15 \
  --n3 2 \
  --seed 50 \
  --goal_jitter_pos 3.0 \
  --goal_tries 100 \
  --window 32 \
  --num_scale 8 \
  2>&1 | tee "${RUN_ROOT}/generator.log"

# This exact gate is the stage's success condition.  A different yet valid pair
# exits nonzero and therefore cannot satisfy an afterok dependency.
"${CONTAINER[@]}" "${REPO_ROOT}/${AUDITOR_REL}" \
  --raw-root "${RAW_ROOT}" \
  --generator-root "${GENERATOR_ROOT}" \
  --generator-commit "${EXPECTED_GENERATOR_COMMIT}" \
  --generator-sha256 "${EXPECTED_GENERATOR_SHA}" \
  --scene-glb "${SCENE_GLB}" \
  --navmesh "${NAVMESH}" \
  --out "${RUN_ROOT}/raw_audit.json" \
  | tee "${RUN_ROOT}/raw_audit_stdout.json"

echo "NLSR historical raw gap fill passed exact summary audit: ${RUN_ROOT}"
