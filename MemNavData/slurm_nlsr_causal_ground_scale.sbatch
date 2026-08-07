#!/usr/bin/env bash
# Stage 5a: estimate one strictly causal LingBot metric scale per source episode.
# The Goal-B/C manifest is content-pinned; only the prefix before the earliest
# decision is visible.  Submit after the multistage manifest audit succeeds.
#SBATCH --job-name=nlsr_scale
#SBATCH --account=torch_pr_769_tandon_advanced
#SBATCH --partition=h100_tandon
#SBATCH --qos=gpu48
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --output=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/%x_%j.out
#SBATCH --error=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/%x_%j.err

set -euo pipefail
umask 0022

: "${REPO_ROOT:?export REPO_ROOT to the exact clean producer checkout}"
: "${EXPECTED_COMMIT:?export EXPECTED_COMMIT to its full commit SHA}"
: "${RUN_TAG:?reuse the audited NLSR run tag}"
[[ "${RUN_TAG}" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]] || {
  echo "ABORT: invalid RUN_TAG=${RUN_TAG}" >&2
  exit 2
}

RESULT_BASE="${RESULT_BASE:-/scratch/yz11502/Research/Nav-axis-uturn-results/nlsr_gapfill_20260807}"
RUN_ROOT="${RESULT_BASE}/${RUN_TAG}"
MANIFEST="${RUN_ROOT}/manifest_multistage/candidate_manifest_multistage.json"
OUTPUT_ROOT="${RUN_ROOT}/causal_ground_scale"
OUTPUT="${OUTPUT_ROOT}/causal_ground_scale.json"
PRODUCER_REL="MemNavData/build_causal_ground_scale.py"
STREAM_REL="InternNav/internnav/model/basemodel/memnav/lingbot_stream.py"
MEMNAV_PY="${MEMNAV_PY:-/scratch/lg154/conda-envs/memnav/bin/python}"
BASE_SIF="${BASE_SIF:-/share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif}"
OVERLAY="${OVERLAY:-/scratch/lg154/Research/datasets/_overlays/mp3d_revisit_v0_pt1.sqf}"
LINGBOT_REPO="${LINGBOT_REPO:-/scratch/lg154/Research/Nav/NavDP/baselines/memnav/lingbot-map}"
WEIGHTS="${LINGBOT_REPO}/weights/lingbot-map-long.pt"
RAW_SCENE="${RUN_ROOT}/raw/mp3d_3leg/YmJkqBEsHnH"
EPISODE_ROOT="/mp3d_revisit_v0/vln_n1/traj_data/mp3d_3leg"

EXPECTED_MANIFEST_SHA="bc6bf58536f6c159d1898ac03abe365eadba65c22cd246e39401821962abb34c"
EXPECTED_PRODUCER_SHA="523126bb1f56f4754d0b8a0a34b327e3d334fbc01bd0ae51f12c27c50ffebf20"
EXPECTED_FRONTIER_BUILDER_SHA="d2e5889736488f21449fd5d9a054d945cd17bdb509bc311eb23dd58c4de23aa5"
EXPECTED_FLOW_ROUTER_SHA="4896a77fa35871588f4add5c83c10e48fbb9bf45a15845f0a5d4332bc63a0035"
EXPECTED_PRODUCER_BUNDLE_SHA="8da2a4c42de010e699c5ab65a2f6bbc9600c29acc96b71ef82d89126d8b4ca26"
EXPECTED_STREAM_SHA="b5142b76f9062634db826a0e7f2c3da19e4b07e21c2325ecbc60a44cc52370b0"
EXPECTED_LINGBOT_COMMIT="7ff6f3ed0913d4d326f8f13bbb429c4ffc0195c2"
EXPECTED_WEIGHT_SHA="832bc82cbae0bc9bbe946ef5ee1f7226abd8c0e183ccf8beddbb3d133576f409"
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

command -v singularity >/dev/null || fail "singularity is unavailable"
command -v nvidia-smi >/dev/null || fail "nvidia-smi is unavailable"
command -v flock >/dev/null || fail "flock is unavailable"
[[ -d "${REPO_ROOT}/.git" || -f "${REPO_ROOT}/.git" ]] || \
  fail "producer is not a git checkout"
[[ "$(git -C "${REPO_ROOT}" rev-parse HEAD)" == "${EXPECTED_COMMIT}" ]] || \
  fail "producer commit mismatch"
if [[ -n "$(git -C "${REPO_ROOT}" status --porcelain)" ]]; then
  git -C "${REPO_ROOT}" status --short >&2
  fail "producer checkout is not clean"
fi
full_sha_is "${REPO_ROOT}/${PRODUCER_REL}" "${EXPECTED_PRODUCER_SHA}" \
  "causal-scale producer"
full_sha_is "${REPO_ROOT}/MemNavData/build_novel_frontier_candidates.py" \
  "${EXPECTED_FRONTIER_BUILDER_SHA}" "causal-scale frontier dependency"
full_sha_is "${REPO_ROOT}/MemNavData/flow_cache_routing.py" \
  "${EXPECTED_FLOW_ROUTER_SHA}" "causal-scale route dependency"
full_sha_is "${REPO_ROOT}/${STREAM_REL}" "${EXPECTED_STREAM_SHA}" \
  "LingBot stream source"
full_sha_is "${MANIFEST}" "${EXPECTED_MANIFEST_SHA}" "multistage manifest"
full_sha_is "${MEMNAV_PY}" "${EXPECTED_MEMNAV_PY_SHA}" "MemNav Python"
full_sha_is "${WEIGHTS}" "${EXPECTED_WEIGHT_SHA}" "LingBot weights"
[[ "$(awk 'NR==1 {print $1}' "${MANIFEST}.sha256")" == \
  "${EXPECTED_MANIFEST_SHA}" ]] || fail "manifest SHA sidecar mismatch"
[[ -f "${BASE_SIF}" && "$(stat -c '%s' "${BASE_SIF}")" == \
  "${EXPECTED_SIF_BYTES}" ]] || fail "base SIF identity differs"
[[ "$(dd if="${BASE_SIF}" bs=1048576 count=1 status=none | sha256sum | awk '{print $1}')" == \
  "${EXPECTED_SIF_HEAD_SHA}" ]] || fail "base SIF header SHA mismatch"
[[ -f "${OVERLAY}" && "$(stat -c '%s' "${OVERLAY}")" == \
  "${EXPECTED_OVERLAY_BYTES}" ]] || fail "overlay identity differs"
[[ "$(dd if="${OVERLAY}" bs=1048576 count=1 status=none | sha256sum | awk '{print $1}')" == \
  "${EXPECTED_OVERLAY_HEAD_SHA}" ]] || fail "overlay header SHA mismatch"
[[ -d "${RAW_SCENE}" ]] || fail "audited YmJ raw scene is absent"

lingbot_commit="$(git -c safe.directory="${LINGBOT_REPO}" \
  -C "${LINGBOT_REPO}" rev-parse HEAD)"
[[ "${lingbot_commit}" == "${EXPECTED_LINGBOT_COMMIT}" ]] || \
  fail "LingBot commit mismatch: ${lingbot_commit}"
if [[ -n "$(git -c safe.directory="${LINGBOT_REPO}" \
    -C "${LINGBOT_REPO}" status --porcelain)" ]]; then
  git -c safe.directory="${LINGBOT_REPO}" \
    -C "${LINGBOT_REPO}" status --short >&2
  fail "LingBot checkout is not clean"
fi
export GIT_CONFIG_COUNT=1
export GIT_CONFIG_KEY_0=safe.directory
export GIT_CONFIG_VALUE_0="${LINGBOT_REPO}"

exec 9>"${RUN_ROOT}/.causal_ground_scale.lock"
flock -n 9 || fail "another causal-scale stage is active"
[[ ! -e "${OUTPUT_ROOT}" ]] || fail "output already exists: ${OUTPUT_ROOT}"

CONTAINER=(singularity exec --nv
  --overlay "${OVERLAY}:ro"
  -B /scratch/lg154
  -B /scratch/yz11502
  -B "${RAW_SCENE}:${EPISODE_ROOT}/YmJkqBEsHnH:ro"
  "${BASE_SIF}")

"${CONTAINER[@]}" "${MEMNAV_PY}" - "${MANIFEST}" <<'PY'
import importlib.util
import json
from pathlib import Path
import sys
import torch

missing = [name for name in ("numpy", "PIL", "torch")
           if importlib.util.find_spec(name) is None]
if missing:
    raise SystemExit(f"missing causal-scale dependencies: {missing}")
if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable in the exact causal-scale runtime")
manifest = json.loads(Path(sys.argv[1]).read_text())
summary = manifest.get("summary", {})
expected = {"scene_count": 50, "episode_count": 100, "sample_count": 600}
for key, value in expected.items():
    if summary.get(key) != value:
        raise SystemExit(f"manifest {key} changed: {summary.get(key)} != {value}")
print(json.dumps({"causal_scale_preflight": "passed", **expected}, sort_keys=True))
PY

mkdir -p "${OUTPUT_ROOT}"
echo "host=$(hostname) job=${SLURM_JOB_ID:-manual} run_tag=${RUN_TAG}"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
"${CONTAINER[@]}" "${MEMNAV_PY}" "${REPO_ROOT}/${PRODUCER_REL}" \
  --manifest "${MANIFEST}" \
  --expected-manifest-sha "${EXPECTED_MANIFEST_SHA}" \
  --lingbot-repo "${LINGBOT_REPO}" \
  --weights "${WEIGHTS}" \
  --expected-lingbot-commit "${EXPECTED_LINGBOT_COMMIT}" \
  --expected-weights-sha "${EXPECTED_WEIGHT_SHA}" \
  --expected-stream-source-sha "${EXPECTED_STREAM_SHA}" \
  --prefix-frame-cap 64 \
  --num-scale-frames 8 \
  --window 32 \
  --max-frame-num 4096 \
  --device cuda \
  --out "${OUTPUT}" \
  | tee "${OUTPUT_ROOT}/producer_stdout.json"

"${CONTAINER[@]}" "${MEMNAV_PY}" - \
  "${OUTPUT}" "${EXPECTED_MANIFEST_SHA}" "${EXPECTED_PRODUCER_BUNDLE_SHA}" \
  "${EXPECTED_PRODUCER_SHA}" "${EXPECTED_FRONTIER_BUILDER_SHA}" \
  "${EXPECTED_FLOW_ROUTER_SHA}" <<'PY'
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
    raise SystemExit("causal-scale artifact is not canonical")
digest = hashlib.sha256(raw).hexdigest()
if Path(f"{path}.sha256").read_text().split()[0] != digest:
    raise SystemExit("causal-scale sidecar mismatch")
if record.get("schema_version") != "nlsr_v2_causal_ground_scale_v1":
    raise SystemExit("causal-scale schema changed")
provenance = record.get("provenance", {})
if provenance.get("input_manifest_sha256") != sys.argv[2]:
    raise SystemExit("causal-scale manifest binding changed")
if provenance.get("producer_source_sha256") != sys.argv[3]:
    raise SystemExit("causal-scale producer binding changed")
source_files = provenance.get("producer_source_files", {})
expected_sources = {
    "build_causal_ground_scale.py": sys.argv[4],
    "build_novel_frontier_candidates.py": sys.argv[5],
    "flow_cache_routing.py": sys.argv[6],
}
if source_files != expected_sources:
    raise SystemExit("causal-scale producer source-file bindings changed")
summary = record.get("summary", {})
expected = {
    "scene_count": 50,
    "episode_count": 100,
    "valid_episode_count": 100,
    "invalid_episode_count": 0,
    "future_frames_consumed": 0,
}
for key, value in expected.items():
    if summary.get(key) != value:
        raise SystemExit(f"unexpected {key}: {summary.get(key)} != {value}")
print(json.dumps({"causal_scale_cross_audit": "passed", "sha256": digest,
                  **expected}, sort_keys=True))
PY

echo "NLSR causal ground scale passed: ${OUTPUT}"
