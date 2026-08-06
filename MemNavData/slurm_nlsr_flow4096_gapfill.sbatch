#!/usr/bin/env bash
# Stage 2: afterok-only generation of three missing flow-cache pairs and a
# provenance-locked, zero-copy 97-base + 3-patch route registry.
#SBATCH --job-name=nlsr_flow4096
#SBATCH --account=torch_pr_769_tandon_advanced
#SBATCH --partition=h100_tandon
#SBATCH --qos=gpu48
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --output=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/%x_%j.out
#SBATCH --error=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/%x_%j.err

set -euo pipefail
umask 0022

: "${REPO_ROOT:?export REPO_ROOT to the clean committed wrapper worktree}"
: "${EXPECTED_COMMIT:?export EXPECTED_COMMIT to the full SHA of that worktree}"
: "${RUN_TAG:?reuse the exact RUN_TAG from the successful raw job}"
[[ "${RUN_TAG}" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]] || {
  echo "ABORT: invalid RUN_TAG=${RUN_TAG}" >&2
  exit 2
}

RESULT_BASE="${RESULT_BASE:-/scratch/yz11502/Research/Nav-axis-uturn-results/nlsr_gapfill_20260807}"
RUN_ROOT="${RESULT_BASE}/${RUN_TAG}"
RAW_ROOT="${RUN_ROOT}/raw"
RAW_SCENE="${RAW_ROOT}/mp3d_3leg/YmJkqBEsHnH"
RAW_AUDIT="${RUN_ROOT}/raw_audit.json"
FLOW_CODE_ROOT="${FLOW_CODE_ROOT:-/scratch/yz11502/Research/Nav-flow-b3dfae9d}"
PRECOMPUTE_REL="InternNav/scripts/dataset_converters/precompute_lingbot_features.py"
CACHE_SCHEMA_REL="InternNav/internnav/model/basemodel/memnav/cache_schema.py"
MERGER_REL="MemNavData/build_nlsr_merged_flow.py"
ROUTER_REL="MemNavData/flow_cache_routing.py"
SPLIT_REL="MemNavData/router_multiscene_split_20260805.json"
MEMNAV_PY="${MEMNAV_PY:-/scratch/lg154/conda-envs/memnav/bin/python}"
BASE_SIF="${BASE_SIF:-/share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif}"
OVERLAY="${OVERLAY:-/scratch/lg154/Research/datasets/_overlays/mp3d_revisit_v0_pt1.sqf}"
LINGBOT_REPO="${LINGBOT_REPO:-/scratch/lg154/Research/Nav/NavDP/baselines/memnav/lingbot-map}"
WEIGHTS="${LINGBOT_REPO}/weights/lingbot-map-long.pt"
OFFICIAL_FLOW_ROOT="/scratch/lg154/Research/datasets/mp3d_revisit_v0_feat_flowgate/vln_n1/traj_data/mp3d_3leg"
PATCH_OUT_ROOT="${RUN_ROOT}/flow_patch/vln_n1/traj_data"
PATCH_FLOW_ROOT="${PATCH_OUT_ROOT}/mp3d_3leg"
FLOW_ROUTE_ROOT="${RUN_ROOT}/flow_routes"
LIST_ROOT="${RUN_ROOT}/flow_lists"
EPISODE_ROOT="/mp3d_revisit_v0/vln_n1/traj_data/mp3d_3leg"
ROOT_DIRS="/mp3d_revisit_v0/vln_n1/traj_data"

EXPECTED_FLOW_CODE_COMMIT="b3dfae9d5647165927a5ade0d8193fe1d0d3c19d"
EXPECTED_PRECOMPUTE_SHA="ddf1e425318ea59fed51bb78d5d0e52d54860cab0c60589fcae1b264a87fcbb2"
EXPECTED_CACHE_SCHEMA_SHA="412fdc9ba11947406297ba639790b9d3acf7d9c3ab6125dea8e1a1246e38f227"
EXPECTED_MERGER_SHA="75d327a38823baf36635c0c8ce74c7a474744dbd21a2d91be94148d1bec5377c"
EXPECTED_ROUTER_SHA="4896a77fa35871588f4add5c83c10e48fbb9bf45a15845f0a5d4332bc63a0035"
EXPECTED_SPLIT_SHA="97309c183e25cb3dd65472908748d55a94798a636db6157ab6fe120fca05cf7a"
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
command -v flock >/dev/null || fail "flock is unavailable"
clean_commit_is "${REPO_ROOT}" "${EXPECTED_COMMIT}" "wrapper"
clean_commit_is "${FLOW_CODE_ROOT}" "${EXPECTED_FLOW_CODE_COMMIT}" "flow-code"
full_sha_is "${FLOW_CODE_ROOT}/${PRECOMPUTE_REL}" \
  "${EXPECTED_PRECOMPUTE_SHA}" "precompute script"
full_sha_is "${REPO_ROOT}/${CACHE_SCHEMA_REL}" \
  "${EXPECTED_CACHE_SCHEMA_SHA}" "cache schema"
full_sha_is "${REPO_ROOT}/${MERGER_REL}" \
  "${EXPECTED_MERGER_SHA}" "flow merger"
full_sha_is "${REPO_ROOT}/${ROUTER_REL}" \
  "${EXPECTED_ROUTER_SHA}" "flow route resolver"
full_sha_is "${REPO_ROOT}/${SPLIT_REL}" "${EXPECTED_SPLIT_SHA}" "strict split"
full_sha_is "${MEMNAV_PY}" "${EXPECTED_MEMNAV_PY_SHA}" "MemNav Python"
full_sha_is "${WEIGHTS}" "${EXPECTED_WEIGHT_SHA}" "LingBot weights"

# The shared LingBot checkout belongs to another Unix user.  Process-scoped Git
# config keeps the provenance subprocess auditable without mutating ~/.gitconfig.
lingbot_commit="$(git -c safe.directory="${LINGBOT_REPO}" \
  -C "${LINGBOT_REPO}" rev-parse HEAD)"
[[ "${lingbot_commit}" == "${EXPECTED_LINGBOT_COMMIT}" ]] || \
  fail "LingBot commit mismatch: ${lingbot_commit}"
if [[ -n "$(git -c safe.directory="${LINGBOT_REPO}" \
    -C "${LINGBOT_REPO}" status --porcelain)" ]]; then
  git -c safe.directory="${LINGBOT_REPO}" \
    -C "${LINGBOT_REPO}" status --short >&2
  fail "LingBot worktree is not clean"
fi
export GIT_CONFIG_COUNT=1
export GIT_CONFIG_KEY_0=safe.directory
export GIT_CONFIG_VALUE_0="${LINGBOT_REPO}"

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

[[ -d "${RUN_ROOT}" ]] || fail "raw-stage RUN_ROOT is absent: ${RUN_ROOT}"
[[ -d "${RAW_SCENE}" ]] || fail "audited YmJ raw scene is absent: ${RAW_SCENE}"
[[ -f "${RAW_AUDIT}" && -f "${RAW_AUDIT}.sha256" ]] || \
  fail "raw audit artifact pair is absent"
raw_audit_sha="$(sha256sum "${RAW_AUDIT}" | awk '{print $1}')"
[[ "$(awk 'NR==1 {print $1}' "${RAW_AUDIT}.sha256")" == "${raw_audit_sha}" ]] || \
  fail "raw audit SHA sidecar mismatch"
exec 9>"${RUN_ROOT}/.flow_stage.lock"
flock -n 9 || fail "another flow stage is active for RUN_TAG=${RUN_TAG}"
[[ ! -e "${FLOW_ROUTE_ROOT}" ]] || \
  fail "flow route output already exists: ${FLOW_ROUTE_ROOT}"

CONTAINER=(singularity exec --nv
  --overlay "${OVERLAY}:ro"
  -B /scratch/lg154
  -B /scratch/yz11502
  -B "${RAW_SCENE}:${EPISODE_ROOT}/YmJkqBEsHnH:ro"
  "${BASE_SIF}")

# Same container, overlay, bind and interpreter as the real precompute.
"${CONTAINER[@]}" "${MEMNAV_PY}" - \
  "${RAW_AUDIT}" "${EPISODE_ROOT}" <<'PY'
import importlib.util
import json
from pathlib import Path
import sys

required = ("cv2", "numpy", "PIL", "torch", "tqdm")
missing = [name for name in required if importlib.util.find_spec(name) is None]
if missing:
    raise SystemExit(f"missing flow-precompute dependencies: {missing}")
audit = json.loads(Path(sys.argv[1]).read_text())
expected = {
    "episode_0000": {"n_frames": 871, "switches": [218, 548]},
    "episode_0001": {"n_frames": 591, "switches": [192, 296]},
}
if audit.get("status") != "audited_historical_summary_match":
    raise SystemExit("raw audit status is not successful")
if audit.get("historical_three_leg_reference") != expected:
    raise SystemExit("raw audit historical summaries differ")
for episode, summary in expected.items():
    meta = Path(sys.argv[2]) / "YmJkqBEsHnH" / episode / "meta/gen_meta.json"
    record = json.loads(meta.read_text())
    actual = {"n_frames": record.get("n_frames"), "switches": record.get("switches")}
    if actual != summary:
        raise SystemExit(f"bound raw summary mismatch for {episode}: {actual}")
print(json.dumps({"flow_preflight": "passed", "raw_exact_gate": expected}, sort_keys=True))
PY

mkdir -p "${PATCH_OUT_ROOT}" "${LIST_ROOT}"
printf '%s\n' 'mp3d_3leg/B6ByNegPMKs/episode_0001' > \
  "${LIST_ROOT}/b6_ep1.txt"
printf '%s\n' 'mp3d_3leg/YmJkqBEsHnH/episode_0000' > \
  "${LIST_ROOT}/ymj_ep0.txt"
printf '%s\n' 'mp3d_3leg/YmJkqBEsHnH/episode_0001' > \
  "${LIST_ROOT}/ymj_ep1.txt"

echo "host=$(hostname) job=${SLURM_JOB_ID:-manual} run_tag=${RUN_TAG}"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

run_precompute() {
  local name="$1" threshold="$2" list="$3"
  echo "precompute start: ${name} threshold=${threshold} $(date --iso-8601=seconds)"
  "${CONTAINER[@]}" "${MEMNAV_PY}" \
    "${FLOW_CODE_ROOT}/${PRECOMPUTE_REL}" \
    --root_dirs "${ROOT_DIRS}" \
    --out_root "${PATCH_OUT_ROOT}" \
    --lingbot_repo "${LINGBOT_REPO}" \
    --weights "${WEIGHTS}" \
    --trajectory_list "${list}" \
    --flow_threshold "${threshold}" \
    --max_non_keyframe_gap 30 \
    --skip_scale \
    --use_sdpa \
    --kv_cache_sliding_window 32 \
    --num_scale_frames 8 \
    --max_frame_num 4096 \
    --num_shards 1 \
    --shard 0 \
    --limit 0 \
    2>&1 | tee "${RUN_ROOT}/flow_${name}.log"
}

run_precompute b6_ep1 60 "${LIST_ROOT}/b6_ep1.txt"
run_precompute ymj_ep0 25 "${LIST_ROOT}/ymj_ep0.txt"
run_precompute ymj_ep1 20 "${LIST_ROOT}/ymj_ep1.txt"

"${CONTAINER[@]}" "${MEMNAV_PY}" \
  "${REPO_ROOT}/${MERGER_REL}" \
  --split-manifest "${REPO_ROOT}/${SPLIT_REL}" \
  --episode-root "${EPISODE_ROOT}" \
  --official-flow-root "${OFFICIAL_FLOW_ROOT}" \
  --patch-flow-root "${PATCH_FLOW_ROOT}" \
  --flow-route-root "${FLOW_ROUTE_ROOT}" \
  --cache-schema "${REPO_ROOT}/${CACHE_SCHEMA_REL}" \
  --raw-audit "${RAW_AUDIT}" \
  | tee "${RUN_ROOT}/flow_route_stdout.json"

echo "NLSR flow4096 patch and zero-copy routing passed: ${FLOW_ROUTE_ROOT}"
