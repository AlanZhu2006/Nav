#!/usr/bin/env bash
# Stage A: materialize the exact LingBot DINO embedding universe.  This stage
# has no Habitat or geometry-label authority and is resumable by signed shards.
#SBATCH --job-name=nlsr_teacher_a
#SBATCH --account=torch_pr_769_tandon_advanced
#SBATCH --partition=a100_tandon
#SBATCH --qos=gpu48
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:1
#SBATCH --requeue
#SBATCH --output=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/%x_%j.out
#SBATCH --error=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/%x_%j.err

set -euo pipefail
umask 0022

: "${REPO_ROOT:?export REPO_ROOT to the exact clean Stage-A checkout}"
: "${EXPECTED_COMMIT:?export its full commit SHA}"
: "${EXPECTED_LAUNCHER_SHA:?export the committed launcher SHA256}"
: "${RUN_TAG:?export the audited NLSR run tag}"

RESULT_BASE="${RESULT_BASE:-/scratch/yz11502/Research/Nav-axis-uturn-results/nlsr_gapfill_20260807}"
RUN_ROOT="${RESULT_BASE}/${RUN_TAG}"
MANIFEST="${RUN_ROOT}/manifest_multistage/candidate_manifest_multistage.json"
ACCEPTANCE_ROOT="${RUN_ROOT}/causal_ground_scale_acceptance_c8f499dc9c4f_f6f28e20b9f2"
ACCEPTANCE="${ACCEPTANCE_ROOT}/causal_ground_scale_acceptance.json"
OUTPUT_ROOT="${RUN_ROOT}/causal_covisibility_teacher_${EXPECTED_COMMIT:0:12}_bc6bf58536f6"
PROGRESS_ROOT="${OUTPUT_ROOT}/stage_a_progress"
EMBEDDING_BUNDLE="${OUTPUT_ROOT}/dino_embedding_bundle"
STDOUT_TMP="${OUTPUT_ROOT}/.stage_a_stdout.${SLURM_JOB_ID:-manual}.tmp"
STDOUT_FINAL="${OUTPUT_ROOT}/stage_a_stdout.json"

ENTRYPOINT_REL="MemNavData/build_manifest_causal_dino_embeddings.py"
PRODUCER_REL="MemNavData/build_manifest_causal_covisibility_teacher.py"
EXACT_LOADER_REL="MemNavData/diag_distill_geometry_router.py"
LAUNCHER_REL="MemNavData/slurm_nlsr_causal_teacher_embeddings.sbatch"
MEMNAV_PY="/scratch/lg154/conda-envs/memnav/bin/python"
EXPECTED_PYTHON_TARGET="/scratch/lg154/conda-envs/memnav/bin/python3.10"
BASE_SIF="/share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif"
OVERLAY="/scratch/lg154/Research/datasets/_overlays/mp3d_revisit_v0_pt1.sqf"
LINGBOT_REPO="/scratch/lg154/Research/Nav/NavDP/baselines/memnav/lingbot-map"
WEIGHTS="${LINGBOT_REPO}/weights/lingbot-map-long.pt"
RAW_SCENE="${RUN_ROOT}/raw/mp3d_3leg/YmJkqBEsHnH"
EPISODE_ROOT="/mp3d_revisit_v0/vln_n1/traj_data/mp3d_3leg"

EXPECTED_MANIFEST_SHA="bc6bf58536f6c159d1898ac03abe365eadba65c22cd246e39401821962abb34c"
EXPECTED_ACCEPTANCE_SHA="0ced1b3a56fe0e089bad35c5807c219da27daad48e6bbd4bb29436ef2b8eb3ac"
EXPECTED_PRODUCER_SHA="5bcba961491d2d2b6750d477ac9499b45c3b0eae90265b6ab1062d658b686aa1"
EXPECTED_ENTRYPOINT_SHA="f3775608491e0748b67ec33bb8f5ba1635e8ac9b4be3297f6544cf0b76d7e6bf"
EXPECTED_EXACT_LOADER_SHA="34f63e4c8a2e2225f8e4fbec06a560e72014cc1aa4797295a9a19291a00bec00"
EXPECTED_DINO_VIT_SHA="a5237d722a5ce205d13f976a405b1f611c419dace6d3798ce3d4aff93a8a46df"
EXPECTED_DINO_PREPROCESSOR_SHA="25d8edb5bed6f1e64d788b1cfe6ffbc34b63b39fb78235b2e7921a31fdb56bdb"
EXPECTED_LINGBOT_COMMIT="7ff6f3ed0913d4d326f8f13bbb429c4ffc0195c2"
EXPECTED_LINGBOT_TREE="479d4192e8f4fabb6bd1687e2fbd1258de924fe7"
EXPECTED_WEIGHTS_SHA="832bc82cbae0bc9bbe946ef5ee1f7226abd8c0e183ccf8beddbb3d133576f409"
EXPECTED_PYTHON_SHA="cf18c572f59ed93caec89258837eb1cbc5b4a6893e272888c3abebb90ae5ff86"
EXPECTED_SIF_BYTES="10035580928"
EXPECTED_SIF_HEAD_SHA="990c1377491fbf58c1d7d37bcf8043249826bc7a094eacc4c7a443b9905126c7"
EXPECTED_OVERLAY_BYTES="128854888448"
EXPECTED_OVERLAY_HEAD_SHA="43d803c2b39735e764f0c6df8f7c40a7035a1c23789e22fcc9831d29928c801b"

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

prefix_identity_is() {
  local path="$1" bytes="$2" head_sha="$3" label="$4" actual
  [[ -f "${path}" && ! -L "${path}" && \
    "$(stat -c '%s' "${path}")" == "${bytes}" ]] || fail "${label} byte identity changed"
  actual="$(dd if="${path}" bs=1048576 count=1 status=none | sha256sum | awk '{print $1}')"
  [[ "${actual}" == "${head_sha}" ]] || fail "${label} header SHA changed"
}

pinned_python_is() {
  local resolved actual
  [[ -L "${MEMNAV_PY}" && "$(readlink "${MEMNAV_PY}")" == "python3.10" ]] || \
    fail "MemNav Python symlink changed"
  resolved="$(readlink -f "${MEMNAV_PY}")"
  [[ "${resolved}" == "${EXPECTED_PYTHON_TARGET}" ]] || \
    fail "resolved MemNav Python path changed: ${resolved}"
  actual="$(sha256sum "${resolved}" | awk '{print $1}')"
  [[ "${actual}" == "${EXPECTED_PYTHON_SHA}" ]] || fail "MemNav Python SHA changed"
}

[[ "${RUN_TAG}" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]] || fail "invalid RUN_TAG"
[[ "${EXPECTED_COMMIT}" =~ ^[0-9a-f]{40}$ ]] || fail "invalid commit pin"
[[ "${EXPECTED_LAUNCHER_SHA}" =~ ^[0-9a-f]{64}$ ]] || fail "invalid launcher pin"
for command_name in git sha256sum singularity flock nvidia-smi; do
  command -v "${command_name}" >/dev/null || fail "${command_name} is unavailable"
done
gpu_name="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)"
[[ "${gpu_name}" == NVIDIA\ A100* ]] || fail "Stage A requires the pinned A100 class: ${gpu_name}"
[[ -d "${REPO_ROOT}/.git" || -f "${REPO_ROOT}/.git" ]] || fail "not a checkout"
[[ "$(git -C "${REPO_ROOT}" rev-parse HEAD)" == "${EXPECTED_COMMIT}" ]] || \
  fail "Stage-A checkout commit changed"
if [[ -n "$(git -C "${REPO_ROOT}" status --porcelain)" ]]; then
  git -C "${REPO_ROOT}" status --short >&2
  fail "Stage-A checkout is not clean"
fi
full_sha_is "$0" "${EXPECTED_LAUNCHER_SHA}" "spooled Stage-A launcher"
full_sha_is "${REPO_ROOT}/${LAUNCHER_REL}" "${EXPECTED_LAUNCHER_SHA}" \
  "repository Stage-A launcher"
full_sha_is "${REPO_ROOT}/${PRODUCER_REL}" "${EXPECTED_PRODUCER_SHA}" \
  "causal teacher producer"
full_sha_is "${REPO_ROOT}/${ENTRYPOINT_REL}" "${EXPECTED_ENTRYPOINT_SHA}" \
  "Stage-A entrypoint"
full_sha_is "${REPO_ROOT}/${EXACT_LOADER_REL}" "${EXPECTED_EXACT_LOADER_SHA}" \
  "exact DINO loader"
full_sha_is "${MANIFEST}" "${EXPECTED_MANIFEST_SHA}" "formal causal manifest"
full_sha_is "${ACCEPTANCE}" "${EXPECTED_ACCEPTANCE_SHA}" "causal-scale acceptance"
pinned_python_is
prefix_identity_is "${BASE_SIF}" "${EXPECTED_SIF_BYTES}" "${EXPECTED_SIF_HEAD_SHA}" \
  "base SIF"
prefix_identity_is "${OVERLAY}" "${EXPECTED_OVERLAY_BYTES}" \
  "${EXPECTED_OVERLAY_HEAD_SHA}" "dataset overlay"
full_sha_is "${WEIGHTS}" "${EXPECTED_WEIGHTS_SHA}" "LingBot weights"
full_sha_is "${LINGBOT_REPO}/lingbot_map/layers/vision_transformer.py" \
  "${EXPECTED_DINO_VIT_SHA}" "LingBot DINO implementation"
full_sha_is "${LINGBOT_REPO}/lingbot_map/utils/load_fn.py" \
  "${EXPECTED_DINO_PREPROCESSOR_SHA}" "LingBot image preprocessor"
[[ -d "${RAW_SCENE}" ]] || fail "audited YmJ raw scene is absent"

lingbot_commit="$(git -c safe.directory="${LINGBOT_REPO}" -C "${LINGBOT_REPO}" rev-parse HEAD)"
lingbot_tree="$(git -c safe.directory="${LINGBOT_REPO}" -C "${LINGBOT_REPO}" rev-parse 'HEAD^{tree}')"
[[ "${lingbot_commit}" == "${EXPECTED_LINGBOT_COMMIT}" ]] || fail "LingBot commit changed"
[[ "${lingbot_tree}" == "${EXPECTED_LINGBOT_TREE}" ]] || fail "LingBot tree changed"
if [[ -n "$(git -c safe.directory="${LINGBOT_REPO}" -C "${LINGBOT_REPO}" \
  status --porcelain=v1 --untracked-files=all)" ]]; then
  fail "LingBot worktree is not fully clean"
fi
export GIT_CONFIG_COUNT=1
export GIT_CONFIG_KEY_0=safe.directory
export GIT_CONFIG_VALUE_0="${LINGBOT_REPO}"

exec 9>"${RUN_ROOT}/.causal_teacher_stage_a.lock"
flock -n 9 || fail "another causal teacher Stage A is active"
mkdir -p "${OUTPUT_ROOT}"
resume_args=()
if [[ -e "${PROGRESS_ROOT}" || -e "${EMBEDDING_BUNDLE}" ]]; then
  resume_args+=(--resume)
fi

CONTAINER=(singularity exec --nv
  --overlay "${OVERLAY}:ro"
  -B /scratch/lg154
  -B /scratch/yz11502
  -B "${RAW_SCENE}:${EPISODE_ROOT}/YmJkqBEsHnH:ro"
  "${BASE_SIF}")
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export APPTAINERENV_PYTHONPATH="${PYTHONPATH}"
export SINGULARITYENV_PYTHONPATH="${PYTHONPATH}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

"${CONTAINER[@]}" "${MEMNAV_PY}" - "${MANIFEST}" <<'PY'
import importlib.util
import json
from pathlib import Path
import sys
import torch

manifest = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if manifest.get("summary", {}).get("sample_count") != 600:
    raise SystemExit("formal manifest sample count changed")
if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable in Stage A")
required = ("numpy", "PIL", "torch", "torchvision")
missing = [name for name in required if importlib.util.find_spec(name) is None]
if missing:
    raise SystemExit(f"Stage-A dependencies are missing: {missing}")
print(json.dumps({"stage_a_preflight": "passed", "sample_count": 600}, sort_keys=True))
PY

echo "host=$(hostname) job=${SLURM_JOB_ID:-manual} stage=causal_teacher_a"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
rm -f "${STDOUT_TMP}"
trap 'rm -f "${STDOUT_TMP}"' EXIT
"${CONTAINER[@]}" "${MEMNAV_PY}" "${REPO_ROOT}/${ENTRYPOINT_REL}" \
  --manifest "${MANIFEST}" \
  --expected-manifest-sha256 "${EXPECTED_MANIFEST_SHA}" \
  --expected-sample-count 600 \
  --lingbot-repo "${LINGBOT_REPO}" \
  --expected-lingbot-commit "${EXPECTED_LINGBOT_COMMIT}" \
  --expected-lingbot-tree "${EXPECTED_LINGBOT_TREE}" \
  --lingbot-weights "${WEIGHTS}" \
  --expected-python-sha256 "${EXPECTED_PYTHON_SHA}" \
  --expected-lingbot-weights-sha256 "${EXPECTED_WEIGHTS_SHA}" \
  --expected-dino-loader-sha256 "${EXPECTED_EXACT_LOADER_SHA}" \
  --expected-dino-vit-sha256 "${EXPECTED_DINO_VIT_SHA}" \
  --expected-dino-preprocessor-sha256 "${EXPECTED_DINO_PREPROCESSOR_SHA}" \
  --dino-device cuda \
  --dino-batch-size 16 \
  --embedding-chunk-size 256 \
  --expected-producer-sha256 "${EXPECTED_PRODUCER_SHA}" \
  --expected-entrypoint-sha256 "${EXPECTED_ENTRYPOINT_SHA}" \
  --progress-dir "${PROGRESS_ROOT}" \
  --out-dir "${EMBEDDING_BUNDLE}" \
  "${resume_args[@]}" | tee "${STDOUT_TMP}"
mv -f "${STDOUT_TMP}" "${STDOUT_FINAL}"
trap - EXIT

"${CONTAINER[@]}" "${MEMNAV_PY}" - "${EMBEDDING_BUNDLE}" \
  "${EXPECTED_MANIFEST_SHA}" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

root = Path(sys.argv[1])
receipt = root / "embedding_receipt.json"
raw = receipt.read_bytes()
digest = hashlib.sha256(raw).hexdigest()
sidecar = (root / "embedding_receipt.json.sha256").read_text(encoding="ascii")
if sidecar != f"{digest}  embedding_receipt.json\n":
    raise SystemExit("Stage-A receipt sidecar changed")
record = json.loads(raw)
if record.get("status") != "complete":
    raise SystemExit("Stage-A receipt is incomplete")
if record.get("manifest", {}).get("sha256") != sys.argv[2]:
    raise SystemExit("Stage-A receipt is bound to another manifest")
cover = record.get("exact_cover", {})
if not cover.get("all_manifest_query_and_causal_candidate_rgb_inputs"):
    raise SystemExit("Stage-A input cover is incomplete")
print(json.dumps({"stage_a": "passed", "receipt_sha256": digest,
                  "input_count": cover.get("input_count")}, sort_keys=True))
PY

echo "NLSR causal teacher Stage A passed: ${EMBEDDING_BUNDLE}"
