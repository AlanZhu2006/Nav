#!/usr/bin/env bash
# Frozen 24-session contract/timing smoke for Unknown-Goal MRC-v0.
# This job does not authorize an effectiveness claim or threshold selection.
#SBATCH --job-name=mrc_v0_smoke
#SBATCH --account=torch_pr_769_tandon_advanced
#SBATCH --partition=a100_tandon
#SBATCH --qos=gpu48
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1
#SBATCH --requeue
#SBATCH --output=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/%x_%j.out
#SBATCH --error=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/%x_%j.err

set -euo pipefail
umask 0022

: "${SOURCE_BUNDLE:?submit with an immutable source-bundle directory}"
: "${EXPECTED_BUNDLE_MANIFEST_SHA:?submit with the bundle-manifest SHA256}"

BASE_REPO_ROOT="${BASE_REPO_ROOT:-/scratch/yz11502/Research/Nav-graph-blind}"
EXPECTED_BASE_COMMIT="32145f3a3c1ae9f98d9fddbfa8e942e332c7356e"
RESULT_BASE="${RESULT_BASE:-/scratch/yz11502/Research/Nav-axis-uturn-results/unknown_goal_mrc_v0_20260812}"
RUN_TAG="nlsr_routes_41c1773_20260807_v1"
UPSTREAM_ROOT="/scratch/yz11502/Research/Nav-axis-uturn-results/nlsr_gapfill_20260807/${RUN_TAG}"
UPSTREAM="${UPSTREAM_ROOT}/causal_covisibility_teacher_a55ae5d40ba3_bc6bf58536f6/teacher_bundle"
MANIFEST="${UPSTREAM_ROOT}/manifest_multistage/candidate_manifest_multistage.json"
SCALE_ARTIFACT="${UPSTREAM_ROOT}/causal_ground_scale/causal_ground_scale.json"
SCALE_ACCEPTANCE="${UPSTREAM_ROOT}/causal_ground_scale_acceptance_c8f499dc9c4f_f6f28e20b9f2/causal_ground_scale_acceptance.json"
TEACHER_CSV="${UPSTREAM}/teacher.csv"
TEACHER_AUDIT="${UPSTREAM}/audit.json"
SPLIT_MANIFEST="${BASE_REPO_ROOT}/MemNavData/router_multiscene_split_20260805.json"

BUNDLE_MANIFEST="${SOURCE_BUNDLE}/source_bundle_manifest.json"
ENTRYPOINT="${SOURCE_BUNDLE}/MemNavData/diag_lingbot_goal_loop_closure.py"
AUDITOR="${SOURCE_BUNDLE}/MemNavData/audit_unknown_goal_mrc_smoke.py"
SESSION_MANIFEST="${SOURCE_BUNDLE}/MemNavData/unknown_goal_mrc_v0_smoke_sessions_20260812.json"

MEMNAV_PY="/scratch/lg154/conda-envs/memnav/bin/python"
EXPECTED_PYTHON_TARGET="/scratch/lg154/conda-envs/memnav/bin/python3.10"
BASE_SIF="/share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif"
OVERLAY="/scratch/lg154/Research/datasets/_overlays/mp3d_revisit_v0_pt1.sqf"
LINGBOT_REPO="/scratch/lg154/Research/Nav/NavDP/baselines/memnav/lingbot-map"
WEIGHTS="${LINGBOT_REPO}/weights/lingbot-map-long.pt"
RAW_GAP_SCENE="${UPSTREAM_ROOT}/raw/mp3d_3leg/YmJkqBEsHnH"
EPISODE_ROOT="/mp3d_revisit_v0/vln_n1/traj_data/mp3d_3leg"

EXPECTED_MANIFEST_SHA="bc6bf58536f6c159d1898ac03abe365eadba65c22cd246e39401821962abb34c"
EXPECTED_TEACHER_SHA="fd52dcfcb7e8d1bdac0703f7ba0b6d10f04067d4b8793386adf65ede6bc8e313"
EXPECTED_TEACHER_AUDIT_SHA="cd7091bb8c398ac5a7f558003a3fb467cc450a9b65e4416e29b7d27e1fad14c6"
EXPECTED_SCALE_SHA="f6f28e20b9f21a764132af04b425f23b65e1dc779798d831a571a7df26bd78db"
EXPECTED_SCALE_ACCEPTANCE_SHA="0ced1b3a56fe0e089bad35c5807c219da27daad48e6bbd4bb29436ef2b8eb3ac"
EXPECTED_SCALE_ACCEPTANCE_COMMIT="c8f499dc9c4f9112087d26a0daba46f3c3c82d59"
EXPECTED_SCALE_PRODUCER_SHA="8da2a4c42de010e699c5ab65a2f6bbc9600c29acc96b71ef82d89126d8b4ca26"
EXPECTED_SCALE_CONFIG_SHA="f1b01a8b6cd07a8cfb0ee306083e36d6399d21f22f242b3e2eb0f5ee7624945d"
EXPECTED_LINGBOT_COMMIT="7ff6f3ed0913d4d326f8f13bbb429c4ffc0195c2"
EXPECTED_LINGBOT_TREE="479d4192e8f4fabb6bd1687e2fbd1258de924fe7"
EXPECTED_WEIGHTS_SHA="832bc82cbae0bc9bbe946ef5ee1f7226abd8c0e183ccf8beddbb3d133576f409"
EXPECTED_STREAM_SHA="b5142b76f9062634db826a0e7f2c3da19e4b07e21c2325ecbc60a44cc52370b0"
EXPECTED_CACHE_SCHEMA_SHA="412fdc9ba11947406297ba639790b9d3acf7d9c3ab6125dea8e1a1246e38f227"
EXPECTED_SPLIT_SHA="97309c183e25cb3dd65472908748d55a94798a636db6157ab6fe120fca05cf7a"
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
  [[ "${actual}" == "${expected}" ]] || fail "${label} SHA mismatch"
}

sidecar_is() {
  local artifact="$1" expected="$2" sidecar
  sidecar="${artifact}.sha256"
  [[ -f "${sidecar}" && ! -L "${sidecar}" ]] || fail "missing sidecar ${sidecar}"
  [[ "$(<"${sidecar}")" == "${expected}  $(basename "${artifact}")" ]] || \
    fail "sidecar differs for ${artifact}"
}

prefix_identity_is() {
  local path="$1" bytes="$2" head_sha="$3" label="$4" actual
  [[ -f "${path}" && ! -L "${path}" && "$(stat -c '%s' "${path}")" == "${bytes}" ]] || \
    fail "${label} byte identity changed"
  actual="$(dd if="${path}" bs=1048576 count=1 status=none | sha256sum | awk '{print $1}')"
  [[ "${actual}" == "${head_sha}" ]] || fail "${label} header changed"
}

for command_name in git sha256sum singularity flock nvidia-smi; do
  command -v "${command_name}" >/dev/null || fail "${command_name} is unavailable"
done
[[ "${EXPECTED_BUNDLE_MANIFEST_SHA}" =~ ^[0-9a-f]{64}$ ]] || \
  fail "invalid bundle-manifest pin"
SOURCE_BUNDLE="$(realpath "${SOURCE_BUNDLE}")"
[[ "${SOURCE_BUNDLE}" == /scratch/yz11502/Research/Nav-axis-uturn-source-bundles/* ]] || \
  fail "source bundle is outside the immutable bundle root"
full_sha_is "${BUNDLE_MANIFEST}" "${EXPECTED_BUNDLE_MANIFEST_SHA}" "source bundle manifest"

[[ "$(git -C "${BASE_REPO_ROOT}" rev-parse HEAD)" == "${EXPECTED_BASE_COMMIT}" ]] || \
  fail "base checkout commit changed"
git -C "${BASE_REPO_ROOT}" diff --quiet -- InternNav || \
  fail "base InternNav tracked files are dirty"
full_sha_is "${BASE_REPO_ROOT}/InternNav/internnav/model/basemodel/memnav/lingbot_stream.py" \
  "${EXPECTED_STREAM_SHA}" "LingBot stream adapter"
full_sha_is "${BASE_REPO_ROOT}/InternNav/internnav/model/basemodel/memnav/cache_schema.py" \
  "${EXPECTED_CACHE_SCHEMA_SHA}" "LingBot cache schema"
full_sha_is "${SPLIT_MANIFEST}" "${EXPECTED_SPLIT_SHA}" "scene split"

full_sha_is "${MANIFEST}" "${EXPECTED_MANIFEST_SHA}" "causal manifest"
full_sha_is "${TEACHER_CSV}" "${EXPECTED_TEACHER_SHA}" "causal teacher CSV"
full_sha_is "${TEACHER_AUDIT}" "${EXPECTED_TEACHER_AUDIT_SHA}" "teacher audit"
sidecar_is "${TEACHER_AUDIT}" "${EXPECTED_TEACHER_AUDIT_SHA}"
full_sha_is "${SCALE_ARTIFACT}" "${EXPECTED_SCALE_SHA}" "causal scale artifact"
full_sha_is "${SCALE_ACCEPTANCE}" "${EXPECTED_SCALE_ACCEPTANCE_SHA}" "scale acceptance"
sidecar_is "${SCALE_ACCEPTANCE}" "${EXPECTED_SCALE_ACCEPTANCE_SHA}"
full_sha_is "${WEIGHTS}" "${EXPECTED_WEIGHTS_SHA}" "LingBot weights"
prefix_identity_is "${BASE_SIF}" "${EXPECTED_SIF_BYTES}" "${EXPECTED_SIF_HEAD_SHA}" "base SIF"
prefix_identity_is "${OVERLAY}" "${EXPECTED_OVERLAY_BYTES}" \
  "${EXPECTED_OVERLAY_HEAD_SHA}" "dataset overlay"
[[ -d "${RAW_GAP_SCENE}" ]] || fail "audited gap-fill scene is absent"
[[ -L "${MEMNAV_PY}" && "$(readlink -f "${MEMNAV_PY}")" == "${EXPECTED_PYTHON_TARGET}" ]] || \
  fail "MemNav Python target changed"
full_sha_is "${EXPECTED_PYTHON_TARGET}" "${EXPECTED_PYTHON_SHA}" "MemNav Python"
lingbot_commit="$(git -c safe.directory="${LINGBOT_REPO}" -C "${LINGBOT_REPO}" rev-parse HEAD)"
lingbot_tree="$(git -c safe.directory="${LINGBOT_REPO}" -C "${LINGBOT_REPO}" rev-parse 'HEAD^{tree}')"
[[ "${lingbot_commit}" == "${EXPECTED_LINGBOT_COMMIT}" && \
   "${lingbot_tree}" == "${EXPECTED_LINGBOT_TREE}" ]] || fail "LingBot identity changed"
[[ -z "$(git -c safe.directory="${LINGBOT_REPO}" -C "${LINGBOT_REPO}" \
  status --porcelain=v1 --untracked-files=all)" ]] || fail "LingBot checkout is dirty"

CONTAINER=(singularity exec --nv --overlay "${OVERLAY}:ro"
  -B /scratch/lg154
  -B /scratch/yz11502
  -B "${RAW_GAP_SCENE}:${EPISODE_ROOT}/YmJkqBEsHnH:ro"
  "${BASE_SIF}")
export PYTHONPATH="${SOURCE_BUNDLE}:${BASE_REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export APPTAINERENV_PYTHONPATH="${PYTHONPATH}"
export SINGULARITYENV_PYTHONPATH="${PYTHONPATH}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export GIT_CONFIG_COUNT=1
export GIT_CONFIG_KEY_0=safe.directory
export GIT_CONFIG_VALUE_0="${LINGBOT_REPO}"

"${CONTAINER[@]}" "${MEMNAV_PY}" - "${SOURCE_BUNDLE}" "${BUNDLE_MANIFEST}" <<'PY'
import hashlib
import json
from pathlib import Path, PurePosixPath
import sys

root = Path(sys.argv[1]).resolve()
manifest_path = Path(sys.argv[2])
manifest = json.loads(manifest_path.read_text())
if manifest.get("schema_version") != "unknown_goal_mrc_v0_source_bundle_v1":
    raise SystemExit("source-bundle schema changed")
required = {
    "MemNavData/diag_lingbot_goal_loop_closure.py",
    "MemNavData/test_lingbot_goal_loop_closure.py",
    "MemNavData/slurm_lingbot_native_localizer.sbatch",
    "MemNavData/external_causal_scale_contract.py",
    "MemNavData/phase_b_feature_schema.py",
    "MemNavData/phase_b_upstream_receipts.py",
    "MemNavData/test_phase_b_upstream_receipts.py",
    "MemNavData/flow_cache_routing.py",
    "MemNavData/audit_unknown_goal_mrc_smoke.py",
    "MemNavData/test_audit_unknown_goal_mrc_smoke.py",
    "MemNavData/unknown_goal_mrc_v0_smoke_sessions_20260812.json",
    "MemNavData/UNKNOWN_GOAL_MRC_V0_PROTOCOL_20260812.md",
    "MemNavData/slurm_unknown_goal_mrc_v0_smoke.sbatch",
}
files = manifest.get("files")
if not isinstance(files, dict) or set(files) != required:
    raise SystemExit("source-bundle file cover changed")
for relative, expected in sorted(files.items()):
    path = PurePosixPath(relative)
    if path.is_absolute() or ".." in path.parts or path.as_posix() != relative:
        raise SystemExit(f"non-canonical bundle path: {relative}")
    physical = root / relative
    if not physical.is_file() or physical.is_symlink():
        raise SystemExit(f"non-physical bundle file: {relative}")
    digest = hashlib.sha256(physical.read_bytes()).hexdigest()
    if digest != expected:
        raise SystemExit(f"bundle file SHA mismatch: {relative}")
print(json.dumps({"source_bundle": "verified", "files": len(files)}, sort_keys=True))
PY

"${CONTAINER[@]}" "${MEMNAV_PY}" - "${TEACHER_CSV}" "${SESSION_MANIFEST}" <<'PY'
import hashlib
import json
from pathlib import Path
import pandas as pd
import sys

teacher = pd.read_csv(sys.argv[1])
manifest = json.loads(Path(sys.argv[2]).read_text())
kind = "manifest_causal_goal_localization_train"
sessions = sorted(teacher.loc[teacher["kind"].eq(kind), "session_id"].unique())
if len(sessions) != manifest["source_session_count"] or len(sessions) != 480:
    raise SystemExit("train session universe changed")
salt = manifest["salt"]
selected = sorted(
    sessions,
    key=lambda value: (
        hashlib.sha256(f"{salt}\0{value}".encode()).hexdigest(), value),
)[:manifest["selected_session_count"]]
if selected != manifest["sessions"]:
    raise SystemExit("frozen hash sample does not match the teacher")
print(json.dumps({"session_sample": "verified", "sessions": len(selected)},
                 sort_keys=True))
PY

mapfile -t SELECTED_SESSIONS < <(
  "${CONTAINER[@]}" "${MEMNAV_PY}" - "${SESSION_MANIFEST}" <<'PY'
import json
from pathlib import Path
import sys
for session in json.loads(Path(sys.argv[1]).read_text())["sessions"]:
    print(session)
PY
)
[[ "${#SELECTED_SESSIONS[@]}" -eq 24 ]] || fail "session argument count changed"

BUNDLE_TAG="${EXPECTED_BUNDLE_MANIFEST_SHA:0:16}"
RUN_ROOT="${RESULT_BASE}/smoke_${BUNDLE_TAG}_job_${SLURM_JOB_ID}"
mkdir -p "${RESULT_BASE}"
exec 9>"${RESULT_BASE}/.smoke_${BUNDLE_TAG}.lock"
flock -n 9 || fail "another identical MRC-v0 smoke is active"
if [[ -e "${RUN_ROOT}" && ! -f "${RUN_ROOT}/lingbot_goal_loop_closure_checkpoint.sqlite3" ]]; then
  fail "partial run root lacks a resumable checkpoint: ${RUN_ROOT}"
fi

gpu_name="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)"
[[ "${gpu_name}" == NVIDIA\ A100* || "${gpu_name}" == NVIDIA\ H100* ]] || \
  fail "MRC smoke requires an A100/H100 class GPU: ${gpu_name}"
echo "host=$(hostname) job=${SLURM_JOB_ID} bundle=${BUNDLE_TAG} sessions=24"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

COMMON_ARGS=(
  --internnav-root "${BASE_REPO_ROOT}/InternNav"
  --teacher-csv "${TEACHER_CSV}"
  --split-manifest "${SPLIT_MANIFEST}"
  --allowed-role train
  --lingbot-repo "${LINGBOT_REPO}"
  --weights "${WEIGHTS}"
  --expected-weight-sha "${EXPECTED_WEIGHTS_SHA}"
  --expected-lingbot-commit "${EXPECTED_LINGBOT_COMMIT}"
  --causal-manifest "${MANIFEST}"
  --expected-causal-manifest-sha256 "${EXPECTED_MANIFEST_SHA}"
  --external-causal-scale-artifact "${SCALE_ARTIFACT}"
  --expected-external-causal-scale-sha256 "${EXPECTED_SCALE_SHA}"
  --expected-external-scale-producer-sha256 "${EXPECTED_SCALE_PRODUCER_SHA}"
  --expected-external-scale-configuration-sha256 "${EXPECTED_SCALE_CONFIG_SHA}"
  --expected-external-scale-lingbot-commit "${EXPECTED_LINGBOT_COMMIT}"
  --expected-external-scale-weights-sha256 "${EXPECTED_WEIGHTS_SHA}"
  --expected-external-scale-stream-source-sha256 "${EXPECTED_STREAM_SHA}"
  --causal-teacher-audit "${TEACHER_AUDIT}"
  --expected-causal-teacher-audit-sha256 "${EXPECTED_TEACHER_AUDIT_SHA}"
  --causal-scale-acceptance "${SCALE_ACCEPTANCE}"
  --expected-causal-scale-acceptance-sha256 "${EXPECTED_SCALE_ACCEPTANCE_SHA}"
  --expected-causal-scale-acceptance-commit "${EXPECTED_SCALE_ACCEPTANCE_COMMIT}"
  --out-dir "${RUN_ROOT}"
  --selection-mode deployment
  --kind manifest_causal_goal_localization_train
  --max-sessions 0
  --top-k 1
  --candidate-min-gap 16
  --neighbor-offset=0
  --adaptive-neighbor-radius 4
  --adaptive-neighbor-count 3
  --warm 64
  --full-replay
  --num-scale 8
  --window 32
  --max-frame-num 4096
  --camera-num-iterations 4
  --pixel-stride 10
  --confidence-quantile 0.5
  --max-points 768
  --max-cached-episodes 1
  --device cuda
)
for session in "${SELECTED_SESSIONS[@]}"; do
  COMMON_ARGS+=(--session "${session}")
done

"${CONTAINER[@]}" "${MEMNAV_PY}" -m py_compile \
  "${ENTRYPOINT}" "${AUDITOR}"
"${CONTAINER[@]}" "${MEMNAV_PY}" -m unittest -v \
  MemNavData.test_phase_b_upstream_receipts \
  MemNavData.test_lingbot_goal_loop_closure \
  MemNavData.test_audit_unknown_goal_mrc_smoke
"${CONTAINER[@]}" "${MEMNAV_PY}" "${ENTRYPOINT}" \
  "${COMMON_ARGS[@]}" --preflight-only

resume_args=()
[[ -f "${RUN_ROOT}/lingbot_goal_loop_closure_checkpoint.sqlite3" ]] && \
  resume_args+=(--resume)
mkdir -p "${RUN_ROOT}"
"${CONTAINER[@]}" "${MEMNAV_PY}" -u "${ENTRYPOINT}" \
  "${COMMON_ARGS[@]}" "${resume_args[@]}"

"${CONTAINER[@]}" "${MEMNAV_PY}" "${AUDITOR}" \
  --rows "${RUN_ROOT}/lingbot_goal_loop_closure_rows.csv" \
  --report "${RUN_ROOT}/diagnostic_lingbot_goal_loop_closure.json" \
  --progress "${RUN_ROOT}/lingbot_goal_loop_closure_progress.json" \
  --session-manifest "${SESSION_MANIFEST}" \
  --out "${RUN_ROOT}/unknown_goal_mrc_v0_smoke_receipt.json"
echo "MRC_V0_SMOKE_ROOT=${RUN_ROOT}"
