#!/usr/bin/env bash
# Frozen two-stage train-only audit of LightGlue+LingBot relocalization.
# Stage A: all 480 train sessions, image-only DINO top-8 localization evidence.
# Stage B defaults to the frozen 24-session v2 confirmation.  The explicit
# train40_all480 scope instead characterizes every train session with the same
# fixed three-view LingBot-depth PnP and symmetric support certificate.
# Neither stage is a closed-loop navigation claim.
#SBATCH --job-name=lg_rel_v2
#SBATCH --account=torch_pr_769_tandon_advanced
#SBATCH --partition=a100_tandon
#SBATCH --qos=gpu48
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --gres=gpu:1
#SBATCH --requeue
#SBATCH --output=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/%x_%j.out
#SBATCH --error=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/%x_%j.err

set -euo pipefail
umask 0022

: "${SOURCE_BUNDLE:?submit with an immutable source bundle}"
: "${EXPECTED_BUNDLE_MANIFEST_SHA:?submit with the exact bundle-manifest SHA}"

BASE_REPO_ROOT="${BASE_REPO_ROOT:-/scratch/yz11502/Research/Nav-graph-blind}"
EXPECTED_BASE_COMMIT="32145f3a3c1ae9f98d9fddbfa8e942e332c7356e"
RUN_SCOPE="${RUN_SCOPE:-confirm24}"
case "${RUN_SCOPE}" in
  confirm24)
    DEFAULT_RESULT_BASE="/scratch/yz11502/Research/Nav-axis-uturn-results/lingbot_lightglue_relocalization_v2_20260812"
    SESSION_MANIFEST_RELATIVE="MemNavData/unknown_goal_relocalization_v2_confirm_sessions_20260812.json"
    EXPECTED_SESSION_COUNT=24
    PNP_DIRECTORY="pnp_three_view_24"
    ;;
  train40_all480)
    DEFAULT_RESULT_BASE="/scratch/yz11502/Research/Nav-axis-uturn-results/lingbot_lightglue_train40_challenge_20260814"
    SESSION_MANIFEST_RELATIVE="MemNavData/train40_certificate_challenge_manifest_20260814.json"
    EXPECTED_SESSION_COUNT=480
    PNP_DIRECTORY="pnp_three_view_480"
    ;;
  *)
    echo "ABORT: unsupported RUN_SCOPE=${RUN_SCOPE}" >&2
    exit 2
    ;;
esac
RESULT_BASE="${RESULT_BASE:-${DEFAULT_RESULT_BASE}}"
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
STATIC_AUDIT="${SOURCE_BUNDLE}/MemNavData/audit_lightglue_open_set_localization.py"
SUMMARIZER="${SOURCE_BUNDLE}/MemNavData/summarize_lingbot_lightglue_localization.py"
CONTRACT_AUDITOR="${SOURCE_BUNDLE}/MemNavData/audit_unknown_goal_mrc_smoke.py"
SESSION_MANIFEST="${SOURCE_BUNDLE}/${SESSION_MANIFEST_RELATIVE}"
LIGHTGLUE_REPO="${SOURCE_BUNDLE}/third_party/LightGlue"
DEPENDENCY_ROOT="${SOURCE_BUNDLE}/third_party/python"

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

fail() { echo "ABORT: $*" >&2; exit 2; }
full_sha_is() {
  local path="$1" expected="$2" label="$3" actual
  [[ -f "${path}" && ! -L "${path}" ]] || fail "missing physical ${label}: ${path}"
  actual="$(sha256sum "${path}" | awk '{print $1}')"
  [[ "${actual}" == "${expected}" ]] || fail "${label} SHA mismatch"
}
prefix_identity_is() {
  local path="$1" bytes="$2" head_sha="$3" label="$4" actual
  [[ -f "${path}" && "$(stat -c '%s' "${path}")" == "${bytes}" ]] || \
    fail "${label} byte identity changed"
  actual="$(dd if="${path}" bs=1048576 count=1 status=none | sha256sum | awk '{print $1}')"
  [[ "${actual}" == "${head_sha}" ]] || fail "${label} header changed"
}

for command_name in git sha256sum singularity flock nvidia-smi; do
  command -v "${command_name}" >/dev/null || fail "${command_name} unavailable"
done
SOURCE_BUNDLE="$(realpath "${SOURCE_BUNDLE}")"
[[ "${SOURCE_BUNDLE}" == /scratch/yz11502/Research/Nav-axis-uturn-source-bundles/* ]] || \
  fail "source bundle is outside immutable root"
full_sha_is "${BUNDLE_MANIFEST}" "${EXPECTED_BUNDLE_MANIFEST_SHA}" "bundle manifest"

"${MEMNAV_PY}" - "${SOURCE_BUNDLE}" "${BUNDLE_MANIFEST}" <<'PY'
import hashlib, json, sys
from pathlib import Path, PurePosixPath
root = Path(sys.argv[1]).resolve()
manifest = json.loads(Path(sys.argv[2]).read_text())
if manifest.get("schema_version") != "lingbot_lightglue_relocalization_bundle_v1":
    raise SystemExit("bundle schema changed")
required = {
    "MemNavData/diag_lingbot_goal_loop_closure.py",
    "MemNavData/slurm_lingbot_native_localizer.sbatch",
    "MemNavData/lingbot_colored_registration.py",
    "MemNavData/lingbot_pnp_localization.py",
    "MemNavData/audit_lightglue_open_set_localization.py",
    "MemNavData/summarize_lingbot_lightglue_localization.py",
    "MemNavData/audit_unknown_goal_mrc_smoke.py",
    "MemNavData/unknown_goal_relocalization_v2_confirm_sessions_20260812.json",
    "MemNavData/train40_certificate_challenge_manifest_20260814.json",
    "MemNavData/slurm_lingbot_lightglue_relocalization.sbatch",
    "third_party/LightGlue/BUNDLED_COMMIT",
    "third_party/LightGlue/lightglue/__init__.py",
    "third_party/python/kornia/__init__.py",
    "third_party/python/kornia_rs/kornia_rs.cpython-310-x86_64-linux-gnu.so",
    "torch_home/hub/checkpoints/superpoint_v1.pth",
    "torch_home/hub/checkpoints/superpoint_lightglue_v0-1_arxiv.pth",
}
files = manifest.get("files")
if not isinstance(files, dict) or not required <= set(files):
    raise SystemExit("bundle required-file cover changed")
for relative, expected in sorted(files.items()):
    canonical = PurePosixPath(relative)
    if canonical.is_absolute() or ".." in canonical.parts:
        raise SystemExit(f"non-canonical bundle path: {relative}")
    physical = root / relative
    if not physical.is_file() or physical.is_symlink():
        raise SystemExit(f"non-physical bundle file: {relative}")
    digest = hashlib.sha256(physical.read_bytes()).hexdigest()
    if digest != expected:
        raise SystemExit(f"bundle SHA mismatch: {relative}")
print(json.dumps({"bundle": "verified", "files": len(files)}, sort_keys=True))
PY

[[ "$(git -C "${BASE_REPO_ROOT}" rev-parse HEAD)" == "${EXPECTED_BASE_COMMIT}" ]] || \
  fail "base checkout commit changed"
full_sha_is "${BASE_REPO_ROOT}/InternNav/internnav/model/basemodel/memnav/lingbot_stream.py" \
  "${EXPECTED_STREAM_SHA}" "LingBot stream adapter"
full_sha_is "${BASE_REPO_ROOT}/InternNav/internnav/model/basemodel/memnav/cache_schema.py" \
  "${EXPECTED_CACHE_SCHEMA_SHA}" "LingBot cache schema"
full_sha_is "${SPLIT_MANIFEST}" "${EXPECTED_SPLIT_SHA}" "scene split"
full_sha_is "${MANIFEST}" "${EXPECTED_MANIFEST_SHA}" "causal manifest"
full_sha_is "${TEACHER_CSV}" "${EXPECTED_TEACHER_SHA}" "causal teacher"
full_sha_is "${TEACHER_AUDIT}" "${EXPECTED_TEACHER_AUDIT_SHA}" "teacher audit"
full_sha_is "${SCALE_ARTIFACT}" "${EXPECTED_SCALE_SHA}" "causal scale"
full_sha_is "${SCALE_ACCEPTANCE}" "${EXPECTED_SCALE_ACCEPTANCE_SHA}" "scale acceptance"
full_sha_is "${WEIGHTS}" "${EXPECTED_WEIGHTS_SHA}" "LingBot weights"
prefix_identity_is "${BASE_SIF}" "${EXPECTED_SIF_BYTES}" "${EXPECTED_SIF_HEAD_SHA}" "base SIF"
prefix_identity_is "${OVERLAY}" "${EXPECTED_OVERLAY_BYTES}" \
  "${EXPECTED_OVERLAY_HEAD_SHA}" "dataset overlay"
[[ -d "${RAW_GAP_SCENE}" ]] || fail "audited gap-fill scene absent"
[[ -L "${MEMNAV_PY}" && "$(readlink -f "${MEMNAV_PY}")" == "${EXPECTED_PYTHON_TARGET}" ]] || \
  fail "MemNav Python target changed"
full_sha_is "${EXPECTED_PYTHON_TARGET}" "${EXPECTED_PYTHON_SHA}" "MemNav Python"
lingbot_commit="$(git -c safe.directory="${LINGBOT_REPO}" -C "${LINGBOT_REPO}" rev-parse HEAD)"
lingbot_tree="$(git -c safe.directory="${LINGBOT_REPO}" -C "${LINGBOT_REPO}" rev-parse 'HEAD^{tree}')"
[[ "${lingbot_commit}" == "${EXPECTED_LINGBOT_COMMIT}" && \
   "${lingbot_tree}" == "${EXPECTED_LINGBOT_TREE}" ]] || fail "LingBot identity changed"

CONTAINER=(singularity exec --nv --overlay "${OVERLAY}:ro"
  -B /scratch/lg154 -B /scratch/yz11502
  -B "${RAW_GAP_SCENE}:${EPISODE_ROOT}/YmJkqBEsHnH:ro" "${BASE_SIF}")
export TORCH_HOME="${SOURCE_BUNDLE}/torch_home"
export PYTHONPATH="${SOURCE_BUNDLE}:${BASE_REPO_ROOT}:${DEPENDENCY_ROOT}:${LIGHTGLUE_REPO}${PYTHONPATH:+:${PYTHONPATH}}"
export APPTAINERENV_PYTHONPATH="${PYTHONPATH}"
export SINGULARITYENV_PYTHONPATH="${PYTHONPATH}"
export APPTAINERENV_TORCH_HOME="${TORCH_HOME}"
export SINGULARITYENV_TORCH_HOME="${TORCH_HOME}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export GIT_CONFIG_COUNT=2
export GIT_CONFIG_KEY_0=safe.directory
export GIT_CONFIG_VALUE_0="${LINGBOT_REPO}"
export GIT_CONFIG_KEY_1=safe.directory
export GIT_CONFIG_VALUE_1="${LIGHTGLUE_REPO}"

mapfile -t SELECTED_SESSIONS < <(
  "${CONTAINER[@]}" "${MEMNAV_PY}" - \
    "${TEACHER_CSV}" "${SESSION_MANIFEST}" "${RUN_SCOPE}" <<'PY'
import hashlib, json, sys
from pathlib import Path
import pandas as pd
teacher = pd.read_csv(sys.argv[1])
manifest = json.loads(Path(sys.argv[2]).read_text())
scope = sys.argv[3]
sessions = sorted(teacher.loc[
    teacher["kind"].eq("manifest_causal_goal_localization_train"),
    "session_id"].unique())
if len(sessions) != 480:
    raise SystemExit("train session universe changed")
if scope == "train40_all480":
    if manifest.get("schema_version") != (
            "train40_certificate_challenge_manifest_v1"):
        raise SystemExit("train40 challenge manifest schema changed")
    if manifest.get("selection_algorithm") != (
            "complete_sorted_train_session_universe_v1"):
        raise SystemExit("train40 challenge selection algorithm changed")
    if manifest.get("selection_uses_labels") is not False:
        raise SystemExit("train40 challenge selection is not label blind")
    if manifest.get("source_teacher_sha256") != (
            "fd52dcfcb7e8d1bdac0703f7ba0b6d10f04067d4b8793386adf65ede6bc8e313"):
        raise SystemExit("train40 challenge teacher identity changed")
    if manifest.get("selected_session_count") != 480:
        raise SystemExit("train40 challenge session count changed")
    if manifest.get("selected_scene_count") != 40:
        raise SystemExit("train40 challenge scene count changed")
    if manifest.get("sessions") != sessions:
        raise SystemExit("train40 challenge is not the complete session universe")
    universe_sha = hashlib.sha256(
        "".join(f"{session}\n" for session in sessions).encode()).hexdigest()
    if manifest.get("session_universe_sha256") != universe_sha:
        raise SystemExit("train40 challenge universe receipt changed")
    selected = sessions
elif scope == "confirm24":
    if len(sessions) != manifest["source_session_count"]:
        raise SystemExit("confirmation source session count changed")
    if manifest.get("schema_version") != (
            "unknown_goal_mrc_v0_smoke_sessions_v1"):
        raise SystemExit("confirmation manifest schema changed")
    if manifest.get("selection_algorithm") != "scene_balanced_disjoint_v1":
        raise SystemExit("confirmation selection algorithm changed")
    if manifest.get("parent_rows_sha256") != (
            "900a8c4eaef9c194b68d7055454eb90a00ca9160fac1ba6ac98ec033e86ecdd7"):
        raise SystemExit("v1 design receipt identity changed")
    if manifest.get("local_design_rows_sha256") != (
            "112c48c8aa90d867bfe1a833381149a565713d7853296e9441ae3d32b4903166"):
        raise SystemExit("local design receipt identity changed")
    salt = manifest["salt"]
    excluded = set(map(str, manifest.get("excluded_scenes", [])))
    if len(excluded) != manifest.get("excluded_scene_count"):
        raise SystemExit("excluded scene cover changed")
    scene_sessions = {}
    for session in sessions:
        scene_sessions.setdefault(session.split("/")[1], []).append(session)
    if not excluded <= set(scene_sessions):
        raise SystemExit("excluded scene absent from train universe")
    eligible_scenes = sorted(set(scene_sessions) - excluded)
    if len(eligible_scenes) != manifest["selected_scene_count"]:
        raise SystemExit("unseen scene universe changed")
    rank = lambda tag, value: hashlib.sha256(
        f"{salt}\0{tag}\0{value}".encode()).hexdigest()
    selected_scenes = sorted(
        eligible_scenes, key=lambda value: (rank("scene", value), value))
    primary = [
        sorted(scene_sessions[scene],
               key=lambda value: (rank("primary", value), value))[0]
        for scene in selected_scenes
    ]
    primary_set = set(primary)
    remaining = [
        session for scene in selected_scenes for session in scene_sessions[scene]
        if session not in primary_set
    ]
    extras = sorted(
        remaining, key=lambda value: (rank("extra", value), value)
    )[:manifest["additional_session_count"]]
    selected = primary + extras
    if len(primary) != manifest["primary_session_count"]:
        raise SystemExit("primary session count changed")
    if selected != manifest["sessions"]:
        raise SystemExit("frozen session sample changed")
    if {session.split("/")[1] for session in selected} & excluded:
        raise SystemExit("design-exposed scene entered confirmation")
else:
    raise SystemExit(f"unsupported scope: {scope}")
for session in selected:
    print(session)
PY
)
[[ "${#SELECTED_SESSIONS[@]}" -eq "${EXPECTED_SESSION_COUNT}" ]] || \
  fail "session sample count changed"

BUNDLE_TAG="${EXPECTED_BUNDLE_MANIFEST_SHA:0:16}"
RUN_ROOT="${RESULT_BASE}/bundle_${BUNDLE_TAG}_job_${SLURM_JOB_ID}"
STATIC_ROOT="${RUN_ROOT}/static_top8_480"
PNP_ROOT="${RUN_ROOT}/${PNP_DIRECTORY}"
mkdir -p "${RUN_ROOT}"
exec 9>"${RESULT_BASE}/.bundle_${BUNDLE_TAG}.lock"
flock -n 9 || fail "another identical relocalization job is active"
gpu_name="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)"
[[ "${gpu_name}" == NVIDIA\ A100* || "${gpu_name}" == NVIDIA\ H100* ]] || \
  fail "A100/H100 required: ${gpu_name}"
echo "host=$(hostname) job=${SLURM_JOB_ID} scope=${RUN_SCOPE} bundle=${BUNDLE_TAG} gpu=${gpu_name}"

"${CONTAINER[@]}" "${MEMNAV_PY}" -m py_compile \
  "${ENTRYPOINT}" "${STATIC_AUDIT}" "${SUMMARIZER}"
"${CONTAINER[@]}" "${MEMNAV_PY}" -m pytest -q \
  "${SOURCE_BUNDLE}/MemNavData/test_lingbot_pnp_localization.py" \
  "${SOURCE_BUNDLE}/MemNavData/test_lingbot_goal_loop_closure.py" \
  "${SOURCE_BUNDLE}/MemNavData/test_audit_lightglue_open_set_localization.py" \
  "${SOURCE_BUNDLE}/MemNavData/test_audit_unknown_goal_mrc_smoke.py" \
  "${SOURCE_BUNDLE}/MemNavData/test_build_train40_certificate_challenge_manifest.py" \
  "${SOURCE_BUNDLE}/MemNavData/test_summarize_lingbot_lightglue_localization.py"

if [[ ! -f "${STATIC_ROOT}/lightglue_open_set_report.json" ]]; then
  [[ ! -e "${STATIC_ROOT}" ]] || fail "partial non-resumable static output exists"
  "${CONTAINER[@]}" "${MEMNAV_PY}" -u "${STATIC_AUDIT}" \
    --teacher-csv "${TEACHER_CSV}" \
    --lightglue-repo "${LIGHTGLUE_REPO}" \
    --dependency-root "${DEPENDENCY_ROOT}" \
    --out-dir "${STATIC_ROOT}" \
    --kind manifest_causal_goal_localization_train \
    --top-k 8 --candidate-min-gap 4 --minimum-anchor 8 \
    --print-every 500 --device cuda
fi
"${CONTAINER[@]}" "${MEMNAV_PY}" - "${STATIC_ROOT}/lightglue_open_set_report.json" <<'PY'
import json, sys
report = json.load(open(sys.argv[1]))
if report["n_sessions"] != 480 or report["n_candidates"] != 3840:
    raise SystemExit("static 480x8 cover changed")
if report["config"]["decision_threshold_fitted"]:
    raise SystemExit("static audit fitted a threshold")
if report["config"].get("minimum_anchor") != 8:
    raise SystemExit("static audit used an incompatible causal anchor floor")
print(json.dumps({"static_cover": "verified", "sessions": 480, "rows": 3840}))
PY
STATIC_CSV="${STATIC_ROOT}/lightglue_open_set_rows.csv"
STATIC_REPORT="${STATIC_ROOT}/lightglue_open_set_report.json"
STATIC_CSV_SHA="$(sha256sum "${STATIC_CSV}" | awk '{print $1}')"
STATIC_REPORT_SHA="$(sha256sum "${STATIC_REPORT}" | awk '{print $1}')"

COMMON_ARGS=(
  --internnav-root "${BASE_REPO_ROOT}/InternNav"
  --teacher-csv "${TEACHER_CSV}"
  --split-manifest "${SPLIT_MANIFEST}" --allowed-role train
  --lingbot-repo "${LINGBOT_REPO}" --weights "${WEIGHTS}"
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
  --out-dir "${PNP_ROOT}" --selection-mode lightglue_ranked
  --kind manifest_causal_goal_localization_train --max-sessions 0 --top-k 8
  --candidate-min-gap 4
  --lightglue-selection-csv "${STATIC_CSV}"
  --lightglue-selection-report "${STATIC_REPORT}"
  --expected-lightglue-selection-csv-sha256 "${STATIC_CSV_SHA}"
  --expected-lightglue-selection-report-sha256 "${STATIC_REPORT_SHA}"
  --neighbor-offset=0
  --adaptive-neighbor-radius 4 --adaptive-neighbor-count 3
  --warm 64 --full-replay --num-scale 8 --window 32 --max-frame-num 4096
  --camera-num-iterations 4 --pixel-stride 10 --confidence-quantile 0.5
  --max-points 768 --max-cached-episodes 1 --device cuda
  --pnp-lightglue --lightglue-repo "${LIGHTGLUE_REPO}"
  --lightglue-dependency-root "${DEPENDENCY_ROOT}"
)
if [[ "${RUN_SCOPE}" == "confirm24" ]]; then
  for session in "${SELECTED_SESSIONS[@]}"; do
    COMMON_ARGS+=(--session "${session}")
  done
fi

"${CONTAINER[@]}" "${MEMNAV_PY}" "${ENTRYPOINT}" "${COMMON_ARGS[@]}" --preflight-only
resume_args=()
[[ -f "${PNP_ROOT}/lingbot_goal_loop_closure_checkpoint.sqlite3" ]] && \
  resume_args+=(--resume)
"${CONTAINER[@]}" "${MEMNAV_PY}" -u "${ENTRYPOINT}" \
  "${COMMON_ARGS[@]}" "${resume_args[@]}"

"${CONTAINER[@]}" "${MEMNAV_PY}" "${CONTRACT_AUDITOR}" \
  --rows "${PNP_ROOT}/lingbot_goal_loop_closure_rows.csv" \
  --report "${PNP_ROOT}/diagnostic_lingbot_goal_loop_closure.json" \
  --progress "${PNP_ROOT}/lingbot_goal_loop_closure_progress.json" \
  --session-manifest "${SESSION_MANIFEST}" \
  --expected-selection-mode lightglue_ranked \
  --expected-top-k 8 \
  --expected-selection-origin lightglue_fundamental_rank_v1 \
  --expected-session-count "${EXPECTED_SESSION_COUNT}" \
  --require-lightglue-pnp \
  --out "${PNP_ROOT}/contract_receipt.json"
"${CONTAINER[@]}" "${MEMNAV_PY}" "${SUMMARIZER}" \
  --rows "${PNP_ROOT}/lingbot_goal_loop_closure_rows.csv" \
  --report "${PNP_ROOT}/diagnostic_lingbot_goal_loop_closure.json" \
  --session-manifest "${SESSION_MANIFEST}" \
  --static-rows "${STATIC_ROOT}/lightglue_open_set_rows.csv" \
  --static-report "${STATIC_ROOT}/lightglue_open_set_report.json" \
  --out "${RUN_ROOT}/actionability_audit.json"
echo "LINGBOT_LIGHTGLUE_RELOCALIZATION_ROOT=${RUN_ROOT}"
