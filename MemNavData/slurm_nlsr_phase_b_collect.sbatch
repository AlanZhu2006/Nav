#!/usr/bin/env bash
# Formal receipt-bound LingBot feature join for one frozen scene role.
# The collector is session-atomic and may be relaunched against the same role
# directory; an existing compatible SQLite checkpoint is resumed automatically.
#SBATCH --job-name=nlsr_pB_feat
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

: "${REPO_ROOT:?export the exact clean Phase-B checkout}"
: "${EXPECTED_COMMIT:?export its full commit SHA}"
: "${EXPECTED_LAUNCHER_SHA:?export the committed launcher SHA256}"
: "${RUN_TAG:?export the audited NLSR run tag}"
: "${PHASE_B_ROLE:?export train or development}"

RESULT_BASE="${RESULT_BASE:-/scratch/yz11502/Research/Nav-axis-uturn-results/nlsr_gapfill_20260807}"
RUN_ROOT="${RESULT_BASE}/${RUN_TAG}"
UPSTREAM="${RUN_ROOT}/causal_covisibility_teacher_a55ae5d40ba3_bc6bf58536f6/teacher_bundle"
MANIFEST="${RUN_ROOT}/manifest_multistage/candidate_manifest_multistage.json"
SCALE_ARTIFACT="${RUN_ROOT}/causal_ground_scale/causal_ground_scale.json"
SCALE_ACCEPTANCE="${RUN_ROOT}/causal_ground_scale_acceptance_c8f499dc9c4f_f6f28e20b9f2/causal_ground_scale_acceptance.json"
TEACHER_CSV="${UPSTREAM}/teacher.csv"
TEACHER_AUDIT="${UPSTREAM}/audit.json"
SPLIT_MANIFEST="${REPO_ROOT}/MemNavData/router_multiscene_split_20260805.json"
OUTPUT_ROOT="${RUN_ROOT}/phase_b_lingbot_features_${EXPECTED_COMMIT:0:12}"

ENTRYPOINT_REL="MemNavData/diag_lingbot_goal_loop_closure.py"
HELPER_REL="MemNavData/phase_b_upstream_receipts.py"
TEST_REL="MemNavData/test_phase_b_upstream_receipts.py"
COLLECTOR_TEST_REL="MemNavData/test_lingbot_goal_loop_closure.py"
LAUNCHER_REL="MemNavData/slurm_nlsr_phase_b_collect.sbatch"

MEMNAV_PY="/scratch/lg154/conda-envs/memnav/bin/python"
EXPECTED_PYTHON_TARGET="/scratch/lg154/conda-envs/memnav/bin/python3.10"
BASE_SIF="/share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif"
OVERLAY="/scratch/lg154/Research/datasets/_overlays/mp3d_revisit_v0_pt1.sqf"
LINGBOT_REPO="/scratch/lg154/Research/Nav/NavDP/baselines/memnav/lingbot-map"
WEIGHTS="${LINGBOT_REPO}/weights/lingbot-map-long.pt"
RAW_SCENE="${RUN_ROOT}/raw/mp3d_3leg/YmJkqBEsHnH"
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

committed_file_is() {
  local relative="$1" actual expected
  actual="$(sha256sum "${REPO_ROOT}/${relative}" | awk '{print $1}')"
  expected="$(git -C "${REPO_ROOT}" show "${EXPECTED_COMMIT}:${relative}" | \
    sha256sum | awk '{print $1}')"
  [[ "${actual}" == "${expected}" ]] || fail "${relative} differs from commit"
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
[[ "${RUN_TAG}" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]] || fail "invalid RUN_TAG"
[[ "${PHASE_B_ROLE}" =~ ^(train|development)$ ]] || fail "invalid PHASE_B_ROLE"
[[ "${EXPECTED_COMMIT}" =~ ^[0-9a-f]{40}$ ]] || fail "invalid commit pin"
[[ "${EXPECTED_LAUNCHER_SHA}" =~ ^[0-9a-f]{64}$ ]] || fail "invalid launcher pin"
[[ "$(git -C "${REPO_ROOT}" rev-parse HEAD)" == "${EXPECTED_COMMIT}" ]] || \
  fail "Phase-B checkout commit changed"
[[ -z "$(git -C "${REPO_ROOT}" status --porcelain)" ]] || fail "checkout is not clean"
full_sha_is "$0" "${EXPECTED_LAUNCHER_SHA}" "spooled collector launcher"
for relative in "${ENTRYPOINT_REL}" "${HELPER_REL}" "${TEST_REL}" \
                "${COLLECTOR_TEST_REL}" "${LAUNCHER_REL}" \
                "MemNavData/external_causal_scale_contract.py" \
                "MemNavData/flow_cache_routing.py" \
                "MemNavData/router_multiscene_split_20260805.json"; do
  committed_file_is "${relative}"
done

full_sha_is "${MANIFEST}" "${EXPECTED_MANIFEST_SHA}" "causal manifest"
full_sha_is "${TEACHER_CSV}" "${EXPECTED_TEACHER_SHA}" "causal teacher CSV"
full_sha_is "${TEACHER_AUDIT}" "${EXPECTED_TEACHER_AUDIT_SHA}" "teacher audit"
sidecar_is "${TEACHER_AUDIT}" "${EXPECTED_TEACHER_AUDIT_SHA}"
full_sha_is "${SCALE_ARTIFACT}" "${EXPECTED_SCALE_SHA}" "causal scale artifact"
full_sha_is "${SCALE_ACCEPTANCE}" "${EXPECTED_SCALE_ACCEPTANCE_SHA}" "scale acceptance"
sidecar_is "${SCALE_ACCEPTANCE}" "${EXPECTED_SCALE_ACCEPTANCE_SHA}"
full_sha_is "${SPLIT_MANIFEST}" "${EXPECTED_SPLIT_SHA}" "scene split"
full_sha_is "${WEIGHTS}" "${EXPECTED_WEIGHTS_SHA}" "LingBot weights"
prefix_identity_is "${BASE_SIF}" "${EXPECTED_SIF_BYTES}" "${EXPECTED_SIF_HEAD_SHA}" "base SIF"
prefix_identity_is "${OVERLAY}" "${EXPECTED_OVERLAY_BYTES}" \
  "${EXPECTED_OVERLAY_HEAD_SHA}" "dataset overlay"
[[ -d "${RAW_SCENE}" ]] || fail "audited YmJ raw scene is absent"
[[ -L "${MEMNAV_PY}" && "$(readlink -f "${MEMNAV_PY}")" == "${EXPECTED_PYTHON_TARGET}" ]] || \
  fail "MemNav Python target changed"
full_sha_is "${EXPECTED_PYTHON_TARGET}" "${EXPECTED_PYTHON_SHA}" "MemNav Python"
lingbot_commit="$(git -c safe.directory="${LINGBOT_REPO}" -C "${LINGBOT_REPO}" rev-parse HEAD)"
lingbot_tree="$(git -c safe.directory="${LINGBOT_REPO}" -C "${LINGBOT_REPO}" rev-parse 'HEAD^{tree}')"
[[ "${lingbot_commit}" == "${EXPECTED_LINGBOT_COMMIT}" && \
   "${lingbot_tree}" == "${EXPECTED_LINGBOT_TREE}" ]] || fail "LingBot identity changed"
[[ -z "$(git -c safe.directory="${LINGBOT_REPO}" -C "${LINGBOT_REPO}" \
  status --porcelain=v1 --untracked-files=all)" ]] || fail "LingBot checkout is dirty"

if [[ "${PHASE_B_ROLE}" == "train" ]]; then
  SELECTION_MODE="train_augmented"
  KIND="manifest_causal_goal_localization_train"
  TOP_K=2
  EXPECTED_SESSIONS=480
  EXPECTED_SEEDS=1098
  EXPECTED_SCENES=40
else
  SELECTION_MODE="deployment"
  KIND="manifest_causal_goal_localization_development"
  TOP_K=8
  EXPECTED_SESSIONS=120
  EXPECTED_SEEDS=933
  EXPECTED_SCENES=10
fi
ROLE_ROOT="${OUTPUT_ROOT}/${PHASE_B_ROLE}_top${TOP_K}_gap16"
mkdir -p "${OUTPUT_ROOT}"
exec 9>"${OUTPUT_ROOT}/.${PHASE_B_ROLE}.lock"
flock -n 9 || fail "another ${PHASE_B_ROLE} collector is active"
if [[ -e "${ROLE_ROOT}" && ! -f "${ROLE_ROOT}/lingbot_goal_loop_closure_checkpoint.sqlite3" ]]; then
  fail "partial role directory lacks a resumable checkpoint: ${ROLE_ROOT}"
fi

CONTAINER=(singularity exec --nv --overlay "${OVERLAY}:ro"
  -B /scratch/lg154
  -B /scratch/yz11502
  -B "${RAW_SCENE}:${EPISODE_ROOT}/YmJkqBEsHnH:ro"
  "${BASE_SIF}")
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export APPTAINERENV_PYTHONPATH="${PYTHONPATH}"
export SINGULARITYENV_PYTHONPATH="${PYTHONPATH}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export GIT_CONFIG_COUNT=1
export GIT_CONFIG_KEY_0=safe.directory
export GIT_CONFIG_VALUE_0="${LINGBOT_REPO}"

gpu_name="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)"
[[ "${gpu_name}" == NVIDIA\ A100* || "${gpu_name}" == NVIDIA\ H100* ]] || \
  fail "collector requires an A100/H100 class GPU: ${gpu_name}"
echo "host=$(hostname) job=${SLURM_JOB_ID:-manual} role=${PHASE_B_ROLE} seeds=${EXPECTED_SEEDS}"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

COMMON_ARGS=(
  --internnav-root "${REPO_ROOT}/InternNav"
  --teacher-csv "${TEACHER_CSV}"
  --split-manifest "${SPLIT_MANIFEST}"
  --allowed-role "${PHASE_B_ROLE}"
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
  --out-dir "${ROLE_ROOT}"
  --selection-mode "${SELECTION_MODE}"
  --kind "${KIND}"
  --max-sessions 0
  --top-k "${TOP_K}"
  --candidate-min-gap 16
  --neighbor-offset=0
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

"${CONTAINER[@]}" "${MEMNAV_PY}" -m py_compile \
  "${REPO_ROOT}/${ENTRYPOINT_REL}" "${REPO_ROOT}/${HELPER_REL}"
"${CONTAINER[@]}" "${MEMNAV_PY}" - <<'PY'
import importlib
import json
import torch
required = ("numpy", "pandas", "PIL", "sklearn")
for name in required:
    importlib.import_module(name)
from internnav.model.basemodel.memnav.lingbot_stream import LingBotStream  # noqa: F401
if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable in the collector runtime")
print(json.dumps({"dependency_preflight": "passed",
                  "torch": torch.__version__,
                  "cuda_device": torch.cuda.get_device_name(0)}, sort_keys=True))
PY
"${CONTAINER[@]}" "${MEMNAV_PY}" -m unittest -v \
  MemNavData.test_phase_b_upstream_receipts \
  MemNavData.test_lingbot_goal_loop_closure
"${CONTAINER[@]}" "${MEMNAV_PY}" "${REPO_ROOT}/${ENTRYPOINT_REL}" \
  "${COMMON_ARGS[@]}" --preflight-only

if [[ -f "${ROLE_ROOT}/lingbot_goal_loop_closure_progress.json" ]] && \
   "${CONTAINER[@]}" "${MEMNAV_PY}" - "${ROLE_ROOT}" \
     "${EXPECTED_SESSIONS}" "${EXPECTED_SEEDS}" <<'PY'
import json
from pathlib import Path
import sys
root = Path(sys.argv[1])
progress = json.loads((root / "lingbot_goal_loop_closure_progress.json").read_text())
complete = (
    progress.get("status") == "complete"
    and progress.get("completed_sessions") == int(sys.argv[2])
    and progress.get("completed_seeds") == int(sys.argv[3])
)
raise SystemExit(0 if complete else 1)
PY
then
  echo "[complete] ${PHASE_B_ROLE} feature join already finished: ${ROLE_ROOT}"
  exit 0
fi

resume_args=()
[[ -f "${ROLE_ROOT}/lingbot_goal_loop_closure_checkpoint.sqlite3" ]] && resume_args+=(--resume)
mkdir -p "${ROLE_ROOT}"
"${CONTAINER[@]}" "${MEMNAV_PY}" -u "${REPO_ROOT}/${ENTRYPOINT_REL}" \
  "${COMMON_ARGS[@]}" "${resume_args[@]}"

"${CONTAINER[@]}" "${MEMNAV_PY}" - "${ROLE_ROOT}" \
  "${EXPECTED_SESSIONS}" "${EXPECTED_SEEDS}" "${EXPECTED_SCENES}" <<'PY'
import json
from pathlib import Path
import pandas as pd
import sys
root = Path(sys.argv[1])
progress = json.loads((root / "lingbot_goal_loop_closure_progress.json").read_text())
rows = pd.read_csv(root / "lingbot_goal_loop_closure_rows.csv")
if progress.get("status") != "complete":
    raise SystemExit("collector did not finish")
if progress.get("completed_sessions") != int(sys.argv[2]):
    raise SystemExit("completed-session count changed")
if progress.get("completed_seeds") != int(sys.argv[3]) or len(rows) != int(sys.argv[3]):
    raise SystemExit("seed/row exact cover changed")
if rows["scene"].nunique() != int(sys.argv[4]):
    raise SystemExit("scene cover changed")
print(json.dumps({"phase_b_feature_join": "complete", "rows": len(rows),
                  "sessions": rows["session_id"].nunique()}, sort_keys=True))
PY
