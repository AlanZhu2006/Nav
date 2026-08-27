#!/usr/bin/env bash
# Hardware-paired train-only collection: both proposals in one LingBot process.
#SBATCH --job-name=cdec_pnp
#SBATCH --output=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/%x_%j.out
#SBATCH --error=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/%x_%j.err
#SBATCH --time=16:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --partition=a100_tandon,h100_tandon
#SBATCH --qos=gpu48
#SBATCH --gres=gpu:1
#SBATCH --account=torch_pr_769_tandon_advanced
#SBATCH --requeue
#SBATCH --export=ALL

set -euo pipefail
umask 0022
: "${SOURCE_ROOT:?set immutable task source root}"
: "${SOURCE_RECEIPT:?set source receipt}"
: "${EXPECTED_SOURCE_RECEIPT_SHA:?set source receipt SHA}"
: "${RUN_ROOT:?set unique result root}"

BASE_REPO_ROOT=/scratch/yz11502/Research/Nav-graph-blind
EXPECTED_BASE_COMMIT=32145f3a3c1ae9f98d9fddbfa8e942e332c7356e
DEPENDENCY_BUNDLE=/scratch/yz11502/Research/Nav-axis-uturn-source-bundles/lingbot_lightglue_relocalization_f73016d5ecfebfa7
EXPECTED_DEPENDENCY_MANIFEST_SHA=f73016d5ecfebfa7c0cd28e34ea80890c81832d714365ff712b6d5d58ad57ecb
UPSTREAM_ROOT=/scratch/yz11502/Research/Nav-axis-uturn-results/nlsr_gapfill_20260807/nlsr_routes_41c1773_20260807_v1
UPSTREAM=${UPSTREAM_ROOT}/causal_covisibility_teacher_a55ae5d40ba3_bc6bf58536f6/teacher_bundle
MANIFEST=${UPSTREAM_ROOT}/manifest_multistage/candidate_manifest_multistage.json
SCALE_ARTIFACT=${UPSTREAM_ROOT}/causal_ground_scale/causal_ground_scale.json
SCALE_ACCEPTANCE=${UPSTREAM_ROOT}/causal_ground_scale_acceptance_c8f499dc9c4f_f6f28e20b9f2/causal_ground_scale_acceptance.json
TEACHER_CSV=${UPSTREAM}/teacher.csv
TEACHER_AUDIT=${UPSTREAM}/audit.json
SPLIT_MANIFEST=${BASE_REPO_ROOT}/MemNavData/router_multiscene_split_20260805.json
PRIOR_ROOT=/scratch/yz11502/Research/Nav-axis-uturn-results/lingbot_lightglue_relocalization_v2_20260812/bundle_f73016d5ecfebfa7_job_15634113
STATIC_CSV=${PRIOR_ROOT}/static_top8_480/lightglue_open_set_rows.csv
STATIC_REPORT=${PRIOR_ROOT}/static_top8_480/lightglue_open_set_report.json
CDEC_CSV=${SOURCE_ROOT}/artifacts/cdec_oof_selection_rows.csv
CDEC_REPORT=${SOURCE_ROOT}/artifacts/cdec_pairwise_oof_report.json
ENTRYPOINT=${SOURCE_ROOT}/MemNavData/diag_lingbot_goal_loop_closure.py
ACTIONABILITY=${SOURCE_ROOT}/MemNavData/summarize_lingbot_lightglue_localization.py
LIGHTGLUE_REPO=${DEPENDENCY_BUNDLE}/third_party/LightGlue
DEPENDENCY_ROOT=${DEPENDENCY_BUNDLE}/third_party/python
TORCH_HOME=${DEPENDENCY_BUNDLE}/torch_home
MEMNAV_PY=/scratch/lg154/conda-envs/memnav/bin/python
BASE_SIF=/share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif
OVERLAY=/scratch/lg154/Research/datasets/_overlays/mp3d_revisit_v0_pt1.sqf
LINGBOT_REPO=/scratch/lg154/Research/Nav/NavDP/baselines/memnav/lingbot-map
WEIGHTS=${LINGBOT_REPO}/weights/lingbot-map-long.pt
RAW_GAP_SCENE=${UPSTREAM_ROOT}/raw/mp3d_3leg/YmJkqBEsHnH
EPISODE_ROOT=/mp3d_revisit_v0/vln_n1/traj_data/mp3d_3leg

EXPECTED_MANIFEST_SHA=bc6bf58536f6c159d1898ac03abe365eadba65c22cd246e39401821962abb34c
EXPECTED_TEACHER_SHA=fd52dcfcb7e8d1bdac0703f7ba0b6d10f04067d4b8793386adf65ede6bc8e313
EXPECTED_TEACHER_AUDIT_SHA=cd7091bb8c398ac5a7f558003a3fb467cc450a9b65e4416e29b7d27e1fad14c6
EXPECTED_SCALE_SHA=f6f28e20b9f21a764132af04b425f23b65e1dc779798d831a571a7df26bd78db
EXPECTED_SCALE_ACCEPTANCE_SHA=0ced1b3a56fe0e089bad35c5807c219da27daad48e6bbd4bb29436ef2b8eb3ac
EXPECTED_SCALE_ACCEPTANCE_COMMIT=c8f499dc9c4f9112087d26a0daba46f3c3c82d59
EXPECTED_SCALE_PRODUCER_SHA=8da2a4c42de010e699c5ab65a2f6bbc9600c29acc96b71ef82d89126d8b4ca26
EXPECTED_SCALE_CONFIG_SHA=f1b01a8b6cd07a8cfb0ee306083e36d6399d21f22f242b3e2eb0f5ee7624945d
EXPECTED_LINGBOT_COMMIT=7ff6f3ed0913d4d326f8f13bbb429c4ffc0195c2
EXPECTED_WEIGHTS_SHA=832bc82cbae0bc9bbe946ef5ee1f7226abd8c0e183ccf8beddbb3d133576f409
EXPECTED_STREAM_SHA=b5142b76f9062634db826a0e7f2c3da19e4b07e21c2325ecbc60a44cc52370b0
EXPECTED_STATIC_CSV_SHA=85f9064bff15ce59106ad2a1aa8e5dc4720ee1b1ad894aac1bcedf8581a1d127
EXPECTED_STATIC_REPORT_SHA=5be98839d04a7e9ebc8fcd198f511c0854a3b45686df0a54f58263fe516914c8
EXPECTED_CDEC_CSV_SHA=65a3f7ccc876e7fbf6857b5b4a13f8e9061e092ecc76c4273c0ffec6bf0133c3
EXPECTED_CDEC_REPORT_SHA=88417e3b02dfd16bf94c4dbd0353372a505700bce969a4cf787bb5c3897211e6

fail() { echo "ABORT: $*" >&2; exit 2; }
sha_is() {
  [[ -f "$1" && ! -L "$1" ]] || fail "missing physical input $1"
  [[ "$(sha256sum "$1" | awk '{print $1}')" == "$2" ]] || fail "SHA mismatch $1"
}
sha_is "${SOURCE_RECEIPT}" "${EXPECTED_SOURCE_RECEIPT_SHA}"
(cd "${SOURCE_ROOT}" && sha256sum -c "${SOURCE_RECEIPT}") >/dev/null || \
  fail "task source bundle changed"
sha_is "${DEPENDENCY_BUNDLE}/source_bundle_manifest.json" \
  "${EXPECTED_DEPENDENCY_MANIFEST_SHA}"
[[ "$(git -C "${BASE_REPO_ROOT}" rev-parse HEAD)" == "${EXPECTED_BASE_COMMIT}" ]] || \
  fail "base checkout changed"
sha_is "${MANIFEST}" "${EXPECTED_MANIFEST_SHA}"
sha_is "${TEACHER_CSV}" "${EXPECTED_TEACHER_SHA}"
sha_is "${TEACHER_AUDIT}" "${EXPECTED_TEACHER_AUDIT_SHA}"
sha_is "${SCALE_ARTIFACT}" "${EXPECTED_SCALE_SHA}"
sha_is "${SCALE_ACCEPTANCE}" "${EXPECTED_SCALE_ACCEPTANCE_SHA}"
sha_is "${WEIGHTS}" "${EXPECTED_WEIGHTS_SHA}"
sha_is "${STATIC_CSV}" "${EXPECTED_STATIC_CSV_SHA}"
sha_is "${STATIC_REPORT}" "${EXPECTED_STATIC_REPORT_SHA}"
sha_is "${CDEC_CSV}" "${EXPECTED_CDEC_CSV_SHA}"
sha_is "${CDEC_REPORT}" "${EXPECTED_CDEC_REPORT_SHA}"
[[ -f "${BASE_SIF}" && -f "${OVERLAY}" && -d "${RAW_GAP_SCENE}" ]] || \
  fail "container/overlay/gap scene missing"
[[ "$(git -c safe.directory="${LINGBOT_REPO}" -C "${LINGBOT_REPO}" rev-parse HEAD)" == \
   "${EXPECTED_LINGBOT_COMMIT}" ]] || fail "LingBot commit changed"

CONTAINER=(singularity exec --nv --overlay "${OVERLAY}:ro"
  -B /scratch/lg154 -B /scratch/yz11502
  -B "${RAW_GAP_SCENE}:${EPISODE_ROOT}/YmJkqBEsHnH:ro" "${BASE_SIF}")
export PYTHONPYCACHEPREFIX=${SLURM_TMPDIR:-/tmp}/cdec_dual_pnp_pycache_${SLURM_JOB_ID}
mkdir -p "${PYTHONPYCACHEPREFIX}"
export TORCH_HOME
export PYTHONPATH=${SOURCE_ROOT}:${BASE_REPO_ROOT}:${DEPENDENCY_ROOT}:${LIGHTGLUE_REPO}${PYTHONPATH:+:${PYTHONPATH}}
export APPTAINERENV_PYTHONPATH=${PYTHONPATH}
export SINGULARITYENV_PYTHONPATH=${PYTHONPATH}
export APPTAINERENV_TORCH_HOME=${TORCH_HOME}
export SINGULARITYENV_TORCH_HOME=${TORCH_HOME}
export APPTAINERENV_PYTHONPYCACHEPREFIX=${PYTHONPYCACHEPREFIX}
export SINGULARITYENV_PYTHONPYCACHEPREFIX=${PYTHONPYCACHEPREFIX}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export GIT_CONFIG_COUNT=2 GIT_CONFIG_KEY_0=safe.directory
export GIT_CONFIG_VALUE_0=${LINGBOT_REPO}
export GIT_CONFIG_KEY_1=safe.directory GIT_CONFIG_VALUE_1=${LIGHTGLUE_REPO}

POLICY=geometry_first_cdec_reject_fallback
SELECTION_ARGS=(
  --selection-mode cdec_geometry_dual_ranked
  --lightglue-selection-csv "${STATIC_CSV}"
  --lightglue-selection-report "${STATIC_REPORT}"
  --expected-lightglue-selection-csv-sha256 "${EXPECTED_STATIC_CSV_SHA}"
  --expected-lightglue-selection-report-sha256 "${EXPECTED_STATIC_REPORT_SHA}"
  --cdec-selection-csv "${CDEC_CSV}"
  --cdec-selection-report "${CDEC_REPORT}"
  --expected-cdec-selection-csv-sha256 "${EXPECTED_CDEC_CSV_SHA}"
  --expected-cdec-selection-report-sha256 "${EXPECTED_CDEC_REPORT_SHA}"
  --cdec-training-rows-csv "${STATIC_CSV}"
  --expected-cdec-training-rows-sha256 "${EXPECTED_STATIC_CSV_SHA}"
)
OUT=${RUN_ROOT}/dual
mkdir -p "${RUN_ROOT}"
[[ ! -e "${OUT}/diagnostic_lingbot_goal_loop_closure.json" ]] || \
  fail "completed output already exists: ${OUT}"

COMMON=(
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
  --out-dir "${OUT}" --kind manifest_causal_goal_localization_train
  --max-sessions 0 --top-k 8 --candidate-min-gap 4
  --neighbor-offset=0
  --warm 64 --full-replay --num-scale 8 --window 32 --max-frame-num 4096
  --camera-num-iterations 4 --pixel-stride 10 --confidence-quantile 0.5
  --max-points 768 --max-cached-episodes 1 --device cuda
  --pnp-lightglue --lightglue-repo "${LIGHTGLUE_REPO}"
  --lightglue-dependency-root "${DEPENDENCY_ROOT}"
  "${SELECTION_ARGS[@]}"
)

echo "host=$(hostname) job=${SLURM_JOB_ID} policy=${POLICY} gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1)"
"${CONTAINER[@]}" "${MEMNAV_PY}" -m py_compile "${ENTRYPOINT}" "${ACTIONABILITY}"
"${CONTAINER[@]}" "${MEMNAV_PY}" -m pytest -q -p no:cacheprovider \
  "${SOURCE_ROOT}/MemNavData/test_lingbot_goal_loop_closure.py" \
  "${SOURCE_ROOT}/MemNavData/test_lingbot_pnp_localization.py"
"${CONTAINER[@]}" "${MEMNAV_PY}" "${ENTRYPOINT}" "${COMMON[@]}" --preflight-only
resume=()
[[ -f "${OUT}/lingbot_goal_loop_closure_checkpoint.sqlite3" ]] && resume+=(--resume)
"${CONTAINER[@]}" "${MEMNAV_PY}" -u "${ENTRYPOINT}" "${COMMON[@]}" "${resume[@]}"
sha256sum "${OUT}/lingbot_goal_loop_closure_rows.csv" \
  "${OUT}/diagnostic_lingbot_goal_loop_closure.json" > "${OUT}/OUTPUTS.sha256"
chmod a-w "${OUT}/lingbot_goal_loop_closure_rows.csv" \
  "${OUT}/diagnostic_lingbot_goal_loop_closure.json" "${OUT}/OUTPUTS.sha256"
echo "COMPLETE policy=${POLICY} out=${OUT}"
