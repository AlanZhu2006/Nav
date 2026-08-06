#!/bin/bash
# Collect deployment-style LingBot-native loop-factor and metric-pose data.
# The same file serves smoke and full runs; only session/candidate counts vary.
# It never writes the source dataset, feature cache, LingBot checkout, or weights.
#
# Example development smoke:
#   sbatch --export=ALL,EXPECTED_COMMIT=<sha>,RUN_PREFIX=dev_smoke,\
#SELECTION_MODE=deployment,KIND=cross_episode_evaluation,SCENE_ROLE=development,\
#LOOP_SESSIONS='<positive>|<no-match>',MAX_SESSIONS=0,TOP_K=4,\
#NEIGHBOR_OFFSETS='-4|0|4' \
#     MemNavData/slurm_lingbot_native_localizer.sbatch
#
# Example frozen development collection after the smoke passes:
#   sbatch --export=ALL,EXPECTED_COMMIT=<sha>,RUN_PREFIX=dev_full,\
#SELECTION_MODE=deployment,KIND=cross_episode_evaluation,SCENE_ROLE=development,\
#LOOP_SESSIONS=all,MAX_SESSIONS=0,TOP_K=4,NEIGHBOR_OFFSETS=0 \
#     MemNavData/slurm_lingbot_native_localizer.sbatch
#SBATCH --job-name=lb_native_loc
#SBATCH --output=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_lb_native_%j.out
#SBATCH --error=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_lb_native_%j.err
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --partition=h100_tandon
#SBATCH --gres=gpu:h100:1
#SBATCH --account=torch_pr_769_tandon_advanced
#SBATCH --export=ALL

set -euo pipefail
umask 0022

: "${EXPECTED_COMMIT:?submit with EXPECTED_COMMIT=<full git SHA>}"

RUN_PREFIX=${RUN_PREFIX:-smoke}
SELECTION_MODE=${SELECTION_MODE:-balanced}
KIND=${KIND:-revisit_b}
SCENE_ROLE=${SCENE_ROLE:-train}
LOOP_SESSIONS=${LOOP_SESSIONS:-17DRP5sb8fy/episode_0000/revisit_b}
MAX_SESSIONS=${MAX_SESSIONS:-1}
PER_CLASS=${PER_CLASS:-2}
TOP_K=${TOP_K:-4}
FULL_REPLAY=${FULL_REPLAY:-1}
NEIGHBOR_OFFSETS=${NEIGHBOR_OFFSETS:-0}
MAX_CACHED_EPISODES=${MAX_CACHED_EPISODES:-1}
RESUME_RUN_ROOT=${RESUME_RUN_ROOT:-}

[[ "${RUN_PREFIX}" =~ ^[A-Za-z0-9_.-]+$ ]] || {
  echo "ABORT: invalid RUN_PREFIX=${RUN_PREFIX}" >&2; exit 1; }
[[ "${SELECTION_MODE}" =~ ^(balanced|deployment)$ ]] || {
  echo "ABORT: invalid SELECTION_MODE=${SELECTION_MODE}" >&2; exit 1; }
[[ "${KIND}" =~ ^[A-Za-z0-9_]+$ ]] || {
  echo "ABORT: invalid KIND=${KIND}" >&2; exit 1; }
[[ "${SCENE_ROLE}" =~ ^(train|development|final_reserved)$ ]] || {
  echo "ABORT: invalid SCENE_ROLE=${SCENE_ROLE}" >&2; exit 1; }
[[ "${MAX_SESSIONS}" =~ ^[0-9]+$ ]] || {
  echo "ABORT: MAX_SESSIONS must be a non-negative integer" >&2; exit 1; }
[[ "${PER_CLASS}" =~ ^[1-9][0-9]*$ ]] || {
  echo "ABORT: PER_CLASS must be a positive integer" >&2; exit 1; }
[[ "${TOP_K}" =~ ^[1-9][0-9]*$ ]] || {
  echo "ABORT: TOP_K must be a positive integer" >&2; exit 1; }
[[ "${FULL_REPLAY}" =~ ^[01]$ ]] || {
  echo "ABORT: FULL_REPLAY must be 0 or 1" >&2; exit 1; }
[[ "${MAX_CACHED_EPISODES}" =~ ^[1-9][0-9]*$ ]] || {
  echo "ABORT: MAX_CACHED_EPISODES must be positive" >&2; exit 1; }
[[ "${NEIGHBOR_OFFSETS}" =~ ^-?[0-9]+(\|-?[0-9]+)*$ ]] || {
  echo "ABORT: invalid NEIGHBOR_OFFSETS=${NEIGHBOR_OFFSETS}" >&2; exit 1; }
[[ "|${NEIGHBOR_OFFSETS}|" == *"|0|"* ]] || {
  echo "ABORT: NEIGHBOR_OFFSETS must contain 0" >&2; exit 1; }
if [[ "${LOOP_SESSIONS}" != "all" ]]; then
  [[ "${LOOP_SESSIONS}" =~ ^[A-Za-z0-9_./|-]+$ ]] || {
    echo "ABORT: invalid LOOP_SESSIONS=${LOOP_SESSIONS}" >&2; exit 1; }
fi

ROOT=${ROOT:-/scratch/yz11502/Research/Nav-graph-blind}
RESULT_BASE=${RESULT_BASE:-/scratch/yz11502/Research/Nav-axis-uturn-results/lingbot_native_localizer_20260806}
RESUME=0
if [[ -n "${RESUME_RUN_ROOT}" ]]; then
  RUN_ROOT=$(realpath -m "${RESUME_RUN_ROOT}")
  result_base_absolute=$(realpath -m "${RESULT_BASE}")
  [[ "${RUN_ROOT}" == "${result_base_absolute}"/* ]] || {
    echo "ABORT: resume root must be below ${result_base_absolute}" >&2
    exit 1
  }
  RESUME=1
else
  RUN_ROOT=${RESULT_BASE}/${RUN_PREFIX}_job_${SLURM_JOB_ID}
fi
BASE_SIF=/share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif
OVERLAY=/scratch/lg154/Research/datasets/_overlays/mp3d_revisit_v0_pt1.sqf
MEMNAV_PY=/scratch/lg154/conda-envs/memnav/bin/python
LINGBOT_REPO=/scratch/lg154/Research/Nav/NavDP/baselines/memnav/lingbot-map
WEIGHTS=${LINGBOT_REPO}/weights/lingbot-map-long.pt
FEATURE_ROOT=/scratch/lg154/Research/datasets/mp3d_revisit_v0_feat_flowgate/vln_n1/traj_data
# Keep loop-factor labels aligned with the frozen listwise router/cache used by
# the graph evaluation.  Job 15315411 has the same candidate keys as the older
# table, but contains the corrected directional co-visibility labels.
TEACHER_CSV=/scratch/yz11502/Research/Nav-axis-uturn-results/patch_router_multiscene_20260805/full_job_15315411/covisibility_teacher_pairs.csv
SPLIT_MANIFEST=${ROOT}/MemNavData/router_multiscene_split_20260805.json
DIAGNOSTIC=${ROOT}/MemNavData/diag_lingbot_goal_loop_closure.py

EXPECTED_WEIGHT_SHA=832bc82cbae0bc9bbe946ef5ee1f7226abd8c0e183ccf8beddbb3d133576f409
EXPECTED_LINGBOT_COMMIT=7ff6f3ed0913d4d326f8f13bbb429c4ffc0195c2
EXPECTED_TEACHER_SHA=927d0e87fc6b53561d5e5546bbcd5d1b94e11907e0aaecd5c810f4fa8911110d
EXPECTED_SPLIT_SHA=97309c183e25cb3dd65472908748d55a94798a636db6157ab6fe120fca05cf7a
EXPECTED_OVERLAY_BYTES=128854888448
EXPECTED_OVERLAY_HEAD_SHA=43d803c2b39735e764f0c6df8f7c40a7035a1c23789e22fcc9831d29928c801b

for required in "${BASE_SIF}" "${OVERLAY}" "${MEMNAV_PY}" \
                "${WEIGHTS}" "${FEATURE_ROOT}" "${TEACHER_CSV}" \
                "${SPLIT_MANIFEST}" "${DIAGNOSTIC}"; do
  test -r "${required}" || {
    echo "ABORT: missing dependency ${required}" >&2; exit 1; }
done
[[ "$(git -C "${ROOT}" rev-parse HEAD)" == "${EXPECTED_COMMIT}" ]] || {
  echo "ABORT: remote worktree commit mismatch" >&2; exit 1; }
for task_file in \
  MemNavData/diag_lingbot_goal_loop_closure.py \
  MemNavData/test_lingbot_goal_loop_closure.py \
  MemNavData/slurm_lingbot_native_localizer.sbatch; do
  git -C "${ROOT}" diff --quiet -- "${task_file}" || {
    echo "ABORT: unstaged task-file change: ${task_file}" >&2; exit 1; }
  git -C "${ROOT}" diff --cached --quiet -- "${task_file}" || {
    echo "ABORT: staged task-file change: ${task_file}" >&2; exit 1; }
done
if [[ "${RESUME}" -eq 1 ]]; then
  test -r "${RUN_ROOT}/lingbot_goal_loop_closure_checkpoint.sqlite3" || {
    echo "ABORT: resume checkpoint is missing from ${RUN_ROOT}" >&2
    exit 1
  }
else
  [[ ! -e "${RUN_ROOT}" ]] || {
    echo "ABORT: output already exists: ${RUN_ROOT}" >&2; exit 1; }
fi

[[ "$(sha256sum "${TEACHER_CSV}" | awk '{print $1}')" == \
    "${EXPECTED_TEACHER_SHA}" ]] || {
  echo "ABORT: teacher CSV SHA mismatch" >&2; exit 1; }
[[ "$(sha256sum "${SPLIT_MANIFEST}" | awk '{print $1}')" == \
    "${EXPECTED_SPLIT_SHA}" ]] || {
  echo "ABORT: split manifest SHA mismatch" >&2; exit 1; }
[[ "$(stat -c '%s' "${OVERLAY}")" == "${EXPECTED_OVERLAY_BYTES}" ]] || {
  echo "ABORT: overlay size mismatch" >&2; exit 1; }
overlay_head_sha=$(dd if="${OVERLAY}" bs=1048576 count=1 status=none | \
  sha256sum | awk '{print $1}')
[[ "${overlay_head_sha}" == "${EXPECTED_OVERLAY_HEAD_SHA}" ]] || {
  echo "ABORT: overlay header SHA mismatch" >&2; exit 1; }

mkdir -p "${RESULT_BASE}"
cd "${ROOT}"
echo "host=$(hostname) job=${SLURM_JOB_ID} commit=${EXPECTED_COMMIT} mode=${SELECTION_MODE} kind=${KIND} role=${SCENE_ROLE} full_replay=${FULL_REPLAY} resume=${RESUME} max_cached_episodes=${MAX_CACHED_EPISODES}"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

COMMON_ARGS=(
  --internnav-root "${ROOT}/InternNav"
  --teacher-csv "${TEACHER_CSV}"
  --split-manifest "${SPLIT_MANIFEST}"
  --allowed-role "${SCENE_ROLE}"
  --feature-root "${FEATURE_ROOT}"
  --lingbot-repo "${LINGBOT_REPO}"
  --weights "${WEIGHTS}"
  --expected-weight-sha "${EXPECTED_WEIGHT_SHA}"
  --expected-lingbot-commit "${EXPECTED_LINGBOT_COMMIT}"
  --out-dir "${RUN_ROOT}"
  --selection-mode "${SELECTION_MODE}"
  --kind "${KIND}"
  --max-sessions "${MAX_SESSIONS}"
  --per-class "${PER_CLASS}"
  --top-k "${TOP_K}"
  --candidate-min-gap 4
  --warm 64
  --window 32
  --pixel-stride 10
  --max-points 768
  --max-cached-episodes "${MAX_CACHED_EPISODES}"
)
if [[ "${FULL_REPLAY}" == 1 ]]; then
  COMMON_ARGS+=(--full-replay)
fi
IFS='|' read -r -a neighbor_offsets <<< "${NEIGHBOR_OFFSETS}"
for offset in "${neighbor_offsets[@]}"; do
  COMMON_ARGS+=(--neighbor-offset="${offset}")
done
if [[ "${LOOP_SESSIONS}" != "all" ]]; then
  IFS='|' read -r -a selected_sessions <<< "${LOOP_SESSIONS}"
  for session in "${selected_sessions[@]}"; do
    [[ -n "${session}" ]] && COMMON_ARGS+=(--session "${session}")
  done
fi
if [[ "${RESUME}" -eq 1 ]]; then
  COMMON_ARGS+=(--resume)
fi

# Preflight and the real run execute in the exact same container and overlay.
CONTAINER=(singularity exec --nv --overlay "${OVERLAY}:ro"
  -B /scratch/lg154 -B /scratch/yz11502 "${BASE_SIF}" "${MEMNAV_PY}")
"${CONTAINER[@]}" -m unittest -v \
  MemNavData.test_lingbot_goal_loop_closure
"${CONTAINER[@]}" -m py_compile \
  "${DIAGNOSTIC}" "${ROOT}/MemNavData/test_lingbot_goal_loop_closure.py"
"${CONTAINER[@]}" "${DIAGNOSTIC}" "${COMMON_ARGS[@]}" --preflight-only

mkdir -p "${RUN_ROOT}"
exec "${CONTAINER[@]}" "${DIAGNOSTIC}" "${COMMON_ARGS[@]}"
