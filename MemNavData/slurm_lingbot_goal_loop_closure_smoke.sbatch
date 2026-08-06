#!/bin/bash
# Zero-training LingBot-native loop-closure feasibility smoke.
# Four candidate seeds (2 positive, 2 hard negative) x three nearby anchors.
# Every hypothesis replays continuously from the scale block through its anchor,
# matching the online pose-only controller instead of using a fixed warm prefix.
# Source data, caches, weights, and the MP3D overlay are mounted read-only.
#
# Submit with:
#   sbatch --export=ALL,EXPECTED_COMMIT=<full-sha> \
#     MemNavData/slurm_lingbot_goal_loop_closure_smoke.sbatch
#SBATCH --job-name=lb_loop_smoke
#SBATCH --output=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_lb_loop_%j.out
#SBATCH --error=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_lb_loop_%j.err
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=h200_public
#SBATCH --gres=gpu:h200:1
#SBATCH --account=torch_pr_769_tandon_advanced
#SBATCH --export=ALL

set -euo pipefail
umask 0022

: "${EXPECTED_COMMIT:?submit with EXPECTED_COMMIT=<full git SHA>}"

# Defaults reproduce the original 2-positive/2-negative single-session smoke.
# Overrides permit a larger development-only audit without maintaining a
# second Slurm script.
RUN_PREFIX=${RUN_PREFIX:-smoke}
LOOP_SESSIONS=${LOOP_SESSIONS:-17DRP5sb8fy/episode_0000/revisit_b}
MAX_SESSIONS=${MAX_SESSIONS:-1}
PER_CLASS=${PER_CLASS:-2}
[[ "${RUN_PREFIX}" =~ ^[A-Za-z0-9_.-]+$ ]] || {
  echo "ABORT: invalid RUN_PREFIX=${RUN_PREFIX}" >&2; exit 1; }
[[ "${MAX_SESSIONS}" =~ ^[0-9]+$ ]] || {
  echo "ABORT: MAX_SESSIONS must be a non-negative integer" >&2; exit 1; }
[[ "${PER_CLASS}" =~ ^[1-9][0-9]*$ ]] || {
  echo "ABORT: PER_CLASS must be a positive integer" >&2; exit 1; }
if [[ "${LOOP_SESSIONS}" != "all" ]]; then
  [[ "${LOOP_SESSIONS}" =~ ^[A-Za-z0-9_./|-]+$ ]] || {
    echo "ABORT: invalid LOOP_SESSIONS=${LOOP_SESSIONS}" >&2; exit 1; }
fi

ROOT=${ROOT:-/scratch/yz11502/Research/Nav-axis-uturn}
RESULT_BASE=${RESULT_BASE:-/scratch/yz11502/Research/Nav-axis-uturn-results/lingbot_goal_loop_closure_20260805}
RUN_ROOT=${RESULT_BASE}/${RUN_PREFIX}_job_${SLURM_JOB_ID}
BASE_SIF=/share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif
OVERLAY=/scratch/lg154/Research/datasets/_overlays/mp3d_revisit_v0_pt1.sqf
MEMNAV_PY=/scratch/lg154/conda-envs/memnav/bin/python
LINGBOT_REPO=/scratch/lg154/Research/Nav/NavDP/baselines/memnav/lingbot-map
WEIGHTS=${LINGBOT_REPO}/weights/lingbot-map-long.pt
FEATURE_ROOT=/scratch/lg154/Research/datasets/mp3d_revisit_v0_feat_flowgate/vln_n1/traj_data
TEACHER_CSV=/scratch/yz11502/Research/Nav-axis-uturn-results/patch_router_multiscene_20260805/full_job_15315411/covisibility_teacher_pairs.csv
DIAGNOSTIC=${ROOT}/MemNavData/diag_lingbot_goal_loop_closure.py

EXPECTED_WEIGHT_SHA=832bc82cbae0bc9bbe946ef5ee1f7226abd8c0e183ccf8beddbb3d133576f409
EXPECTED_LINGBOT_COMMIT=7ff6f3ed0913d4d326f8f13bbb429c4ffc0195c2
EXPECTED_TEACHER_SHA=927d0e87fc6b53561d5e5546bbcd5d1b94e11907e0aaecd5c810f4fa8911110d
EXPECTED_OVERLAY_BYTES=128854888448
EXPECTED_OVERLAY_HEAD_SHA=43d803c2b39735e764f0c6df8f7c40a7035a1c23789e22fcc9831d29928c801b

mkdir -p "${RESULT_BASE}"
for required in "${BASE_SIF}" "${OVERLAY}" "${MEMNAV_PY}" \
                "${WEIGHTS}" "${FEATURE_ROOT}" "${TEACHER_CSV}" \
                "${DIAGNOSTIC}"; do
  test -r "${required}" || {
    echo "ABORT: missing dependency ${required}" >&2
    exit 1
  }
done

actual_commit=$(git -C "${ROOT}" rev-parse HEAD)
[[ "${actual_commit}" == "${EXPECTED_COMMIT}" ]] || {
  echo "ABORT: code commit ${actual_commit} != ${EXPECTED_COMMIT}" >&2
  exit 1
}
git -C "${ROOT}" diff --quiet -- \
  MemNavData/diag_lingbot_goal_loop_closure.py \
  MemNavData/slurm_lingbot_goal_loop_closure_smoke.sbatch || {
    echo "ABORT: loop-closure task files differ from commit" >&2
    exit 1
  }
git -C "${ROOT}" diff --cached --quiet -- \
  MemNavData/diag_lingbot_goal_loop_closure.py \
  MemNavData/slurm_lingbot_goal_loop_closure_smoke.sbatch || {
    echo "ABORT: staged loop-closure task files differ from commit" >&2
    exit 1
  }

teacher_sha=$(sha256sum "${TEACHER_CSV}" | awk '{print $1}')
[[ "${teacher_sha}" == "${EXPECTED_TEACHER_SHA}" ]] || {
  echo "ABORT: teacher CSV SHA mismatch ${teacher_sha}" >&2
  exit 1
}
overlay_bytes=$(stat -c '%s' "${OVERLAY}")
overlay_head_sha=$(dd if="${OVERLAY}" bs=1048576 count=1 status=none | \
  sha256sum | awk '{print $1}')
[[ "${overlay_bytes}" == "${EXPECTED_OVERLAY_BYTES}" ]] || {
  echo "ABORT: overlay size mismatch ${overlay_bytes}" >&2
  exit 1
}
[[ "${overlay_head_sha}" == "${EXPECTED_OVERLAY_HEAD_SHA}" ]] || {
  echo "ABORT: overlay header SHA mismatch ${overlay_head_sha}" >&2
  exit 1
}

echo "host=$(hostname) job=${SLURM_JOB_ID} commit=${actual_commit}"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

COMMON_ARGS=(
  --internnav-root "${ROOT}/InternNav"
  --teacher-csv "${TEACHER_CSV}"
  --feature-root "${FEATURE_ROOT}"
  --lingbot-repo "${LINGBOT_REPO}"
  --weights "${WEIGHTS}"
  --expected-weight-sha "${EXPECTED_WEIGHT_SHA}"
  --expected-lingbot-commit "${EXPECTED_LINGBOT_COMMIT}"
  --out-dir "${RUN_ROOT}"
  --max-sessions "${MAX_SESSIONS}"
  --per-class "${PER_CLASS}"
  --candidate-min-gap 4
  --neighbor-offset=-4
  --neighbor-offset=0
  --neighbor-offset=4
  --warm 64
  --full-replay
  --window 32
  --pixel-stride 10
  --max-points 768
)
# Use ``|`` rather than comma because Slurm's --export syntax reserves comma
# as its own variable delimiter.
if [[ "${LOOP_SESSIONS}" != "all" ]]; then
  IFS='|' read -r -a SELECTED_SESSIONS <<< "${LOOP_SESSIONS}"
  for session in "${SELECTED_SESSIONS[@]}"; do
    [[ -n "${session}" ]] && COMMON_ARGS+=(--session "${session}")
  done
fi

# The Python preflight checks imports, teacher columns, selected images, paired
# cache schema, LingBot commit, and the complete 4.4-GB weight SHA before the
# model is allocated on GPU.
singularity exec --nv \
  --overlay "${OVERLAY}:ro" \
  -B /scratch/lg154 \
  -B /scratch/yz11502 \
  "${BASE_SIF}" \
  "${MEMNAV_PY}" "${DIAGNOSTIC}" "${COMMON_ARGS[@]}" --preflight-only

mkdir -p "${RUN_ROOT}"
exec singularity exec --nv \
  --overlay "${OVERLAY}:ro" \
  -B /scratch/lg154 \
  -B /scratch/yz11502 \
  "${BASE_SIF}" \
  "${MEMNAV_PY}" "${DIAGNOSTIC}" "${COMMON_ARGS[@]}"
