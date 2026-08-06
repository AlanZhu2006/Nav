#!/bin/bash
# Strict CPU Phase-B training gate for the LingBot-native localizer.
#
# Submit only with an afterok dependency on the complete train collector.  The
# script still performs its own semantic artifact audit because Slurm afterok
# validates only the upstream process exit status.
#
# Example:
#   sbatch --dependency=afterok:<collector-job> --kill-on-invalid-dep=yes \
#     --export=ALL,EXPECTED_COMMIT=<trainer-sha>,\
#TRAIN_RUN_ROOT=<collector-output>,RUN_PREFIX=phase_b \
#     MemNavData/slurm_train_lingbot_native_localizer.sbatch
#SBATCH --job-name=lb_phase_b
#SBATCH --output=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_lb_phase_b_%j.out
#SBATCH --error=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_lb_phase_b_%j.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=cpu_short
#SBATCH --account=torch_pr_769_tandon_advanced
#SBATCH --export=ALL

set -euo pipefail
umask 0022

: "${EXPECTED_COMMIT:?submit with the exact trainer commit}"
: "${TRAIN_RUN_ROOT:?submit with the completed collector output directory}"

RUN_PREFIX=${RUN_PREFIX:-lingbot_native_phase_b}
WANDB_MODE=${WANDB_MODE:-online}
WANDB_PROJECT=${WANDB_PROJECT:-memnav}
WANDB_NAME=${WANDB_NAME:-${RUN_PREFIX}_${SLURM_JOB_ID}}
[[ "${RUN_PREFIX}" =~ ^[A-Za-z0-9_.-]+$ ]] || {
  echo "ABORT: invalid RUN_PREFIX=${RUN_PREFIX}" >&2; exit 1; }
[[ "${WANDB_MODE}" =~ ^(disabled|offline|online)$ ]] || {
  echo "ABORT: invalid WANDB_MODE=${WANDB_MODE}" >&2; exit 1; }
if [[ "${WANDB_MODE}" == "online" && -z "${WANDB_API_KEY:-}" ]]; then
  echo "ABORT: WANDB_MODE=online requires WANDB_API_KEY" >&2
  exit 1
fi

ROOT=${ROOT:-/scratch/yz11502/Research/Nav-graph-blind-phaseb}
RESULT_BASE=${RESULT_BASE:-/scratch/yz11502/Research/Nav-axis-uturn-results/lingbot_native_phase_b_20260807}
RUN_ROOT=${RESULT_BASE}/${RUN_PREFIX}_job_${SLURM_JOB_ID}
COLLECTOR_RESULT_BASE=/scratch/yz11502/Research/Nav-axis-uturn-results/lingbot_native_localizer_20260806
TRAIN_RUN_ROOT=$(realpath -m "${TRAIN_RUN_ROOT}")
[[ "${TRAIN_RUN_ROOT}" == "${COLLECTOR_RESULT_BASE}"/* ]] || {
  echo "ABORT: TRAIN_RUN_ROOT is outside the frozen collector result base" >&2
  exit 1
}

BASE_SIF=/share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif
MEMNAV_PY=/scratch/lg154/conda-envs/memnav/bin/python
TEACHER_CSV=/scratch/yz11502/Research/Nav-axis-uturn-results/patch_router_multiscene_20260805/full_job_15315411/covisibility_teacher_pairs.csv
SPLIT_MANIFEST=${ROOT}/MemNavData/router_multiscene_split_20260805.json
DEVELOPMENT_CSV=/scratch/yz11502/Research/Nav-axis-uturn-results/lingbot_native_localizer_20260806/dev_deploy_full_top4_center_job_15421650/lingbot_goal_loop_closure_rows.csv
AUDITOR=${ROOT}/MemNavData/audit_lingbot_native_localizer_artifact.py
TRAINER=${ROOT}/MemNavData/train_lingbot_native_localizer.py
TRAINING_TEST=${ROOT}/MemNavData/test_lingbot_native_training.py
SET_TRAINER=${ROOT}/MemNavData/train_neural_set_localizer.py

EXPECTED_COLLECTOR_COMMIT=32145f3a3c1ae9f98d9fddbfa8e942e332c7356e
EXPECTED_TEACHER_SHA=927d0e87fc6b53561d5e5546bbcd5d1b94e11907e0aaecd5c810f4fa8911110d
EXPECTED_SPLIT_SHA=97309c183e25cb3dd65472908748d55a94798a636db6157ab6fe120fca05cf7a
EXPECTED_DEVELOPMENT_SHA=f5f028ad2f6c0ea34e1d972da721ac5cc7d04c027e26517312b96d47f1e32d06

for required in \
  "${BASE_SIF}" \
  "${MEMNAV_PY}" \
  "${TRAIN_RUN_ROOT}/lingbot_goal_loop_closure_checkpoint.sqlite3" \
  "${TRAIN_RUN_ROOT}/lingbot_goal_loop_closure_progress.json" \
  "${TRAIN_RUN_ROOT}/lingbot_goal_loop_closure_rows.csv" \
  "${TRAIN_RUN_ROOT}/diagnostic_lingbot_goal_loop_closure.json" \
  "${TEACHER_CSV}" \
  "${SPLIT_MANIFEST}" \
  "${DEVELOPMENT_CSV}" \
  "${AUDITOR}" \
  "${TRAINER}" \
  "${TRAINING_TEST}" \
  "${SET_TRAINER}"; do
  test -r "${required}" || {
    echo "ABORT: missing dependency ${required}" >&2; exit 1; }
done

[[ "$(git -C "${ROOT}" rev-parse HEAD)" == "${EXPECTED_COMMIT}" ]] || {
  echo "ABORT: trainer worktree commit mismatch" >&2; exit 1; }
TASK_FILES=(
  MemNavData/audit_lingbot_native_localizer_artifact.py
  MemNavData/train_lingbot_native_localizer.py
  MemNavData/test_lingbot_native_training.py
  MemNavData/slurm_train_lingbot_native_localizer.sbatch
  MemNavData/train_neural_set_localizer.py
  MemNavData/router_multiscene_split_20260805.json
)
git -C "${ROOT}" diff --quiet -- "${TASK_FILES[@]}" || {
  echo "ABORT: unstaged training dependency change" >&2; exit 1; }
git -C "${ROOT}" diff --cached --quiet -- "${TASK_FILES[@]}" || {
  echo "ABORT: staged training dependency change" >&2; exit 1; }
[[ "$(sha256sum "${TEACHER_CSV}" | awk '{print $1}')" == \
    "${EXPECTED_TEACHER_SHA}" ]] || {
  echo "ABORT: corrected teacher SHA mismatch" >&2; exit 1; }
[[ "$(sha256sum "${SPLIT_MANIFEST}" | awk '{print $1}')" == \
    "${EXPECTED_SPLIT_SHA}" ]] || {
  echo "ABORT: split manifest SHA mismatch" >&2; exit 1; }
[[ "$(sha256sum "${DEVELOPMENT_CSV}" | awk '{print $1}')" == \
    "${EXPECTED_DEVELOPMENT_SHA}" ]] || {
  echo "ABORT: development exact-row SHA mismatch" >&2; exit 1; }
[[ ! -e "${RUN_ROOT}" ]] || {
  echo "ABORT: output already exists: ${RUN_ROOT}" >&2; exit 1; }

mkdir -p "${RESULT_BASE}"
cd "${ROOT}"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export PYTHONHASHSEED=0

CONTAINER=(singularity exec
  -B /scratch/lg154
  -B /scratch/yz11502
  "${BASE_SIF}"
  "${MEMNAV_PY}")

"${CONTAINER[@]}" -m py_compile \
  "${AUDITOR}" "${TRAINER}" "${TRAINING_TEST}" "${SET_TRAINER}"
"${CONTAINER[@]}" -m unittest -v \
  MemNavData.test_lingbot_native_training \
  MemNavData.test_neural_set_localizer

mkdir -p "${RUN_ROOT}"
AUDIT_REPORT=${RUN_ROOT}/train_artifact_audit.json
"${CONTAINER[@]}" -u "${AUDITOR}" \
  --run-dir "${TRAIN_RUN_ROOT}" \
  --teacher-csv "${TEACHER_CSV}" \
  --split-manifest "${SPLIT_MANIFEST}" \
  --expected-role train \
  --expected-scenes 40 \
  --expected-sessions 311 \
  --expected-seeds 1244 \
  --expected-rows 1244 \
  --expected-source-commit "${EXPECTED_COLLECTOR_COMMIT}" \
  --expected-teacher-sha256 "${EXPECTED_TEACHER_SHA}" \
  --expected-split-sha256 "${EXPECTED_SPLIT_SHA}" \
  --out-report "${AUDIT_REPORT}"

TRAIN_ARGS=(
  --train-run-dir "${TRAIN_RUN_ROOT}"
  --train-audit "${AUDIT_REPORT}"
  --development-csv "${DEVELOPMENT_CSV}"
  --teacher-csv "${TEACHER_CSV}"
  --split-manifest "${SPLIT_MANIFEST}"
  --out-dir "${RUN_ROOT}/model"
  --epochs 200
  --patience 25
  --batch-size 32
  --learning-rate 3e-4
  --weight-decay 1e-4
  --pose-weight 1.0
  --pose-tail-weight 0.5
  --pose-tail-fraction 0.2
  --hidden-dim 64
  --dropout 0.10
  --device cpu
)

# Exact production-artifact gradient smoke.  It touches no development metric,
# writes no checkpoint, and must show a finite non-zero gradient in every head.
"${CONTAINER[@]}" -u "${TRAINER}" \
  "${TRAIN_ARGS[@]}" \
  --wandb-mode disabled \
  --preflight-only

"${CONTAINER[@]}" -u "${TRAINER}" \
  "${TRAIN_ARGS[@]}" \
  --wandb-mode "${WANDB_MODE}" \
  --wandb-project "${WANDB_PROJECT}" \
  --wandb-name "${WANDB_NAME}"

echo "[complete] audit=${AUDIT_REPORT} model=${RUN_ROOT}/model/report.json"
