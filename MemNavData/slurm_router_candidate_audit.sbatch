#!/bin/bash
# CPU-only follow-up that reuses the frozen full_job_15289458 candidate table.
# Submit with:
#   sbatch --export=ALL,EXPECTED_COMMIT=<full SHA> <this-file>
#SBATCH --job-name=router_recall
#SBATCH --output=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_router_recall_%j.out
#SBATCH --error=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_router_recall_%j.err
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --partition=cpu_short
#SBATCH --account=torch_pr_769_tandon_advanced
#SBATCH --export=ALL

set -euo pipefail
umask 0022

ROOT=/scratch/yz11502/Research/Nav-axis-uturn
SOURCE_RUN=/scratch/yz11502/Research/Nav-axis-uturn-results/patch_router_multiscene_20260805/full_job_15289458
SOURCE_CSV=${SOURCE_RUN}/expanded_teacher_pairs.csv
RESULT_BASE=/scratch/yz11502/Research/Nav-axis-uturn-results/router_candidate_recall_20260805
BASE_SIF=/share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif
OVERLAY=/scratch/lg154/Research/datasets/_overlays/mp3d_revisit_v0_pt1.sqf
MEMNAV_PY=/scratch/lg154/conda-envs/memnav/bin/python
SPLIT_MANIFEST=${ROOT}/MemNavData/router_multiscene_split_20260805.json
RUNNER=${ROOT}/MemNavData/run_router_candidate_audit.sh
EXPECTED_INPUT_SHA=a32d2b59f0503eabea72ad7bca4a04d7ff2d9621f4b7bb1de01168a71177ca08
EXPECTED_OVERLAY_BYTES=128854888448
EXPECTED_OVERLAY_HEAD_SHA=43d803c2b39735e764f0c6df8f7c40a7035a1c23789e22fcc9831d29928c801b
: "${EXPECTED_COMMIT:?submit with EXPECTED_COMMIT=<full git SHA>}"

mkdir -p "${RESULT_BASE}"
for required in "${BASE_SIF}" "${OVERLAY}" "${MEMNAV_PY}" \
                "${SOURCE_CSV}" "${SPLIT_MANIFEST}" "${RUNNER}"; do
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
TASK_FILES=(
  MemNavData/relabel_router_covisibility.py
  MemNavData/audit_router_candidate_recall.py
  MemNavData/test_router_candidate_recall.py
  MemNavData/run_router_candidate_audit.sh
  MemNavData/slurm_router_candidate_audit.sbatch
  MemNavData/router_multiscene_split_20260805.json
)
git -C "${ROOT}" diff --quiet -- "${TASK_FILES[@]}" || {
  echo "ABORT: candidate-audit files differ from commit" >&2; exit 1; }
git -C "${ROOT}" diff --cached --quiet -- "${TASK_FILES[@]}" || {
  echo "ABORT: staged candidate-audit files differ from commit" >&2; exit 1; }

overlay_bytes=$(stat -c '%s' "${OVERLAY}")
overlay_head_sha=$(dd if="${OVERLAY}" bs=1048576 count=1 status=none | \
  sha256sum | awk '{print $1}')
[[ "${overlay_bytes}" == "${EXPECTED_OVERLAY_BYTES}" ]] || {
  echo "ABORT: overlay size mismatch ${overlay_bytes}" >&2; exit 1; }
[[ "${overlay_head_sha}" == "${EXPECTED_OVERLAY_HEAD_SHA}" ]] || {
  echo "ABORT: overlay header SHA mismatch ${overlay_head_sha}" >&2; exit 1; }

echo "host=$(hostname) job=${SLURM_JOB_ID:-none} commit=${actual_commit}"
exec singularity exec \
  --overlay "${OVERLAY}:ro" \
  -B /scratch/lg154 \
  -B /scratch/yz11502 \
  "${BASE_SIF}" \
  env \
    ROOT="${ROOT}" \
    SOURCE_CSV="${SOURCE_CSV}" \
    RUN_ROOT="${RESULT_BASE}/job_${SLURM_JOB_ID}" \
    MEMNAV_PY="${MEMNAV_PY}" \
    SPLIT_MANIFEST="${SPLIT_MANIFEST}" \
    EXPECTED_INPUT_SHA="${EXPECTED_INPUT_SHA}" \
    "${RUNNER}"
