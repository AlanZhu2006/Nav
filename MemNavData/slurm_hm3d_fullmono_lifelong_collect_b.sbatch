#!/usr/bin/env bash
#SBATCH --job-name=h3lifeB
#SBATCH --output=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/%x_%A_%a.out
#SBATCH --error=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/%x_%A_%a.err
# One scene may contain four factual-B rollouts.  Keep one server pair alive
# for the scene, but budget the wall clock for all four 600-step rollouts.
#SBATCH --time=01:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=72G
#SBATCH --partition=h100_tandon,a100_tandon
#SBATCH --qos=gpu48
#SBATCH --gres=gpu:1
#SBATCH --account=torch_pr_769_tandon_advanced

set -euo pipefail
: "${TASK_ROOT:?}" "${BASE_SOURCE_ROOT:?}" "${RUN_ROOT:?}"
: "${PARENT_ROOT:?}" "${PROTOCOL:?}" "${SLURM_ARRAY_TASK_ID:=0}"
BASE_SIF=/share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif
exec singularity exec --nv -B /scratch/lg154 -B /scratch/yz11502 \
  "${BASE_SIF}" env MODE=lifelong_b TASK_ROOT="${TASK_ROOT}" \
  BASE_SOURCE_ROOT="${BASE_SOURCE_ROOT}" RUN_ROOT="${RUN_ROOT}" \
  PARENT_MANIFEST="${PARENT_ROOT}/sealed_inputs/parent_manifest.json" \
  PROTOCOL="${PROTOCOL}" SCENE_INDEX="${SLURM_ARRAY_TASK_ID}" \
  BENCH_ROOT="${RUN_ROOT}/benchmarks_ab/role_pairs" \
  MAX_STEPS="${MAX_STEPS:-600}" \
  HAB_PY=/scratch/lg154/conda-envs/habitat/bin/python \
  MEMNAV_PY=/scratch/lg154/conda-envs/memnav/bin/python \
  TASK_RECEIPT="${TASK_RECEIPT}" \
  EXPECTED_TASK_RECEIPT_SHA="${EXPECTED_TASK_RECEIPT_SHA}" \
  BASE_RECEIPT="${BASE_RECEIPT}" \
  EXPECTED_BASE_RECEIPT_SHA="${EXPECTED_BASE_RECEIPT_SHA}" \
  bash "${TASK_ROOT}/MemNavData/run_hm3d_fullmono_server_scene.sh"
