#!/usr/bin/env bash
#SBATCH --job-name=h3T3SurveyBuild
#SBATCH --output=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/%x_%A_%a.out
#SBATCH --error=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/%x_%A_%a.err
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --partition=h100_tandon,a100_tandon
#SBATCH --gres=gpu:1
#SBATCH --account=torch_pr_769_tandon_advanced

set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
: "${SLURM_ARRAY_TASK_ID:?}" "${SURVEY_SOURCE_ROOT:?}"
: "${SURVEY_SOURCE_RECEIPT:?}" "${EXPECTED_SURVEY_SOURCE_RECEIPT_SHA:?}"
: "${BASE_WRAPPER_ROOT:?}" "${BASE_WRAPPER_RECEIPT:?}"
: "${EXPECTED_BASE_WRAPPER_RECEIPT_SHA:?}"
: "${RUN_ROOT:?}" "${CANDIDATE_PLAN:?}"
: "${EXPECTED_CANDIDATE_PLAN_SHA:?}" "${SURVEY_PROTOCOL:?}"
[[ "$(sha256sum "${SURVEY_SOURCE_RECEIPT}" | awk '{print $1}')" == \
   "${EXPECTED_SURVEY_SOURCE_RECEIPT_SHA}" ]]
(cd "${SURVEY_SOURCE_ROOT}" && sha256sum -c --quiet "${SURVEY_SOURCE_RECEIPT}")
[[ "$(sha256sum "${BASE_WRAPPER_RECEIPT}" | awk '{print $1}')" == \
   "${EXPECTED_BASE_WRAPPER_RECEIPT_SHA}" ]]
(cd "${BASE_WRAPPER_ROOT}" && \
  sha256sum -c --quiet "${BASE_WRAPPER_RECEIPT}")
[[ "$(sha256sum "${CANDIDATE_PLAN}" | awk '{print $1}')" == \
   "${EXPECTED_CANDIDATE_PLAN_SHA}" ]]

BASE_SIF=/share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif
export PYTHONPATH=${SURVEY_SOURCE_ROOT}:${SURVEY_SOURCE_ROOT}/MemNavData:${BASE_WRAPPER_ROOT}:${BASE_WRAPPER_ROOT}/MemNavData
exec singularity exec --nv -B /scratch/lg154 -B /scratch/yz11502 \
  "${BASE_SIF}" /scratch/lg154/conda-envs/habitat/bin/python -u \
  "${SURVEY_SOURCE_ROOT}/MemNavData/construct_hm3d_table3_causal_survey_role_pair.py" \
  --run-root "${RUN_ROOT}" --candidate-plan "${CANDIDATE_PLAN}" \
  --protocol "${SURVEY_PROTOCOL}" --history-index "${SLURM_ARRAY_TASK_ID}"
