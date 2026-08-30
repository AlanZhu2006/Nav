#!/usr/bin/env bash
#SBATCH --job-name=h3T3ExpBuild
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
: "${SLURM_ARRAY_TASK_ID:?}" "${EXPANSION_EXEC_SOURCE_ROOT:?}"
: "${EXPANSION_EXEC_SOURCE_RECEIPT:?}" "${EXPECTED_EXPANSION_EXEC_SOURCE_RECEIPT_SHA:?}"
: "${BASE_WRAPPER_ROOT:?}" "${BASE_WRAPPER_RECEIPT:?}"
: "${EXPECTED_BASE_WRAPPER_RECEIPT_SHA:?}"
: "${EXPANSION_RUN_ROOT:?}" "${EXPANSION_PLAN:?}"
: "${EXPECTED_EXPANSION_PLAN_SHA:?}" "${EXPANSION_CONSTRUCTION_PROTOCOL:?}"
: "${EXPECTED_EXPANSION_PROTOCOL_SHA:?}" "${EXPANSION_PLAN_VERIFICATION:?}"
: "${EXPECTED_EXPANSION_PLAN_VERIFICATION_SHA:?}"

[[ "$(sha256sum "${EXPANSION_EXEC_SOURCE_RECEIPT}" | awk '{print $1}')" == \
   "${EXPECTED_EXPANSION_EXEC_SOURCE_RECEIPT_SHA}" ]]
(cd "${EXPANSION_EXEC_SOURCE_ROOT}" && \
  sha256sum -c --quiet "${EXPANSION_EXEC_SOURCE_RECEIPT}")
[[ "$(sha256sum "${BASE_WRAPPER_RECEIPT}" | awk '{print $1}')" == \
   "${EXPECTED_BASE_WRAPPER_RECEIPT_SHA}" ]]
(cd "${BASE_WRAPPER_ROOT}" && sha256sum -c --quiet "${BASE_WRAPPER_RECEIPT}")
[[ "$(sha256sum "${EXPANSION_PLAN}" | awk '{print $1}')" == \
   "${EXPECTED_EXPANSION_PLAN_SHA}" ]]
[[ "$(sha256sum "${EXPANSION_CONSTRUCTION_PROTOCOL}" | awk '{print $1}')" == \
   "${EXPECTED_EXPANSION_PROTOCOL_SHA}" ]]
[[ "$(sha256sum "${EXPANSION_PLAN_VERIFICATION}" | awk '{print $1}')" == \
   "${EXPECTED_EXPANSION_PLAN_VERIFICATION_SHA}" ]]
/scratch/lg154/conda-envs/memnav/bin/python - "${EXPANSION_PLAN_VERIFICATION}" <<'PY'
import json,sys
p=json.load(open(sys.argv[1]))
assert p['schema_version']=='hm3d_table3_causal_survey_expansion_plan_verification_v1_20260831'
assert p['verified'] is True
assert p['query_policy_outcomes_read'] is False
assert p['navigation_policy_outcomes_read'] is False
assert p['query_policy_evaluation_authorized'] is False
PY

BASE_SIF=/share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif
export PYTHONPATH=${EXPANSION_EXEC_SOURCE_ROOT}:${EXPANSION_EXEC_SOURCE_ROOT}/MemNavData:${BASE_WRAPPER_ROOT}:${BASE_WRAPPER_ROOT}/MemNavData
exec singularity exec --nv -B /scratch/lg154 -B /scratch/yz11502 \
  "${BASE_SIF}" /scratch/lg154/conda-envs/habitat/bin/python -u \
  "${EXPANSION_EXEC_SOURCE_ROOT}/MemNavData/construct_hm3d_table3_causal_survey_role_pair.py" \
  --run-root "${EXPANSION_RUN_ROOT}" --candidate-plan "${EXPANSION_PLAN}" \
  --protocol "${EXPANSION_CONSTRUCTION_PROTOCOL}" \
  --plan-index "${SLURM_ARRAY_TASK_ID}"
