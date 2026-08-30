#!/usr/bin/env bash
#SBATCH --job-name=h3T3ExpPlan
#SBATCH --output=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/%x_%j.out
#SBATCH --error=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/%x_%j.err
#SBATCH --time=00:20:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --partition=cpu_short
#SBATCH --account=torch_pr_769_tandon_advanced

set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
: "${MODE:?freeze or verify}" "${EXPANSION_SOURCE_ROOT:?}"
: "${EXPANSION_SOURCE_RECEIPT:?}" "${EXPECTED_EXPANSION_SOURCE_RECEIPT_SHA:?}"
: "${EXPANSION_RUN_ROOT:?}" "${EXPANSION_SELECTION_PROTOCOL:?}"
[[ "$(sha256sum "${EXPANSION_SOURCE_RECEIPT}" | awk '{print $1}')" == \
   "${EXPECTED_EXPANSION_SOURCE_RECEIPT_SHA}" ]]
(cd "${EXPANSION_SOURCE_ROOT}" && \
  sha256sum -c --quiet "${EXPANSION_SOURCE_RECEIPT}")

export PYTHONPATH=${EXPANSION_SOURCE_ROOT}:${EXPANSION_SOURCE_ROOT}/MemNavData
py=/scratch/lg154/conda-envs/memnav/bin/python
plan_root=${EXPANSION_RUN_ROOT}/expansion_plan
plan=${plan_root}/candidate_plan.json
construction_protocol=${plan_root}/construction_protocol.json
verification=${plan_root}/independent_verification.json

if [[ "${MODE}" == freeze ]]; then
  [[ "$(find /scratch/yz11502/Research/Nav-axis-uturn-results/hm3d_table3_causal_survey_20260830/formal_20260830T150042Z_3d811e0f/construction_fragments \
      -mindepth 2 -maxdepth 2 -name completion.json -type f | wc -l)" -eq 125 ]]
  mkdir -p "${plan_root}"
  "${py}" -u \
    "${EXPANSION_SOURCE_ROOT}/MemNavData/freeze_hm3d_table3_causal_survey_expansion_plan.py" \
    --selection-protocol "${EXPANSION_SELECTION_PROTOCOL}" \
    --out-plan "${plan}" \
    --out-construction-protocol "${construction_protocol}"
elif [[ "${MODE}" == verify ]]; then
  "${py}" -u \
    "${EXPANSION_SOURCE_ROOT}/MemNavData/independent_verify_hm3d_table3_causal_survey_expansion_plan.py" \
    --selection-protocol "${EXPANSION_SELECTION_PROTOCOL}" \
    --plan "${plan}" --construction-protocol "${construction_protocol}" \
    --out "${verification}"
  (cd "${plan_root}" && \
    sha256sum -c --quiet candidate_plan.json.sha256 \
      construction_protocol.json.sha256 independent_verification.json.sha256)
  chmod -R a-w "${plan_root}"
else
  echo "invalid MODE=${MODE}" >&2
  exit 2
fi
