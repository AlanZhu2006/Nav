#!/usr/bin/env bash
#SBATCH --job-name=h3lifeV4Bfix
#SBATCH --output=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/%x_%A_%a.out
#SBATCH --error=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/%x_%A_%a.err
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=72G
#SBATCH --partition=h100_tandon,a100_tandon
#SBATCH --gres=gpu:1
#SBATCH --account=torch_pr_769_tandon_advanced

set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
: "${SLURM_ARRAY_TASK_ID:?}" "${REPAIR_ROOT:?}" "${REPAIR_RECEIPT:?}"
: "${EXPECTED_REPAIR_RECEIPT_SHA:?}" "${TASK_ROOT:?}" "${TASK_RECEIPT:?}"
: "${EXPECTED_TASK_RECEIPT_SHA:?}" "${RUN_ROOT:?}" "${AB_MANIFEST:?}"
: "${EXPECTED_AB_MANIFEST_SHA:?}" "${SHARD_MANIFEST:?}"
: "${EXPECTED_SHARD_MANIFEST_SHA:?}" "${REPAIR_TAG:?}"
[[ "${SLURM_ARRAY_TASK_ID}" == 31 || "${SLURM_ARRAY_TASK_ID}" == 37 ]]
[[ "$(sha256sum "${REPAIR_RECEIPT}" | awk '{print $1}')" == \
   "${EXPECTED_REPAIR_RECEIPT_SHA}" ]]
(cd "${REPAIR_ROOT}" && sha256sum -c --quiet "${REPAIR_RECEIPT}")
[[ "$(sha256sum "${TASK_RECEIPT}" | awk '{print $1}')" == \
   "${EXPECTED_TASK_RECEIPT_SHA}" ]]
(cd "${TASK_ROOT}" && sha256sum -c --quiet "${TASK_RECEIPT}")

/scratch/lg154/conda-envs/memnav/bin/python -u \
  "${REPAIR_ROOT}/MemNavData/repair_hm3d_fullmono_lifelong_natural_v4_factual_b.py" \
  archive --run-root "${RUN_ROOT}" --manifest "${AB_MANIFEST}" \
  --schedule "${SHARD_MANIFEST}" \
  --expected-manifest-sha256 "${EXPECTED_AB_MANIFEST_SHA}" \
  --expected-schedule-sha256 "${EXPECTED_SHARD_MANIFEST_SHA}" \
  --shard-index "${SLURM_ARRAY_TASK_ID}" --repair-tag "${REPAIR_TAG}"

export RUNTIME_ATTEMPT_PREFIX=${REPAIR_TAG}_shard
exec bash \
  "${TASK_ROOT}/MemNavData/slurm_hm3d_fullmono_lifelong_natural_v4_collect_b_shard.sbatch"
