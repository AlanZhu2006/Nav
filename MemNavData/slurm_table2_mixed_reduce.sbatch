#!/usr/bin/env bash
#SBATCH --job-name=table2_mixed_reduce
#SBATCH --output=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/%x_%j.out
#SBATCH --error=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/%x_%j.err
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --partition=cpu_short
#SBATCH --account=torch_pr_769_tandon_advanced
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
: "${REPAIRED_BUNDLE:?}" "${TABLE2_PLAN:?}" "${TABLE2_RUN:?}" "${TABLE2_REDUCE:?}"
: "${EXPECTED_RUNTIME_SHA:?}" "${EXPECTED_PLAN_SHA:?}"
[[ "$(id -un)" == yz11502 ]]
[[ "$(sha256sum "${REPAIRED_BUNDLE}/SOURCE_BUNDLE.sha256" | awk '{print $1}')" == "${EXPECTED_RUNTIME_SHA}" ]]
[[ "$(sha256sum "${TABLE2_PLAN}" | awk '{print $1}')" == "${EXPECTED_PLAN_SHA}" ]]
(cd "${REPAIRED_BUNDLE}" && sha256sum -c --quiet SOURCE_BUNDLE.sha256)
export PYTHONPATH=${REPAIRED_BUNDLE}:${REPAIRED_BUNDLE}/MemNavData
if [[ "${TABLE2_REDUCE}" == select_c ]]; then
  /scratch/lg154/conda-envs/memnav/bin/python "${REPAIRED_BUNDLE}/MemNavData/table2_mixed_hpc.py" select-c \
    --plan "${TABLE2_PLAN}" --run "${TABLE2_RUN}" --out "${TABLE2_RUN}/c_population.json"
  exec bash "${REPAIRED_BUNDLE}/MemNavData/submit_table2_mixed_hpc.sh" c
elif [[ "${TABLE2_REDUCE}" == summary ]]; then
  exec /scratch/lg154/conda-envs/memnav/bin/python "${REPAIRED_BUNDLE}/MemNavData/summarize_table2_mixed_hpc.py" \
    --plan "${TABLE2_PLAN}" --run "${TABLE2_RUN}" --out "${TABLE2_RUN}/paired_summary.json"
else
  echo "Unknown reducer: ${TABLE2_REDUCE}" >&2
  exit 2
fi
