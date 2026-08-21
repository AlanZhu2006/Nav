#!/usr/bin/env bash
#SBATCH --job-name=h3mix_seal
#SBATCH --output=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/%x_%j.out
#SBATCH --error=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/%x_%j.err
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --partition=cpu_short
#SBATCH --account=torch_pr_769_tandon_advanced

set -euo pipefail
: "${TASK_ROOT:?set task bundle}"
: "${EXPECTED_TASK_RECEIPT_SHA:?set task receipt SHA}"
: "${EXEC_SOURCE_ROOT:?set Final14 execution bundle}"
: "${RUN_ROOT:?set run root}"
: "${PARENT_MANIFEST:?set copied parent manifest}"
receipt=${TASK_ROOT}/SOURCE_BUNDLE.sha256
[[ "$(sha256sum "${receipt}" | awk '{print $1}')" == \
   "${EXPECTED_TASK_RECEIPT_SHA}" ]]
(cd "${TASK_ROOT}" && sha256sum -c SOURCE_BUNDLE.sha256 >/dev/null)
PY=/scratch/lg154/conda-envs/memnav/bin/python
export PYTHONPATH=${TASK_ROOT}/MemNavData:${EXEC_SOURCE_ROOT}/MemNavData:${EXEC_SOURCE_ROOT}
"${PY}" -u "${TASK_ROOT}/MemNavData/finalize_hm3d_mixed_role.py" \
  --run-root "${RUN_ROOT}" --parent-manifest "${PARENT_MANIFEST}"
(cd "${RUN_ROOT}/benchmarks" && sha256sum -c BENCHMARK_FILES.sha256 >/dev/null)
chmod -R a-w "${RUN_ROOT}/benchmarks"
