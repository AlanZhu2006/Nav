#!/usr/bin/env bash
# Aggregate native Goal-A traces and finalize the two policy-blind benchmarks.
#SBATCH --job-name=paper_A_sum
#SBATCH --output=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/%x_%j.out
#SBATCH --error=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/%x_%j.err
#SBATCH --time=00:20:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --partition=cpu_short
#SBATCH --account=torch_pr_769_tandon_advanced
#SBATCH --export=ALL

set -euo pipefail
umask 0022
: "${SOURCE_ROOT:?set SOURCE_ROOT}"
: "${SOURCE_RECEIPT:?set SOURCE_RECEIPT}"
: "${EXPECTED_SOURCE_RECEIPT_SHA:?set receipt SHA}"
: "${RUN_ROOT:?set RUN_ROOT}"
: "${MANIFEST:?set frozen source manifest}"
: "${EXPECTED_MANIFEST_SHA:?set manifest SHA}"
FINAL14_POPULATION_MODE=${FINAL14_POPULATION_MODE:-0}
PY=/scratch/lg154/conda-envs/memnav/bin/python

[[ "${FINAL14_POPULATION_MODE}" == 0 || "${FINAL14_POPULATION_MODE}" == 1 ]]

[[ "$(sha256sum "${SOURCE_RECEIPT}" | awk '{print $1}')" == \
   "${EXPECTED_SOURCE_RECEIPT_SHA}" ]]
(cd "${SOURCE_ROOT}" && sha256sum -c "${SOURCE_RECEIPT}") >/dev/null
[[ ! -e "${RUN_ROOT}/online_a_inventory.json" ]]
export PYTHONPATH=${SOURCE_ROOT}:${SOURCE_ROOT}/MemNavData${PYTHONPATH:+:${PYTHONPATH}}
"${PY}" -u "${SOURCE_ROOT}/MemNavData/summarize_paper_online_a.py" \
  --root "${RUN_ROOT}" --manifest "${MANIFEST}" \
  --manifest-sha "${EXPECTED_MANIFEST_SHA}" \
  --out "${RUN_ROOT}/online_a_inventory.json"
sha256sum "${RUN_ROOT}/online_a_inventory.json" \
  > "${RUN_ROOT}/online_a_inventory.json.sha256"
if [[ "${FINAL14_POPULATION_MODE}" == 1 ]]; then
  "${PY}" -u "${SOURCE_ROOT}/MemNavData/finalize_final14_role_pairs.py" \
    --run-root "${RUN_ROOT}" --out "${RUN_ROOT}/benchmarks" \
    --expected-scene-count 14 --natural-target-histories 28 \
    --natural-target-scenes 10 --hard-target-histories 16 \
    --hard-target-scenes 8
  "${PY}" -u "${SOURCE_ROOT}/MemNavData/audit_final14_role_pairs.py" \
    --root "${RUN_ROOT}/benchmarks" \
    --out "${RUN_ROOT}/benchmarks/final14_construction_audit.json" \
    >/dev/null
else
  "${PY}" -u "${SOURCE_ROOT}/MemNavData/finalize_paper_role_pairs.py" \
    --run-root "${RUN_ROOT}" --out "${RUN_ROOT}/benchmarks"
fi
(
  cd "${RUN_ROOT}/benchmarks"
  find . -type f ! -name BENCHMARK_FILES.sha256 \
    ! -name BENCHMARK_FILES.sha256.sha256 ! -name SEALED \
    -print0 | sort -z | xargs -0 sha256sum \
    > BENCHMARK_FILES.sha256
  sha256sum -c BENCHMARK_FILES.sha256 >/dev/null
  sha256sum BENCHMARK_FILES.sha256 > BENCHMARK_FILES.sha256.sha256
  : > SEALED
  chmod -R a-w .
)
chmod -R a-w "${RUN_ROOT}/traces"
