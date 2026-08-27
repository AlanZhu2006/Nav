#!/usr/bin/env bash
#SBATCH --job-name=h3lifeV4PopFV
#SBATCH --output=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/%x_%j.out
#SBATCH --error=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/%x_%j.err
#SBATCH --time=00:40:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --partition=cpu_short
#SBATCH --account=torch_pr_769_tandon_advanced

set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
: "${VERIFY_ROOT:?}" "${VERIFY_RECEIPT:?}" "${EXPECTED_VERIFY_RECEIPT_SHA:?}"
: "${TASK_ROOT:?}" "${RUN_ROOT:?}" "${PROTOCOL:?}"
: "${EXPECTED_PROTOCOL_SHA:?}"
[[ "$(sha256sum "${VERIFY_RECEIPT}" | awk '{print $1}')" == \
   "${EXPECTED_VERIFY_RECEIPT_SHA}" ]]
(cd "${VERIFY_ROOT}" && sha256sum -c --quiet "${VERIFY_RECEIPT}")
[[ "$(sha256sum "${PROTOCOL}" | awk '{print $1}')" == \
   "${EXPECTED_PROTOCOL_SHA}" ]]

# The task-bundle finalizer verifies all task/server/base receipts, requires
# exactly 99 prefix completions, seals the population, and makes every source
# prefix immutable.  Its SBATCH header is inert when invoked as a shell script.
bash "${TASK_ROOT}/MemNavData/slurm_hm3d_fullmono_lifelong_natural_v4_finalize_population.sbatch"

OUT=${RUN_ROOT}/independent_natural_v4_population_verification.json
[[ ! -e "${OUT}" && ! -e "${OUT}.sha256" ]]
PY=/scratch/lg154/conda-envs/memnav/bin/python
export PYTHONPYCACHEPREFIX=${SLURM_TMPDIR:-/tmp}/h3life_v4_pop_fv_${SLURM_JOB_ID}
export PYTHONPATH=${VERIFY_ROOT}/MemNavData
"${PY}" -m unittest -q \
  test_independent_verify_hm3d_fullmono_lifelong_natural_v4_population
"${PY}" -u \
  "${VERIFY_ROOT}/MemNavData/independent_verify_hm3d_fullmono_lifelong_natural_v4_population.py" \
  --run-root "${RUN_ROOT}" --protocol "${PROTOCOL}" --out "${OUT}"
(cd "${RUN_ROOT}" && sha256sum -c --quiet \
  independent_natural_v4_population_verification.json.sha256)
chmod a-w "${OUT}" "${OUT}.sha256"
