#!/usr/bin/env bash
#SBATCH --job-name=h3NatBxMatV
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
: "${TASK_ROOT:?}" "${TASK_RECEIPT:?}" "${EXPECTED_TASK_RECEIPT_SHA:?}"
: "${RUN_ROOT:?}" "${PROTOCOL:?}" "${EXPECTED_PROTOCOL_SHA:?}"
[[ "$(sha256sum "${TASK_RECEIPT}" | awk '{print $1}')" == \
   "${EXPECTED_TASK_RECEIPT_SHA}" ]]
(cd "${TASK_ROOT}" && sha256sum -c --quiet "${TASK_RECEIPT}")
[[ "$(sha256sum "${PROTOCOL}" | awk '{print $1}')" == \
   "${EXPECTED_PROTOCOL_SHA}" ]]
[[ -f "${RUN_ROOT}/ab_population/SEALED" ]]
out=${RUN_ROOT}/independent_natural_b_expansion_materialization_verification.json
[[ ! -e "${out}" && ! -e "${out}.sha256" ]]
py=/scratch/lg154/conda-envs/memnav/bin/python
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPYCACHEPREFIX=${SLURM_TMPDIR:-/tmp}/h3_natbx_matv_${SLURM_JOB_ID}
export PYTHONPATH=${TASK_ROOT}/MemNavData
"${py}" -m unittest -q \
  test_independent_verify_hm3d_fullmono_lifelong_natural_b_expansion_materialization
"${py}" -u \
  "${TASK_ROOT}/MemNavData/independent_verify_hm3d_fullmono_lifelong_natural_b_expansion_materialization.py" \
  --run-root "${RUN_ROOT}" --protocol "${PROTOCOL}" --out "${out}"
(cd "${RUN_ROOT}" && sha256sum -c --quiet \
  independent_natural_b_expansion_materialization_verification.json.sha256)
chmod a-w "${out}" "${out}.sha256"
