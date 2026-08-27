#!/usr/bin/env bash
#SBATCH --job-name=h3lifeV4Verify
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
: "${SOURCE_ROOT:?}" "${SOURCE_RECEIPT:?}"
: "${EXPECTED_SOURCE_RECEIPT_SHA:?}" "${RUN_ROOT:?}"
: "${AUDIT_RUN_ROOT:?}" "${PROTOCOL:?}" "${EXPECTED_PROTOCOL_SHA:?}"
[[ "$(sha256sum "${SOURCE_RECEIPT}" | awk '{print $1}')" == \
   "${EXPECTED_SOURCE_RECEIPT_SHA}" ]]
(cd "${SOURCE_ROOT}" && sha256sum -c --quiet "${SOURCE_RECEIPT}")
[[ "$(sha256sum "${PROTOCOL}" | awk '{print $1}')" == \
   "${EXPECTED_PROTOCOL_SHA}" ]]
[[ -f "${RUN_ROOT}/ab_population/SEALED" ]]
OUT=${RUN_ROOT}/independent_natural_v4_materialization_verification.json
[[ ! -e "${OUT}" && ! -e "${OUT}.sha256" ]]
PY=/scratch/lg154/conda-envs/memnav/bin/python
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPYCACHEPREFIX=${SLURM_TMPDIR:-/tmp}/h3life_v4_verify_${SLURM_JOB_ID}
export PYTHONPATH=${SOURCE_ROOT}/MemNavData
"${PY}" -m unittest -q \
  test_independent_verify_hm3d_fullmono_lifelong_natural_v4_materialization
"${PY}" -u \
  "${SOURCE_ROOT}/MemNavData/independent_verify_hm3d_fullmono_lifelong_natural_v4_materialization.py" \
  --run-root "${RUN_ROOT}" --audit-run-root "${AUDIT_RUN_ROOT}" \
  --protocol "${PROTOCOL}" --out "${OUT}"
(cd "${RUN_ROOT}" && sha256sum -c --quiet \
  independent_natural_v4_materialization_verification.json.sha256)
chmod a-w "${OUT}" "${OUT}.sha256"
