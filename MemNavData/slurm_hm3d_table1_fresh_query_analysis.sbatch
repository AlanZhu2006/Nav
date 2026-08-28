#!/usr/bin/env bash
#SBATCH --job-name=h3T1constructAudit
#SBATCH --output=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/%x_%j.out
#SBATCH --error=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/%x_%j.err
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=cpu_short
#SBATCH --account=torch_pr_769_tandon_advanced

set -euo pipefail
umask 0022
export PYTHONDONTWRITEBYTECODE=1

: "${MODE:?finalize or verify}"
: "${TASK_ROOT:?}" "${TASK_RECEIPT:?}" "${EXPECTED_TASK_RECEIPT_SHA:?}"
: "${SOURCE_RUN_ROOT:?}" "${RUN_ROOT:?}"
MEMNAV_PY=/scratch/lg154/conda-envs/memnav/bin/python
CONSUMED=${SOURCE_RUN_ROOT}/benchmarks/natural_direction/manifest.json
POPULATION=${RUN_ROOT}/population
[[ "$(sha256sum "${TASK_RECEIPT}" | awk '{print $1}')" == \
   "${EXPECTED_TASK_RECEIPT_SHA}" ]]
(cd "${TASK_ROOT}" && sha256sum -c --quiet "${TASK_RECEIPT}")

case "${MODE}" in
  finalize)
    [[ "$(find "${RUN_ROOT}/construction/scenes" -mindepth 2 -maxdepth 2 \
      -name construction_receipt.json -type f | wc -l)" -eq 54 ]]
    [[ ! -e "${POPULATION}" ]]
    PYTHONPATH="${TASK_ROOT}:${TASK_ROOT}/MemNavData" \
      "${MEMNAV_PY}" -u \
      "${TASK_ROOT}/MemNavData/finalize_hm3d_table1_fresh_query_reserve.py" \
        --construction-root "${RUN_ROOT}/construction/scenes" \
        --original-manifest "${CONSUMED}" --out "${POPULATION}" \
        --target-histories 24 --target-scenes 15 \
        --minimum-per-stratum 4
    chmod -R a-w "${POPULATION}"
    ;;
  verify)
    [[ -d "${POPULATION}" ]]
    out=${RUN_ROOT}/hm3d_table1_fresh_query_verification.json
    [[ ! -e "${out}" ]]
    PYTHONPATH="${TASK_ROOT}:${TASK_ROOT}/MemNavData" \
      "${MEMNAV_PY}" -u \
      "${TASK_ROOT}/MemNavData/verify_hm3d_table1_fresh_query_reserve.py" \
        --root "${POPULATION}" --original-manifest "${CONSUMED}" \
        --out "${out}"
    sha256sum "${out}" >"${out}.sha256"
    chmod a-w "${out}" "${out}.sha256"
    ;;
  *) echo "ABORT: invalid MODE=${MODE}" >&2; exit 2 ;;
esac
