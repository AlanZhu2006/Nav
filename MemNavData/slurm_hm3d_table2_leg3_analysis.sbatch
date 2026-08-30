#!/usr/bin/env bash
#SBATCH --job-name=h3T2L3Audit
#SBATCH --output=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/%x_%j.out
#SBATCH --error=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/%x_%j.err
#SBATCH --time=00:45:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --partition=cpu_short
#SBATCH --account=torch_pr_769_tandon_advanced

set -euo pipefail
umask 0022
export PYTHONDONTWRITEBYTECODE=1

: "${MODE:?finalize or verify}"
: "${TASK_ROOT:?}" "${TASK_RECEIPT:?}" "${EXPECTED_TASK_RECEIPT_SHA:?}"
: "${SOURCE_RUN_ROOT:?}" "${RUN_ROOT:?}" "${PROTOCOL:?}"
PY=/scratch/lg154/conda-envs/memnav/bin/python
POPULATION=${RUN_ROOT}/population
[[ "$(sha256sum "${TASK_RECEIPT}" | awk '{print $1}')" == \
   "${EXPECTED_TASK_RECEIPT_SHA}" ]]
(cd "${TASK_ROOT}" && sha256sum -c --quiet "${TASK_RECEIPT}")

case "${MODE}" in
  finalize)
    expected=$("${PY}" - "${PROTOCOL}" <<'PY'
import json,sys
p=json.load(open(sys.argv[1]))
print(int(p['source_population']['actual_AB_successful_histories']))
PY
)
    [[ "${expected}" =~ ^[1-9][0-9]*$ ]]
    [[ "$(find "${RUN_ROOT}/construction/fragments" -mindepth 2 -maxdepth 2 \
      -name completion.json -type f | wc -l)" -eq "${expected}" ]]
    [[ ! -e "${POPULATION}" ]]
    PYTHONPATH="${TASK_ROOT}:${TASK_ROOT}/MemNavData" \
      "${PY}" -u \
      "${TASK_ROOT}/MemNavData/finalize_hm3d_table2_leg3_mixed_role.py" \
        --protocol "${PROTOCOL}" --source-root "${SOURCE_RUN_ROOT}" \
        --fragments "${RUN_ROOT}/construction/fragments" \
        --out "${POPULATION}"
    chmod -R a-w "${POPULATION}"
    ;;
  verify)
    [[ -d "${POPULATION}" ]]
    out=${RUN_ROOT}/hm3d_table2_leg3_construction_verification.json
    [[ ! -e "${out}" ]]
    PYTHONPATH="${TASK_ROOT}:${TASK_ROOT}/MemNavData" \
      "${PY}" -u \
      "${TASK_ROOT}/MemNavData/independent_verify_hm3d_table2_leg3_mixed_role.py" \
        --root "${POPULATION}" --protocol "${PROTOCOL}" --out "${out}"
    sha256sum "${out}" >"${out}.sha256"
    chmod a-w "${out}" "${out}.sha256"
    ;;
  *) echo "ABORT: invalid MODE=${MODE}" >&2; exit 2 ;;
esac
