#!/usr/bin/env bash
#SBATCH --job-name=cecPortAudit
#SBATCH --output=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/%x_%j.out
#SBATCH --error=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/%x_%j.err
#SBATCH --time=00:20:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=cpu_short
#SBATCH --account=torch_pr_769_tandon_advanced

set -euo pipefail
umask 0022
export PYTHONDONTWRITEBYTECODE=1
: "${MODE:?set aggregate or verify}" "${SOURCE_ROOT:?}" "${RUN_ROOT:?}"
: "${SOURCE_RECEIPT:?}" "${EXPECTED_SOURCE_RECEIPT_SHA:?}"
: "${QUERY_MANIFEST:?}" "${EXPECTED_QUERY_MANIFEST_SHA:?}"
PY=/scratch/lg154/conda-envs/memnav/bin/python
[[ "$(sha256sum "${SOURCE_RECEIPT}" | awk '{print $1}')" == \
   "${EXPECTED_SOURCE_RECEIPT_SHA}" ]]
(cd "${SOURCE_ROOT}" && sha256sum -c --quiet "${SOURCE_RECEIPT}")
[[ "$(sha256sum "${QUERY_MANIFEST}" | awk '{print $1}')" == \
   "${EXPECTED_QUERY_MANIFEST_SHA}" ]]
export PYTHONPATH=${SOURCE_ROOT}:${SOURCE_ROOT}/MemNavData

summary=${RUN_ROOT}/cec_proof_locked_portability_pilot_summary.json
verification=${RUN_ROOT}/cec_proof_locked_portability_pilot_independent_verification.json
if [[ "${MODE}" == aggregate ]]; then
  [[ ! -e "${summary}" ]]
  "${PY}" -u \
    "${SOURCE_ROOT}/MemNavData/aggregate_cec_proof_locked_portability.py" \
    --run-root "${RUN_ROOT}" --query-manifest "${QUERY_MANIFEST}" \
    --out "${summary}"
  sha256sum "${summary}" >"${summary}.sha256"
elif [[ "${MODE}" == verify ]]; then
  [[ -f "${summary}" && ! -e "${verification}" ]]
  "${PY}" -u \
    "${SOURCE_ROOT}/MemNavData/independent_verify_cec_proof_locked_portability.py" \
    --run-root "${RUN_ROOT}" --summary "${summary}" \
    --query-manifest "${QUERY_MANIFEST}" --out "${verification}"
  sha256sum "${verification}" >"${verification}.sha256"
  "${PY}" - "${verification}" <<'PY'
import json,sys
if json.load(open(sys.argv[1])).get("verified") is not True:
 raise SystemExit("independent verification failed")
PY
  printf '%s\n' "verified=true" >"${RUN_ROOT}/SEALED"
  chmod a-w "${summary}" "${summary}.sha256" "${verification}" \
    "${verification}.sha256" "${RUN_ROOT}/SEALED"
else
  echo "invalid MODE=${MODE}" >&2
  exit 2
fi
