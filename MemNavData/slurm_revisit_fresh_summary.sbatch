#!/usr/bin/env bash
# Read-only paired/clustered summary after all scene tasks succeed.
#SBATCH --job-name=revfresh_summary
#SBATCH --output=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/%x_%j.out
#SBATCH --error=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/%x_%j.err
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=cs
#SBATCH --qos=cpu48
#SBATCH --account=torch_pr_769_tandon_advanced
#SBATCH --export=ALL

set -euo pipefail
umask 0022
: "${SOURCE_ROOT:?set immutable SOURCE_ROOT}"
: "${RUN_ROOT:?set RUN_ROOT}"
: "${SOURCE_RECEIPT:?set SOURCE_RECEIPT}"
: "${EXPECTED_SOURCE_RECEIPT_SHA:?set EXPECTED_SOURCE_RECEIPT_SHA}"
HAB_PY=/scratch/lg154/conda-envs/habitat/bin/python
MANIFEST=${RUN_ROOT}/data_manifest.json
SUMMARIZER=${SOURCE_ROOT}/MemNavData/summarize_revisit_fresh_confirmation.py
OUT=${RUN_ROOT}/report.json
export PYTHONPYCACHEPREFIX=${SLURM_TMPDIR:-/tmp}/revfresh_summary_pycache_${SLURM_JOB_ID}
mkdir -p "${PYTHONPYCACHEPREFIX}"
[[ "$(sha256sum "${SOURCE_RECEIPT}" | awk '{print $1}')" == \
    "${EXPECTED_SOURCE_RECEIPT_SHA}" ]] || { echo "ABORT: source receipt" >&2; exit 2; }
sha256sum -c "${MANIFEST}.sha256"
[[ ! -e "${OUT}" ]] || { echo "ABORT: report already exists" >&2; exit 2; }
export PYTHONPATH=${SOURCE_ROOT}${PYTHONPATH:+:${PYTHONPATH}}
env PYTHONPYCACHEPREFIX="${PYTHONPYCACHEPREFIX}" \
  "${HAB_PY}" -m py_compile "${SUMMARIZER}"
"${HAB_PY}" -u "${SUMMARIZER}" --manifest "${MANIFEST}" \
  --run-root "${RUN_ROOT}" --out "${OUT}"
sha256sum "${OUT}" > "${OUT}.sha256"
chmod a-w "${OUT}" "${OUT}.sha256"
echo "COMPLETE ${OUT}"
