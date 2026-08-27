#!/usr/bin/env bash
# Analysis-only replacement for the original summary job.  The rollout,
# frozen summarizer, manifest, source receipt, and output path are unchanged;
# only the pre-summary unit-test interpreter is corrected to an environment
# that actually provides pytest.
#SBATCH --job-name=certrel_sum_fix
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
MEMNAV_PY=/scratch/lg154/conda-envs/memnav/bin/python
MANIFEST=${RUN_ROOT}/data_manifest.json
SUMMARIZER=${SOURCE_ROOT}/MemNavData/summarize_certified_relocalization_closed_loop.py
OUT=${RUN_ROOT}/report.json
export PYTHONPYCACHEPREFIX=${SLURM_TMPDIR:-/tmp}/certrel_sum_fix_pycache_${SLURM_JOB_ID}
mkdir -p "${PYTHONPYCACHEPREFIX}"
[[ "$(sha256sum "${SOURCE_RECEIPT}" | awk '{print $1}')" == \
    "${EXPECTED_SOURCE_RECEIPT_SHA}" ]] || {
      echo "ABORT: source receipt" >&2; exit 2; }
(cd "${SOURCE_ROOT}" && sha256sum -c "${SOURCE_RECEIPT}") >/dev/null
(cd "${RUN_ROOT}" && sha256sum -c data_manifest.json.sha256) >/dev/null
(cd "${RUN_ROOT}" && sha256sum -c dependency_receipt.json.sha256) >/dev/null
[[ ! -e "${OUT}" ]] || { echo "ABORT: report already exists" >&2; exit 2; }
export PYTHONPATH=${SOURCE_ROOT}${PYTHONPATH:+:${PYTHONPATH}}
env PYTHONPYCACHEPREFIX="${PYTHONPYCACHEPREFIX}" \
  "${HAB_PY}" -m py_compile "${SUMMARIZER}"
"${MEMNAV_PY}" -m pytest -q \
  "${SOURCE_ROOT}/MemNavData/test_summarize_certified_relocalization_closed_loop.py"
"${HAB_PY}" -u "${SUMMARIZER}" --manifest "${MANIFEST}" \
  --run-root "${RUN_ROOT}" --out "${OUT}"
sha256sum "${OUT}" > "${OUT}.sha256"
chmod a-w "${OUT}" "${OUT}.sha256"
echo "COMPLETE ${OUT}"
