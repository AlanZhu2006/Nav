#!/usr/bin/env bash
#SBATCH --job-name=cecCtlSum
#SBATCH --output=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/%x_%j.out
#SBATCH --error=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/%x_%j.err
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --partition=cpu_short
#SBATCH --account=torch_pr_769_tandon_advanced
#SBATCH --export=ALL

set -euo pipefail
: "${SOURCE_ROOT:?set immutable source root}"
: "${SOURCE_RECEIPT:?set source receipt}"
: "${EXPECTED_SOURCE_RECEIPT_SHA:?set source receipt SHA}"
: "${RUN_ROOT:?set pilot root}"
: "${EXPECTED_HISTORY_INDICES:?set comma-separated history indices}"

[[ "$(sha256sum "${SOURCE_RECEIPT}" | awk '{print $1}')" == \
   "${EXPECTED_SOURCE_RECEIPT_SHA}" ]] || {
  echo "ABORT: source receipt changed" >&2; exit 2;
}
(cd "${SOURCE_ROOT}" && sha256sum -c --quiet "${SOURCE_RECEIPT}")
/scratch/lg154/conda-envs/memnav/bin/python \
  "${SOURCE_ROOT}/MemNavData/summarize_cec_controller_portability_pilot.py" \
  --root "${RUN_ROOT}" --expected-indices "${EXPECTED_HISTORY_INDICES}" \
  --out "${RUN_ROOT}/pilot_summary.json"
sha256sum "${RUN_ROOT}/pilot_summary.json" \
  >"${RUN_ROOT}/pilot_summary.json.sha256"
