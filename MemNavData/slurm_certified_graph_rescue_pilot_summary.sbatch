#!/usr/bin/env bash
#SBATCH --job-name=cgraph3sum
#SBATCH --output=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/%x_%j.out
#SBATCH --error=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/%x_%j.err
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --partition=cs
#SBATCH --qos=cpu48
#SBATCH --account=torch_pr_769_tandon_advanced
#SBATCH --export=ALL

set -euo pipefail
umask 0022
: "${SOURCE_ROOT:?}"; : "${SOURCE_RECEIPT:?}"; : "${EXPECTED_SOURCE_RECEIPT_SHA:?}"
: "${FROZEN_RUN_ROOT:?}"; : "${RUN_ROOT:?}"
HAB_PY=/scratch/lg154/conda-envs/habitat/bin/python
[[ "$(sha256sum "${SOURCE_RECEIPT}" | awk '{print $1}')" == \
   "${EXPECTED_SOURCE_RECEIPT_SHA}" ]]
(cd "${SOURCE_ROOT}" && sha256sum -c "${SOURCE_RECEIPT}") >/dev/null
manifest=${FROZEN_RUN_ROOT}/prepared/benchmark/manifest.json
manifest_sha=$(sha256sum "${manifest}" | awk '{print $1}')
if [[ -e "${RUN_ROOT}/benchmark_manifest.json" ]]; then
  cmp --silent "${manifest}" "${RUN_ROOT}/benchmark_manifest.json" || {
    echo "ABORT: existing benchmark manifest differs" >&2
    exit 2
  }
else
  cp "${manifest}" "${RUN_ROOT}/benchmark_manifest.json"
fi
"${HAB_PY}" "${SOURCE_ROOT}/MemNavData/summarize_certified_graph_rescue_pilot.py" \
  --run-root "${RUN_ROOT}" --expected-manifest-sha "${manifest_sha}" \
  --out "${RUN_ROOT}/report.json"
sha256sum "${RUN_ROOT}/report.json" > "${RUN_ROOT}/report.json.sha256"
chmod a-w "${RUN_ROOT}/report.json" "${RUN_ROOT}/report.json.sha256" \
  "${RUN_ROOT}/benchmark_manifest.json"
