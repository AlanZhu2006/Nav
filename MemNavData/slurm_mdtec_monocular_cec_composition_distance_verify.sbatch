#!/bin/bash
# Independent post-hoc repair: recompute Goal-B success from each retained
# arm metric.csv/final_dist_B.  This never mutates rollout or primary summary.
#SBATCH --job-name=mdtecCecDistV
#SBATCH --output=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/mdtecCecDistV_%j.out
#SBATCH --error=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/mdtecCecDistV_%j.err
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --partition=cpu_short
#SBATCH --account=torch_pr_769_tandon_advanced
#SBATCH --export=ALL

set -euo pipefail
: "${REPAIR_ROOT:?set immutable repair source root}"
: "${ORIGINAL_SOURCE_ROOT:?set original immutable source root}"
: "${RUN_ROOT:?set formal run root}"
: "${EXPECTED_VERIFIER_SHA:?set verifier SHA-256}"

MEMNAV_PY=/scratch/lg154/conda-envs/memnav/bin/python
verifier=${REPAIR_ROOT}/independent_verify_mdtec_monocular_cec_composition.py
protocol=${ORIGINAL_SOURCE_ROOT}/MemNavData/mdtec_monocular_cec_composition_protocol_20260819.json
report=${RUN_ROOT}/POSTHOC/mdtec_cec_composition_summary.json
out_root=${RUN_ROOT}/POSTHOC_DISTANCE_VERIFICATION

[[ -r "${verifier}" && -r "${protocol}" && -r "${report}" ]]
[[ "$(sha256sum "${verifier}" | awk '{print $1}')" == \
   "${EXPECTED_VERIFIER_SHA}" ]]
[[ ! -e "${out_root}" ]] || { echo "distance verification output exists" >&2; exit 1; }
mkdir -p "${out_root}"

PYTHONDONTWRITEBYTECODE=1 "${MEMNAV_PY}" "${verifier}" \
  --run-root "${RUN_ROOT}" \
  --report "${report}" \
  --protocol "${protocol}" \
  --out "${out_root}/mdtec_cec_composition_distance_verification.json"
sha256sum "${out_root}/mdtec_cec_composition_distance_verification.json" > \
  "${out_root}/output_receipt.sha256"
chmod a-w "${out_root}/"*.json "${out_root}/output_receipt.sha256"
