#!/usr/bin/env bash
#SBATCH --job-name=vintCECdir5sum
#SBATCH --output=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/%x_%j.out
#SBATCH --error=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/%x_%j.err
#SBATCH --time=00:20:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --partition=cpu_short
#SBATCH --account=torch_pr_769_tandon_advanced

set -euo pipefail
umask 0022
export PYTHONDONTWRITEBYTECODE=1
: "${SOURCE_ROOT:?}" "${SOURCE_RECEIPT:?}" "${EXPECTED_SOURCE_RECEIPT_SHA:?}"
: "${RUN_ROOT:?}"
MEMNAV_PY=/scratch/lg154/conda-envs/memnav/bin/python

[[ "$(sha256sum "${SOURCE_RECEIPT}" | awk '{print $1}')" == \
   "${EXPECTED_SOURCE_RECEIPT_SHA}" ]]
(cd "${SOURCE_ROOT}" && sha256sum -c --quiet "${SOURCE_RECEIPT}")
test "$(find "${RUN_ROOT}/evaluation" -path '*/vint/direction_triple_audit.json' \
  -type f | wc -l)" -eq 5
env PYTHONPATH="${SOURCE_ROOT}" "${MEMNAV_PY}" -u \
  "${SOURCE_ROOT}/MemNavData/aggregate_vint_cec_bearing_alignment_loss5.py" \
  --run-root "${RUN_ROOT}" --expected-cells 5 \
  --out "${RUN_ROOT}/mechanism_summary.json"
sha256sum "${RUN_ROOT}/mechanism_summary.json" \
  >"${RUN_ROOT}/mechanism_summary.json.sha256"
env PYTHONPATH="${SOURCE_ROOT}" "${MEMNAV_PY}" -u \
  "${SOURCE_ROOT}/MemNavData/independent_verify_vint_cec_bearing_alignment_loss5.py" \
  --run-root "${RUN_ROOT}" --summary "${RUN_ROOT}/mechanism_summary.json" \
  --selection "${SOURCE_ROOT}/MemNavData/vint_cec_direction_loss5_manifest_20260828.json" \
  --out "${RUN_ROOT}/mechanism_independent_verification.json"
sha256sum "${RUN_ROOT}/mechanism_independent_verification.json" \
  >"${RUN_ROOT}/mechanism_independent_verification.json.sha256"
chmod a-w "${RUN_ROOT}/mechanism_summary.json" \
  "${RUN_ROOT}/mechanism_summary.json.sha256" \
  "${RUN_ROOT}/mechanism_independent_verification.json" \
  "${RUN_ROOT}/mechanism_independent_verification.json.sha256"
