#!/usr/bin/env bash
#SBATCH --job-name=goat_cec_verify
#SBATCH --output=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/%x_%j.out
#SBATCH --error=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/%x_%j.err
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --partition=cpu_short
#SBATCH --account=torch_pr_769_tandon_advanced

set -euo pipefail
umask 0022
: "${SOURCE_ROOT:?set immutable source root}"
: "${SOURCE_RECEIPT:?set source receipt}"
: "${EXPECTED_SOURCE_RECEIPT_SHA:?set source receipt SHA-256}"
: "${MANIFEST:?set frozen manifest}"
: "${EXPECTED_MANIFEST_SHA:?set manifest SHA-256}"
: "${RUN_ROOT:?set result root}"

readonly PY="${MEMNAV_PY:-/scratch/lg154/conda-envs/memnav/bin/python}"
readonly SCRIPT="${SOURCE_ROOT}/MemNavData/verify_goat_sequential_revisit.py"
readonly SUMMARY="${RUN_ROOT}/goat_sequential_revisit_summary.json"
readonly OUTPUT="${RUN_ROOT}/goat_sequential_revisit_independent_verification.json"
fail() { echo "ABORT: $*" >&2; exit 2; }
[[ "$(sha256sum "${SOURCE_RECEIPT}" | awk '{print $1}')" == \
   "${EXPECTED_SOURCE_RECEIPT_SHA}" ]] || fail "source receipt changed"
(cd "${SOURCE_ROOT}" && sha256sum -c "${SOURCE_RECEIPT}") >/dev/null || \
  fail "source bundle changed"
[[ "$(sha256sum "${MANIFEST}" | awk '{print $1}')" == \
   "${EXPECTED_MANIFEST_SHA}" ]] || fail "manifest changed"
[[ -s "${SUMMARY}" && ! -e "${OUTPUT}" ]] || \
  fail "summary missing or verification already exists"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${SOURCE_ROOT}"
"${PY}" "${SCRIPT}" --manifest "${MANIFEST}" --run-root "${RUN_ROOT}" \
  --summary "${SUMMARY}" --output "${OUTPUT}"
"${PY}" - "${OUTPUT}" <<'PY'
import json
import sys
assert json.load(open(sys.argv[1]))["verified"] is True
PY
sha256sum "${OUTPUT}" > "${OUTPUT}.sha256"
