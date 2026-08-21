#!/usr/bin/env bash
# Independent raw-record recomputation. This does not import the summarizer.
#SBATCH --job-name=goat_arr_verify
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
: "${SOURCE_ROOT:?set immutable confirmation source root}"
: "${SOURCE_RECEIPT:?set source receipt}"
: "${EXPECTED_SOURCE_RECEIPT_SHA:?set receipt SHA-256}"
: "${MANIFEST:?set frozen formal manifest}"
: "${EXPECTED_MANIFEST_SHA:?set manifest SHA-256}"
: "${RUN_ROOT:?set formal result root}"

readonly PY="${SUMMARY_PY:-/scratch/lg154/conda-envs/memnav/bin/python}"
readonly ENTRY="${SOURCE_ROOT}/MemNavData/verify_goat_certified_arrival.py"
fail() { echo "ABORT: $*" >&2; exit 2; }
[[ "$(sha256sum "${SOURCE_RECEIPT}" | awk '{print $1}')" == \
   "${EXPECTED_SOURCE_RECEIPT_SHA}" ]] || fail "source receipt changed"
(cd "${SOURCE_ROOT}" && sha256sum -c "${SOURCE_RECEIPT}") >/dev/null || \
  fail "source bundle changed"
[[ "$(sha256sum "${MANIFEST}" | awk '{print $1}')" == \
   "${EXPECTED_MANIFEST_SHA}" ]] || fail "manifest changed"
[[ -f "${RUN_ROOT}/summary/SEALED" ]] || fail "sealed summary missing"
readonly OUTPUT="${RUN_ROOT}/independent_verification.json"
[[ ! -e "${OUTPUT}" ]] || fail "verification output already exists"

export PYTHONPATH="${SOURCE_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
"${PY}" -u "${ENTRY}" --manifest "${MANIFEST}" \
  --episode-dir "${RUN_ROOT}/episodes" \
  --summary-dir "${RUN_ROOT}/summary" --output "${OUTPUT}"
sha256sum "${OUTPUT}" > "${OUTPUT}.sha256"

