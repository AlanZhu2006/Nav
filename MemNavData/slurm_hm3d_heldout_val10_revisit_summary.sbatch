#!/usr/bin/env bash
# Summarize all 40 frozen HM3D held-out val10 paired episodes after evaluation.
#SBATCH --job-name=hm3dh10_sum
#SBATCH --output=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/%x_%j.out
#SBATCH --error=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/%x_%j.err
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --partition=cpu_short
#SBATCH --account=torch_pr_769_tandon_advanced

set -euo pipefail
umask 0022

: "${TASK_ROOT:?set immutable task bundle root}"
: "${BASE_SOURCE_ROOT:?set immutable base source root}"
: "${RUN_ROOT:?set run root}"
: "${EXPECTED_TASK_RECEIPT_SHA:?set task receipt SHA}"
: "${EXPECTED_BASE_SOURCE_RECEIPT_SHA:?set base receipt SHA}"

readonly TASK_RECEIPT=${TASK_ROOT}/SOURCE_BUNDLE.sha256
readonly BASE_RECEIPT=${BASE_SOURCE_ROOT}/SOURCE_BUNDLE.sha256
readonly MANIFEST=${RUN_ROOT}/data_manifest.json
readonly MANIFEST_RECEIPT=${MANIFEST}.sha256
readonly SUMMARIZER=${TASK_ROOT}/MemNavData/summarize_hm3d_heldout_val10_revisit.py
readonly TEST=${TASK_ROOT}/MemNavData/test_hm3d_heldout_val10_revisit_summary.py
readonly INTEGRATION_TEST=${TASK_ROOT}/MemNavData/test_hm3d_heldout_val10_revisit_integration.py
readonly REPORT=${RUN_ROOT}/hm3d_heldout_val10_revisit_summary.json
readonly PY=/scratch/lg154/conda-envs/memnav/bin/python

fail() { echo "ABORT: $*" >&2; exit 2; }
[[ "$(sha256sum "${TASK_RECEIPT}" | awk '{print $1}')" == \
    "${EXPECTED_TASK_RECEIPT_SHA}" ]] || fail "task receipt changed"
(cd "${TASK_ROOT}" && sha256sum -c SOURCE_BUNDLE.sha256 >/dev/null) || \
  fail "task bundle changed"
[[ "$(sha256sum "${BASE_RECEIPT}" | awk '{print $1}')" == \
    "${EXPECTED_BASE_SOURCE_RECEIPT_SHA}" ]] || fail "base receipt changed"
for path in "${MANIFEST}" "${MANIFEST_RECEIPT}" "${SUMMARIZER}" \
  "${TEST}" "${INTEGRATION_TEST}" "${PY}"; do
  test -r "${path}" || fail "missing input ${path}"
done
(cd "${RUN_ROOT}" && sha256sum -c data_manifest.json.sha256 >/dev/null) || \
  fail "data manifest changed"
[[ ! -e "${REPORT}" && ! -e "${REPORT}.sha256" ]] || \
  fail "summary output already exists"

export PYTHONPYCACHEPREFIX=${SLURM_TMPDIR:-/tmp}/hm3dh10_sum_${SLURM_JOB_ID}
export PYTHONPATH=${TASK_ROOT}:${BASE_SOURCE_ROOT}${PYTHONPATH:+:${PYTHONPATH}}
mkdir -p "${PYTHONPYCACHEPREFIX}"
"${PY}" -m py_compile "${SUMMARIZER}"
"${PY}" -m unittest -q \
  MemNavData.test_hm3d_heldout_val10_revisit_summary \
  MemNavData.test_hm3d_heldout_val10_revisit_integration
"${PY}" -u "${SUMMARIZER}" --manifest "${MANIFEST}" \
  --run-root "${RUN_ROOT}" --out "${REPORT}"
(cd "${RUN_ROOT}" && \
  sha256sum -c hm3d_heldout_val10_revisit_summary.json.sha256 >/dev/null) || \
  fail "summary receipt invalid"
chmod a-w "${REPORT}.sha256"
echo "COMPLETE ${REPORT}"
