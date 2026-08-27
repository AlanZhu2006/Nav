#!/usr/bin/env bash
# Seal the exact strict constructs left by the failed fresh40 power gate.
#SBATCH --job-name=drev_seal20
#SBATCH --output=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/%x_%j.out
#SBATCH --error=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/%x_%j.err
#SBATCH --time=00:30:00
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
: "${SOURCE_ROOT:?set immutable task source root}"
: "${SOURCE_RECEIPT:?set task source receipt}"
: "${EXPECTED_SOURCE_RECEIPT_SHA:?set task source receipt SHA}"
: "${RUN_ROOT:?set existing failed fresh40 run root}"
: "${UPSTREAM_RUN_ROOT:?set immutable fresh160 run root}"
: "${EXPECTED_UPSTREAM_MANIFEST_SHA:?set fresh160 manifest SHA}"
: "${FAILED_PREP_JOB_ID:?set failed preparation job id}"

HAB_PY=/scratch/lg154/conda-envs/habitat/bin/python
FINALIZER=${SOURCE_ROOT}/MemNavData/finalize_shared_online_double_revisit_fresh20.py
FAILED_STDOUT=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/drev_prep_${FAILED_PREP_JOB_ID}.out
FAILED_STDERR=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/drev_prep_${FAILED_PREP_JOB_ID}.err
export PYTHONPYCACHEPREFIX=${SLURM_TMPDIR:-/tmp}/drev_seal20_pycache_${SLURM_JOB_ID}
export PYTHONPATH=${SOURCE_ROOT}:${SOURCE_ROOT}/MemNavData${PYTHONPATH:+:${PYTHONPATH}}
mkdir -p "${PYTHONPYCACHEPREFIX}"

fail() { echo "ABORT: $*" >&2; exit 2; }
[[ "$(sha256sum "${SOURCE_RECEIPT}" | awk '{print $1}')" == \
   "${EXPECTED_SOURCE_RECEIPT_SHA}" ]] || fail "task source receipt changed"
(cd "${SOURCE_ROOT}" && sha256sum -c "${SOURCE_RECEIPT}") >/dev/null || \
  fail "task source bundle changed"
[[ "$(sha256sum "${UPSTREAM_RUN_ROOT}/data_manifest.json" | awk '{print $1}')" == \
   "${EXPECTED_UPSTREAM_MANIFEST_SHA}" ]] || fail "fresh160 manifest changed"
[[ -f "${RUN_ROOT}/prepared/INCOMPLETE" && ! -e "${RUN_ROOT}/prepared/SEALED" ]] || \
  fail "failed preparation is not in the expected unsealed state"
[[ ! -e "${RUN_ROOT}/scenes" ]] || fail "navigation outcomes already exist"

"${HAB_PY}" -m py_compile "${FINALIZER}"
"${HAB_PY}" -m unittest MemNavData.test_prepare_shared_online_double_revisit_fresh
"${HAB_PY}" -u "${FINALIZER}" \
  --prepared-root "${RUN_ROOT}/prepared" \
  --source-manifest "${UPSTREAM_RUN_ROOT}/data_manifest.json" \
  --expected-source-manifest-sha "${EXPECTED_UPSTREAM_MANIFEST_SHA}" \
  --failed-stdout "${FAILED_STDOUT}" \
  --failed-stderr "${FAILED_STDERR}" \
  --original-submission "${RUN_ROOT}/submission.json" \
  --failed-job-id "${FAILED_PREP_JOB_ID}" \
  --expected-candidates 120 \
  --expected-constructible 20 \
  --expected-scenes 13 \
  --preregistered-target 40

[[ ! -e "${RUN_ROOT}/prepared/INCOMPLETE" && \
   -f "${RUN_ROOT}/prepared/SEALED" ]] || fail "finalizer did not seal"
(cd "${RUN_ROOT}/prepared/benchmark" && sha256sum -c manifest.json.sha256) >/dev/null
echo "COMPLETE ${RUN_ROOT}/prepared"
