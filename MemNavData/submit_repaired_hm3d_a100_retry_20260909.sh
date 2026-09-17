#!/usr/bin/env bash
# Execute on the authenticated yz11502 login shell; reuse the sealed code.
set -euo pipefail
[[ "$(id -un)" == yz11502 ]]
export REPAIRED_BUNDLE=/scratch/yz11502/Research/Nav-axis-uturn-source-bundles/repaired_fullmono_c8cf8c60e7efd55f
export EXPECTED_REPAIR_SHA=c8cf8c60e7efd55f1b360bb5a2d5fd92850dc87d6a310b05e77f0a9239cdfc08
export REPAIRED_EXTENSION_ROOT=/scratch/yz11502/Research/Nav-axis-uturn-results/repaired_hm3d_extension_20260908/a100_retry1_20260909
export PYTHONFAULTHANDLER=1
RETRY_SUBMISSION=/scratch/yz11502/Research/Nav-axis-uturn-results/repaired_hm3d_extension_20260908/submission_a100_retry1_20260909
[[ "$(sha256sum "${REPAIRED_BUNDLE}/SOURCE_BUNDLE.sha256" | awk '{print $1}')" == "${EXPECTED_REPAIR_SHA}" ]]
(cd "${REPAIRED_BUNDLE}" && sha256sum -c --quiet SOURCE_BUNDLE.sha256)
[[ ! -e "${REPAIRED_EXTENSION_ROOT}" && ! -e "${RETRY_SUBMISSION}/submission.jobs" ]]
source "${REPAIRED_BUNDLE}/MemNavData/slurm_safe_submit.sh"
RETRY_TEMPLATE=${REPAIRED_BUNDLE}/MemNavData/slurm_repaired_hm3d_gate_extension.sbatch
bash -n "${RETRY_TEMPLATE}"
safe_sbatch --lint-fatal --test-only --partition=a100_tandon --array=1 \
  --job-name=cec_repaired_a100_retry --export=ALL "${RETRY_TEMPLATE}"
mkdir -p "${RETRY_SUBMISSION}"
safe_sbatch --lint-fatal --parsable --partition=a100_tandon --array=1 \
  --job-name=cec_repaired_a100_retry --export=ALL "${RETRY_TEMPLATE}" \
  | tee "${RETRY_SUBMISSION}/submission.jobs"
