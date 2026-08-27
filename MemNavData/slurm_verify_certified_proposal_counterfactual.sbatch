#!/usr/bin/env bash
# Independent raw-record recomputation; does not import the formal summarizer.
#SBATCH --job-name=cec_prop_verify
#SBATCH --output=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/%x_%j.out
#SBATCH --error=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/%x_%j.err
#SBATCH --time=00:20:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --partition=cpu_short
#SBATCH --account=torch_pr_769_tandon_advanced
#SBATCH --export=ALL

set -euo pipefail
umask 0022
: "${VERIFIER_ROOT:?set immutable verifier source root}"
: "${VERIFIER_RECEIPT:?set verifier source receipt}"
: "${EXPECTED_VERIFIER_RECEIPT_SHA:?set verifier receipt SHA}"
: "${RUN_ROOT:?set formal audit root}"
: "${EXPECTED_COUNT:?set frozen population size}"

PY=/scratch/lg154/conda-envs/memnav/bin/python
[[ "$(sha256sum "${VERIFIER_RECEIPT}" | awk '{print $1}')" == \
   "${EXPECTED_VERIFIER_RECEIPT_SHA}" ]]
(cd "${VERIFIER_ROOT}" && sha256sum -c "${VERIFIER_RECEIPT}") >/dev/null
[[ "${EXPECTED_COUNT}" =~ ^[1-9][0-9]*$ ]]
[[ -f "${RUN_ROOT}/summary.json" ]]
[[ ! -e "${RUN_ROOT}/independent_verification.json" ]]
export PYTHONPYCACHEPREFIX=${SLURM_TMPDIR:-/tmp}/cec_verify_${SLURM_JOB_ID}
"${PY}" "${VERIFIER_ROOT}/MemNavData/independent_verify_certified_proposal_counterfactual.py" \
  --input-root "${RUN_ROOT}/records" \
  --summary "${RUN_ROOT}/summary.json" \
  --expected-count "${EXPECTED_COUNT}" \
  --output "${RUN_ROOT}/independent_verification.json"
sha256sum "${RUN_ROOT}/independent_verification.json" > \
  "${RUN_ROOT}/independent_verification.json.sha256"
