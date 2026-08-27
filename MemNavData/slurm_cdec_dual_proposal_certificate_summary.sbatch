#!/usr/bin/env bash
#SBATCH --job-name=cdec_pnp_sum
#SBATCH --output=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/%x_%j.out
#SBATCH --error=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/%x_%j.err
#SBATCH --time=01:00:00
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
: "${SOURCE_RECEIPT:?set source receipt}"
: "${EXPECTED_SOURCE_RECEIPT_SHA:?set source receipt SHA}"
: "${RUN_ROOT:?set result root}"
REPORT_NAME=${REPORT_NAME:-report.json}
PY=/scratch/lg154/conda-envs/memnav/bin/python
export PYTHONPATH=${SOURCE_ROOT}${PYTHONPATH:+:${PYTHONPATH}}
[[ "${REPORT_NAME}" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*\.json$ ]] || {
  echo "ABORT: invalid report name" >&2; exit 2; }
[[ "$(sha256sum "${SOURCE_RECEIPT}" | awk '{print $1}')" == \
   "${EXPECTED_SOURCE_RECEIPT_SHA}" ]] || { echo ABORT >&2; exit 2; }
(cd "${SOURCE_ROOT}" && sha256sum -c "${SOURCE_RECEIPT}") >/dev/null
(cd "${RUN_ROOT}/dual" && sha256sum -c OUTPUTS.sha256) >/dev/null
OUT=${RUN_ROOT}/${REPORT_NAME}
[[ ! -e "${OUT}" ]] || { echo "ABORT: report exists" >&2; exit 2; }
"${PY}" -m pytest -q \
  "${SOURCE_ROOT}/MemNavData/test_summarize_cdec_dual_proposal_certificate.py" \
  "${SOURCE_ROOT}/MemNavData/test_summarize_lingbot_lightglue_localization.py"
"${PY}" "${SOURCE_ROOT}/MemNavData/summarize_cdec_dual_proposal_certificate.py" \
  --dual-rows "${RUN_ROOT}/dual/lingbot_goal_loop_closure_rows.csv" \
  --dual-report "${RUN_ROOT}/dual/diagnostic_lingbot_goal_loop_closure.json" \
  --out "${OUT}"
sha256sum "${OUT}" > "${OUT}.sha256"
chmod a-w "${OUT}" "${OUT}.sha256"
echo "COMPLETE ${OUT}"
