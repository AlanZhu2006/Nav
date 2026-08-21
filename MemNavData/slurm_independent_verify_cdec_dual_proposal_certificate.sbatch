#!/usr/bin/env bash
#SBATCH --job-name=cdec_ind_verify
#SBATCH --output=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/%x_%j.out
#SBATCH --error=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/%x_%j.err
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --partition=cs
#SBATCH --qos=cpu48
#SBATCH --account=torch_pr_769_tandon_advanced
#SBATCH --export=ALL

set -euo pipefail
umask 0022
: "${SOURCE_ROOT:?set immutable verifier source root}"
: "${SOURCE_RECEIPT:?set verifier source receipt}"
: "${EXPECTED_SOURCE_RECEIPT_SHA:?set verifier source receipt SHA}"
: "${RUN_ROOT:?set CDEC run root}"
: "${OFFICIAL_REPORT_NAME:?set official report basename}"
: "${ROLE_SPLIT:?set frozen role split path}"
: "${EXPECTED_ROLE_SPLIT_SHA:?set frozen role split SHA}"

PY=/scratch/lg154/conda-envs/memnav/bin/python
OUT=${RUN_ROOT}/independent_verification_v1.json
[[ ! -e "${OUT}" ]] || { echo "ABORT: verification exists" >&2; exit 2; }
[[ "$(sha256sum "${SOURCE_RECEIPT}" | awk '{print $1}')" == \
   "${EXPECTED_SOURCE_RECEIPT_SHA}" ]] || { echo "ABORT: bad source receipt" >&2; exit 2; }
(cd "${SOURCE_ROOT}" && sha256sum -c "${SOURCE_RECEIPT}") >/dev/null
[[ "$(sha256sum "${ROLE_SPLIT}" | awk '{print $1}')" == \
   "${EXPECTED_ROLE_SPLIT_SHA}" ]] || { echo "ABORT: role split changed" >&2; exit 2; }
(cd "${RUN_ROOT}/dual" && sha256sum -c OUTPUTS.sha256) >/dev/null
[[ -f "${RUN_ROOT}/${OFFICIAL_REPORT_NAME}" \
   && -f "${RUN_ROOT}/${OFFICIAL_REPORT_NAME}.sha256" ]] || {
  echo "ABORT: official report is incomplete" >&2; exit 2; }
(cd "${RUN_ROOT}" && sha256sum -c "${OFFICIAL_REPORT_NAME}.sha256") >/dev/null

export PYTHONPATH=${SOURCE_ROOT}${PYTHONPATH:+:${PYTHONPATH}}
"${PY}" -m unittest \
  MemNavData.test_independent_verify_cdec_dual_proposal_certificate
"${PY}" "${SOURCE_ROOT}/MemNavData/independent_verify_cdec_dual_proposal_certificate.py" \
  --dual-rows "${RUN_ROOT}/dual/lingbot_goal_loop_closure_rows.csv" \
  --collector-report "${RUN_ROOT}/dual/diagnostic_lingbot_goal_loop_closure.json" \
  --official-report "${RUN_ROOT}/${OFFICIAL_REPORT_NAME}" \
  --role-split "${ROLE_SPLIT}" \
  --out "${OUT}"
sha256sum "${OUT}" > "${OUT}.sha256"
chmod a-w "${OUT}" "${OUT}.sha256"
echo "COMPLETE ${OUT}"
