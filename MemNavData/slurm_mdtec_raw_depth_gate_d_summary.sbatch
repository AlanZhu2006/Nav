#!/bin/bash
#SBATCH --job-name=mdtecDsum
#SBATCH --output=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/mdtecDsum_%j.out
#SBATCH --error=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/mdtecDsum_%j.err
#SBATCH --time=00:20:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=cpu_short
#SBATCH --account=torch_pr_769_tandon_advanced
#SBATCH --export=ALL

set -euo pipefail
: "${SOURCE_ROOT:?set SOURCE_ROOT}"
: "${RUN_ROOT:?set RUN_ROOT}"
: "${SOURCE_RECEIPT:?set SOURCE_RECEIPT}"
: "${EXPECTED_SOURCE_RECEIPT_SHA:?set receipt SHA}"
MEMNAV_PY=/scratch/lg154/conda-envs/memnav/bin/python
[[ "$(sha256sum "${SOURCE_RECEIPT}" | awk '{print $1}')" == \
   "${EXPECTED_SOURCE_RECEIPT_SHA}" ]]
(cd "${SOURCE_ROOT}" && sha256sum -c --quiet "${SOURCE_RECEIPT}")
mkdir -p "${RUN_ROOT}/POSTHOC"
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="${SOURCE_ROOT}" "${MEMNAV_PY}" \
  "${SOURCE_ROOT}/MemNavData/summarize_mdtec_raw_depth_gate_d.py" \
    --run-root "${RUN_ROOT}" \
    --manifest "${SOURCE_ROOT}/MemNavData/expanded_navdp_router_eval_20260805.json" \
    --protocol "${SOURCE_ROOT}/MemNavData/mdtec_raw_depth_gate_d_protocol_20260819.json" \
    --analysis "${SOURCE_ROOT}/MemNavData/mdtec_raw_depth_gate_d_analysis_20260819.json" \
    --expected-protocol-sha 8deb4e13bd169bb39c7696a777656c3e4912f985f48ee364111b4a9a76cf9413 \
    --expected-analysis-sha 636b79d1eff9db42a4c3141438c3c469020363e60ab366171b0fdfa2aa639380 \
    --out "${RUN_ROOT}/POSTHOC/mdtec_gate_d_summary.json"
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="${SOURCE_ROOT}" "${MEMNAV_PY}" \
  "${SOURCE_ROOT}/MemNavData/independent_verify_mdtec_raw_depth_gate_d.py" \
    --run-root "${RUN_ROOT}" \
    --report "${RUN_ROOT}/POSTHOC/mdtec_gate_d_summary.json" \
    --analysis "${SOURCE_ROOT}/MemNavData/mdtec_raw_depth_gate_d_analysis_20260819.json" \
    --expected-analysis-sha 636b79d1eff9db42a4c3141438c3c469020363e60ab366171b0fdfa2aa639380 \
    --out "${RUN_ROOT}/POSTHOC/mdtec_gate_d_independent_verification.json"
sha256sum "${RUN_ROOT}/POSTHOC/"*.json > \
  "${RUN_ROOT}/POSTHOC/output_receipt.sha256"
chmod a-w "${RUN_ROOT}/POSTHOC/"*.json \
  "${RUN_ROOT}/POSTHOC/output_receipt.sha256"
