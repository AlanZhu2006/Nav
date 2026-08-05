#!/bin/bash
# CPU-only postflight, normally submitted after the complete 20-scene array.
#SBATCH --job-name=sum_router20
#SBATCH --output=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_sum_router20_%j.out
#SBATCH --error=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_sum_router20_%j.err
#SBATCH --time=00:20:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --partition=cpu_short
#SBATCH --account=torch_pr_769_tandon_advanced
#SBATCH --export=ALL

set -euo pipefail
: "${RUN_TAG:?submit with the same RUN_TAG as the array}"
: "${EXPECTED_COMMIT:?submit with EXPECTED_COMMIT=<full SHA>}"

ROOT=/scratch/yz11502/Research/Nav-axis-uturn
RUN_ROOT=/scratch/yz11502/Research/Nav-axis-uturn-results/expanded_navdp_router_20260805/${RUN_TAG}
MANIFEST=${ROOT}/MemNavData/expanded_navdp_router_eval_20260805.json
SUMMARIZER=${ROOT}/MemNavData/summarize_expanded_navdp_router_eval.py
PYTHON=/usr/bin/python3

actual_commit=$(git -C "${ROOT}" rev-parse HEAD)
[[ "${actual_commit}" == "${EXPECTED_COMMIT}" ]] || {
  echo "ABORT: code commit mismatch" >&2; exit 1; }
"${PYTHON}" -m py_compile "${SUMMARIZER}"
"${PYTHON}" "${SUMMARIZER}" \
  --manifest "${MANIFEST}" \
  --run-root "${RUN_ROOT}" \
  > "${RUN_ROOT}/summary.json"
cat "${RUN_ROOT}/summary.json"
