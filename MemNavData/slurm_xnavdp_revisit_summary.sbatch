#!/bin/bash
#SBATCH --job-name=xnavdp_summary
#SBATCH --output=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_xnavdp_summary_%j.out
#SBATCH --error=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_xnavdp_summary_%j.err
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --partition=cpu_short
#SBATCH --account=torch_pr_769_tandon_advanced
#SBATCH --export=ALL

set -euo pipefail
umask 0022

: "${RUN_TAG:?export the full-run tag}"
: "${EXPECTED_COMMIT:?export the exact committed source SHA}"
: "${EXPECTED_MANIFEST_SHA:?export the frozen manifest SHA256}"

ROOT=${ROOT:-/scratch/yz11502/Research/Nav-axis-uturn}
RESULT_BASE=${RESULT_BASE:-/scratch/yz11502/Research/Nav-axis-uturn-results/xnavdp_revisit_gate_20260809}
RUN_ROOT=${RESULT_BASE}/${RUN_TAG}
MANIFEST=${ROOT}/MemNavData/expanded_navdp_router_eval_20260805.json
SUMMARIZER=${ROOT}/MemNavData/summarize_xnavdp_revisit_gate.py
HAB_PY=${HAB_PY:-/scratch/lg154/conda-envs/habitat/bin/python}

[[ "$(git -C "${ROOT}" rev-parse HEAD)" == "${EXPECTED_COMMIT}" ]] || {
  echo "ABORT: summary checkout differs from EXPECTED_COMMIT" >&2; exit 1; }
[[ "$(sha256sum "${MANIFEST}" | awk '{print $1}')" == \
    "${EXPECTED_MANIFEST_SHA}" ]] || {
  echo "ABORT: summary manifest SHA256 mismatch" >&2; exit 1; }
test -d "${RUN_ROOT}/scenes" || {
  echo "ABORT: full result root is incomplete" >&2; exit 1; }

temporary=${RUN_ROOT}/xnavdp_revisit_gate_report.json.tmp.${SLURM_JOB_ID}
final=${RUN_ROOT}/xnavdp_revisit_gate_report.json
test ! -e "${final}" || {
  echo "ABORT: summary already exists: ${final}" >&2; exit 1; }
"${HAB_PY}" "${SUMMARIZER}" \
  --manifest "${MANIFEST}" \
  --run-root "${RUN_ROOT}" > "${temporary}"
"${HAB_PY}" - "${temporary}" <<'PY'
import json, sys
report = json.load(open(sys.argv[1]))
if report.get("audit", {}).get("status") != "ok":
    raise SystemExit("summary audit failed")
print(json.dumps(report["frozen_g1_decision"], indent=2, sort_keys=True))
PY
mv "${temporary}" "${final}"
echo "[complete] ${final}"
