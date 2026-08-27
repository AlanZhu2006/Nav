#!/usr/bin/env bash
#SBATCH --job-name=lifeNNRverify
#SBATCH --output=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/%x_%j.out
#SBATCH --error=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/%x_%j.err
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=cpu_short
#SBATCH --account=torch_pr_769_tandon_advanced
#SBATCH --export=ALL

set -euo pipefail
umask 0022
: "${SOURCE_ROOT:?set immutable task source root}"
: "${SOURCE_RECEIPT:?set source receipt}"
: "${EXPECTED_SOURCE_RECEIPT_SHA:?set source receipt SHA}"
: "${RUN_ROOT:?set unique lifelong run root}"
PY=/scratch/lg154/conda-envs/memnav/bin/python
OUT=${RUN_ROOT}/independent_verification.json
[[ ! -e "${OUT}" ]]
[[ "$(sha256sum "${SOURCE_RECEIPT}" | awk '{print $1}')" == \
   "${EXPECTED_SOURCE_RECEIPT_SHA}" ]]
(cd "${SOURCE_ROOT}" && sha256sum -c --quiet "${SOURCE_RECEIPT}")
(cd "${RUN_ROOT}/factual_b_support" && sha256sum -c --quiet population.json.sha256)
export PYTHONPATH=${SOURCE_ROOT}:${SOURCE_ROOT}/MemNavData
export PYTHONPYCACHEPREFIX=${SLURM_TMPDIR:-/tmp}/life_nnr_verify_${SLURM_JOB_ID}
"${PY}" -u \
  "${SOURCE_ROOT}/MemNavData/independent_verify_shared_online_lifelong_nnr.py" \
  --all_prior "${RUN_ROOT}/aggregate/all_prior" \
  --initial_leg_only "${RUN_ROOT}/aggregate/initial_leg_only" \
  --population "${RUN_ROOT}/factual_b_support/population.json" --out "${OUT}"
"${PY}" - "${OUT}" <<'PY'
import json,sys
p=json.load(open(sys.argv[1]))
if p.get("verified") is not True: raise SystemExit("verification is false")
PY
sha256sum "${OUT}" > "${OUT}.sha256"
chmod a-w "${OUT}" "${OUT}.sha256"
printf 'verified=true\n' > "${RUN_ROOT}/VERIFIED"
chmod a-w "${RUN_ROOT}/VERIFIED"
echo "COMPLETE verification=${OUT}"
