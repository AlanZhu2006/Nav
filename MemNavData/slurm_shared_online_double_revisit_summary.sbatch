#!/usr/bin/env bash
# Independent paired and scene-clustered audit after every frozen task succeeds.
#SBATCH --job-name=drev_sum
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
: "${SOURCE_RECEIPT:?set task source receipt}"
: "${EXPECTED_SOURCE_RECEIPT_SHA:?set task source receipt SHA}"
: "${RUN_ROOT:?set run root}"

HAB_PY=/scratch/lg154/conda-envs/habitat/bin/python
AUDITOR=${SOURCE_ROOT}/MemNavData/audit_shared_online_double_revisit_fresh.py
BENCH_ROOT=${RUN_ROOT}/prepared/benchmark
OUT=${RUN_ROOT}/report.json
export PYTHONPYCACHEPREFIX=${SLURM_TMPDIR:-/tmp}/drev_sum_pycache_${SLURM_JOB_ID}
export PYTHONPATH=${SOURCE_ROOT}:${SOURCE_ROOT}/MemNavData${PYTHONPATH:+:${PYTHONPATH}}
mkdir -p "${PYTHONPYCACHEPREFIX}"

[[ "$(sha256sum "${SOURCE_RECEIPT}" | awk '{print $1}')" == \
   "${EXPECTED_SOURCE_RECEIPT_SHA}" ]] || { echo "ABORT: source receipt" >&2; exit 2; }
(cd "${SOURCE_ROOT}" && sha256sum -c "${SOURCE_RECEIPT}") >/dev/null
(cd "${BENCH_ROOT}" && sha256sum -c manifest.json.sha256) >/dev/null
[[ ! -e "${OUT}" ]] || { echo "ABORT: report already exists" >&2; exit 2; }
EXPECTED_MANIFEST_SHA=$(awk '{print $1}' "${BENCH_ROOT}/manifest.json.sha256")
EXPECTED_EPISODES=$("${HAB_PY}" -c \
  'import json,sys; print(len(json.load(open(sys.argv[1]))["episodes"]))' \
  "${BENCH_ROOT}/manifest.json")
[[ "${EXPECTED_EPISODES}" =~ ^[1-9][0-9]*$ ]] || {
  echo "ABORT: invalid frozen episode count" >&2; exit 2; }

"${HAB_PY}" -m py_compile "${AUDITOR}"
"${HAB_PY}" -m unittest MemNavData.test_audit_shared_online_double_revisit_fresh
"${HAB_PY}" -u "${AUDITOR}" \
  --run-root "${RUN_ROOT}" \
  --expected-manifest-sha "${EXPECTED_MANIFEST_SHA}" \
  --expected-episodes "${EXPECTED_EPISODES}" \
  --bootstrap-seed 20260813 \
  --bootstrap-resamples 100000 \
  --out "${OUT}"
sha256sum "${OUT}" > "${OUT}.sha256"
chmod a-w "${OUT}" "${OUT}.sha256"
echo "COMPLETE ${OUT}"
