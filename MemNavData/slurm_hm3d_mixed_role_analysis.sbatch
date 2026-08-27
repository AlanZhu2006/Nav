#!/usr/bin/env bash
#SBATCH --job-name=h3mix_sum
#SBATCH --output=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/%x_%j.out
#SBATCH --error=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/%x_%j.err
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=cpu_short
#SBATCH --account=torch_pr_769_tandon_advanced

set -euo pipefail
: "${MODE:?set summary or verify}"
: "${TASK_ROOT:?set task bundle}"
: "${EXPECTED_TASK_RECEIPT_SHA:?set task receipt SHA}"
: "${RUN_ROOT:?set run root}"
receipt=${TASK_ROOT}/SOURCE_BUNDLE.sha256
[[ "$(sha256sum "${receipt}" | awk '{print $1}')" == \
   "${EXPECTED_TASK_RECEIPT_SHA}" ]]
(cd "${TASK_ROOT}" && sha256sum -c SOURCE_BUNDLE.sha256 >/dev/null)
PY=/scratch/lg154/conda-envs/memnav/bin/python
if [[ "${MODE}" == summary ]]; then
  out=${RUN_ROOT}/hm3d_mixed_role_summary.json
  [[ ! -e "${out}" ]]
  "${PY}" -u "${TASK_ROOT}/MemNavData/summarize_hm3d_mixed_role.py" \
    --root "${RUN_ROOT}" --out "${out}"
  sha256sum "${out}" > "${out}.sha256"
elif [[ "${MODE}" == verify ]]; then
  summary=${RUN_ROOT}/hm3d_mixed_role_summary.json
  out=${RUN_ROOT}/hm3d_mixed_role_independent_verification.json
  [[ -f "${summary}" && ! -e "${out}" ]]
  "${PY}" -u "${TASK_ROOT}/MemNavData/verify_hm3d_mixed_role.py" \
    --root "${RUN_ROOT}" --summary "${summary}" --out "${out}"
  sha256sum "${out}" > "${out}.sha256"
else
  echo "invalid MODE=${MODE}" >&2; exit 2
fi
