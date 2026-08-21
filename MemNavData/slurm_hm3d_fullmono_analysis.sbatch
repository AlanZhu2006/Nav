#!/usr/bin/env bash
#SBATCH --job-name=h3monoSum
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
: "${TASK_ROOT:?}" "${BASE_SOURCE_ROOT:?}" "${RUN_ROOT:?}" "${BENCH_ROOT:?}"
[[ "$(sha256sum "${TASK_RECEIPT}" | awk '{print $1}')" == \
   "${EXPECTED_TASK_RECEIPT_SHA}" ]]
(cd "${TASK_ROOT}" && sha256sum -c --quiet "${TASK_RECEIPT}")
PY=/scratch/lg154/conda-envs/memnav/bin/python
manifest_sha=$(sha256sum "${BENCH_ROOT}/manifest.json" | awk '{print $1}')
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="${TASK_ROOT}:${TASK_ROOT}/MemNavData:${BASE_SOURCE_ROOT}:${BASE_SOURCE_ROOT}/MemNavData"
if [[ "${MODE}" == summary ]]; then
  out=${RUN_ROOT}/hm3d_fullmono_mixed_role_summary.json
  [[ ! -e "${out}" ]]
  "${PY}" -u "${TASK_ROOT}/MemNavData/summarize_hm3d_fullmono_mixed_role.py" \
    --run-root "${RUN_ROOT}" --bench-root "${BENCH_ROOT}" \
    --expected-manifest-sha256 "${manifest_sha}" --out "${out}"
  sha256sum "${out}" >"${out}.sha256"
elif [[ "${MODE}" == verify ]]; then
  summary=${RUN_ROOT}/hm3d_fullmono_mixed_role_summary.json
  out=${RUN_ROOT}/hm3d_fullmono_mixed_role_independent_verification.json
  [[ -f "${summary}" && ! -e "${out}" ]]
  "${PY}" -u \
    "${TASK_ROOT}/MemNavData/independent_verify_hm3d_fullmono_mixed_role.py" \
    --run-root "${RUN_ROOT}" --bench-root "${BENCH_ROOT}" \
    --expected-manifest-sha256 "${manifest_sha}" \
    --summary "${summary}" --out "${out}"
  sha256sum "${out}" >"${out}.sha256"
else
  echo "invalid MODE=${MODE}" >&2; exit 2
fi
