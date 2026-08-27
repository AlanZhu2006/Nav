#!/usr/bin/env bash
#SBATCH --job-name=h3lifeSealAB
#SBATCH --output=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/%x_%j.out
#SBATCH --error=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/%x_%j.err
#SBATCH --time=00:20:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --partition=cpu_short
#SBATCH --account=torch_pr_769_tandon_advanced

set -euo pipefail
: "${TASK_ROOT:?}" "${BASE_SOURCE_ROOT:?}" "${RUN_ROOT:?}"
: "${PARENT_ROOT:?}" "${PROTOCOL:?}"
[[ "$(sha256sum "${TASK_RECEIPT}" | awk '{print $1}')" == \
   "${EXPECTED_TASK_RECEIPT_SHA}" ]]
(cd "${TASK_ROOT}" && sha256sum -c --quiet "${TASK_RECEIPT}")
if [[ -n "${WRAPPER_ROOT:-}" ]]; then
  : "${WRAPPER_RECEIPT:?}" "${EXPECTED_WRAPPER_RECEIPT_SHA:?}"
  [[ "$(sha256sum "${WRAPPER_RECEIPT}" | awk '{print $1}')" == \
     "${EXPECTED_WRAPPER_RECEIPT_SHA}" ]]
  (cd "${WRAPPER_ROOT}" && sha256sum -c --quiet "${WRAPPER_RECEIPT}")
fi
BASE_SIF=/share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif
PY=/scratch/lg154/conda-envs/habitat/bin/python
HAB_REQUESTS_VENDOR=/scratch/lg154/conda-envs/habitat/lib/python3.9/site-packages/pip/_vendor
pycache=${SLURM_TMPDIR:-/tmp}/h3life_seal_ab_${SLURM_JOB_ID}
mkdir -p "${pycache}"
singularity exec -B /scratch/lg154 -B /scratch/yz11502 \
  "${BASE_SIF}" env PYTHONDONTWRITEBYTECODE=1 \
  PYTHONPYCACHEPREFIX="${pycache}" \
  PYTHONPATH="${TASK_ROOT}:${TASK_ROOT}/MemNavData:${BASE_SOURCE_ROOT}:${BASE_SOURCE_ROOT}/MemNavData:${HAB_REQUESTS_VENDOR}" \
  "${PY}" -u "${TASK_ROOT}/MemNavData/finalize_hm3d_fullmono_lifelong_ab.py" \
    --parent-root "${PARENT_ROOT}" --protocol "${PROTOCOL}" \
    --construction-root "${RUN_ROOT}/construct_ab/scenes" \
    --out "${RUN_ROOT}/benchmarks_ab"
(cd "${RUN_ROOT}/benchmarks_ab" && sha256sum -c BENCHMARK_FILES.sha256 >/dev/null)
chmod -R a-w "${RUN_ROOT}/benchmarks_ab"
