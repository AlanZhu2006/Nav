#!/usr/bin/env bash
#SBATCH --job-name=h3lifeV4Prefix
#SBATCH --output=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/%x_%A_%a.out
#SBATCH --error=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/%x_%A_%a.err
#SBATCH --time=00:20:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=40G
#SBATCH --partition=h100_tandon,a100_tandon
#SBATCH --gres=gpu:1
#SBATCH --account=torch_pr_769_tandon_advanced

set -euo pipefail
: "${SLURM_ARRAY_TASK_ID:?candidate history index required}"
: "${TASK_ROOT:?}" "${SERVER_SOURCE_ROOT:?}" "${BASE_SOURCE_ROOT:?}"
: "${RUN_ROOT:?}" "${PROTOCOL:?}"
: "${TASK_RECEIPT:?}" "${EXPECTED_TASK_RECEIPT_SHA:?}"
: "${SERVER_SOURCE_RECEIPT:?}" "${EXPECTED_SERVER_SOURCE_RECEIPT_SHA:?}"
: "${BASE_RECEIPT:?}" "${EXPECTED_BASE_RECEIPT_SHA:?}"
[[ "$(sha256sum "${TASK_RECEIPT}" | awk '{print $1}')" == \
   "${EXPECTED_TASK_RECEIPT_SHA}" ]]
(cd "${TASK_ROOT}" && sha256sum -c --quiet "${TASK_RECEIPT}")
[[ "$(sha256sum "${SERVER_SOURCE_RECEIPT}" | awk '{print $1}')" == \
   "${EXPECTED_SERVER_SOURCE_RECEIPT_SHA}" ]]
(cd "${SERVER_SOURCE_ROOT}" && \
  sha256sum -c --quiet "${SERVER_SOURCE_RECEIPT}")
[[ "$(sha256sum "${BASE_RECEIPT}" | awk '{print $1}')" == \
   "${EXPECTED_BASE_RECEIPT_SHA}" ]]
(cd "${BASE_SOURCE_ROOT}" && sha256sum -c --quiet "${BASE_RECEIPT}")
manifest=${RUN_ROOT}/ab_population/role_pairs/manifest.json
readarray -t identity < <(/scratch/lg154/conda-envs/memnav/bin/python - \
  "${manifest}" "${SLURM_ARRAY_TASK_ID}" <<'PY'
import json,sys
row=json.load(open(sys.argv[1]))['episodes'][int(sys.argv[2])]
print(row['scene']); print(row['episode'])
PY
)
[[ "${#identity[@]}" -eq 2 ]]
label=$(printf '%03d_%s_%s' "${SLURM_ARRAY_TASK_ID}" \
  "${identity[0]}" "${identity[1]}")
OUT=${RUN_ROOT}/prefix_fragments/${label}
BASE_SIF=/share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif
pycache=${SLURM_TMPDIR:-/tmp}/h3life_v4_prefix_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}
mkdir -p "${pycache}"
exec singularity exec --nv -B /scratch/lg154 -B /scratch/yz11502 \
  "${BASE_SIF}" env PYTHONDONTWRITEBYTECODE=1 \
  PYTHONPYCACHEPREFIX="${pycache}" \
  PYTHONPATH="${TASK_ROOT}:${TASK_ROOT}/MemNavData:${SERVER_SOURCE_ROOT}:${SERVER_SOURCE_ROOT}/MemNavData:${BASE_SOURCE_ROOT}:${BASE_SOURCE_ROOT}/MemNavData" \
  /scratch/lg154/conda-envs/habitat/bin/python -u \
    "${TASK_ROOT}/MemNavData/construct_hm3d_fullmono_lifelong_prefix.py" \
    --protocol "${PROTOCOL}" --ab-root "${RUN_ROOT}/ab_population" \
    --b-root "${RUN_ROOT}/factual_b" \
    --history-index "${SLURM_ARRAY_TASK_ID}" --out "${OUT}"
