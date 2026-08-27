#!/usr/bin/env bash
#SBATCH --job-name=h3lifeABFix
#SBATCH --output=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/%x_%A_%a.out
#SBATCH --error=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/%x_%A_%a.err
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=cpu_short
#SBATCH --account=torch_pr_769_tandon_advanced

set -euo pipefail
: "${SLURM_ARRAY_TASK_ID:?scene array index required}"
: "${REPAIR_ROOT:?}" "${ORIGINAL_TASK_ROOT:?}" "${BASE_SOURCE_ROOT:?}"
: "${RUN_ROOT:?}" "${PARENT_ROOT:?}" "${PROTOCOL:?}" "${REPAIR_MANIFEST:?}"
: "${REPAIR_RECEIPT:?}" "${EXPECTED_REPAIR_RECEIPT_SHA:?}"
: "${ORIGINAL_TASK_RECEIPT:?}" "${EXPECTED_ORIGINAL_TASK_RECEIPT_SHA:?}"
: "${BASE_RECEIPT:?}" "${EXPECTED_BASE_RECEIPT_SHA:?}"

[[ "$(sha256sum "${REPAIR_RECEIPT}" | awk '{print $1}')" == \
   "${EXPECTED_REPAIR_RECEIPT_SHA}" ]]
(cd "${REPAIR_ROOT}" && sha256sum -c --quiet "${REPAIR_RECEIPT}")
[[ "$(sha256sum "${ORIGINAL_TASK_RECEIPT}" | awk '{print $1}')" == \
   "${EXPECTED_ORIGINAL_TASK_RECEIPT_SHA}" ]]
(cd "${ORIGINAL_TASK_ROOT}" && sha256sum -c --quiet "${ORIGINAL_TASK_RECEIPT}")
[[ "$(sha256sum "${BASE_RECEIPT}" | awk '{print $1}')" == \
   "${EXPECTED_BASE_RECEIPT_SHA}" ]]
(cd "${BASE_SOURCE_ROOT}" && sha256sum -c --quiet "${BASE_RECEIPT}")

PY=/scratch/lg154/conda-envs/habitat/bin/python
BASE_SIF=/share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif
readarray -t identity < <(/scratch/lg154/conda-envs/memnav/bin/python - \
  "${REPAIR_MANIFEST}" "${PARENT_ROOT}/sealed_inputs/parent_manifest.json" \
  "${SLURM_ARRAY_TASK_ID}" <<'PY'
import json, sys
repair = json.load(open(sys.argv[1]))
parent = json.load(open(sys.argv[2]))
index = int(sys.argv[3])
assert repair["selection_reads_navigation_outcomes"] is False
assert repair["scientific_thresholds_changed"] is False
assert repair["population_rule_changed"] is False
assert index in repair["indices"]
row = {int(item["scene_index"]): item for item in repair["fragments"]}[index]
assert row["scene"] == parent["scenes"][index]
print(row["scene"])
PY
)
[[ "${#identity[@]}" -eq 1 ]] || { echo "bad repair identity" >&2; exit 2; }
SCENE=${identity[0]}
OUT=${RUN_ROOT}/construct_ab/scenes/$(printf '%02d_%s' \
  "${SLURM_ARRAY_TASK_ID}" "${SCENE}")
pycache=${SLURM_TMPDIR:-/tmp}/h3life_abfix_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}
mkdir -p "${pycache}"

exec singularity exec -B /scratch/lg154 -B /scratch/yz11502 \
  "${BASE_SIF}" env PYTHONDONTWRITEBYTECODE=1 \
  PYTHONPYCACHEPREFIX="${pycache}" \
  PYTHONPATH="${REPAIR_ROOT}:${REPAIR_ROOT}/MemNavData:${ORIGINAL_TASK_ROOT}:${ORIGINAL_TASK_ROOT}/MemNavData:${BASE_SOURCE_ROOT}:${BASE_SOURCE_ROOT}/MemNavData" \
  "${PY}" -u \
    "${REPAIR_ROOT}/MemNavData/construct_hm3d_fullmono_lifelong_ab.py" \
    --parent-root "${PARENT_ROOT}" --protocol "${PROTOCOL}" \
    --scene-index "${SLURM_ARRAY_TASK_ID}" --out "${OUT}"
