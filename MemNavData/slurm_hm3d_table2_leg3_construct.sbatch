#!/usr/bin/env bash
#SBATCH --job-name=h3T2L3Build
#SBATCH --output=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/%x_%A_%a.out
#SBATCH --error=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/%x_%A_%a.err
#SBATCH --time=00:45:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=h100_tandon,a100_tandon
#SBATCH --qos=gpu48
#SBATCH --gres=gpu:1
#SBATCH --account=torch_pr_769_tandon_advanced

set -euo pipefail
umask 0022
export PYTHONDONTWRITEBYTECODE=1

: "${SLURM_ARRAY_TASK_ID:?submit as a prefix array}"
: "${TASK_ROOT:?}" "${TASK_RECEIPT:?}" "${EXPECTED_TASK_RECEIPT_SHA:?}"
: "${BASE_SOURCE_ROOT:?}" "${BASE_RECEIPT:?}" "${EXPECTED_BASE_RECEIPT_SHA:?}"
: "${SOURCE_RUN_ROOT:?}" "${RUN_ROOT:?}" "${PROTOCOL:?}"

HAB_PY=/scratch/lg154/conda-envs/habitat/bin/python
SIF=/share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif
fail() { echo "ABORT: $*" >&2; exit 2; }

[[ "$(sha256sum "${TASK_RECEIPT}" | awk '{print $1}')" == \
   "${EXPECTED_TASK_RECEIPT_SHA}" ]] || fail "task receipt changed"
(cd "${TASK_ROOT}" && sha256sum -c --quiet "${TASK_RECEIPT}") || \
  fail "task bundle changed"
[[ "$(sha256sum "${BASE_RECEIPT}" | awk '{print $1}')" == \
   "${EXPECTED_BASE_RECEIPT_SHA}" ]] || fail "base receipt changed"
(cd "${BASE_SOURCE_ROOT}" && sha256sum -c --quiet "${BASE_RECEIPT}") || \
  fail "base source bundle changed"

readarray -t frozen < <("${HAB_PY}" - "${PROTOCOL}" <<'PY'
import json,sys
p=json.load(open(sys.argv[1])); s=p['source_population']
print(s['population_sha256']); print(s['seal_sha256'])
print(s['actual_AB_successful_histories'])
PY
)
[[ "${#frozen[@]}" -eq 3 ]] || fail "protocol reader failed"
population=${SOURCE_RUN_ROOT}/population/population.json
seal=${SOURCE_RUN_ROOT}/population/SEALED
[[ "$(sha256sum "${population}" | awk '{print $1}')" == "${frozen[0]}" ]] || \
  fail "source population changed"
[[ "$(sha256sum "${seal}" | awk '{print $1}')" == "${frozen[1]}" ]] || \
  fail "source population seal changed"
(( SLURM_ARRAY_TASK_ID < frozen[2] )) || fail "prefix index outside protocol"

readarray -t identity < <("${HAB_PY}" - "${population}" \
  "${SLURM_ARRAY_TASK_ID}" <<'PY'
import json,sys
r=json.load(open(sys.argv[1]))['accepted'][int(sys.argv[2])]
print(r['scene']); print(r['episode'])
PY
)
[[ "${#identity[@]}" -eq 2 ]] || fail "source prefix identity missing"
label=$(printf '%03d_%s_%s' "${SLURM_ARRAY_TASK_ID}" \
  "${identity[0]}" "${identity[1]}")
out=${RUN_ROOT}/construction/fragments/${label}
[[ ! -e "${out}" ]] || fail "construction output exists"
mkdir -p "${out%/*}" "${RUN_ROOT}/logs"
pycache=${SLURM_TMPDIR:-/tmp}/h3t2_l3_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}
mkdir -p "${pycache}"

echo "population_index=${SLURM_ARRAY_TASK_ID} scene=${identity[0]} host=$(hostname)"
nvidia-smi --query-gpu=name,uuid,memory.total --format=csv,noheader
singularity exec --nv -B /scratch/lg154 -B /scratch/yz11502 "${SIF}" env \
  PYTHONDONTWRITEBYTECODE=1 PYTHONPYCACHEPREFIX="${pycache}" \
  PYTHONPATH="${TASK_ROOT}:${TASK_ROOT}/MemNavData:${BASE_SOURCE_ROOT}:${BASE_SOURCE_ROOT}/MemNavData" \
  "${HAB_PY}" -u \
  "${TASK_ROOT}/MemNavData/construct_hm3d_table2_leg3_mixed_role.py" \
    --protocol "${PROTOCOL}" --source-root "${SOURCE_RUN_ROOT}" \
    --population-index "${SLURM_ARRAY_TASK_ID}" --out "${out}"
chmod -R a-w "${out}"
echo "COMPLETE ${out}"
