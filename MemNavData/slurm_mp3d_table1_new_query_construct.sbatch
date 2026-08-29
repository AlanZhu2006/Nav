#!/usr/bin/env bash
#SBATCH --job-name=m3T1construct
#SBATCH --output=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/%x_%A_%a.out
#SBATCH --error=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/%x_%A_%a.err
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --partition=h100_tandon,a100_tandon
#SBATCH --qos=gpu48
#SBATCH --gres=gpu:1
#SBATCH --account=torch_pr_769_tandon_advanced

set -euo pipefail
umask 0022
export PYTHONDONTWRITEBYTECODE=1

: "${SLURM_ARRAY_TASK_ID:?submit as an array}"
: "${TASK_ROOT:?}" "${TASK_RECEIPT:?}" "${EXPECTED_TASK_RECEIPT_SHA:?}"
: "${RUN_ROOT:?}" "${SOURCE_LEDGER:?}" "${EXPECTED_SOURCE_LEDGER_SHA:?}"

HAB_PY=/scratch/lg154/conda-envs/habitat/bin/python
SIF=/share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif
fail() { echo "ABORT: $*" >&2; exit 2; }

[[ "$(sha256sum "${TASK_RECEIPT}" | awk '{print $1}')" == \
   "${EXPECTED_TASK_RECEIPT_SHA}" ]] || fail "task receipt changed"
(cd "${TASK_ROOT}" && sha256sum -c --quiet "${TASK_RECEIPT}") || \
  fail "task bundle changed"
[[ "$(sha256sum "${SOURCE_LEDGER}" | awk '{print $1}')" == \
   "${EXPECTED_SOURCE_LEDGER_SHA}" ]] || fail "source ledger changed"
(( SLURM_ARRAY_TASK_ID >= 0 && SLURM_ARRAY_TASK_ID < 20 )) || \
  fail "scene index outside frozen population"

scene=$("${HAB_PY}" - "${SOURCE_LEDGER}" "${SLURM_ARRAY_TASK_ID}" <<'PY'
import json,sys
p=json.load(open(sys.argv[1])); print(p["scenes"][int(sys.argv[2])]["scene"])
PY
)
label=$(printf '%02d_%s' "${SLURM_ARRAY_TASK_ID}" "${scene}")
out=${RUN_ROOT}/construction/scenes/${label}
[[ ! -e "${out}" ]] || fail "construction output already exists"
mkdir -p "${out%/*}" "${RUN_ROOT}/logs"
export PYTHONPYCACHEPREFIX=${SLURM_TMPDIR:-/tmp}/m3t1_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}
mkdir -p "${PYTHONPYCACHEPREFIX}"

echo "scene_index=${SLURM_ARRAY_TASK_ID} scene=${scene} host=$(hostname)"
nvidia-smi --query-gpu=name,uuid,memory.total --format=csv,noheader
singularity exec --nv -B /scratch/lg154 -B /scratch/yz11502 "${SIF}" env \
  PYTHONDONTWRITEBYTECODE=1 PYTHONPYCACHEPREFIX="${PYTHONPYCACHEPREFIX}" \
  PYTHONPATH="${TASK_ROOT}:${TASK_ROOT}/MemNavData" \
  "${HAB_PY}" -u \
  "${TASK_ROOT}/MemNavData/build_mp3d_table1_new_query_scene.py" \
    --source-ledger "${SOURCE_LEDGER}" \
    --scene-index "${SLURM_ARRAY_TASK_ID}" --out "${out}"
sha256sum "${out}/construction_receipt.json" \
  >"${out}/construction_receipt.json.sha256"
chmod -R a-w "${out}"
echo "COMPLETE ${out}"
