#!/usr/bin/env bash
# One scene per task: generate the frozen 8 fresh 2-leg episodes.
#SBATCH --job-name=revfresh_gen
#SBATCH --output=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/%x_%A_%a.out
#SBATCH --error=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/%x_%A_%a.err
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=l40s_public,h100_tandon,h200_public,a100_tandon
#SBATCH --gres=gpu:1
#SBATCH --account=torch_pr_769_tandon_advanced
#SBATCH --export=ALL

set -euo pipefail
umask 0022
: "${SLURM_ARRAY_TASK_ID:?submit as array 0-19}"
: "${SOURCE_ROOT:?set immutable SOURCE_ROOT}"
: "${RUN_ROOT:?set RUN_ROOT}"
: "${SOURCE_RECEIPT:?set SOURCE_RECEIPT}"
: "${EXPECTED_SOURCE_RECEIPT_SHA:?set EXPECTED_SOURCE_RECEIPT_SHA}"

BASE_SIF=/share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif
HAB_PY=/scratch/lg154/conda-envs/habitat/bin/python
PROTOCOL=${SOURCE_ROOT}/MemNavData/revisit_fresh_confirmation_protocol_20260811.json
GENERATOR=${SOURCE_ROOT}/MemNavData/generate_twoleg.py
ASSET_ROOT=/scratch/lg154/Research/datasets/mp3d/mp3d
export PYTHONPYCACHEPREFIX=${SLURM_TMPDIR:-/tmp}/revfresh_pycache_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}
mkdir -p "${PYTHONPYCACHEPREFIX}"
[[ "${SLURM_ARRAY_TASK_ID}" =~ ^([0-9]|1[0-9])$ ]] || { echo ABORT >&2; exit 2; }
[[ "$(sha256sum "${SOURCE_RECEIPT}" | awk '{print $1}')" == \
    "${EXPECTED_SOURCE_RECEIPT_SHA}" ]] || { echo "ABORT: source receipt" >&2; exit 2; }
for path in "${BASE_SIF}" "${HAB_PY}" "${PROTOCOL}" "${GENERATOR}"; do
  test -r "${path}" || { echo "ABORT: missing ${path}" >&2; exit 2; }
done
mkdir -p "${RUN_ROOT}/data" /scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs

readarray -t spec < <("${HAB_PY}" - "${PROTOCOL}" "${SLURM_ARRAY_TASK_ID}" <<'PY'
import json, sys
p=json.load(open(sys.argv[1])); i=int(sys.argv[2])
print(p["scenes"][i]); print(p["episodes_per_scene"])
print(int(p["generation"]["base_seed"])+i)
print(int(p["generation"]["episode_attempt_multiplier"]))
PY
)
scene=${spec[0]}; count=${spec[1]}; seed=${spec[2]}
attempt_multiplier=${spec[3]}
[[ "${attempt_multiplier}" =~ ^[1-9][0-9]*$ ]] || {
  echo "ABORT: invalid EPISODE_ATTEMPT_MULTIPLIER=${attempt_multiplier}" >&2; exit 2; }
scene_file=${ASSET_ROOT}/${scene}/${scene}.glb
test -r "${scene_file}" || { echo "ABORT: missing ${scene_file}" >&2; exit 2; }

echo "host=$(hostname) job=${SLURM_JOB_ID} task=${SLURM_ARRAY_TASK_ID} scene=${scene} seed=${seed} attempt_multiplier=${attempt_multiplier}"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

singularity exec --nv -B /scratch/lg154 -B /scratch/yz11502 "${BASE_SIF}" \
  env PYTHONPYCACHEPREFIX="${PYTHONPYCACHEPREFIX}" \
  "${HAB_PY}" -m py_compile "${GENERATOR}"
singularity exec --nv -B /scratch/lg154 -B /scratch/yz11502 "${BASE_SIF}" \
  env PYTHONPYCACHEPREFIX="${PYTHONPYCACHEPREFIX}" \
  "${HAB_PY}" -u "${GENERATOR}" \
    --scene "${scene_file}" --navmesh "" \
    --out "${RUN_ROOT}/data" --n2 "${count}" \
    --seed "${seed}" --episode_attempt_multiplier "${attempt_multiplier}" \
    --dA_min 3.0 --dA_max 9.0 \
    --window 32 --num_scale 8 --long_term_frac 0.7 \
    --min_recall_gap 32 --covis_lo 0.20 --covis_hi 1.0 \
    --head_max_deg 45.0 --goal_jitter_pos 1.50 \
    --agent_radius 0.30 --max_frame_turn 15.0

scene_out=${RUN_ROOT}/data/mp3d_2leg/${scene}
actual=$(find "${scene_out}" -mindepth 1 -maxdepth 1 -type d -name 'episode_*' | wc -l)
[[ "${actual}" -eq "${count}" ]] || {
  echo "ABORT: generated ${actual}/${count} episodes" >&2; exit 2; }
"${HAB_PY}" - "${scene_out}" "${scene}" "${seed}" "${count}" <<'PY'
import json, sys
from pathlib import Path
root=Path(sys.argv[1]); scene=sys.argv[2]; seed=int(sys.argv[3]); count=int(sys.argv[4])
for i in range(count):
    p=root/f"episode_{i:04d}"/"meta"/"gen_meta.json"
    m=json.loads(p.read_text())
    if m.get("generation_seed") != seed or Path(m.get("scene", "")).stem != scene:
        raise SystemExit(f"generation receipt mismatch: {p}")
print(json.dumps({"scene":scene,"seed":seed,"episodes":count,"status":"complete"}))
PY
