#!/usr/bin/env bash
# One frozen HM3D held-out val10 scene per task; generate four two-leg episodes.
#SBATCH --job-name=hm3dh10_gen
#SBATCH --output=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/%x_%A_%a.out
#SBATCH --error=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/%x_%A_%a.err
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=h100_tandon,a100_tandon,h200_public,l40s_public
#SBATCH --gres=gpu:1
#SBATCH --account=torch_pr_769_tandon_advanced

set -euo pipefail
umask 0022

: "${SLURM_ARRAY_TASK_ID:?submit as array 0-9}"
: "${TASK_ROOT:?set immutable task bundle root}"
: "${BASE_SOURCE_ROOT:?set immutable base source root}"
: "${RUN_ROOT:?set run root}"
: "${DATA_ROOT:?set HM3D held-out val10 data root}"
: "${EXPECTED_TASK_RECEIPT_SHA:?set task receipt SHA}"
: "${EXPECTED_BASE_SOURCE_RECEIPT_SHA:?set base receipt SHA}"

readonly TASK_RECEIPT=${TASK_ROOT}/SOURCE_BUNDLE.sha256
readonly BASE_RECEIPT=${BASE_SOURCE_ROOT}/SOURCE_BUNDLE.sha256
readonly PROTOCOL=${TASK_ROOT}/MemNavData/hm3d_heldout_val10_revisit_protocol_20260816.json
readonly GENERATOR=${TASK_ROOT}/MemNavData/generate_twoleg.py
readonly HAB_PY=/scratch/lg154/conda-envs/habitat/bin/python
readonly BASE_SIF=/share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif

fail() { echo "ABORT: $*" >&2; exit 2; }
[[ "${SLURM_ARRAY_TASK_ID}" =~ ^[0-9]$ ]] || fail "array index must be 0..9"
[[ "$(sha256sum "${TASK_RECEIPT}" | awk '{print $1}')" == \
    "${EXPECTED_TASK_RECEIPT_SHA}" ]] || fail "task receipt changed"
(cd "${TASK_ROOT}" && sha256sum -c SOURCE_BUNDLE.sha256 >/dev/null) || \
  fail "task bundle changed"
[[ "$(sha256sum "${BASE_RECEIPT}" | awk '{print $1}')" == \
    "${EXPECTED_BASE_SOURCE_RECEIPT_SHA}" ]] || fail "base receipt changed"
for path in "${PROTOCOL}" "${GENERATOR}" "${HAB_PY}" "${BASE_SIF}" \
  "${DATA_ROOT}/data/scene_datasets/hm3d/heldout_val10/SEALED"; do
  test -r "${path}" || fail "missing input ${path}"
done

readarray -t spec < <("${HAB_PY}" - "${PROTOCOL}" \
  "${SLURM_ARRAY_TASK_ID}" <<'PY'
import json,sys
p=json.load(open(sys.argv[1])); row=p["scenes"][int(sys.argv[2])]
g=p["generation"]
print(row["scene_id"]); print(row["directory"])
print(g["episodes_per_scene"]); print(g["base_seed"]+row["index"])
print(g["episode_attempt_multiplier"])
PY
)
[[ "${#spec[@]}" -eq 5 ]] || fail "protocol reader failed"
scene=${spec[0]}; directory=${spec[1]}; count=${spec[2]}
seed=${spec[3]}; attempt_multiplier=${spec[4]}
asset_dir=${DATA_ROOT}/data/scene_datasets/hm3d/heldout_val10/${directory}
scene_file=${asset_dir}/${scene}.basis.glb
navmesh_file=${asset_dir}/${scene}.basis.navmesh
scene_out=${RUN_ROOT}/data/hm3d_2leg/${scene}
for path in "${scene_file}" "${navmesh_file}"; do
  test -r "${path}" || fail "missing scene asset ${path}"
done
[[ ! -e "${scene_out}" ]] || fail "scene output already exists: ${scene_out}"
mkdir -p "${scene_out}" /scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs

export PYTHONPYCACHEPREFIX=${SLURM_TMPDIR:-/tmp}/hm3dh10_gen_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}
mkdir -p "${PYTHONPYCACHEPREFIX}"
echo "host=$(hostname) job=${SLURM_JOB_ID} task=${SLURM_ARRAY_TASK_ID} scene=${scene} seed=${seed}"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

singularity exec --nv -B /scratch/lg154 -B /scratch/yz11502 \
  "${BASE_SIF}" env PYTHONPYCACHEPREFIX="${PYTHONPYCACHEPREFIX}" \
  MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet \
  "${HAB_PY}" -u "${GENERATOR}" \
    --scene "${scene_file}" --navmesh "${navmesh_file}" \
    --out "${scene_out}" --n "${count}" --n_legs 2 \
    --seed "${seed}" --episode_attempt_multiplier "${attempt_multiplier}" \
    --dA_min 3.0 --dA_max 9.0 \
    --window 32 --num_scale 8 --long_term_frac 0.7 \
    --min_recall_gap 32 --covis_lo 0.20 --covis_hi 1.0 \
    --head_max_deg 45.0 --goal_jitter_pos 1.50 \
    --agent_radius 0.30 --max_frame_turn 15.0

"${HAB_PY}" - "${scene_out}" "${scene}" "${seed}" "${count}" <<'PY'
import json,pathlib,sys
root=pathlib.Path(sys.argv[1]); scene=sys.argv[2]
seed=int(sys.argv[3]); count=int(sys.argv[4])
summary=json.loads((root/"generation_summary.json").read_text())
assert summary["complete"] is True
assert summary["generated_episodes"] == count
for index in range(count):
    path=root/f"episode_{index:04d}"/"meta/gen_meta.json"
    metadata=json.loads(path.read_text())
    assert metadata["generation_seed"] == seed
    assert metadata["scene"] == f"{scene}.basis.glb"
print(json.dumps({"status":"complete","scene":scene,"episodes":count}))
PY
