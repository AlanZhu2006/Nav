#!/usr/bin/env bash
# Generate four frozen source episodes for one fresh HM3D scene.
#SBATCH --job-name=h3freshGen
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
: "${SLURM_ARRAY_TASK_ID:?array 0-53 required}"
: "${TASK_ROOT:?}" "${BASE_SOURCE_ROOT:?}" "${RUN_ROOT:?}" \
  "${DATA_ROOT:?}" "${EXPECTED_TASK_RECEIPT_SHA:?}" \
  "${EXPECTED_BASE_RECEIPT_SHA:?}"
[[ "${SLURM_ARRAY_TASK_ID}" =~ ^([0-9]|[1-4][0-9]|5[0-3])$ ]] || {
  echo "bad scene index" >&2; exit 2; }
TASK_RECEIPT=${TASK_ROOT}/SOURCE_BUNDLE.sha256
BASE_RECEIPT=${BASE_SOURCE_ROOT}/source_inputs.sha256
PROTOCOL=${TASK_ROOT}/MemNavData/hm3d_fresh_fullmono_mixed_role_protocol_20260820.json
GENERATOR=${TASK_ROOT}/MemNavData/generate_twoleg.py
HAB_PY=/scratch/lg154/conda-envs/habitat/bin/python
SIF=/share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif
[[ "$(sha256sum "${TASK_RECEIPT}" | awk '{print $1}')" == \
   "${EXPECTED_TASK_RECEIPT_SHA}" ]]
(cd "${TASK_ROOT}" && sha256sum -c --quiet SOURCE_BUNDLE.sha256)
[[ "$(sha256sum "${BASE_RECEIPT}" | awk '{print $1}')" == \
   "${EXPECTED_BASE_RECEIPT_SHA}" ]]
(cd "${BASE_SOURCE_ROOT}" && sha256sum -c --quiet source_inputs.sha256)

readarray -t spec < <("${HAB_PY}" - "${PROTOCOL}" \
  "${SLURM_ARRAY_TASK_ID}" <<'PY'
import json,sys
p=json.load(open(sys.argv[1])); row=p["dataset"]["scenes"][int(sys.argv[2])]
g=p["source_generation"]
print(row["scene_id"]); print(row["directory"])
print(g["episodes_per_scene"]); print(g["base_seed"]+row["rank"])
print(g["episode_attempt_multiplier"])
PY
)
[[ "${#spec[@]}" -eq 5 ]]
scene=${spec[0]}; directory=${spec[1]}; count=${spec[2]}
seed=${spec[3]}; multiplier=${spec[4]}
asset_dir=${DATA_ROOT}/data/scene_datasets/hm3d/fresh_fullmono54/${directory}
scene_file=${asset_dir}/${scene}.basis.glb
navmesh_file=${asset_dir}/${scene}.basis.navmesh
scene_out=${RUN_ROOT}/source_generation/${scene}
[[ -r "${asset_dir}/../SEALED" || -r "${DATA_ROOT}/data/scene_datasets/hm3d/fresh_fullmono54/SEALED" ]]
[[ -r "${scene_file}" && -r "${navmesh_file}" && ! -e "${scene_out}" ]]
mkdir -p "${scene_out}" /scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs
pycache=${SLURM_TMPDIR:-/tmp}/h3fresh_gen_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}
mkdir -p "${pycache}"
singularity exec --nv -B /scratch/lg154 -B /scratch/yz11502 \
  "${SIF}" env PYTHONPYCACHEPREFIX="${pycache}" MAGNUM_LOG=quiet \
  HABITAT_SIM_LOG=quiet "${HAB_PY}" -u "${GENERATOR}" \
    --scene "${scene_file}" --navmesh "${navmesh_file}" \
    --out "${scene_out}" --n "${count}" --n_legs 2 \
    --seed "${seed}" --episode_attempt_multiplier "${multiplier}" \
    --allow_incomplete --dA_min 3.0 --dA_max 9.0 \
    --window 32 --num_scale 8 --long_term_frac 0.7 \
    --min_recall_gap 32 --covis_lo 0.20 --covis_hi 1.0 \
    --head_max_deg 45.0 --goal_jitter_pos 1.50 \
    --agent_radius 0.30 --max_frame_turn 15.0

"${HAB_PY}" - "${scene_out}" "${scene}" "${seed}" "${count}" \
  "${SLURM_ARRAY_TASK_ID}" <<'PY'
import hashlib,json,pathlib,sys
root=pathlib.Path(sys.argv[1]); scene=sys.argv[2]; seed=int(sys.argv[3])
count=int(sys.argv[4]); rank=int(sys.argv[5])
summary_path=root/"generation_summary.json"
if not summary_path.is_file(): raise SystemExit("generation summary missing")
summary=json.loads(summary_path.read_text())
if summary.get("requested_episodes") != count or summary.get("n_legs") != 2:
    raise SystemExit("generation summary contract changed")
generated=int(summary.get("generated_episodes",-1))
complete=summary.get("complete") is True and generated == count
if complete:
    for index in range(count):
        meta=root/f"episode_{index:04d}"/"meta/gen_meta.json"
        payload=json.loads(meta.read_text())
        assert payload["generation_seed"] == seed
        assert payload["scene"] == f"{scene}.basis.glb"
status="complete" if complete else "constructibility_attrition"
receipt={
 "schema_version":"hm3d_fresh_source_generation_scene_v1_20260820",
 "status":status,"scene":scene,"scene_rank":rank,"seed":seed,
 "requested_episodes":count,"generated_episodes":generated,
 "complete_scene_required":True,"partial_episodes_excluded":not complete,
 "fixed_attempt_budget":True,"retry_or_parameter_change":False,
 "generation_summary_sha256":hashlib.sha256(summary_path.read_bytes()).hexdigest(),
}
path=root/"scene_generation_receipt.json"
path.write_text(json.dumps(receipt,indent=2,sort_keys=True)+"\n")
(root/"scene_generation_receipt.json.sha256").write_text(
    hashlib.sha256(path.read_bytes()).hexdigest()+"  scene_generation_receipt.json\n")
print(json.dumps(receipt,sort_keys=True))
PY
