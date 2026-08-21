#!/usr/bin/env bash
# Build one C goal from one factual online-A trace; no C rollout is read.
#SBATCH --job-name=nnr_build
#SBATCH --output=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/%x_%A_%a.out
#SBATCH --error=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/%x_%A_%a.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --partition=l40s_public,h100_tandon,h200_public,a100_tandon
#SBATCH --qos=gpu48
#SBATCH --gres=gpu:1
#SBATCH --account=torch_pr_769_tandon_advanced
#SBATCH --export=ALL

set -euo pipefail
umask 0022
: "${SLURM_ARRAY_TASK_ID:?submit 0-21}"
: "${SOURCE_ROOT:?set immutable source root}"
: "${SOURCE_RECEIPT:?set source receipt}"
: "${EXPECTED_SOURCE_RECEIPT_SHA:?set receipt SHA}"
: "${RUN_ROOT:?set run root}"
: "${GENERATION_ROOT:?set strict-v4 source}"
: "${ASSET_ROOT:?set MP3D asset root}"

HAB_PY=/scratch/lg154/conda-envs/habitat/bin/python
BASE_SIF=/share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif
TRACE_ROOT=${RUN_ROOT}/prepared/traces
PARTS_ROOT=${RUN_ROOT}/prepared/parts
fail() { echo "ABORT: $*" >&2; exit 2; }
[[ "${SLURM_ARRAY_TASK_ID}" =~ ^([0-9]|1[0-9]|2[01])$ ]] || \
  fail "array index outside 0-21"
[[ -f "${RUN_ROOT}/prepared/EXTRACTED" ]] || fail "traces not sealed"
(cd "${TRACE_ROOT}" && sha256sum -c manifest.json.sha256) >/dev/null || \
  fail "trace manifest changed"
[[ "$(sha256sum "${SOURCE_RECEIPT}" | awk '{print $1}')" == \
   "${EXPECTED_SOURCE_RECEIPT_SHA}" ]] || fail "source receipt changed"
(cd "${SOURCE_ROOT}" && sha256sum -c "${SOURCE_RECEIPT}") >/dev/null || \
  fail "source bundle changed"
[[ -f "${BASE_SIF}" ]] || fail "container missing"

readarray -t identity < <("${HAB_PY}" - "${TRACE_ROOT}/manifest.json" \
  "${SLURM_ARRAY_TASK_ID}" <<'PY'
import json,sys
r=json.load(open(sys.argv[1]))["accepted"][int(sys.argv[2])]
print(r["scene"]); print(r["episode"])
PY
)
[[ "${#identity[@]}" -eq 2 ]] || fail "trace identity failed"
scene=${identity[0]}; episode=${identity[1]}
scene_asset=${ASSET_ROOT}/${scene}/${scene}.glb
source_scene=${GENERATION_ROOT}/${scene}
trace_scene=${TRACE_ROOT}/${scene}
label=$(printf '%03d' "${SLURM_ARRAY_TASK_ID}")_${scene}_${episode}
out=${PARTS_ROOT}/${label}/${scene}
[[ -f "${scene_asset}" && -d "${source_scene}/${episode}" ]] || \
  fail "source episode/asset missing"
[[ ! -e "${PARTS_ROOT:?}/${label}" ]] || fail "part output exists"
mkdir -p "${PARTS_ROOT}/${label}"

export PYTHONPYCACHEPREFIX=${SLURM_TMPDIR:-/tmp}/nnr_build_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}
export PYTHONPATH=${SOURCE_ROOT}:${SOURCE_ROOT}/MemNavData${PYTHONPATH:+:${PYTHONPATH}}
mkdir -p "${PYTHONPYCACHEPREFIX}"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
singularity exec --nv -B /scratch/qw2440 -B /scratch/lg154 \
  -B /scratch/yz11502 "${BASE_SIF}" env \
  PYTHONPYCACHEPREFIX="${PYTHONPYCACHEPREFIX}" PYTHONPATH="${PYTHONPATH}" \
  "${HAB_PY}" -u \
    "${SOURCE_ROOT}/MemNavData/build_shared_online_novel_revisit.py" \
    --episode-root "${source_scene}" --episode-ids "${episode}" \
    --trace-root "${trace_scene}" --scene-asset "${scene_asset}" \
    --out "${out}" --allow-empty

"${HAB_PY}" - "${out}/manifest.json" "${scene}" "${episode}" <<'PY'
import json,sys
m=json.load(open(sys.argv[1]))
assert m["scene"]==sys.argv[2]
rows=list(m["accepted"])+list(m["rejected"])
assert len(rows)==1 and rows[0]["episode"]==sys.argv[3]
assert m["selected_before_c_navigation"] is True
PY
echo "COMPLETE ${label}"
