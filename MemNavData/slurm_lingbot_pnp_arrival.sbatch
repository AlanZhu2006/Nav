#!/usr/bin/env bash
# Exact-state train-only LingBot/PnP arrival-certificate collection.
#SBATCH --job-name=pnp_arrival
#SBATCH --account=torch_pr_769_tandon_advanced
#SBATCH --partition=a100_tandon,h100_tandon
#SBATCH --qos=gpu48
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1
#SBATCH --requeue
#SBATCH --output=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/%x_%A_%a.out
#SBATCH --error=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/%x_%A_%a.err

set -euo pipefail
umask 0022

: "${SOURCE_BUNDLE:?SOURCE_BUNDLE is required}"
: "${EXPECTED_BUNDLE_MANIFEST_SHA:?bundle SHA is required}"
: "${RUN_ROOT:?RUN_ROOT is required}"
RUN_MODE="${RUN_MODE:-formal}"
SHARD_COUNT="${SHARD_COUNT:-8}"

fail() { echo "ABORT: $*" >&2; exit 2; }
[[ "${RUN_MODE}" == "smoke" || "${RUN_MODE}" == "formal" ]] || \
  fail "RUN_MODE must be smoke or formal"

SOURCE_BUNDLE="$(realpath "${SOURCE_BUNDLE}")"
BUNDLE_MANIFEST="${SOURCE_BUNDLE}/source_bundle_manifest.json"
BASE_REPO_ROOT="/scratch/yz11502/Research/Nav-graph-blind"
BASE_SIF="/share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif"
OVERLAY="/scratch/lg154/Research/datasets/_overlays/mp3d_revisit_v0_pt1.sqf"
MEMNAV_PY="/scratch/lg154/conda-envs/memnav/bin/python"
LINGBOT_REPO="/scratch/lg154/Research/Nav/NavDP/baselines/memnav/lingbot-map"
WEIGHTS="${LINGBOT_REPO}/weights/lingbot-map-long.pt"
RAW_GAP_SCENE="/scratch/yz11502/Research/Nav-axis-uturn-results/nlsr_gapfill_20260807/nlsr_routes_41c1773_20260807_v1/raw/mp3d_3leg/YmJkqBEsHnH"
EPISODE_ROOT="/mp3d_revisit_v0/vln_n1/traj_data/mp3d_3leg"
LIGHTGLUE_REPO="${SOURCE_BUNDLE}/third_party/LightGlue"
DEPENDENCY_ROOT="${SOURCE_BUNDLE}/third_party/python"

ENTRYPOINT="${SOURCE_BUNDLE}/MemNavData/audit_lingbot_pnp_arrival.py"
STATES="${SOURCE_BUNDLE}/inputs/states.csv"
SAMPLES="${SOURCE_BUNDLE}/inputs/samples.csv"
INVENTORY="${SOURCE_BUNDLE}/inputs/inventory.json"
FLOW_ROUTES="${SOURCE_BUNDLE}/inputs/FLOW_ROUTE_PROVENANCE.json"
CAUSAL_SCALE="${SOURCE_BUNDLE}/inputs/causal_ground_scale.json"

[[ "$(sha256sum "${BUNDLE_MANIFEST}" | awk '{print $1}')" == \
   "${EXPECTED_BUNDLE_MANIFEST_SHA}" ]] || fail "bundle manifest SHA changed"
"${MEMNAV_PY}" - "${SOURCE_BUNDLE}" "${BUNDLE_MANIFEST}" <<'PY'
import hashlib, json, sys
from pathlib import Path, PurePosixPath
root=Path(sys.argv[1]).resolve(); manifest=json.load(open(sys.argv[2]))
if manifest.get("schema_version") != "lingbot_pnp_arrival_bundle_v1":
    raise SystemExit("bundle schema changed")
for relative, expected in manifest.get("files", {}).items():
    canonical=PurePosixPath(relative)
    if canonical.is_absolute() or ".." in canonical.parts:
        raise SystemExit(f"non-canonical bundle path: {relative}")
    path=root/relative
    if not path.is_file() or path.is_symlink():
        raise SystemExit(f"non-physical bundle member: {relative}")
    actual=hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != expected:
        raise SystemExit(f"bundle member changed: {relative}")
print(json.dumps({"bundle":"verified","files":len(manifest["files"])}))
PY

for path in "${BASE_SIF}" "${OVERLAY}" "${WEIGHTS}" "${ENTRYPOINT}" \
            "${STATES}" "${SAMPLES}" "${INVENTORY}" "${FLOW_ROUTES}" \
            "${CAUSAL_SCALE}"; do
  [[ -e "${path}" ]] || fail "missing required input ${path}"
done
[[ -d "${RAW_GAP_SCENE}" ]] || fail "audited gap scene is absent"
gpu_name="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
[[ "${gpu_name}" == NVIDIA\ A100* || "${gpu_name}" == NVIDIA\ H100* ]] || \
  fail "formal GPU must be A100/H100, got ${gpu_name}"

CONTAINER=(singularity exec --nv --overlay "${OVERLAY}:ro"
  -B /scratch/lg154 -B /scratch/yz11502
  -B "${RAW_GAP_SCENE}:${EPISODE_ROOT}/YmJkqBEsHnH:ro" "${BASE_SIF}")
export TORCH_HOME="${SOURCE_BUNDLE}/torch_home"
export PYTHONPATH="${SOURCE_BUNDLE}:${BASE_REPO_ROOT}:${DEPENDENCY_ROOT}:${LIGHTGLUE_REPO}${PYTHONPATH:+:${PYTHONPATH}}"
export APPTAINERENV_PYTHONPATH="${PYTHONPATH}"
export SINGULARITYENV_PYTHONPATH="${PYTHONPATH}"
export APPTAINERENV_TORCH_HOME="${TORCH_HOME}"
export SINGULARITYENV_TORCH_HOME="${TORCH_HOME}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

if [[ "${RUN_MODE}" == "smoke" ]]; then
  shard_index=0
  shard_count=1
  max_episodes=1
  out_dir="${RUN_ROOT}/smoke"
  "${CONTAINER[@]}" "${MEMNAV_PY}" -m py_compile \
    "${ENTRYPOINT}" \
    "${SOURCE_BUNDLE}/MemNavData/summarize_lingbot_pnp_arrival.py"
  "${CONTAINER[@]}" "${MEMNAV_PY}" -m unittest \
    MemNavData.test_lingbot_pnp_arrival -v
else
  shard_index="${SLURM_ARRAY_TASK_ID:?formal mode requires an array task}"
  shard_count="${SHARD_COUNT}"
  max_episodes=0
  printf -v shard_name 'shard_%02d' "${shard_index}"
  out_dir="${RUN_ROOT}/shards/${shard_name}"
fi
[[ ! -e "${out_dir}" ]] || fail "immutable output already exists: ${out_dir}"
mkdir -p "$(dirname "${out_dir}")"

echo "host=$(hostname) gpu=${gpu_name} mode=${RUN_MODE} shard=${shard_index}/${shard_count}"
"${CONTAINER[@]}" "${MEMNAV_PY}" -u "${ENTRYPOINT}" \
  --states-csv "${STATES}" --samples-csv "${SAMPLES}" \
  --inventory-json "${INVENTORY}" \
  --flow-route-provenance "${FLOW_ROUTES}" \
  --causal-scale-artifact "${CAUSAL_SCALE}" \
  --internnav-root "${BASE_REPO_ROOT}/InternNav" \
  --lingbot-repo "${LINGBOT_REPO}" --weights "${WEIGHTS}" \
  --lightglue-repo "${LIGHTGLUE_REPO}" \
  --dependency-root "${DEPENDENCY_ROOT}" \
  --out-dir "${out_dir}" --device cuda --num-scale 8 --window 32 \
  --max-frame-num 4096 --camera-num-iterations 4 --max-keypoints 2048 \
  --chunk-size 16 --shard-index "${shard_index}" \
  --shard-count "${shard_count}" --max-episodes "${max_episodes}"
echo "PNP_ARRIVAL_OUTPUT=${out_dir}"
