#!/usr/bin/env bash
# Train-only Cyclic Goal Compass observability gate.  The eight-hour request
# covers pinned Habitat rendering, exact frozen-NavDP feature extraction, and
# 5-fold x 3-seed OOF training; it is not an eight-hour hyperparameter sweep.
#SBATCH --job-name=cgc_oof
#SBATCH --account=torch_pr_769_tandon_advanced
#SBATCH --partition=h100_tandon
#SBATCH --qos=gpu48
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --output=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/%x_%j.out
#SBATCH --error=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/%x_%j.err

set -euo pipefail
umask 0022

: "${CGC_SOURCE_ROOT:?export the immutable staged source root}"
: "${EXPECTED_BUNDLE_MANIFEST_SHA:?export its bundle_manifest.json SHA256}"
: "${EXPECTED_LAUNCHER_SHA:?export the spooled launcher SHA256}"
: "${RUN_TAG:?export a unique alphanumeric run tag}"

[[ "${RUN_TAG}" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]] || {
  echo "ABORT: invalid RUN_TAG=${RUN_TAG}" >&2; exit 2; }
SMOKE="${SMOKE:-0}"
[[ "${SMOKE}" =~ ^[01]$ ]] || {
  echo "ABORT: SMOKE must be 0 or 1" >&2; exit 2; }
CGC_DEVICE="${CGC_DEVICE:-cuda}"
[[ "${CGC_DEVICE}" =~ ^(cpu|cuda)$ ]] || {
  echo "ABORT: CGC_DEVICE must be cpu or cuda" >&2; exit 2; }
[[ "${SMOKE}" == 1 || "${CGC_DEVICE}" == cuda ]] || {
  echo "ABORT: the formal run is frozen to CUDA" >&2; exit 2; }

RESULT_BASE="/scratch/yz11502/Research/Nav-axis-uturn-results/cgc_observability_20260809"
RUN_ROOT="${RESULT_BASE}/${RUN_TAG}_job_${SLURM_JOB_ID:-manual}"
MANIFEST="/scratch/yz11502/Research/Nav-axis-uturn-results/nlsr_gapfill_20260807/nlsr_routes_41c1773_20260807_v1/manifest_multistage/candidate_manifest_multistage.json"
SPLIT_MANIFEST="${CGC_SOURCE_ROOT}/MemNavData/router_multiscene_split_20260805.json"
GEOMETRY_ROOT="/scratch/yz11502/Research/Nav-axis-uturn-results/nlsr_gapfill_20260807/nlsr_routes_41c1773_20260807_v1/frozen_geometry_map_1c9163c8ea09_8a7e71e9514b"
GEOMETRY_MAP="${GEOMETRY_ROOT}/frozen_geometry_map.json"
EPISODE_ROOT="/mp3d_revisit_v0/vln_n1/traj_data/mp3d_3leg"
GAP_SCENE="/scratch/yz11502/Research/Nav-axis-uturn-results/nlsr_gapfill_20260807/nlsr_routes_41c1773_20260807_v1/raw/mp3d_3leg/YmJkqBEsHnH"
GAP_EPISODE_ROOT="/scratch/yz11502/Research/Nav-axis-uturn-results/nlsr_gapfill_20260807/nlsr_routes_41c1773_20260807_v1/raw/mp3d_3leg"
SCENE_ROOT="/scratch/lg154/Research/datasets/mp3d/mp3d"
NAVDP_ROOT="/scratch/lg154/Research/Nav/NavDP/baselines/navdp"
NAVDP_CHECKPOINT="/scratch/yz11502/Research/cgc_assets_20260809/navdp_checkpoint.ckpt"
BASE_SIF="/share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif"
OVERLAY="/scratch/lg154/Research/datasets/_overlays/mp3d_revisit_v0_pt1.sqf"
HABITAT_PY="/scratch/lg154/conda-envs/habitat/bin/python"
NAVDP_PY="/scratch/lg154/conda-envs/navdp/bin/python"

EXPECTED_MANIFEST_SHA="bc6bf58536f6c159d1898ac03abe365eadba65c22cd246e39401821962abb34c"
EXPECTED_SPLIT_SHA="97309c183e25cb3dd65472908748d55a94798a636db6157ab6fe120fca05cf7a"
EXPECTED_GEOMETRY_MAP_SHA="7664ea0d65bdd125c9246d947e2fd470b5d3e90c83e407898da27c4f62e983ec"
EXPECTED_NAVDP_CHECKPOINT_SHA="3bb3ad4ab241e857bb57a4021cc6aab76d5263e81fbf80298d579053ef011947"
EXPECTED_NAVDP_BACKBONE_SHA="2adb42f025eea514887f3724dab3469f1fcb425a2cb4d82616c8af28056e0fcf"
EXPECTED_OVERLAY_BYTES="128854888448"
EXPECTED_OVERLAY_HEAD_SHA="43d803c2b39735e764f0c6df8f7c40a7035a1c23789e22fcc9831d29928c801b"

fail() { echo "ABORT: $*" >&2; exit 2; }
for command_name in singularity sha256sum; do
  command -v "${command_name}" >/dev/null || fail "${command_name} unavailable"
done
if [[ "${CGC_DEVICE}" == cuda ]]; then
  command -v nvidia-smi >/dev/null || fail "nvidia-smi unavailable"
fi
for required_path in "$0" "${CGC_SOURCE_ROOT}/bundle_manifest.json" \
    "${MANIFEST}" "${SPLIT_MANIFEST}" "${GEOMETRY_MAP}" \
    "${NAVDP_CHECKPOINT}" "${NAVDP_ROOT}/policy_backbone.py"; do
  [[ -f "${required_path}" && ! -L "${required_path}" ]] || \
    fail "missing or symbolic-link input: ${required_path}"
done
actual_launcher_sha="$(sha256sum "$0" | awk '{print $1}')"
actual_bundle_sha="$(sha256sum "${CGC_SOURCE_ROOT}/bundle_manifest.json" | awk '{print $1}')"
actual_manifest_sha="$(sha256sum "${MANIFEST}" | awk '{print $1}')"
actual_split_sha="$(sha256sum "${SPLIT_MANIFEST}" | awk '{print $1}')"
actual_geometry_sha="$(sha256sum "${GEOMETRY_MAP}" | awk '{print $1}')"
actual_checkpoint_sha="$(sha256sum "${NAVDP_CHECKPOINT}" | awk '{print $1}')"
actual_backbone_sha="$(sha256sum "${NAVDP_ROOT}/policy_backbone.py" | awk '{print $1}')"
[[ "${actual_launcher_sha}" == "${EXPECTED_LAUNCHER_SHA}" ]] || \
  fail "spooled launcher SHA256 mismatch"
[[ "${actual_bundle_sha}" == "${EXPECTED_BUNDLE_MANIFEST_SHA}" ]] || \
  fail "source bundle manifest SHA256 mismatch"
[[ "${actual_manifest_sha}" == "${EXPECTED_MANIFEST_SHA}" ]] || \
  fail "expert manifest SHA256 mismatch"
[[ "${actual_split_sha}" == "${EXPECTED_SPLIT_SHA}" ]] || \
  fail "scene split SHA256 mismatch"
[[ "${actual_geometry_sha}" == "${EXPECTED_GEOMETRY_MAP_SHA}" ]] || \
  fail "geometry map SHA256 mismatch"
[[ "${actual_checkpoint_sha}" == "${EXPECTED_NAVDP_CHECKPOINT_SHA}" ]] || \
  fail "frozen NavDP checkpoint SHA256 mismatch"
[[ "${actual_backbone_sha}" == "${EXPECTED_NAVDP_BACKBONE_SHA}" ]] || \
  fail "NavDP ImageGoal backbone source SHA256 mismatch"
[[ -d "${GAP_SCENE}" && -d "${SCENE_ROOT}" ]] || \
  fail "dataset roots are incomplete"
[[ -f "${OVERLAY}" && "$(stat -c '%s' "${OVERLAY}")" == \
  "${EXPECTED_OVERLAY_BYTES}" ]] || fail "overlay identity changed"
[[ "$(dd if="${OVERLAY}" bs=1048576 count=1 status=none | \
  sha256sum | awk '{print $1}')" == "${EXPECTED_OVERLAY_HEAD_SHA}" ]] || \
  fail "overlay header SHA changed"

"${NAVDP_PY}" - "${CGC_SOURCE_ROOT}/bundle_manifest.json" \
  "${CGC_SOURCE_ROOT}" <<'PY'
import hashlib, json
from pathlib import Path
import sys
manifest_path, root = map(Path, sys.argv[1:])
manifest = json.loads(manifest_path.read_text())
if manifest.get("schema_version") != "cgc_source_bundle_v1":
    raise SystemExit("bad source-bundle schema")
for relative, expected in manifest["files"].items():
    path = root / relative
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != expected:
        raise SystemExit(f"source hash mismatch: {relative}")
print(json.dumps({"source_bundle_preflight": "passed",
                  "files": len(manifest["files"])}, sort_keys=True))
PY

[[ ! -e "${RUN_ROOT}" ]] || fail "output already exists: ${RUN_ROOT}"
mkdir -p "${RUN_ROOT}"
export PYTHONPATH="${CGC_SOURCE_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export APPTAINERENV_PYTHONPATH="${PYTHONPATH}"
export SINGULARITYENV_PYTHONPATH="${PYTHONPATH}"
export PYTHONHASHSEED=0
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

CONTAINER=(singularity exec)
if [[ "${CGC_DEVICE}" == cuda ]]; then
  CONTAINER+=(--nv)
fi
CONTAINER+=(--overlay "${OVERLAY}:ro"
  -B /scratch/lg154 -B /scratch/yz11502
  "${BASE_SIF}")

echo "host=$(hostname) job=${SLURM_JOB_ID:-manual} smoke=${SMOKE} device=${CGC_DEVICE} run_root=${RUN_ROOT}"
if [[ "${CGC_DEVICE}" == cuda ]]; then
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
fi
"${CONTAINER[@]}" "${HABITAT_PY}" -m py_compile \
  "${CGC_SOURCE_ROOT}/MemNavData/circular_goal_compass.py" \
  "${CGC_SOURCE_ROOT}/MemNavData/build_cgc_multiyaw_dataset.py"
"${CONTAINER[@]}" "${NAVDP_PY}" -m py_compile \
  "${CGC_SOURCE_ROOT}/MemNavData/train_cgc_scene_oof.py"
"${CONTAINER[@]}" "${HABITAT_PY}" -m unittest -v \
  MemNavData.test_circular_goal_compass
"${CONTAINER[@]}" "${NAVDP_PY}" -m unittest -v \
  MemNavData.test_circular_goal_compass

BUILD_ARGS=(
  --manifest "${MANIFEST}"
  --expected-manifest-sha256 "${EXPECTED_MANIFEST_SHA}"
  --split-manifest "${SPLIT_MANIFEST}"
  --expected-split-sha256 "${EXPECTED_SPLIT_SHA}"
  --geometry-map "${GEOMETRY_MAP}"
  --expected-geometry-map-sha256 "${EXPECTED_GEOMETRY_MAP_SHA}"
  --episode-root "${EPISODE_ROOT}"
  --episode-fallback-root "${GAP_EPISODE_ROOT}"
  --scene-root "${SCENE_ROOT}"
  --output "${RUN_ROOT}/teacher"
)
TRAIN_ARGS=(
  --dataset-root "${RUN_ROOT}/teacher"
  --episode-root "${EPISODE_ROOT}"
  --episode-fallback-root "${GAP_EPISODE_ROOT}"
  --navdp-root "${NAVDP_ROOT}"
  --navdp-checkpoint "${NAVDP_CHECKPOINT}"
  --expected-navdp-checkpoint-sha256 "${EXPECTED_NAVDP_CHECKPOINT_SHA}"
  --output "${RUN_ROOT}/oof"
  --device "${CGC_DEVICE}"
)
if [[ "${SMOKE}" == 1 ]]; then
  BUILD_ARGS+=(--max-scenes 2)
  TRAIN_ARGS+=(--smoke --folds 2 --epochs 2 --seeds 11)
fi

"${CONTAINER[@]}" "${HABITAT_PY}" -u \
  "${CGC_SOURCE_ROOT}/MemNavData/build_cgc_multiyaw_dataset.py" \
  "${BUILD_ARGS[@]}" | tee "${RUN_ROOT}/teacher_stdout.jsonl"
TEACHER_REPORT_SHA="$(sha256sum "${RUN_ROOT}/teacher/report.json" | awk '{print $1}')"
TRAIN_ARGS+=(--expected-teacher-report-sha256 "${TEACHER_REPORT_SHA}")
"${CONTAINER[@]}" "${NAVDP_PY}" -u \
  "${CGC_SOURCE_ROOT}/MemNavData/train_cgc_scene_oof.py" \
  "${TRAIN_ARGS[@]}" | tee "${RUN_ROOT}/training_stdout.jsonl"

sha256sum "${RUN_ROOT}/teacher/report.json" \
  "${RUN_ROOT}/oof/report.json" > "${RUN_ROOT}/FINAL_ARTIFACTS.sha256"
echo "CGC observability complete: ${RUN_ROOT}"
