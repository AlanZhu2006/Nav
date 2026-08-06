#!/usr/bin/env bash
#SBATCH --job-name=nlsr_manifest
#SBATCH --account=torch_pr_769_tandon_advanced
#SBATCH --partition=cpu_short
#SBATCH --qos=cpu48
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/%x_%j.out
#SBATCH --error=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/%x_%j.err

set -euo pipefail

: "${REPO_ROOT:?export REPO_ROOT to a clean child worktree}"
: "${EXPECTED_COMMIT:?export EXPECTED_COMMIT to the exact audited commit}"

PYTHON_BIN="${PYTHON_BIN:-/scratch/lg154/conda-envs/memnav/bin/python}"
BASE_SIF="${BASE_SIF:-/share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif}"
OVERLAY="${OVERLAY:-/scratch/lg154/Research/datasets/_overlays/mp3d_revisit_v0_pt1.sqf}"
EPISODE_ROOT="${EPISODE_ROOT:-/mp3d_revisit_v0/vln_n1/traj_data/mp3d_3leg}"
FLOW_CACHE_ROOT="${FLOW_CACHE_ROOT:-/scratch/lg154/Research/datasets/mp3d_revisit_v0_feat_flowgate/vln_n1/traj_data/mp3d_3leg}"
ASSET_ROOT="${ASSET_ROOT:-/scratch/lg154/Research/datasets/mp3d/mp3d}"
RESULT_ROOT="${RESULT_ROOT:-/scratch/yz11502/Research/Nav-axis-uturn-results/nlsr_candidate_pipeline_20260807}"
RUN_NAME="${RUN_NAME:-manifest_full_v1}"
SPLIT_REL="MemNavData/router_multiscene_split_20260805.json"
EXPECTED_SPLIT_SHA="97309c183e25cb3dd65472908748d55a94798a636db6157ab6fe120fca05cf7a"
EXPECTED_OVERLAY_BYTES="128854888448"
EXPECTED_OVERLAY_HEAD_SHA="43d803c2b39735e764f0c6df8f7c40a7035a1c23789e22fcc9831d29928c801b"
EXPECTED_SIF_BYTES="10035580928"
EXPECTED_SIF_HEAD_SHA="990c1377491fbf58c1d7d37bcf8043249826bc7a094eacc4c7a443b9905126c7"

cd "${REPO_ROOT}"

actual_commit="$(git rev-parse HEAD)"
if [[ "${actual_commit}" != "${EXPECTED_COMMIT}" ]]; then
  echo "commit mismatch: ${actual_commit} != ${EXPECTED_COMMIT}" >&2
  exit 2
fi
if [[ -n "$(git status --porcelain)" ]]; then
  echo "worktree is not clean" >&2
  git status --short >&2
  exit 2
fi

for required in \
  "${PYTHON_BIN}" \
  "${BASE_SIF}" \
  "${OVERLAY}" \
  "${FLOW_CACHE_ROOT}" \
  "${ASSET_ROOT}" \
  "${SPLIT_REL}" \
  MemNavData/build_novel_candidate_manifest.py; do
  if [[ ! -e "${required}" ]]; then
    echo "missing dependency: ${required}" >&2
    exit 2
  fi
done
command -v singularity >/dev/null || {
  echo "missing dependency: singularity" >&2
  exit 2
}

actual_split_sha="$(sha256sum "${SPLIT_REL}" | awk '{print $1}')"
if [[ "${actual_split_sha}" != "${EXPECTED_SPLIT_SHA}" ]]; then
  echo "split SHA mismatch: ${actual_split_sha}" >&2
  exit 2
fi

file_signature() {
  local path="$1"
  local expected_bytes="$2"
  local expected_head_sha="$3"
  local actual_bytes actual_head_sha
  actual_bytes="$(stat -c '%s' "${path}")"
  actual_head_sha="$(dd if="${path}" bs=1048576 count=1 status=none | \
    sha256sum | awk '{print $1}')"
  if [[ "${actual_bytes}" != "${expected_bytes}" \
        || "${actual_head_sha}" != "${expected_head_sha}" ]]; then
    echo "dependency signature mismatch: ${path}" >&2
    echo "bytes=${actual_bytes} head_sha=${actual_head_sha}" >&2
    exit 2
  fi
}
file_signature "${OVERLAY}" "${EXPECTED_OVERLAY_BYTES}" \
  "${EXPECTED_OVERLAY_HEAD_SHA}"
file_signature "${BASE_SIF}" "${EXPECTED_SIF_BYTES}" \
  "${EXPECTED_SIF_HEAD_SHA}"

CONTAINER=(singularity exec --overlay "${OVERLAY}:ro"
  -B /scratch/lg154 -B /scratch/yz11502 "${BASE_SIF}")

"${CONTAINER[@]}" "${PYTHON_BIN}" - <<'PY'
import importlib.util
import json
import platform

required = ("numpy", "pyarrow")
missing = [name for name in required if importlib.util.find_spec(name) is None]
if missing:
    raise SystemExit(f"missing Python dependencies: {missing}")
import numpy
import pyarrow
print(json.dumps({
    "python": platform.python_version(),
    "numpy": numpy.__version__,
    "pyarrow": pyarrow.__version__,
}, sort_keys=True))
PY

# Fail before expensive per-file hashing unless the frozen split is visible in
# the exact overlay and every scene exposes at least two three-leg episodes.
"${CONTAINER[@]}" "${PYTHON_BIN}" - \
  "${REPO_ROOT}/${SPLIT_REL}" "${EPISODE_ROOT}" <<'PY'
import json
from pathlib import Path
import sys

split = json.loads(Path(sys.argv[1]).read_text())
root = Path(sys.argv[2])
missing = []
too_small = []
for role in ("train", "development"):
    for scene in split[role]:
        scene_root = root / scene
        if not scene_root.is_dir():
            missing.append(scene)
            continue
        episodes = sorted(
            path.name for path in scene_root.glob("episode_*")
            if (path / "meta/gen_meta.json").is_file())
        if len(episodes) < 2:
            too_small.append({"scene": scene, "episodes": episodes})
if missing or too_small:
    raise SystemExit(
        f"overlay scene preflight failed: missing={missing} too_small={too_small}")
print(json.dumps({
    "overlay_scene_preflight": "passed",
    "scenes": len(split["train"]) + len(split["development"]),
}, sort_keys=True))
PY

job_id="${SLURM_JOB_ID:-manual}"
out_dir="${RESULT_ROOT}/${RUN_NAME}_job_${job_id}"
if [[ -e "${out_dir}" ]]; then
  echo "output directory already exists: ${out_dir}" >&2
  exit 2
fi
mkdir -p "${out_dir}"

manifest="${out_dir}/candidate_manifest.json"
"${CONTAINER[@]}" "${PYTHON_BIN}" \
  "${REPO_ROOT}/MemNavData/build_novel_candidate_manifest.py" \
  --split-manifest "${REPO_ROOT}/${SPLIT_REL}" \
  --episode-root "${EPISODE_ROOT}" \
  --flow-cache-root "${FLOW_CACHE_ROOT}" \
  --environment-root "${ASSET_ROOT}" \
  --navmesh-root "${ASSET_ROOT}" \
  --role train \
  --role development \
  --out "${manifest}" \
  | tee "${out_dir}/builder_stdout.json"

"${CONTAINER[@]}" "${PYTHON_BIN}" - \
  "${manifest}" "${EXPECTED_COMMIT}" <<'PY'
import json
from pathlib import Path
import sys

path = Path(sys.argv[1])
expected_commit = sys.argv[2]
record = json.loads(path.read_text())
summary = record.get("summary", {})
expected = {
    "scene_count": 50,
    "episode_count": 100,
    "sample_count": 400,
}
for key, value in expected.items():
    if summary.get(key) != value:
        raise SystemExit(f"manifest {key} mismatch: {summary.get(key)} != {value}")
roles = {row.get("split_role") for row in record.get("scenes", [])}
if roles != {"train", "development"}:
    raise SystemExit(f"unexpected scene roles: {sorted(roles)}")
if any(row.get("split_role") == "final_reserved"
       for row in record.get("scenes", [])):
    raise SystemExit("final-reserved scene leaked into manifest")
report = {
    "status": "manifest_audited",
    "commit": expected_commit,
    "summary": summary,
    "manifest_sha256": path.with_suffix(path.suffix + ".sha256")
        .read_text().split()[0],
    "missing_flow_caches": record.get("missing_flow_caches", []),
    "downstream_ready": bool(summary.get("all_flow_caches_complete")),
}
(path.parent / "manifest_audit.json").write_text(
    json.dumps(report, indent=2, sort_keys=True) + "\n")
print(json.dumps(report, indent=2, sort_keys=True))
PY

echo "NLSR manifest completed: ${out_dir}"
