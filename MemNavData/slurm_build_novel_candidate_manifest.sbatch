#!/usr/bin/env bash
#SBATCH --job-name=nlsr_manifest
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
EPISODE_ROOT="${EPISODE_ROOT:-/scratch/lg154/Research/datasets/mp3d_revisit_v0/vln_n1/traj_data/mp3d_3leg}"
FLOW_CACHE_ROOT="${FLOW_CACHE_ROOT:-/scratch/lg154/Research/datasets/mp3d_revisit_v0_feat_flowgate/vln_n1/traj_data/mp3d_3leg}"
ASSET_ROOT="${ASSET_ROOT:-/scratch/lg154/Research/datasets/mp3d/mp3d}"
RESULT_ROOT="${RESULT_ROOT:-/scratch/yz11502/Research/Nav-axis-uturn-results/nlsr_candidate_pipeline_20260807}"
RUN_NAME="${RUN_NAME:-manifest_full_v1}"
SPLIT_REL="MemNavData/router_multiscene_split_20260805.json"
EXPECTED_SPLIT_SHA="97309c183e25cb3dd65472908748d55a94798a636db6157ab6fe120fca05cf7a"

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
  "${EPISODE_ROOT}" \
  "${FLOW_CACHE_ROOT}" \
  "${ASSET_ROOT}" \
  "${SPLIT_REL}" \
  MemNavData/build_novel_candidate_manifest.py; do
  if [[ ! -e "${required}" ]]; then
    echo "missing dependency: ${required}" >&2
    exit 2
  fi
done

actual_split_sha="$(sha256sum "${SPLIT_REL}" | awk '{print $1}')"
if [[ "${actual_split_sha}" != "${EXPECTED_SPLIT_SHA}" ]]; then
  echo "split SHA mismatch: ${actual_split_sha}" >&2
  exit 2
fi

"${PYTHON_BIN}" - <<'PY'
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

job_id="${SLURM_JOB_ID:-manual}"
out_dir="${RESULT_ROOT}/${RUN_NAME}_job_${job_id}"
if [[ -e "${out_dir}" ]]; then
  echo "output directory already exists: ${out_dir}" >&2
  exit 2
fi
mkdir -p "${out_dir}"

manifest="${out_dir}/candidate_manifest.json"
"${PYTHON_BIN}" MemNavData/build_novel_candidate_manifest.py \
  --split-manifest "${SPLIT_REL}" \
  --episode-root "${EPISODE_ROOT}" \
  --flow-cache-root "${FLOW_CACHE_ROOT}" \
  --environment-root "${ASSET_ROOT}" \
  --navmesh-root "${ASSET_ROOT}" \
  --role train \
  --role development \
  --out "${manifest}" \
  | tee "${out_dir}/builder_stdout.json"

"${PYTHON_BIN}" - "${manifest}" "${EXPECTED_COMMIT}" <<'PY'
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
