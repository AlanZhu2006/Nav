#!/usr/bin/env bash
# Selectively extract exactly the 36 HM3D validation scenes referenced by the
# official GOAT-Bench val_unseen split.
#SBATCH --job-name=goat_hm3d_extract
#SBATCH --output=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/%x_%j.out
#SBATCH --error=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/%x_%j.err
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --partition=cpu_short
#SBATCH --account=torch_pr_769_tandon_advanced

set -euo pipefail
umask 0022

readonly ROOT="${GOAT_HM3D_ROOT:-/scratch/yz11502/Research/datasets/goat_bench_20260814}"
readonly ARCHIVE="${ROOT}/downloads/hm3d-val-habitat-v0.2.tar"
readonly CONTENT="${ROOT}/data/datasets/goat_bench/hm3d/v1/val_unseen/content"
readonly TARGET="${ROOT}/data/scene_datasets/hm3d/val"
readonly RECEIPT_DIR="${ROOT}/receipts"
readonly MEMBERS="${RECEIPT_DIR}/goat-val-unseen-selected-members.txt"
readonly PREFLIGHT="${RECEIPT_DIR}/goat-val-unseen-scene-preflight.json"
readonly FILE_HASHES="${RECEIPT_DIR}/goat-val-unseen-scene-files.sha256"
readonly RECEIPT="${RECEIPT_DIR}/goat-val-unseen-extraction.json"

test -f "${ARCHIVE}"
test -d "${CONTENT}"
mkdir -p "${TARGET}" "${RECEIPT_DIR}"

exec 9>"${ROOT}/hm3d-val-goat36.extract.lock"
flock -n 9 || {
  echo "another GOAT HM3D extraction owns the lock" >&2
  exit 3
}

if [[ -e "${TARGET}/SEALED" ]]; then
  echo "target_already_sealed=yes"
  test -f "${RECEIPT}"
  exit 0
fi

if find "${TARGET}" -mindepth 1 -print -quit | grep -q .; then
  echo "unsealed extraction target is not empty" >&2
  exit 4
fi

python3 - "${ARCHIVE}" "${CONTENT}" "${MEMBERS}" "${PREFLIGHT}" <<'PY'
import json
import pathlib
import re
import sys
import tarfile

archive = pathlib.Path(sys.argv[1])
content = pathlib.Path(sys.argv[2])
members_out = pathlib.Path(sys.argv[3])
preflight_out = pathlib.Path(sys.argv[4])

required = sorted(path.name[:-8] for path in content.glob("*.json.gz"))
if len(required) != 36 or len(set(required)) != 36:
    raise SystemExit(f"expected 36 unique GOAT scene IDs, got {len(required)}")

with tarfile.open(archive, "r:") as handle:
    names = handle.getnames()

for name in names:
    path = pathlib.PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts:
        raise SystemExit(f"unsafe tar member: {name}")

selected = []
scene_roots = {}
for scene_id in required:
    roots = sorted(
        {
            name.split("/", 1)[0]
            for name in names
            if re.fullmatch(rf"\d{{5}}-{re.escape(scene_id)}/.*", name)
        }
    )
    if len(roots) != 1:
        raise SystemExit(f"scene {scene_id} has {len(roots)} archive roots")
    root = roots[0]
    expected = {
        f"{root}/{scene_id}.basis.glb",
        f"{root}/{scene_id}.basis.navmesh",
    }
    present = {name for name in names if name.startswith(root + "/") and not name.endswith("/")}
    if present != expected:
        raise SystemExit(
            f"scene {scene_id} members mismatch: expected={sorted(expected)} present={sorted(present)}"
        )
    scene_roots[scene_id] = root
    # Select only the two exact file members.  Python tarfile normalizes
    # directory entries by dropping their trailing slash, whereas GNU tar's
    # textual listing preserves it.  File-only selection avoids coupling the
    # extraction contract to that representational difference; GNU tar creates
    # the parent directory when extracting each file.
    selected.extend(sorted(expected))

members_out.write_text("\n".join(selected) + "\n")
payload = {
    "schema_version": "goat_val_unseen_scene_preflight_v1_20260814",
    "required_scene_count": len(required),
    "matched_scene_count": len(scene_roots),
    "selected_tar_member_count": len(selected),
    "scene_roots": scene_roots,
    "complete": True,
}
preflight_out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
print(json.dumps(payload, sort_keys=True))
PY

tar -xf "${ARCHIVE}" -C "${TARGET}" --verbatim-files-from --files-from "${MEMBERS}"

hash_tmp="${FILE_HASHES}.tmp.${SLURM_JOB_ID:-local}"
find "${TARGET}" -type f \( -name '*.basis.glb' -o -name '*.basis.navmesh' \) \
  -print0 | sort -z | xargs -0 sha256sum > "${hash_tmp}"
mv "${hash_tmp}" "${FILE_HASHES}"

python3 - "${CONTENT}" "${TARGET}" "${FILE_HASHES}" "${RECEIPT}" <<'PY'
import datetime
import hashlib
import json
import pathlib
import sys

content = pathlib.Path(sys.argv[1])
target = pathlib.Path(sys.argv[2])
hashes = pathlib.Path(sys.argv[3])
receipt = pathlib.Path(sys.argv[4])

required = sorted(path.name[:-8] for path in content.glob("*.json.gz"))
directories = sorted(path for path in target.iterdir() if path.is_dir())
found = {}
total_bytes = 0
for directory in directories:
    if "-" not in directory.name:
        raise SystemExit(f"unexpected scene directory {directory.name}")
    scene_id = directory.name.split("-", 1)[1]
    glb = directory / f"{scene_id}.basis.glb"
    navmesh = directory / f"{scene_id}.basis.navmesh"
    if not glb.is_file() or not navmesh.is_file():
        raise SystemExit(f"incomplete extracted scene {scene_id}")
    found[scene_id] = directory.name
    total_bytes += glb.stat().st_size + navmesh.stat().st_size

if sorted(found) != required:
    raise SystemExit("extracted scene IDs differ from GOAT val_unseen")
hash_lines = hashes.read_text().splitlines()
if len(hash_lines) != 2 * len(required):
    raise SystemExit("scene hash receipt does not contain exactly two assets per scene")

payload = {
    "schema_version": "goat_val_unseen_scene_extraction_v1_20260814",
    "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "scene_count": len(found),
    "glb_count": len(required),
    "navmesh_count": len(required),
    "extracted_bytes": total_bytes,
    "file_hash_receipt_sha256": hashlib.sha256(hashes.read_bytes()).hexdigest(),
    "scene_directories": found,
    "complete": True,
}
receipt.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
print(json.dumps(payload, sort_keys=True))
PY

: > "${TARGET}/SEALED"
