#!/usr/bin/env bash
# Select, validate, and seal ten outcome-disjoint HM3D v0.2 val assets.
#SBATCH --job-name=hm3dh10_prep
#SBATCH --output=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/%x_%j.out
#SBATCH --error=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/%x_%j.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --partition=cpu_short
#SBATCH --account=torch_pr_769_tandon_advanced

set -euo pipefail
umask 0022

: "${TASK_ROOT:?set immutable task bundle root}"
: "${DATA_ROOT:?set HM3D held-out val10 data root}"
: "${EXPECTED_TASK_RECEIPT_SHA:?set task bundle receipt SHA}"

readonly TASK_RECEIPT=${TASK_ROOT}/SOURCE_BUNDLE.sha256
readonly PROTOCOL=${TASK_ROOT}/MemNavData/hm3d_heldout_val10_revisit_protocol_20260816.json
readonly SELECTION_RECEIPT=${TASK_ROOT}/hm3d_heldout_scene_selection_verification.json
readonly CREDENTIALS=${MATTERPORT_NETRC:-${HOME}/.config/hm3d/netrc}
readonly URL=https://api.matterport.com/resources/habitat/hm3d-val-habitat-v0.2.tar
readonly DOWNLOAD_DIR=${DATA_ROOT}/downloads
readonly DOWNLOAD_ARCHIVE=${DOWNLOAD_DIR}/hm3d-val-habitat-v0.2.tar
readonly REUSE_ARCHIVE=${HM3D_SOURCE_ARCHIVE:-/scratch/yz11502/Research/datasets/goat_bench_20260814/downloads/hm3d-val-habitat-v0.2.tar}
readonly TARGET=${DATA_ROOT}/data/scene_datasets/hm3d/heldout_val10
readonly RECEIPT_DIR=${DATA_ROOT}/receipts
readonly MEMBERS=${RECEIPT_DIR}/hm3d-heldout-val10-selected-members.txt
readonly FILE_HASHES=${RECEIPT_DIR}/hm3d-heldout-val10-files.sha256
readonly RECEIPT=${RECEIPT_DIR}/hm3d-heldout-val10-extraction.json

fail() { echo "ABORT: $*" >&2; exit 2; }
[[ "$(sha256sum "${TASK_RECEIPT}" | awk '{print $1}')" == \
    "${EXPECTED_TASK_RECEIPT_SHA}" ]] || fail "task receipt changed"
(cd "${TASK_ROOT}" && sha256sum -c SOURCE_BUNDLE.sha256 >/dev/null) || \
  fail "task bundle changed"
for path in "${PROTOCOL}" "${SELECTION_RECEIPT}"; do
  test -r "${path}" || fail "missing input ${path}"
done
python3 - "${SELECTION_RECEIPT}" <<'PY'
import json,sys
p=json.load(open(sys.argv[1]))
assert p["verified"] is True
assert p["archive_scene_count"] == 100
assert p["consumed_scene_count"] == 36
assert p["unconsumed_scene_count"] == 64
assert p["selected_overlap_with_consumed"] == []
assert p["outcome_fields_read_for_selection"] is False
assert len(p["selected_scenes"]) == 10
PY
mkdir -p "${DOWNLOAD_DIR}" "${TARGET}" "${RECEIPT_DIR}" \
  /scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs

exec 9>"${DATA_ROOT}/hm3d-heldout-val10.prepare.lock"
flock -n 9 || fail "another held-out-val10 preparation owns the lock"

if [[ -e "${TARGET}/SEALED" ]]; then
  test -s "${RECEIPT}" && test -s "${FILE_HASHES}" || \
    fail "sealed target lacks receipts"
  python3 - "${RECEIPT}" "${PROTOCOL}" "${SELECTION_RECEIPT}" <<'PY'
import hashlib,json,sys
def sha(path):
    return hashlib.sha256(open(path,"rb").read()).hexdigest()
p=json.load(open(sys.argv[1]))
assert p["schema_version"] == "hm3d_heldout_val10_asset_receipt_v1_20260816"
assert p["complete"] is True and p["scene_count"] == 10
assert p["protocol_sha256"] == sha(sys.argv[2])
assert p["selection_receipt_sha256"] == sha(sys.argv[3])
PY
  echo "already_sealed=${TARGET}"
  exit 0
fi

if [[ -r "${REUSE_ARCHIVE}" ]]; then
  readonly ARCHIVE=${REUSE_ARCHIVE}
  readonly ARCHIVE_SOURCE=validated_hm3d_val_archive
  echo "archive_source=reused_validated_val_archive"
else
  test -r "${CREDENTIALS}" || fail "missing HM3D credentials and reusable archive"
  readonly ARCHIVE=${DOWNLOAD_ARCHIVE}
  readonly ARCHIVE_SOURCE=downloaded_hm3d_val_archive
  curl --netrc-file "${CREDENTIALS}" --location --fail --show-error \
    --retry 8 --retry-all-errors --continue-at - --output "${ARCHIVE}" "${URL}"
  echo "archive_source=downloaded_val_archive"
fi
test -s "${ARCHIVE}" || fail "downloaded archive is empty"
tar -tf "${ARCHIVE}" >/dev/null || fail "archive integrity check failed"

python3 - "${ARCHIVE}" "${PROTOCOL}" "${MEMBERS}" <<'PY'
import json,pathlib,re,sys,tarfile
archive=pathlib.Path(sys.argv[1]); protocol=json.load(open(sys.argv[2]))
out=pathlib.Path(sys.argv[3]); selected=[]
with tarfile.open(archive,"r:") as handle:
    members={member.name:member for member in handle.getmembers()}
names=list(members)
for name in names:
    path=pathlib.PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts:
        raise SystemExit(f"unsafe tar member: {name}")
for row in protocol["scenes"]:
    directory=row["directory"]; scene=row["scene_id"]
    roots=sorted({name.split("/",1)[0] for name in names
                  if re.fullmatch(re.escape(directory)+r"/.*",name)})
    if roots != [directory]:
        raise SystemExit(f"archive root mismatch for {scene}: {roots}")
    expected={f"{directory}/{scene}.basis.glb",
              f"{directory}/{scene}.basis.navmesh"}
    present={name for name in names
             if name.startswith(directory+"/") and not name.endswith("/")}
    if present != expected:
        raise SystemExit(
            f"archive members mismatch for {scene}: {sorted(present)}")
    if not all(members[name].isfile() for name in expected):
        raise SystemExit(f"selected asset is not a regular file: {scene}")
    selected.extend(sorted(expected))
if len(selected) != 20:
    raise SystemExit("expected exactly twenty selected asset files")
out.write_text("\n".join(selected)+"\n")
PY

if find "${TARGET}" -mindepth 1 -print -quit | grep -q .; then
  fail "unsealed extraction target is not empty"
fi
tar -xf "${ARCHIVE}" -C "${TARGET}" \
  --verbatim-files-from --files-from "${MEMBERS}"
find "${TARGET}" -type f \( -name '*.basis.glb' -o -name '*.basis.navmesh' \) \
  -print0 | sort -z | xargs -0 sha256sum > "${FILE_HASHES}"

python3 - "${PROTOCOL}" "${TARGET}" "${ARCHIVE}" \
  "${FILE_HASHES}" "${RECEIPT}" "${ARCHIVE_SOURCE}" \
  "${SELECTION_RECEIPT}" <<'PY'
import datetime,hashlib,json,pathlib,sys
protocol=json.load(open(sys.argv[1])); target=pathlib.Path(sys.argv[2])
archive=pathlib.Path(sys.argv[3]); hashes=pathlib.Path(sys.argv[4])
receipt=pathlib.Path(sys.argv[5]); source=sys.argv[6]
selection_receipt=pathlib.Path(sys.argv[7]); assets={}; total=0
for row in protocol["scenes"]:
    directory=row["directory"]; scene=row["scene_id"]
    glb=target/directory/f"{scene}.basis.glb"
    nav=target/directory/f"{scene}.basis.navmesh"
    if not glb.is_file() or not nav.is_file():
        raise SystemExit(f"missing extracted assets for {scene}")
    assets[scene]={"directory":directory,"glb_bytes":glb.stat().st_size,
                   "navmesh_bytes":nav.stat().st_size}
    total += glb.stat().st_size + nav.stat().st_size
lines=hashes.read_text().splitlines()
if len(lines) != 20:
    raise SystemExit("asset hash receipt must contain twenty files")
archive_digest=hashlib.sha256()
with archive.open("rb") as handle:
    for block in iter(lambda: handle.read(8 << 20), b""):
        archive_digest.update(block)
payload={
 "schema_version":"hm3d_heldout_val10_asset_receipt_v1_20260816",
 "created_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),
 "archive_source":source,"archive_path":str(archive.resolve()),
 "archive_bytes":archive.stat().st_size,
 "archive_sha256":archive_digest.hexdigest(),
 "protocol_sha256":hashlib.sha256(pathlib.Path(sys.argv[1]).read_bytes()).hexdigest(),
 "selection_receipt_sha256":hashlib.sha256(selection_receipt.read_bytes()).hexdigest(),
 "asset_bytes":total,"scene_count":len(assets),"assets":assets,
 "file_hash_receipt_sha256":hashlib.sha256(hashes.read_bytes()).hexdigest(),
 "complete":True,
}
receipt.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n")
print(json.dumps(payload,sort_keys=True))
PY

: > "${TARGET}/SEALED"
chmod a-w "${TARGET}/SEALED" "${RECEIPT}" "${FILE_HASHES}" "${MEMBERS}"
