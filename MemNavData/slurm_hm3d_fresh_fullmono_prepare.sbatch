#!/usr/bin/env bash
# Extract and seal all 54 prospectively frozen HM3D scene assets.
#SBATCH --job-name=h3freshPrep
#SBATCH --output=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/%x_%j.out
#SBATCH --error=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/%x_%j.err
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=cs
#SBATCH --qos=cpu48
#SBATCH --account=torch_pr_769_tandon_advanced

set -euo pipefail
umask 0022
: "${TASK_ROOT:?}" "${DATA_ROOT:?}" "${EXPECTED_TASK_RECEIPT_SHA:?}"

TASK_RECEIPT=${TASK_ROOT}/SOURCE_BUNDLE.sha256
PROTOCOL=${TASK_ROOT}/MemNavData/hm3d_fresh_fullmono_mixed_role_protocol_20260820.json
MEMBER_LIST=${TASK_ROOT}/.diagnostics/datasets/goat-bench/hm3d_val_receipts/hm3d-val-habitat-v0.2.members.txt
PRIOR_AUDIT=${TASK_ROOT}/MemNavData/hm3d_consumed_scene_audit_20260816.json
PRIOR_PROTOCOL=${TASK_ROOT}/MemNavData/hm3d_heldout_val10_revisit_protocol_20260816.json
AUDITOR=${TASK_ROOT}/MemNavData/audit_hm3d_fresh_fullmono_selection.py
ARCHIVE=/scratch/yz11502/Research/datasets/goat_bench_20260814/downloads/hm3d-val-habitat-v0.2.tar
TARGET=${DATA_ROOT}/data/scene_datasets/hm3d/fresh_fullmono54
RECEIPTS=${DATA_ROOT}/receipts
SELECTION_RECEIPT=${RECEIPTS}/fresh_fullmono54_selection.json
ASSET_RECEIPT=${RECEIPTS}/fresh_fullmono54_assets.json
FILE_HASHES=${RECEIPTS}/fresh_fullmono54_assets.sha256
MEMBERS=${RECEIPTS}/fresh_fullmono54_members.txt

fail() { echo "ABORT: $*" >&2; exit 2; }
[[ "$(sha256sum "${TASK_RECEIPT}" | awk '{print $1}')" == \
   "${EXPECTED_TASK_RECEIPT_SHA}" ]] || fail "task receipt changed"
(cd "${TASK_ROOT}" && sha256sum -c --quiet SOURCE_BUNDLE.sha256) || \
  fail "task bundle changed"
for path in "${PROTOCOL}" "${MEMBER_LIST}" "${PRIOR_AUDIT}" \
            "${PRIOR_PROTOCOL}" "${AUDITOR}" "${ARCHIVE}"; do
  [[ -r "${path}" ]] || fail "missing ${path}"
done
mkdir -p "${TARGET}" "${RECEIPTS}" \
  /scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs
exec 9>"${DATA_ROOT}/fresh_fullmono54.prepare.lock"
flock -n 9 || fail "another preparation owns the lock"

if [[ -e "${TARGET}/SEALED" ]]; then
  [[ -s "${SELECTION_RECEIPT}" && -s "${ASSET_RECEIPT}" && \
     -s "${FILE_HASHES}" ]] || fail "sealed asset root lacks receipts"
  python3 - "${ASSET_RECEIPT}" "${PROTOCOL}" <<'PY'
import hashlib,json,sys
p=json.load(open(sys.argv[1])); protocol=open(sys.argv[2],"rb").read()
assert p["complete"] is True and p["scene_count"] == 54
assert p["protocol_sha256"] == hashlib.sha256(protocol).hexdigest()
PY
  echo "already_sealed=${TARGET}"
  exit 0
fi
[[ -z "$(find "${TARGET}" -mindepth 1 -print -quit)" ]] || \
  fail "unsealed extraction target is not empty"
tar -tf "${ARCHIVE}" >/dev/null || fail "archive integrity check failed"

/scratch/lg154/conda-envs/memnav/bin/python "${AUDITOR}" \
  --protocol "${PROTOCOL}" --member-list "${MEMBER_LIST}" \
  --prior-audit "${PRIOR_AUDIT}" \
  --heldout10-protocol "${PRIOR_PROTOCOL}" --out "${SELECTION_RECEIPT}" \
  >/dev/null

python3 - "${ARCHIVE}" "${PROTOCOL}" "${MEMBERS}" <<'PY'
import json,pathlib,sys,tarfile
archive=pathlib.Path(sys.argv[1]); protocol=json.load(open(sys.argv[2]))
out=pathlib.Path(sys.argv[3]); selected=[]
with tarfile.open(archive,"r:") as handle:
    members={m.name:m for m in handle.getmembers()}
for name in members:
    path=pathlib.PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts:
        raise SystemExit(f"unsafe tar member: {name}")
for row in protocol["dataset"]["scenes"]:
    directory=row["directory"]; scene=row["scene_id"]
    expected={f"{directory}/{scene}.basis.glb",
              f"{directory}/{scene}.basis.navmesh"}
    present={name for name in members
             if name.startswith(directory+"/") and not name.endswith("/")}
    if present != expected or not all(members[name].isfile() for name in expected):
        raise SystemExit(f"archive members changed for {scene}: {sorted(present)}")
    selected.extend(sorted(expected))
if len(selected) != 108:
    raise SystemExit("expected 108 selected asset files")
out.write_text("\n".join(selected)+"\n")
PY

tar -xf "${ARCHIVE}" -C "${TARGET}" \
  --verbatim-files-from --files-from "${MEMBERS}"
find "${TARGET}" -type f \( -name '*.basis.glb' -o -name '*.basis.navmesh' \) \
  -print0 | sort -z | xargs -0 sha256sum >"${FILE_HASHES}"

python3 - "${PROTOCOL}" "${TARGET}" "${ARCHIVE}" "${FILE_HASHES}" \
  "${ASSET_RECEIPT}" "${SELECTION_RECEIPT}" <<'PY'
import datetime,hashlib,json,pathlib,sys
protocol_path=pathlib.Path(sys.argv[1]); protocol=json.load(open(protocol_path))
target=pathlib.Path(sys.argv[2]); archive=pathlib.Path(sys.argv[3])
hashes=pathlib.Path(sys.argv[4]); receipt=pathlib.Path(sys.argv[5])
selection=pathlib.Path(sys.argv[6]); assets={}; total=0
for row in protocol["dataset"]["scenes"]:
    directory=row["directory"]; scene=row["scene_id"]
    glb=target/directory/f"{scene}.basis.glb"
    nav=target/directory/f"{scene}.basis.navmesh"
    if not glb.is_file() or not nav.is_file():
        raise SystemExit(f"missing extracted asset {scene}")
    assets[scene]={"directory":directory,"glb_bytes":glb.stat().st_size,
                   "navmesh_bytes":nav.stat().st_size}
    total += glb.stat().st_size + nav.stat().st_size
if len(hashes.read_text().splitlines()) != 108:
    raise SystemExit("asset hash receipt must contain 108 rows")
digest=hashlib.sha256()
with archive.open("rb") as handle:
    for block in iter(lambda:handle.read(8<<20),b""): digest.update(block)
payload={
 "schema_version":"hm3d_fresh_fullmono54_asset_receipt_v1_20260820",
 "created_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),
 "complete":True,"scene_count":len(assets),"asset_bytes":total,
 "archive_path":str(archive.resolve()),"archive_bytes":archive.stat().st_size,
 "archive_sha256":digest.hexdigest(),"assets":assets,
 "protocol_sha256":hashlib.sha256(protocol_path.read_bytes()).hexdigest(),
 "selection_receipt_sha256":hashlib.sha256(selection.read_bytes()).hexdigest(),
 "file_hash_receipt_sha256":hashlib.sha256(hashes.read_bytes()).hexdigest(),
}
receipt.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n")
PY

: >"${TARGET}/SEALED"
chmod -R a-w "${TARGET}"
chmod a-w "${SELECTION_RECEIPT}" "${ASSET_RECEIPT}" "${FILE_HASHES}" \
  "${MEMBERS}"
