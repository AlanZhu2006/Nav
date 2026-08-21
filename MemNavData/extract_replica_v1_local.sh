#!/usr/bin/env bash
# Safely materialize the already-downloaded official Replica-v1 archive.

set -euo pipefail
umask 0022

ARCHIVE_ROOT=${ARCHIVE_ROOT:-/home/asus/Research/Pi3/Replica-Dataset}
OUT_ROOT=${OUT_ROOT:-/home/asus/Research/Pi3/data/replica_v1_full_20260814}
PARTIAL_ROOT=${OUT_ROOT}.partial
PY=${PY:-/home/asus/miniconda3/envs/memnav/bin/python}

fail() { echo "ABORT: $*" >&2; exit 2; }
[[ "${OUT_ROOT}" == /home/asus/Research/Pi3/data/* ]] || \
  fail "output must stay under the explicit Pi3 data root"
[[ ! -e "${OUT_ROOT}" && ! -e "${PARTIAL_ROOT}" ]] || \
  fail "output or partial output already exists"
[[ -r "${ARCHIVE_ROOT}/assets/additional_habitat_configs.zip" ]] || \
  fail "additional Habitat configs are missing"
command -v unpigz >/dev/null || fail "unpigz is unavailable"
command -v unzip >/dev/null || fail "unzip is unavailable"

parts=()
for suffix in {a..q}; do
  part=${ARCHIVE_ROOT}/replica_v1_0.tar.gz.parta${suffix}
  [[ -r "${part}" ]] || fail "missing archive part ${part}"
  parts+=("${part}")
done
[[ "${#parts[@]}" -eq 17 ]] || fail "expected 17 archive parts"
for index in $(seq 0 15); do
  [[ "$(stat -c '%s' "${parts[${index}]}")" -eq 2000000000 ]] || \
    fail "unexpected size for ${parts[${index}]}"
done
[[ "$(stat -c '%s' "${parts[16]}")" -eq 1859047808 ]] || \
  fail "unexpected final archive-part size"

mkdir -p "${PARTIAL_ROOT}"
echo "[replica] extracting 17 verified-size parts into ${PARTIAL_ROOT}"
cat "${parts[@]}" | unpigz -p 8 | tar -x -C "${PARTIAL_ROOT}"
unzip -qn "${ARCHIVE_ROOT}/assets/additional_habitat_configs.zip" \
  -d "${PARTIAL_ROOT}"

"${PY}" - "${ARCHIVE_ROOT}" "${PARTIAL_ROOT}" <<'PY'
import hashlib,json,sys
from pathlib import Path

archive=Path(sys.argv[1]); root=Path(sys.argv[2])
expected={
 "apartment_0","apartment_1","apartment_2","frl_apartment_0",
 "frl_apartment_1","frl_apartment_2","frl_apartment_3","frl_apartment_4",
 "frl_apartment_5","hotel_0","office_0","office_1","office_2","office_3",
 "office_4","room_0","room_1","room_2",
}
required=("mesh.ply","habitat/replica_stage.stage_config.json",
          "habitat/mesh_semantic.ply","habitat/mesh_semantic.navmesh")
found={path.name for path in root.iterdir() if path.is_dir() and path.name in expected}
if found != expected:
    raise SystemExit(f"Replica scene population differs: missing={sorted(expected-found)} extra={sorted(found-expected)}")
rows=[]
for scene in sorted(expected):
    scene_root=root/scene
    missing=[name for name in required if not (scene_root/name).is_file()]
    if missing: raise SystemExit(f"{scene}: missing {missing}")
    rows.append({
      "scene":scene,
      "required_file_sizes":{name:(scene_root/name).stat().st_size for name in required},
    })
parts=[]
for suffix in "abcdefghijklmnopq":
    path=archive/f"replica_v1_0.tar.gz.parta{suffix}"
    parts.append({"name":path.name,"bytes":path.stat().st_size})
config=root/"replica.scene_dataset_config.json"
if not config.is_file(): raise SystemExit("root Replica scene-dataset config is missing")
digest=hashlib.sha256(config.read_bytes()).hexdigest()
receipt={
 "schema_version":"replica_v1_local_extraction_v1_20260814",
 "source":"official facebookresearch/Replica-Dataset v1.0 split archive",
 "archive_parts":parts,"scene_count":len(rows),"scenes":rows,
 "scene_dataset_config_sha256":digest,"gzip_and_tar_integrity_passed":True,
}
(root/"EXTRACTION_RECEIPT.json").write_text(
    json.dumps(receipt,indent=2,sort_keys=True)+"\n")
PY

mv "${PARTIAL_ROOT}" "${OUT_ROOT}"
echo "[replica] complete: ${OUT_ROOT}"
