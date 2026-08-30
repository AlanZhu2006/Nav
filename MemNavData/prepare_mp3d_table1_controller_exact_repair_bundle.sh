#!/usr/bin/env bash
# Build a minimal immutable wrapper bundle for the frozen MP3D Table-1 repair.
set -euo pipefail
umask 0022

ROOT=${ROOT:-$(git rev-parse --show-toplevel)}
LOCAL_MEMNAV_PY=${LOCAL_MEMNAV_PY:-/home/asus/miniconda3/envs/memnav/bin/python}
LOCAL_BUNDLE_PARENT=${LOCAL_BUNDLE_PARENT:-${ROOT}/.diagnostics/source_bundles}
fail() { echo "ABORT: $*" >&2; exit 2; }

required=(
  MemNavData/MP3D_TABLE1_CONTROLLER_EXACT_REPAIR_PROTOCOL_20260829.md
  MemNavData/mp3d_table1_controller_exact_repair_protocol_20260829.json
  MemNavData/MP3D_TABLE1_NAVDP_AUTHORITY_CACHE_COMPOSITION_REPAIR_20260829.md
  MemNavData/mp3d_table1_navdp_authority_cache_composition_repair_20260829.json
  MemNavData/audit_mp3d_table1_controller_exact_repair_bundle.py
  MemNavData/test_audit_mp3d_table1_controller_exact_repair_bundle.py
  MemNavData/slurm_hm3d_table1_navdp_pair.sbatch
  MemNavData/slurm_hm3d_table1_vint_pair.sbatch
  MemNavData/slurm_port_pair.sh
  MemNavData/test_slurm_port_pair.sh
  MemNavData/slurm_safe_submit.sh
  MemNavData/bundle_selftest.sh
  MemNavData/submit_mp3d_table1_controller_exact_repair_remote.sh
  MemNavData/submit_mp3d_table1_navdp_authority_cache_repair_remote.sh
)
for path in "${required[@]}"; do
  [[ -f "${ROOT}/${path}" && ! -L "${ROOT}/${path}" ]] || \
    fail "missing physical ${path}"
done
[[ -x "${LOCAL_MEMNAV_PY}" ]] || fail "MemNav interpreter missing"

cd "${ROOT}"
"${LOCAL_MEMNAV_PY}" -m json.tool \
  MemNavData/mp3d_table1_controller_exact_repair_protocol_20260829.json \
  >/dev/null
"${LOCAL_MEMNAV_PY}" -m json.tool \
  MemNavData/mp3d_table1_navdp_authority_cache_composition_repair_20260829.json \
  >/dev/null
"${LOCAL_MEMNAV_PY}" -m py_compile \
  MemNavData/audit_mp3d_table1_controller_exact_repair_bundle.py
"${LOCAL_MEMNAV_PY}" -m pytest -q -p no:cacheprovider \
  MemNavData/test_audit_mp3d_table1_controller_exact_repair_bundle.py \
  MemNavData/test_hm3d_table1_navdp_transport_contract.py
bash -n \
  MemNavData/slurm_hm3d_table1_navdp_pair.sbatch \
  MemNavData/slurm_hm3d_table1_vint_pair.sbatch \
  MemNavData/slurm_port_pair.sh \
  MemNavData/test_slurm_port_pair.sh \
  MemNavData/slurm_safe_submit.sh \
  MemNavData/bundle_selftest.sh \
  MemNavData/submit_mp3d_table1_controller_exact_repair_remote.sh \
  MemNavData/submit_mp3d_table1_navdp_authority_cache_repair_remote.sh \
  MemNavData/prepare_mp3d_table1_controller_exact_repair_bundle.sh
bash MemNavData/test_slurm_port_pair.sh
source MemNavData/slurm_safe_submit.sh
lint_sbatch_template MemNavData/slurm_hm3d_table1_navdp_pair.sbatch || \
  fail "NavDP sbatch lint failed"
lint_sbatch_template MemNavData/slurm_hm3d_table1_vint_pair.sbatch || \
  fail "ViNT sbatch lint failed"

mkdir -p "${LOCAL_BUNDLE_PARENT}"
stage=$(mktemp -d "${LOCAL_BUNDLE_PARENT}/mp3d_table1_exact_repair.partial.XXXXXX")
mkdir -p "${stage}/MemNavData"
for path in "${required[@]}"; do
  cp --preserve=mode,timestamps "${ROOT}/${path}" \
    "${stage}/MemNavData/$(basename "${path}")"
done
local_head=$(git -C "${ROOT}" rev-parse HEAD)
"${LOCAL_MEMNAV_PY}" - "${stage}" "${local_head}" <<'PY'
import hashlib,json,sys
from pathlib import Path
root=Path(sys.argv[1]); files={}
for path in sorted(root.rglob('*')):
    if path.is_symlink(): raise SystemExit('bundle symlink: '+str(path))
    if path.is_file() and path.name not in {
        'SOURCE_BUNDLE.sha256','source_bundle_manifest.json'}:
        files[path.relative_to(root).as_posix()]=hashlib.sha256(
            path.read_bytes()).hexdigest()
payload={
    'schema_version':'mp3d_table1_controller_exact_repair_bundle_v1_20260829',
    'local_git_head_context':sys.argv[2],
    'scientific_method_or_population_changed':False,
    'frozen_histories':42,'scene_clusters':25,
    'navdp_exact_history_indices':[29,30],
    'vint_exact_history_indices':[24],
    'files':files,
}
(root/'source_bundle_manifest.json').write_text(
    json.dumps(payload,indent=2,sort_keys=True)+'\n')
PY
(
  cd "${stage}"
  find . -type f ! -name SOURCE_BUNDLE.sha256 -print0 | sort -z | \
    xargs -0 sha256sum >SOURCE_BUNDLE.sha256
  sha256sum -c --quiet SOURCE_BUNDLE.sha256
)

entries=$(mktemp "${LOCAL_BUNDLE_PARENT}/mp3d_table1_exact_repair.entries.XXXXXX")
"${LOCAL_MEMNAV_PY}" - "${entries}" "${LOCAL_MEMNAV_PY}" <<'PY'
from pathlib import Path
import sys
Path(sys.argv[1]).write_text(
    f"{sys.argv[2]} import MemNavData.audit_mp3d_table1_controller_exact_repair_bundle\n"
    f"{sys.argv[2]} run MemNavData/audit_mp3d_table1_controller_exact_repair_bundle.py --root .\n"
)
PY
SELFTEST_BUNDLE_SUBPATHS=MemNavData \
  bash "${stage}/MemNavData/bundle_selftest.sh" "${stage}" "${entries}"
rm -f -- "${entries}"

receipt_sha=$(sha256sum "${stage}/SOURCE_BUNDLE.sha256" | awk '{print $1}')
target=${LOCAL_BUNDLE_PARENT}/mp3d_table1_controller_exact_repair_${receipt_sha:0:16}
[[ ! -e "${target}" ]] || fail "content-addressed bundle already exists: ${target}"
chmod -R a-w "${stage}"
mv "${stage}" "${target}"
printf 'BUNDLE=%s\nRECEIPT_SHA256=%s\n' "${target}" "${receipt_sha}"
