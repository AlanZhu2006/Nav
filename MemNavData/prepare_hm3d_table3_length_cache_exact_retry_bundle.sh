#!/usr/bin/env bash
set -euo pipefail
umask 0022

ROOT=${ROOT:-$(git rev-parse --show-toplevel)}
PY=${LOCAL_MEMNAV_PY:-/home/asus/miniconda3/envs/memnav/bin/python}
PARENT=${LOCAL_BUNDLE_PARENT:-${ROOT}/.diagnostics/source_bundles}
fail() { echo "ABORT: $*" >&2; exit 2; }

required=(
  MemNavData/HM3D_TABLE3_LENGTH_CACHE_EXACT_RETRY_PROTOCOL_20260902.md
  MemNavData/hm3d_table3_length_cache_exact_retry_protocol_20260902.json
  MemNavData/submit_hm3d_table3_length_cache_exact_retry_remote.sh
  MemNavData/test_hm3d_table3_length_cache_exact_retry_contract.py
  MemNavData/slurm_safe_submit.sh
)
for path in "${required[@]}"; do
  [[ -f "${ROOT}/${path}" && ! -L "${ROOT}/${path}" ]] || \
    fail "missing physical file ${path}"
done
[[ -x "${PY}" ]] || fail "missing MemNav interpreter"

cd "${ROOT}"
"${PY}" -m json.tool \
  MemNavData/hm3d_table3_length_cache_exact_retry_protocol_20260902.json \
  >/dev/null
"${PY}" -m pytest -q -p no:cacheprovider \
  MemNavData/test_hm3d_table3_length_cache_exact_retry_contract.py \
  MemNavData/test_navdp_monocular_transaction.py
bash -n MemNavData/submit_hm3d_table3_length_cache_exact_retry_remote.sh \
  MemNavData/slurm_safe_submit.sh \
  MemNavData/prepare_hm3d_table3_length_cache_exact_retry_bundle.sh

mkdir -p "${PARENT}"
stage=$(mktemp -d "${PARENT}/hm3d_table3_length_cache_retry.partial.XXXXXX")
mkdir -p "${stage}/MemNavData"
for path in "${required[@]}"; do
  cp --preserve=mode,timestamps "${ROOT}/${path}" \
    "${stage}/MemNavData/$(basename "${path}")"
done
head=$(git -C "${ROOT}" rev-parse HEAD)
"${PY}" - "${stage}" "${head}" <<'PY'
import hashlib, json, sys
from pathlib import Path
root=Path(sys.argv[1]); files={}
for path in sorted(root.rglob('*')):
    if path.is_symlink(): raise SystemExit('bundle symlink: '+str(path))
    if path.is_file() and path.name not in {'SOURCE_BUNDLE.sha256','source_bundle_manifest.json'}:
        files[path.relative_to(root).as_posix()]=hashlib.sha256(path.read_bytes()).hexdigest()
payload={
 'schema_version':'hm3d_table3_length_cache_exact_retry_bundle_v1_20260902',
 'local_git_head_context':sys.argv[2], 'scientific_method_or_population_changed':False,
 'retained_completion_count':45, 'exact_retry_indices':[35,41,44], 'files':files,
}
(root/'source_bundle_manifest.json').write_text(json.dumps(payload,indent=2,sort_keys=True)+'\n')
PY
(
  cd "${stage}"
  find . -type f ! -name SOURCE_BUNDLE.sha256 -print0 | sort -z | \
    xargs -0 sha256sum >SOURCE_BUNDLE.sha256
  sha256sum -c --quiet SOURCE_BUNDLE.sha256
)
receipt_sha=$(sha256sum "${stage}/SOURCE_BUNDLE.sha256" | awk '{print $1}')
target=${PARENT}/hm3d_table3_length_cache_exact_retry_${receipt_sha:0:16}
[[ ! -e "${target}" ]] || fail "content-addressed bundle already exists: ${target}"
chmod -R a-w "${stage}"
mv "${stage}" "${target}"
printf 'BUNDLE=%s\nRECEIPT_SHA256=%s\n' "${target}" "${receipt_sha}"
