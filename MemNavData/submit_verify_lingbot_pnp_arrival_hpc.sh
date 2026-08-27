#!/usr/bin/env bash
set -euo pipefail
umask 0022

: "${SUMMARY_JOB_ID:?SUMMARY_JOB_ID is required}"
: "${RUN_ROOT:?RUN_ROOT is required}"
LOCAL_ROOT="$(git rev-parse --show-toplevel)"
REMOTE_HOST="${REMOTE_HOST:-alantorch}"
REMOTE_BUNDLE_BASE="${REMOTE_BUNDLE_BASE:-/scratch/yz11502/Research/source_bundles}"
PY="/home/asus/miniconda3/envs/memnav/bin/python"

"${PY}" -m py_compile "${LOCAL_ROOT}/MemNavData/verify_lingbot_pnp_arrival.py"
(cd "${LOCAL_ROOT}" && "${PY}" -m unittest \
  MemNavData.test_verify_lingbot_pnp_arrival -v)
bash -n "${LOCAL_ROOT}/MemNavData/slurm_verify_lingbot_pnp_arrival.sbatch"

stage="$(mktemp -d /tmp/pnp_arrival_verify.XXXXXX)"
trap 'test ! -d "${stage}" || find "${stage}" -depth -delete' EXIT
mkdir -p "${stage}/MemNavData" "${stage}/inputs"
for relative in \
  MemNavData/verify_lingbot_pnp_arrival.py \
  MemNavData/test_verify_lingbot_pnp_arrival.py \
  MemNavData/summarize_lingbot_pnp_arrival.py \
  MemNavData/slurm_verify_lingbot_pnp_arrival.sbatch; do
  cp --preserve=mode,timestamps "${LOCAL_ROOT}/${relative}" "${stage}/${relative}"
done
cp --preserve=mode,timestamps \
  "${LOCAL_ROOT}/.diagnostics/navdp_arrival_consensus_merged_20260815/states.csv" \
  "${stage}/inputs/states.csv"
"${PY}" - "${stage}" <<'PY'
import hashlib,json,sys
from pathlib import Path
root=Path(sys.argv[1]); files={}
for path in sorted(root.rglob("*")):
    if path.is_symlink(): raise SystemExit(f"symlink: {path}")
    if path.is_file() and path.name!="source_bundle_manifest.json":
        files[path.relative_to(root).as_posix()]=hashlib.sha256(
            path.read_bytes()).hexdigest()
(root/"source_bundle_manifest.json").write_text(json.dumps({
    "schema_version":"lingbot_pnp_arrival_verifier_bundle_v1",
    "files":files},indent=2,sort_keys=True)+"\n")
PY
sha="$(sha256sum "${stage}/source_bundle_manifest.json" | awk '{print $1}')"
bundle="${REMOTE_BUNDLE_BASE}/lingbot_pnp_arrival_verifier_${sha:0:16}"
ssh -o BatchMode=yes "${REMOTE_HOST}" "mkdir -p '${bundle}'"
rsync -a --chmod=Du=rwx,Dgo=rx,Fu=rw,Fgo=r "${stage}/" \
  "${REMOTE_HOST}:${bundle}/"
remote_sha="$(ssh -o BatchMode=yes "${REMOTE_HOST}" \
  "sha256sum '${bundle}/source_bundle_manifest.json' | awk '{print \$1}'")"
[[ "${remote_sha}" == "${sha}" ]] || { echo "remote verifier differs" >&2; exit 2; }
job="$(ssh -o BatchMode=yes "${REMOTE_HOST}" \
  "sbatch --parsable --dependency=afterok:${SUMMARY_JOB_ID} --export='ALL,VERIFY_BUNDLE=${bundle},EXPECTED_VERIFY_MANIFEST_SHA=${sha},RUN_ROOT=${RUN_ROOT}' '${bundle}/MemNavData/slurm_verify_lingbot_pnp_arrival.sbatch'")"
printf 'PNP_ARRIVAL_VERIFY_JOB=%s\n' "${job}"
printf 'PNP_ARRIVAL_VERIFY_BUNDLE=%s\n' "${bundle}"
printf 'PNP_ARRIVAL_VERIFY_MANIFEST_SHA=%s\n' "${sha}"
