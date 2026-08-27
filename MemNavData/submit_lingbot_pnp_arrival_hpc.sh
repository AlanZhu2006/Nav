#!/usr/bin/env bash
# Freeze and submit smoke -> 8-shard formal -> deterministic train summary.
set -euo pipefail
umask 0022

LOCAL_ROOT="$(git rev-parse --show-toplevel)"
REMOTE_HOST="${REMOTE_HOST:-alantorch}"
REMOTE_BUNDLE_BASE="${REMOTE_BUNDLE_BASE:-/scratch/yz11502/Research/source_bundles}"
REMOTE_RESULT_BASE="${REMOTE_RESULT_BASE:-/scratch/yz11502/Research/Nav-axis-uturn-results/lingbot_pnp_arrival_20260815}"
MEMNAV_PY="${MEMNAV_PY:-/home/asus/miniconda3/envs/memnav/bin/python}"
SHARD_COUNT="${SHARD_COUNT:-8}"

FILES=(
  MemNavData/audit_lingbot_pnp_arrival.py
  MemNavData/summarize_lingbot_pnp_arrival.py
  MemNavData/test_lingbot_pnp_arrival.py
  MemNavData/lingbot_pnp_localization.py
  MemNavData/lingbot_colored_registration.py
  MemNavData/certified_relocalization_runtime.py
  MemNavData/LINGBOT_PNP_ARRIVAL_PROTOCOL_20260815.md
  MemNavData/slurm_lingbot_pnp_arrival.sbatch
  MemNavData/slurm_summarize_lingbot_pnp_arrival.sbatch
)
for relative in "${FILES[@]}"; do
  [[ -f "${LOCAL_ROOT}/${relative}" && ! -L "${LOCAL_ROOT}/${relative}" ]] || {
    echo "ABORT: missing physical source ${relative}" >&2; exit 2; }
done
[[ "${SHARD_COUNT}" =~ ^[1-9][0-9]*$ ]] || {
  echo "ABORT: SHARD_COUNT must be positive" >&2; exit 2; }

"${MEMNAV_PY}" -m py_compile \
  "${LOCAL_ROOT}/MemNavData/audit_lingbot_pnp_arrival.py" \
  "${LOCAL_ROOT}/MemNavData/summarize_lingbot_pnp_arrival.py"
(cd "${LOCAL_ROOT}" && "${MEMNAV_PY}" -m unittest \
  MemNavData.test_lingbot_pnp_arrival -v)
bash -n "${LOCAL_ROOT}/MemNavData/slurm_lingbot_pnp_arrival.sbatch"
bash -n "${LOCAL_ROOT}/MemNavData/slurm_summarize_lingbot_pnp_arrival.sbatch"

stage="$(mktemp -d /tmp/lingbot_pnp_arrival_bundle.XXXXXX)"
trap 'test ! -d "${stage}" || find "${stage}" -depth -delete' EXIT
mkdir -p "${stage}/MemNavData" "${stage}/inputs" \
  "${stage}/third_party/LightGlue" "${stage}/third_party/python" \
  "${stage}/torch_home/hub/checkpoints"
for relative in "${FILES[@]}"; do
  cp --preserve=mode,timestamps "${LOCAL_ROOT}/${relative}" "${stage}/${relative}"
done
cp --preserve=mode,timestamps \
  "${LOCAL_ROOT}/.diagnostics/navdp_arrival_consensus_merged_20260815/states.csv" \
  "${LOCAL_ROOT}/.diagnostics/navdp_arrival_consensus_merged_20260815/samples.csv" \
  "${LOCAL_ROOT}/.diagnostics/navdp_arrival_consensus_merged_20260815/inventory.json" \
  "${stage}/inputs/"
cp --preserve=mode,timestamps \
  "${LOCAL_ROOT}/.diagnostics/lingbot_pnp_arrival_inputs_20260815/FLOW_ROUTE_PROVENANCE.json" \
  "${LOCAL_ROOT}/.diagnostics/lingbot_pnp_arrival_inputs_20260815/causal_ground_scale.json" \
  "${stage}/inputs/"
cp -a "${LOCAL_ROOT}/.diagnostics/dependencies/LightGlue/lightglue" \
  "${stage}/third_party/LightGlue/"
cp --preserve=mode,timestamps \
  "${LOCAL_ROOT}/.diagnostics/dependencies/LightGlue/LICENSE" \
  "${stage}/third_party/LightGlue/LICENSE"
cp --preserve=mode,timestamps \
  "${LOCAL_ROOT}/MemNavData/lightglue_bundled_commit.txt" \
  "${stage}/third_party/LightGlue/BUNDLED_COMMIT"
for dependency in kornia kornia-0.8.1.dist-info \
                  kornia_rs kornia_rs-0.1.9.dist-info; do
  cp -a "${LOCAL_ROOT}/.diagnostics/dependencies/python/${dependency}" \
    "${stage}/third_party/python/"
done
cp --preserve=mode,timestamps \
  /home/asus/.cache/torch/hub/checkpoints/superpoint_v1.pth \
  /home/asus/.cache/torch/hub/checkpoints/superpoint_lightglue_v0-1_arxiv.pth \
  "${stage}/torch_home/hub/checkpoints/"

"${MEMNAV_PY}" - "${stage}" <<'PY'
import hashlib, json, sys
from pathlib import Path
root=Path(sys.argv[1]); files={}
for path in sorted(root.rglob("*")):
    if path.is_symlink(): raise SystemExit(f"bundle symlink: {path}")
    if path.is_file() and path.name != "source_bundle_manifest.json":
        files[path.relative_to(root).as_posix()]=hashlib.sha256(
            path.read_bytes()).hexdigest()
payload={"schema_version":"lingbot_pnp_arrival_bundle_v1","files":files}
(root/"source_bundle_manifest.json").write_text(
    json.dumps(payload,indent=2,sort_keys=True)+"\n")
PY
manifest_sha="$(sha256sum "${stage}/source_bundle_manifest.json" | awk '{print $1}')"
remote_bundle="${REMOTE_BUNDLE_BASE}/lingbot_pnp_arrival_${manifest_sha:0:16}"
ssh -o BatchMode=yes "${REMOTE_HOST}" "mkdir -p '${remote_bundle}' '${REMOTE_RESULT_BASE}'"
rsync -a --chmod=Du=rwx,Dgo=rx,Fu=rw,Fgo=r "${stage}/" \
  "${REMOTE_HOST}:${remote_bundle}/"
remote_sha="$(ssh -o BatchMode=yes "${REMOTE_HOST}" \
  "sha256sum '${remote_bundle}/source_bundle_manifest.json' | awk '{print \$1}'")"
[[ "${remote_sha}" == "${manifest_sha}" ]] || {
  echo "ABORT: remote bundle differs" >&2; exit 2; }

tag="$(date -u +%Y%m%dT%H%M%SZ)"
run_root="${REMOTE_RESULT_BASE}/run_${manifest_sha:0:16}_${tag}"
common="ALL,SOURCE_BUNDLE=${remote_bundle},EXPECTED_BUNDLE_MANIFEST_SHA=${manifest_sha},RUN_ROOT=${run_root},SHARD_COUNT=${SHARD_COUNT}"
smoke_job="$(ssh -o BatchMode=yes "${REMOTE_HOST}" \
  "sbatch --parsable --time=00:30:00 --export='${common},RUN_MODE=smoke' '${remote_bundle}/MemNavData/slurm_lingbot_pnp_arrival.sbatch'")"
formal_job="$(ssh -o BatchMode=yes "${REMOTE_HOST}" \
  "sbatch --parsable --dependency=afterok:${smoke_job} --array=0-$((SHARD_COUNT-1))%4 --export='${common},RUN_MODE=formal' '${remote_bundle}/MemNavData/slurm_lingbot_pnp_arrival.sbatch'")"
summary_job="$(ssh -o BatchMode=yes "${REMOTE_HOST}" \
  "sbatch --parsable --dependency=afterok:${formal_job} --export='${common}' '${remote_bundle}/MemNavData/slurm_summarize_lingbot_pnp_arrival.sbatch'")"
printf 'PNP_ARRIVAL_SMOKE_JOB=%s\n' "${smoke_job}"
printf 'PNP_ARRIVAL_FORMAL_JOB=%s\n' "${formal_job}"
printf 'PNP_ARRIVAL_SUMMARY_JOB=%s\n' "${summary_job}"
printf 'PNP_ARRIVAL_BUNDLE=%s\n' "${remote_bundle}"
printf 'PNP_ARRIVAL_MANIFEST_SHA=%s\n' "${manifest_sha}"
printf 'PNP_ARRIVAL_RUN_ROOT=%s\n' "${run_root}"
