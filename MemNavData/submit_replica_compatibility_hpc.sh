#!/usr/bin/env bash
set -euo pipefail
umask 0022

LOCAL_ROOT=${LOCAL_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
REPLICA_ROOT=${REPLICA_ROOT:-/home/asus/Research/Pi3/data/replica_v1}
REMOTE_HOST=${REMOTE_HOST:-alantorch}
SSH_CONTROL_PATH=${SSH_CONTROL_PATH:-}
REMOTE_BUNDLE_BASE=${REMOTE_BUNDLE_BASE:-/scratch/yz11502/Research/source_bundles}
REMOTE_RUN_BASE=${REMOTE_RUN_BASE:-/scratch/yz11502/Research/Nav-axis-uturn-results/replica_cross_dataset_20260814}
DRY_RUN=${DRY_RUN:-0}
EXPECTED_COMPLETE_SCENES=${EXPECTED_COMPLETE_SCENES:-18}
MINIMUM_ELIGIBLE_SCENES=${MINIMUM_ELIGIBLE_SCENES:-8}

fail() { echo "ABORT: $*" >&2; exit 2; }
SSH_ARGS=(-T -o BatchMode=yes)
RSYNC_RSH="ssh -o BatchMode=yes"
if [[ -n "${SSH_CONTROL_PATH}" ]]; then
  [[ "${SSH_CONTROL_PATH}" =~ ^/[A-Za-z0-9._/-]+$ \
     && -S "${SSH_CONTROL_PATH}" ]] || fail "invalid SSH control socket"
  SSH_ARGS+=(-S "${SSH_CONTROL_PATH}" -o ControlMaster=no)
  RSYNC_RSH="ssh -o BatchMode=yes -S ${SSH_CONTROL_PATH} -o ControlMaster=no"
fi
remote() { ssh "${SSH_ARGS[@]}" "${REMOTE_HOST}" "$@"; }

[[ "${EXPECTED_COMPLETE_SCENES}" =~ ^[1-9][0-9]*$ ]] || {
  echo "invalid EXPECTED_COMPLETE_SCENES" >&2; exit 2; }
[[ "${MINIMUM_ELIGIBLE_SCENES}" =~ ^[1-9][0-9]*$ ]] || {
  echo "invalid MINIMUM_ELIGIBLE_SCENES" >&2; exit 2; }

for required in \
  "${LOCAL_ROOT}/MemNavData/audit_replica_habitat_compatibility.py" \
  "${LOCAL_ROOT}/MemNavData/generate_twoleg.py" \
  "${LOCAL_ROOT}/MemNavData/slurm_replica_compatibility.sbatch" \
  "${REPLICA_ROOT}/replica.scene_dataset_config.json"; do
  [[ -r "${required}" ]] || { echo "missing ${required}" >&2; exit 2; }
done

staging=$(mktemp -d)
trap 'rm -rf -- "${staging}"' EXIT
mkdir -p "${staging}/MemNavData" "${staging}/replica_v1"
cp --preserve=mode,timestamps \
  "${LOCAL_ROOT}/MemNavData/audit_replica_habitat_compatibility.py" \
  "${LOCAL_ROOT}/MemNavData/generate_twoleg.py" \
  "${LOCAL_ROOT}/MemNavData/slurm_replica_compatibility.sbatch" \
  "${staging}/MemNavData/"
cp --preserve=mode,timestamps \
  "${REPLICA_ROOT}/replica.scene_dataset_config.json" \
  "${staging}/replica_v1/"

scene_count=0
for scene_root in "${REPLICA_ROOT}"/*; do
  [[ -d "${scene_root}" ]] || continue
  required=(
    "${scene_root}/mesh.ply"
    "${scene_root}/habitat/replica_stage.stage_config.json"
    "${scene_root}/habitat/mesh_semantic.ply"
    "${scene_root}/habitat/mesh_semantic.navmesh"
  )
  complete=1
  for path in "${required[@]}"; do [[ -f "${path}" ]] || complete=0; done
  [[ "${complete}" -eq 1 ]] || continue
  scene=$(basename "${scene_root}")
  mkdir -p "${staging}/replica_v1/${scene}/habitat"
  cp --preserve=mode,timestamps "${required[0]}" \
    "${staging}/replica_v1/${scene}/mesh.ply"
  cp --preserve=mode,timestamps "${required[1]}" \
    "${staging}/replica_v1/${scene}/habitat/replica_stage.stage_config.json"
  cp --preserve=mode,timestamps "${required[2]}" \
    "${staging}/replica_v1/${scene}/habitat/mesh_semantic.ply"
  cp --preserve=mode,timestamps "${required[3]}" \
    "${staging}/replica_v1/${scene}/habitat/mesh_semantic.navmesh"
  scene_count=$((scene_count + 1))
done
[[ "${scene_count}" -eq "${EXPECTED_COMPLETE_SCENES}" ]] || {
  echo "expected ${EXPECTED_COMPLETE_SCENES} complete scenes, got ${scene_count}" >&2
  exit 2
}

python - "${staging}" "${EXPECTED_COMPLETE_SCENES}" \
  "${MINIMUM_ELIGIBLE_SCENES}" <<'PY'
import hashlib,json,sys
from pathlib import Path
root=Path(sys.argv[1]); files={}
for path in sorted(root.rglob('*')):
    if path.is_symlink(): raise SystemExit(f'symlink forbidden: {path}')
    if path.is_file() and path.name not in {'SOURCE_BUNDLE.sha256','source_bundle_manifest.json'}:
        files[path.relative_to(root).as_posix()]=hashlib.sha256(path.read_bytes()).hexdigest()
payload={
 'schema_version':'replica_compatibility_task_bundle_v1_20260814',
 'scope':'simulator_sensor_navmesh_gate_no_policy_outcomes',
 'expected_complete_scenes':int(sys.argv[2]),
 'minimum_eligible_scenes':int(sys.argv[3]),
 'files':files,
}
(root/'source_bundle_manifest.json').write_text(json.dumps(payload,indent=2,sort_keys=True)+'\n')
PY
(
  cd "${staging}"
  find . -type f ! -name SOURCE_BUNDLE.sha256 -print0 | sort -z | \
    xargs -0 sha256sum > SOURCE_BUNDLE.sha256
  sha256sum -c SOURCE_BUNDLE.sha256 >/dev/null
)
receipt_sha=$(sha256sum "${staging}/SOURCE_BUNDLE.sha256" | awk '{print $1}')
manifest_sha=$(sha256sum "${staging}/source_bundle_manifest.json" | awk '{print $1}')
remote_bundle=${REMOTE_BUNDLE_BASE}/replica_compat_${manifest_sha:0:16}
remote_partial=${remote_bundle}.partial-$$
run_root=${REMOTE_RUN_BASE}/compat_${manifest_sha:0:16}

if [[ "${DRY_RUN}" == 1 ]]; then
  echo "SOURCE_RECEIPT_SHA=${receipt_sha}"
  echo "REMOTE_BUNDLE=${remote_bundle}"
  echo "RUN_ROOT=${run_root}"
  exit 0
fi

if remote \
  "test -d '${remote_bundle}' && test \"\$(sha256sum '${remote_bundle}/SOURCE_BUNDLE.sha256' | awk '{print \$1}')\" = '${receipt_sha}' && cd '${remote_bundle}' && sha256sum -c SOURCE_BUNDLE.sha256 >/dev/null"; then
  echo "reusing ${remote_bundle}"
else
  remote "test ! -e '${remote_bundle}' && test ! -e '${remote_partial}' && mkdir -p '${remote_partial}'"
  rsync -e "${RSYNC_RSH}" -a --chmod=Du=rwx,Dgo=rx,Fu=rw,Fgo=r \
    "${staging}/" "${REMOTE_HOST}:${remote_partial}/"
  remote \
    "cd '${remote_partial}' && sha256sum -c SOURCE_BUNDLE.sha256 >/dev/null && chmod -R a-w '${remote_partial}' && mv '${remote_partial}' '${remote_bundle}'"
fi

exports="ALL,SOURCE_ROOT=${remote_bundle},SOURCE_RECEIPT=${remote_bundle}/SOURCE_BUNDLE.sha256,EXPECTED_SOURCE_RECEIPT_SHA=${receipt_sha},RUN_ROOT=${run_root},EXPECTED_COMPLETE_SCENES=${EXPECTED_COMPLETE_SCENES},MINIMUM_ELIGIBLE_SCENES=${MINIMUM_ELIGIBLE_SCENES}"
remote \
  "test ! -e '${run_root}' && sbatch --test-only --export='${exports}' '${remote_bundle}/MemNavData/slurm_replica_compatibility.sbatch' >/dev/null"
job_raw=$(remote \
  "sbatch --parsable --export='${exports}' '${remote_bundle}/MemNavData/slurm_replica_compatibility.sbatch'")
job_id=${job_raw%%;*}
[[ "${job_id}" =~ ^[0-9]+$ ]] || { echo "invalid job id: ${job_raw}" >&2; exit 2; }
python - "${LOCAL_ROOT}/.diagnostics/replica_compatibility_submission_20260814.json" \
  "${job_id}" "${remote_bundle}" "${receipt_sha}" "${run_root}" <<'PY'
import json,sys
path,job,bundle,receipt,run=sys.argv[1:]
with open(path,'x') as f:
 json.dump({'schema_version':'replica_compatibility_submission_v1_20260814','job_id':int(job),'source_bundle':bundle,'source_receipt_sha256':receipt,'run_root':run},f,indent=2,sort_keys=True);f.write('\n')
PY
echo "JOB_ID=${job_id}"
echo "SOURCE_BUNDLE=${remote_bundle}"
echo "SOURCE_RECEIPT_SHA=${receipt_sha}"
echo "RUN_ROOT=${run_root}"
