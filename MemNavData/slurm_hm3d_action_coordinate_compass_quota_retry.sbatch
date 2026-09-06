#!/usr/bin/env bash
#SBATCH --job-name=h3ActCoordQR
#SBATCH --output=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/%x_%A_%a.out
#SBATCH --error=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/%x_%A_%a.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=72G
#SBATCH --partition=h100_tandon,a100_tandon
#SBATCH --qos=gpu48
#SBATCH --gres=gpu:1
#SBATCH --account=torch_pr_769_tandon_advanced

set -euo pipefail
umask 0022
export PYTHONDONTWRITEBYTECODE=1

: "${SLURM_ARRAY_TASK_ID:?submit exact indices 18-47}"
: "${REPAIR_SOURCE_ROOT:?}" "${REPAIR_SOURCE_RECEIPT:?}"
: "${EXPECTED_REPAIR_SOURCE_RECEIPT_SHA:?}"
: "${SOURCE_ROOT:?}" "${SOURCE_RECEIPT:?}" "${EXPECTED_SOURCE_RECEIPT_SHA:?}"
: "${RUN_ROOT:?}" "${REPAIR_STATE_ROOT:?}"

ORIGINAL_PAIR=${SOURCE_ROOT}/MemNavData/slurm_hm3d_action_coordinate_compass_pair.sbatch
PY=/scratch/lg154/conda-envs/memnav/bin/python

fail() { echo "ABORT: $*" >&2; exit 2; }
verify_bundle() {
  local root=$1 receipt=$2 expected=$3 label=$4
  [[ -d "${root}" && -f "${receipt}" ]] || fail "missing ${label} bundle"
  [[ "$(sha256sum "${receipt}" | awk '{print $1}')" == "${expected}" ]] ||
    fail "${label} receipt changed"
  (cd "${root}" && sha256sum -c --quiet "${receipt}") ||
    fail "${label} bundle content changed"
}

[[ "$(id -un)" == yz11502 ]] || fail "wrong HPC identity"
[[ "${SLURM_ARRAY_TASK_ID}" =~ ^(1[89]|[2-3][0-9]|4[0-7])$ ]] ||
  fail "retry index escaped exact 18-47 set"
verify_bundle "${REPAIR_SOURCE_ROOT}" "${REPAIR_SOURCE_RECEIPT}" \
  "${EXPECTED_REPAIR_SOURCE_RECEIPT_SHA}" repair
verify_bundle "${SOURCE_ROOT}" "${SOURCE_RECEIPT}" \
  "${EXPECTED_SOURCE_RECEIPT_SHA}" original_scientific
[[ -r "${ORIGINAL_PAIR}" ]] || fail "original paired wrapper is not readable"

# Run the exact immutable scientific wrapper in a child shell. Its final exec(2)
# replaces only that child, allowing this infrastructure wrapper to clean the
# regenerable frame buffer after the sealed completion has been verified.
set +e
bash "${ORIGINAL_PAIR}"
rc=$?
set -e
if [[ "${rc}" -ne 0 ]]; then
  echo "ORIGINAL_PAIR_FAILED index=${SLURM_ARRAY_TASK_ID} rc=${rc}" >&2
  exit "${rc}"
fi

label=$(printf '%03d' "${SLURM_ARRAY_TASK_ID}")
readarray -t resolved < <("${PY}" - "${RUN_ROOT}" "${label}" <<'PY'
import glob, pathlib, sys
root=pathlib.Path(sys.argv[1]); label=sys.argv[2]
completions=glob.glob(str(root/'development/action_coordinate_compass'/f'{label}_*'/'completion.json'))
buffers=glob.glob(str(root/'runtime'/f'eval_*_actCoordPair{label}'/'buffer'))
if len(completions) != 1:
    raise SystemExit(f'expected one sealed completion, found {completions}')
if len(buffers) != 1:
    raise SystemExit(f'expected one runtime buffer, found {buffers}')
print(completions[0]); print(buffers[0])
PY
)
[[ "${#resolved[@]}" -eq 2 ]] || fail "post-run path resolution failed"
completion=${resolved[0]}
buffer=${resolved[1]}
sidecar=${completion}.sha256
[[ -f "${completion}" && -f "${sidecar}" ]] || fail "completion is not sealed"
expected=$(awk 'NR==1{print $1}' "${sidecar}")
actual=$(sha256sum "${completion}" | awk '{print $1}')
[[ "${expected}" == "${actual}" ]] || fail "completion sidecar mismatch"
canonical_buffer=$(realpath -m "${buffer}")
[[ "${canonical_buffer}" == "${RUN_ROOT}/runtime/"eval_*_actCoordPair"${label}"/buffer ]] ||
  fail "buffer escaped exact runtime namespace: ${canonical_buffer}"

buffer_files=$(find "${buffer}" -type f -printf x | wc -c)
buffer_bytes=$(du -sb "${buffer}" | awk '{print $1}')
rm -rf -- "${buffer}"
[[ ! -e "${buffer}" ]] || fail "runtime buffer cleanup failed"

cleanup_root=${REPAIR_STATE_ROOT}/successful_task_buffer_cleanup
mkdir -p "${cleanup_root}"
cleanup=${cleanup_root}/${label}.json
[[ ! -e "${cleanup}" ]] || fail "cleanup receipt already exists"
"${PY}" - "${cleanup}" "${SLURM_ARRAY_TASK_ID}" "${canonical_buffer}" \
  "${buffer_files}" "${buffer_bytes}" "${actual}" <<'PY'
import datetime, json, os, sys
path,index,target,files,bytes_,completion_sha=sys.argv[1:]
payload={
 'schema_version':'hm3d_action_coordinate_task_buffer_cleanup_v1_20260902',
 'created_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),
 'index':int(index), 'target':target, 'files_removed':int(files),
 'bytes_removed':int(bytes_), 'completion_sha256':completion_sha,
 'completion_verified_before_cleanup':True,
 'navigation_outcomes_read':False,
 'data_class':'regenerable_runtime_rgb_buffer',
 'scientific_execution_changed':False,
}
fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o444)
with os.fdopen(fd,'w') as f:
    json.dump(payload,f,indent=2,sort_keys=True); f.write('\n')
PY
sha256sum "${cleanup}" >"${cleanup}.sha256"
chmod a-w "${cleanup}" "${cleanup}.sha256"
printf 'COMPLETE index=%s completion_sha=%s removed_files=%s removed_bytes=%s\n' \
  "${SLURM_ARRAY_TASK_ID}" "${actual}" "${buffer_files}" "${buffer_bytes}"
