#!/usr/bin/env bash
# Exact infrastructure-only repair for Final14 zero-depth history index 19.
# The original cell failed before evaluation because a shared-node TCP port was
# claimed concurrently.  This script reuses every frozen scientific input and
# changes only the port allocation contract.
set -euo pipefail
umask 0022

ROOT=${ROOT:-/home/asus/Research/Nav-graph-blind}
SSH_ALIAS=${SSH_ALIAS:-alantorch}
EXPECTED_SSH_USER=${EXPECTED_SSH_USER:-yz11502}
SUBMIT=${SUBMIT:-0}
LOCAL_PY=${LOCAL_PY:-/home/asus/miniconda3/envs/memnav/bin/python}
ORIGINAL_ARRAY=16499701
SUPERSEDED_ANALYSIS=16499709
FAILED_INDEX=19
FAILED_RAW_JOB=16501938
LABEL=019_Z6MFQCViBuw_episode_0003
RUN_ROOT=/scratch/yz11502/Research/Nav-axis-uturn-results/final14_zero_depth_20260828/formal_4c061bd6b86da365
ARCHIVE_ROOT=${RUN_ROOT}/failed_attempts/port_collision_16499701_19
BASE_SOURCE_ROOT=/scratch/yz11502/Research/Nav-axis-uturn-source-bundles/final14_mono_factorial_5690569a4373f2d2
BASE_SOURCE_RECEIPT=${BASE_SOURCE_ROOT}/source_inputs.sha256
EXPECTED_BASE_SOURCE_RECEIPT_SHA=5690569a4373f2d2768671418f0c604c4a03aa4b0ffe01baf70b288af03ba216
REFERENCE_ROOT=/scratch/yz11502/Research/Nav-axis-uturn-results/final14_mono_factorial_20260819/formal_20260819T124820Z_5690569a
BENCH_ROOT=/scratch/yz11502/Research/Nav-axis-uturn-results/final14_cec_learned_20260817/final14_learned_20260817T115533Z_attempt7_handoff/benchmarks/natural_direction
SOURCE_OVERLAY=/scratch/lg154/Research/datasets/_overlays/mp3d_revisit_v0_pt1.sqf
EXPECTED_SOURCE_OVERLAY_BYTES=128854888448

cd "${ROOT}"
fail() { echo "ABORT: $*" >&2; exit 2; }
job_id() { tr -d '\r' | awk -F';' '/^[0-9]+(;|$)/ {print $1; exit}'; }
SSH_CONTROL_PATH=${SSH_CONTROL_PATH:-$(
  ssh -G "${SSH_ALIAS}" 2>/dev/null |
    awk '$1=="controlpath"{value=$2} END{print value}'
)}
[[ "${SUBMIT}" =~ ^[01]$ ]] || fail "SUBMIT must be 0 or 1"
[[ -S "${SSH_CONTROL_PATH}" ]] || fail "authoritative shared SSH socket missing"
timeout 15 ssh -O check -S "${SSH_CONTROL_PATH}" "${SSH_ALIAS}" \
  >/dev/null 2>&1 || fail "authoritative SSH master is not responsive"
remote() {
  timeout 240 ssh -n -T -o BatchMode=yes -o ControlMaster=no \
    -o ServerAliveInterval=15 -S "${SSH_CONTROL_PATH}" "${SSH_ALIAS}" "$@"
}
[[ "$(remote 'id -un')" == "${EXPECTED_SSH_USER}" ]] || \
  fail "remote identity differs"

# Reuse the normal preparation path so the repaired runner receives the same
# local/remote tests, production dependencies, immutable hashing and Slurm
# linting as a full submission.  It stages only; this script owns the exact
# one-index repair submission below.
prepare_output=$(SUBMIT=0 SSH_CONTROL_PATH="${SSH_CONTROL_PATH}" \
  bash MemNavData/prepare_final14_zero_depth_hpc.sh)
repair_root=$(printf '%s\n' "${prepare_output}" |
  awk -F= '$1=="SOURCE_ROOT"{value=$2} END{print value}')
[[ "${repair_root}" == \
  /scratch/yz11502/Research/Nav-axis-uturn-source-bundles/final14_zero_depth_* ]] || \
  fail "could not resolve repaired immutable bundle"
repair_receipt=${repair_root}/SOURCE_BUNDLE.sha256
receipt_sha=$(remote "sha256sum '${repair_receipt}' | cut -d ' ' -f 1" |
  tr -d '\r')
[[ "${receipt_sha}" =~ ^[0-9a-f]{64}$ ]] || fail "bad source receipt hash"
repair_record_root=${RUN_ROOT}/repair/port_pair_${receipt_sha:0:16}

remote "set -euo pipefail
test \"\$(id -un)\" = '${EXPECTED_SSH_USER}'
test \"\$(sacct -j '${ORIGINAL_ARRAY}_${FAILED_INDEX}' -X -n -o State | xargs)\" = FAILED
test \"\$(sacct -j '${ORIGINAL_ARRAY}_${FAILED_INDEX}' -X -n -o ExitCode | xargs)\" = 2:0
test \"\$(sacct -j '${ORIGINAL_ARRAY}_20' -X -n -o State | xargs)\" = COMPLETED
test \"\$(find '${RUN_ROOT}' -name completion.json -type f | wc -l)\" -eq 20
test -d '${ARCHIVE_ROOT}/${LABEL}'
test ! -e '${RUN_ROOT}/tasks/${LABEL}'
test ! -e '${RUN_ROOT}/evaluation/natural_direction/${LABEL}'
test ! -e '${RUN_ROOT}/POSTHOC'
test ! -e '${repair_record_root}'
test \"\$(sha256sum '${repair_receipt}' | cut -d ' ' -f 1)\" = '${receipt_sha}'
cd '${repair_root}' && sha256sum -c --quiet SOURCE_BUNDLE.sha256
ROOT='${repair_root}' bash '${repair_root}/MemNavData/test_slurm_port_pair.sh'"

common="ALL,REPAIR_ROOT=${repair_root},REPAIR_RECEIPT=${repair_receipt},EXPECTED_REPAIR_RECEIPT_SHA=${receipt_sha},BASE_SOURCE_ROOT=${BASE_SOURCE_ROOT},BASE_SOURCE_RECEIPT=${BASE_SOURCE_RECEIPT},EXPECTED_BASE_SOURCE_RECEIPT_SHA=${EXPECTED_BASE_SOURCE_RECEIPT_SHA},REFERENCE_ROOT=${REFERENCE_ROOT},BENCH_ROOT=${BENCH_ROOT},SOURCE_OVERLAY=${SOURCE_OVERLAY},EXPECTED_SOURCE_OVERLAY_BYTES=${EXPECTED_SOURCE_OVERLAY_BYTES}"
gpu_script=${repair_root}/MemNavData/slurm_final14_zero_depth.sbatch
analysis_script=${repair_root}/MemNavData/slurm_final14_zero_depth_analysis.sbatch
remote "source '${repair_root}/MemNavData/slurm_safe_submit.sh'; safe_sbatch --lint-fatal --test-only --qos=gpu48 --array=${FAILED_INDEX} --export='${common},RUN_ROOT=${RUN_ROOT},SMOKE=0,MAX_STEPS=600' '${gpu_script}' >/dev/null"
remote "source '${repair_root}/MemNavData/slurm_safe_submit.sh'; safe_sbatch --lint-fatal --test-only --partition=cpu_short --export='${common},RUN_ROOT=${RUN_ROOT}' '${analysis_script}' >/dev/null"

if [[ "${SUBMIT}" == 0 ]]; then
  printf 'PREPARED_ONLY=1\nSOURCE_ROOT=%s\nRUN_ROOT=%s\n' \
    "${repair_root}" "${RUN_ROOT}"
  exit 0
fi

repair_job=$(remote "source '${repair_root}/MemNavData/slurm_safe_submit.sh'; safe_sbatch --lint-fatal --parsable --qos=gpu48 --array=${FAILED_INDEX} --export='${common},RUN_ROOT=${RUN_ROOT},SMOKE=0,MAX_STEPS=600' '${gpu_script}'" | job_id)
[[ "${repair_job}" =~ ^[0-9]+$ ]] || fail "bad repair job id"
analysis_job=$(remote "source '${repair_root}/MemNavData/slurm_safe_submit.sh'; safe_sbatch --lint-fatal --parsable --partition=cpu_short --dependency=afterok:${repair_job} --kill-on-invalid-dep=yes --export='${common},RUN_ROOT=${RUN_ROOT}' '${analysis_script}'" | job_id)
[[ "${analysis_job}" =~ ^[0-9]+$ ]] || fail "bad analysis job id"

remote "set -euo pipefail
mkdir -p '${repair_record_root}'
python3 - '${repair_record_root}/submission_receipt.json' <<'PY'
import json
p={
 'schema_version':'final14_zero_depth_port_repair_submission_v1_20260828',
 'scope':'infrastructure-only exact repair; no scientific variable changed',
 'original_array':${ORIGINAL_ARRAY},
 'failed_array_index':${FAILED_INDEX},
 'failed_raw_job':${FAILED_RAW_JOB},
 'superseded_analysis_job':${SUPERSEDED_ANALYSIS},
 'archived_partial':'${ARCHIVE_ROOT}/${LABEL}',
 'source_bundle':'${repair_root}',
 'source_bundle_receipt_sha256':'${receipt_sha}',
 'repair_job':int('${repair_job}'),
 'replacement_analysis_job':int('${analysis_job}'),
 'navigation_outcomes_read_before_repair':False,
}
open('${repair_record_root}/submission_receipt.json','x').write(
 json.dumps(p,indent=2,sort_keys=True)+'\\n')
PY
sha256sum '${repair_record_root}/submission_receipt.json' >'${repair_record_root}/submission_receipt.json.sha256'
chmod -R a-w '${repair_record_root}'"

local_receipt=MemNavData/FINAL14_ZERO_DEPTH_PORT_REPAIR_SUBMISSION_20260828.json
[[ ! -e "${local_receipt}" ]] || fail "local repair receipt exists"
"${LOCAL_PY}" - "${local_receipt}" "${repair_root}" "${receipt_sha}" \
  "${repair_job}" "${analysis_job}" <<'PY'
import json,sys
path,bundle,digest,repair,analysis=sys.argv[1:]
p={
 "schema_version":"final14_zero_depth_port_repair_submission_v1_20260828",
 "scope":"infrastructure-only exact repair; no scientific variable changed",
 "original_array":16499701,"failed_array_index":19,
 "failed_raw_job":16501938,"superseded_analysis_job":16499709,
 "source_bundle":bundle,"source_bundle_receipt_sha256":digest,
 "repair_job":int(repair),"replacement_analysis_job":int(analysis),
 "navigation_outcomes_read_before_repair":False,
}
open(path,"x").write(json.dumps(p,indent=2,sort_keys=True)+"\n")
PY
printf 'REPAIR_JOB=%s\nANALYSIS_JOB=%s\nSOURCE_ROOT=%s\n' \
  "${repair_job}" "${analysis_job}" "${repair_root}"
