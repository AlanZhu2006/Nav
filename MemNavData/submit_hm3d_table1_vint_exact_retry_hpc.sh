#!/usr/bin/env bash
# Preserve one incomplete Table-1 ViNT cell, rerun only that frozen index, and
# repair the dependency of the not-yet-started aggregate without reading SR.
set -euo pipefail
umask 0022

ROOT=${ROOT:-$(git rev-parse --show-toplevel)}
REMOTE_HOST=${REMOTE_HOST:-alantorch}
SSH_CONTROL_PATH=${SSH_CONTROL_PATH:-$(
  ssh -G "${REMOTE_HOST}" 2>/dev/null |
    awk '$1=="controlpath"{value=$2} END{print value}'
)}
DRY_RUN=${DRY_RUN:-0}

TASK_ROOT=/scratch/yz11502/Research/Nav-axis-uturn-source-bundles/hm3d_table1_controller_portability_c0c373bf2ed63087
TASK_RECEIPT=${TASK_ROOT}/SOURCE_BUNDLE.sha256
EXPECTED_TASK_RECEIPT_SHA=c0c373bf2ed630873751e72769643c7d52ee0493f17a8a7bece381f9d52ff955
FORMAL_RUN_ROOT=/scratch/yz11502/Research/Nav-axis-uturn-results/hm3d_table1_controller_portability_20260829/formal_20260828T231109Z
BENCH_ROOT=/scratch/yz11502/Research/Nav-axis-uturn-results/hm3d_table1_fresh_query_reserve_20260829/construction_20260828T212552Z_bb757914/population/natural_direction
CONSTRUCTION_VERIFICATION=/scratch/yz11502/Research/Nav-axis-uturn-results/hm3d_table1_fresh_query_reserve_20260829/construction_20260828T212552Z_bb757914/hm3d_table1_fresh_query_verification.json
EXPECTED_CONSTRUCTION_VERIFICATION_SHA=2a7b8f86f61a6f55762640dcbaef4b975539ec3d93cfb06649bddd6fa4c96dc8
BASE_SOURCE_ROOT=/scratch/yz11502/Research/Nav-axis-uturn-source-bundles/certified_relocalization_closed_loop_d3bd281fc374cc80
BASE_SOURCE_RECEIPT_SHA=74001a9e0150c38c599a206fa0f4dd5e1279b9bed5d167119f4d14cb77995e98
DEPENDENCY_RECEIPT=/scratch/yz11502/Research/Nav-axis-uturn-results/shared_online_double_revisit_fresh_20260813/double_revisit_fresh40_20260813T200121Z/dependency_receipt.json
EXPECTED_DEPENDENCY_RECEIPT_SHA=4eb0ca6479a26f8e04f85a31d906cee4e68b1785f66cfd3ac23bf65424d36e5e
PORTABILITY_ENV_ROOT=/scratch/yz11502/Research/Nav-axis-uturn-envs/controller_portability_a9ec7146bce7_v1
PORTABILITY_CHECKPOINT_ROOT=/scratch/yz11502/Research/Nav-axis-uturn-checkpoints/controller_portability_50387aa89be8

ORIGINAL_ARRAY=16526731
FAILED_INDEX=18
FAILED_LABEL=018_b28CWbpQvor_episode_0001
VINT_AGGREGATE=16526745
VINT_VERIFY=16526759
NAVDP_VERIFY=16528385
JOINT_SEAL=16528391
REPAIR_TAG=vint_exact_retry1_20260829
REPAIR_ROOT=${FORMAL_RUN_ROOT}/repairs/${REPAIR_TAG}
PARTIAL_CELL=${FORMAL_RUN_ROOT}/formal/vint/evaluation/${FAILED_LABEL}/vint
ARCHIVE_ROOT=${REPAIR_ROOT}/failed_attempts/${FAILED_LABEL}
PAIR_SBATCH=${TASK_ROOT}/MemNavData/slurm_hm3d_table1_vint_pair.sbatch
ANALYSIS_SBATCH=${TASK_ROOT}/MemNavData/slurm_hm3d_table1_vint_analysis.sbatch
SEAL_SBATCH=${TASK_ROOT}/MemNavData/slurm_hm3d_table1_controller_seal.sbatch

fail() { echo "ABORT: $*" >&2; exit 2; }
remote() {
  timeout 300 ssh -n -tt -o BatchMode=yes -o ControlMaster=no \
    -S "${SSH_CONTROL_PATH}" "${REMOTE_HOST}" "$@" | tr -d '\r'
}
job_id() { awk -F';' 'NR==1{print $1}'; }

[[ "${DRY_RUN}" =~ ^[01]$ ]] || fail "DRY_RUN must be 0 or 1"
[[ -S "${SSH_CONTROL_PATH}" ]] || fail "authoritative SSH master missing"
[[ -f "${ROOT}/MemNavData/HM3D_TABLE1_VINT_EXACT_RETRY_PROTOCOL_20260829.md" ]] || \
  fail "repair protocol missing"
timeout 15 ssh -O check -S "${SSH_CONTROL_PATH}" "${REMOTE_HOST}" \
  >/dev/null 2>&1 || fail "authoritative SSH master is not responsive"

common="ALL,TASK_ROOT=${TASK_ROOT},TASK_RECEIPT=${TASK_RECEIPT},EXPECTED_TASK_RECEIPT_SHA=${EXPECTED_TASK_RECEIPT_SHA},FORMAL_RUN_ROOT=${FORMAL_RUN_ROOT},BENCH_ROOT=${BENCH_ROOT},CONSTRUCTION_VERIFICATION=${CONSTRUCTION_VERIFICATION},EXPECTED_CONSTRUCTION_VERIFICATION_SHA=${EXPECTED_CONSTRUCTION_VERIFICATION_SHA},BASE_SOURCE_ROOT=${BASE_SOURCE_ROOT},BASE_SOURCE_RECEIPT_SHA=${BASE_SOURCE_RECEIPT_SHA},DEPENDENCY_RECEIPT=${DEPENDENCY_RECEIPT},EXPECTED_DEPENDENCY_RECEIPT_SHA=${EXPECTED_DEPENDENCY_RECEIPT_SHA},PORTABILITY_ENV_ROOT=${PORTABILITY_ENV_ROOT},PORTABILITY_CHECKPOINT_ROOT=${PORTABILITY_CHECKPOINT_ROOT},PHASE=formal"

remote "set -euo pipefail
test \"\$(id -un)\" = yz11502
test \"\$(sha256sum '${TASK_RECEIPT}' | awk '{print \$1}')\" = '${EXPECTED_TASK_RECEIPT_SHA}'
cd '${TASK_ROOT}' && sha256sum -c --quiet '${TASK_RECEIPT}'
test \"\$(sha256sum '${CONSTRUCTION_VERIFICATION}' | awk '{print \$1}')\" = '${EXPECTED_CONSTRUCTION_VERIFICATION_SHA}'
test \"\$(sha256sum '${BASE_SOURCE_ROOT}/SOURCE_BUNDLE.sha256' | awk '{print \$1}')\" = '${BASE_SOURCE_RECEIPT_SHA}'
test \"\$(sha256sum '${DEPENDENCY_RECEIPT}' | awk '{print \$1}')\" = '${EXPECTED_DEPENDENCY_RECEIPT_SHA}'
test \"\$(sacct -X -j '${ORIGINAL_ARRAY}_${FAILED_INDEX}' -n -o State | xargs)\" = FAILED
test \"\$(sacct -X -j '${ORIGINAL_ARRAY}_${FAILED_INDEX}' -n -o ExitCode | xargs)\" = 6:0
grep -q 'Aborted.*core dumped' '/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/h3T1ViNT_${ORIGINAL_ARRAY}_${FAILED_INDEX}.err'
test -d '${PARTIAL_CELL}'
test ! -e '${PARTIAL_CELL}/controller_native_pair_audit.json'
test ! -e '${ARCHIVE_ROOT}'
test ! -e '${FORMAL_RUN_ROOT}/formal/vint/vint_table1_summary.json'
test ! -e '${FORMAL_RUN_ROOT}/formal/vint/vint_table1_independent_verification.json'
test ! -e '${FORMAL_RUN_ROOT}/hm3d_table1_controller_portability_receipt.json'
test \"\$(sacct -X -j '${VINT_AGGREGATE}' -n -o State | xargs)\" = PENDING
test \"\$(sacct -X -j '${VINT_VERIFY}' -n -o State | xargs)\" = PENDING
test \"\$(sacct -X -j '${JOINT_SEAL}' -n -o State | xargs)\" = PENDING
sbatch --test-only --job-name=h3T1ViNTR1 --partition=h100_tandon,a100_tandon --account=torch_pr_769_tandon_advanced --qos=gpu48 --gres=gpu:1 --cpus-per-task=12 --mem=96G --time=01:00:00 --array='${FAILED_INDEX}' --dependency=afterany:${ORIGINAL_ARRAY} --kill-on-invalid-dep=yes --export='${common}' '${PAIR_SBATCH}' >/dev/null
sbatch --test-only --dependency=afterok:${ORIGINAL_ARRAY} --kill-on-invalid-dep=yes --export='${common},MODE=aggregate' '${ANALYSIS_SBATCH}' >/dev/null
sbatch --test-only --export='${common}' '${SEAL_SBATCH}' >/dev/null"

if [[ "${DRY_RUN}" == 1 ]]; then
  echo "DRY_RUN_OK index=${FAILED_INDEX} partial=${PARTIAL_CELL} archive=${ARCHIVE_ROOT}"
  exit 0
fi

result=$(remote "set -euo pipefail
mkdir -p '${ARCHIVE_ROOT}'
mv '${PARTIAL_CELL}' '${ARCHIVE_ROOT}/vint'
(cd '${ARCHIVE_ROOT}' && find vint -type f -print0 | sort -z | xargs -0 sha256sum >partial_files.sha256)
chmod -R a-w '${ARCHIVE_ROOT}'
repair_raw=\$(sbatch --parsable --job-name=h3T1ViNTR1 --partition=h100_tandon,a100_tandon --account=torch_pr_769_tandon_advanced --qos=gpu48 --gres=gpu:1 --cpus-per-task=12 --mem=96G --time=01:00:00 --array='${FAILED_INDEX}' --dependency=afterany:${ORIGINAL_ARRAY} --kill-on-invalid-dep=yes --export='${common}' '${PAIR_SBATCH}')
repair_id=\${repair_raw%%;*}
[[ \"\${repair_id}\" =~ ^[0-9]+\$ ]]
aggregate_raw=\$(sbatch --parsable --job-name=h3T1ViNAnaR1 --dependency=afterok:\${repair_id} --kill-on-invalid-dep=yes --export='${common},MODE=aggregate' '${ANALYSIS_SBATCH}')
aggregate_id=\${aggregate_raw%%;*}
[[ \"\${aggregate_id}\" =~ ^[0-9]+\$ ]]
verify_raw=\$(sbatch --parsable --job-name=h3T1ViNVerR1 --dependency=afterok:\${aggregate_id} --kill-on-invalid-dep=yes --export='${common},MODE=verify' '${ANALYSIS_SBATCH}')
verify_id=\${verify_raw%%;*}
[[ \"\${verify_id}\" =~ ^[0-9]+\$ ]]
seal_raw=\$(sbatch --parsable --job-name=h3T1SealR1 --dependency=afterok:${NAVDP_VERIFY}:\${verify_id} --kill-on-invalid-dep=yes --export='${common}' '${SEAL_SBATCH}')
seal_id=\${seal_raw%%;*}
[[ \"\${seal_id}\" =~ ^[0-9]+\$ ]]
scancel '${VINT_AGGREGATE}' '${VINT_VERIFY}' '${JOINT_SEAL}'
mkdir -p '${REPAIR_ROOT}'
/scratch/lg154/conda-envs/memnav/bin/python - \"\${repair_id}\" \"\${aggregate_id}\" \"\${verify_id}\" \"\${seal_id}\" '${REPAIR_ROOT}/repair_submission.json' <<'PY'
import json, os, sys
repair_id, aggregate_id, verify_id, seal_id, path = sys.argv[1:]
payload = {
    'schema_version': 'hm3d_table1_vint_exact_retry_submission_v1_20260829',
    'failure_class': 'habitat_native_sigabrt_before_pair_audit',
    'original_array': 16526731,
    'failed_indices': [18],
    'repair_job': int(repair_id),
    'superseded_vint_aggregate': 16526745,
    'superseded_vint_verify': 16526759,
    'retained_navdp_verify': 16528385,
    'superseded_joint_seal': 16528391,
    'replacement_vint_aggregate': int(aggregate_id),
    'replacement_vint_verify': int(verify_id),
    'replacement_joint_seal': int(seal_id),
    'partial_policy_outcomes_read_before_repair': False,
    'method_or_population_changed': False,
    'archive_root': '${ARCHIVE_ROOT}',
    'task_bundle': '${TASK_ROOT}',
    'task_receipt_sha256': '${EXPECTED_TASK_RECEIPT_SHA}',
}
fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
with os.fdopen(fd, 'w') as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write('\\n')
PY
sha256sum '${REPAIR_ROOT}/repair_submission.json' >'${REPAIR_ROOT}/repair_submission.json.sha256'
chmod a-w '${REPAIR_ROOT}/repair_submission.json' '${REPAIR_ROOT}/repair_submission.json.sha256'
printf 'REPAIR=%s AGGREGATE=%s VERIFY=%s SEAL=%s ARCHIVE=%s\\n' \"\${repair_id}\" \"\${aggregate_id}\" \"\${verify_id}\" \"\${seal_id}\" '${ARCHIVE_ROOT}'")

printf '%s\n' "${result}"
