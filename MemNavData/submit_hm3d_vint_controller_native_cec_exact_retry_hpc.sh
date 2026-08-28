#!/usr/bin/env bash
# Preserve the two SIGABRT partial cells from formal array 16482393, rerun only
# indices 23 and 27 with the original immutable bundle, then rebuild the
# formal aggregate and independent verification.
set -euo pipefail
umask 0022

ROOT=${ROOT:-$(git rev-parse --show-toplevel)}
REMOTE_HOST=${REMOTE_HOST:-alantorch}
SSH_CONTROL_PATH=${SSH_CONTROL_PATH:-$(
  ssh -G "${REMOTE_HOST}" 2>/dev/null |
    awk '$1=="controlpath"{value=$2} END{print value}'
)}
DRY_RUN=${DRY_RUN:-0}

SOURCE_ROOT=/scratch/yz11502/Research/Nav-axis-uturn-source-bundles/hm3d_vint_cec_3c8da4454ad11c64
SOURCE_RECEIPT=${SOURCE_ROOT}/SOURCE_BUNDLE.sha256
EXPECTED_SOURCE_RECEIPT_SHA=735f6e39012bfa5bd02c1ddfcbaa8c0a2e17d0369f892260c1e3709fd16796a9
RUN_ROOT=/scratch/yz11502/Research/Nav-axis-uturn-results/hm3d_vint_controller_native_cec_20260828/hm3d_vint_cec_table1_20260828
FRESH_ROOT=/scratch/yz11502/Research/Nav-axis-uturn-results/hm3d_fresh_fullmono_mixed_role_20260820/formal_20260820T143609Z_e6dd44c6
BASE_SOURCE_ROOT=/scratch/yz11502/Research/Nav-axis-uturn-source-bundles/certified_relocalization_closed_loop_d3bd281fc374cc80
BASE_SOURCE_RECEIPT_SHA=74001a9e0150c38c599a206fa0f4dd5e1279b9bed5d167119f4d14cb77995e98
DEPENDENCY_RECEIPT=/scratch/yz11502/Research/Nav-axis-uturn-results/shared_online_double_revisit_fresh_20260813/double_revisit_fresh40_20260813T200121Z/dependency_receipt.json
EXPECTED_DEPENDENCY_RECEIPT_SHA=4eb0ca6479a26f8e04f85a31d906cee4e68b1785f66cfd3ac23bf65424d36e5e
PORTABILITY_ENV_ROOT=/scratch/yz11502/Research/Nav-axis-uturn-envs/controller_portability_a9ec7146bce7_v1
PORTABILITY_CHECKPOINT_ROOT=/scratch/yz11502/Research/Nav-axis-uturn-checkpoints/controller_portability_50387aa89be8
ORIGINAL_ARRAY=16482393
SUPERSEDED_AGGREGATE=16482395
SUPERSEDED_VERIFY=16482397
REPAIR_INDICES=23,27
REPAIR_TAG=vint_cec_exact_retry1_20260828
ARCHIVE_ROOT=${RUN_ROOT}/formal/failed_attempts/${REPAIR_TAG}
REMOTE_REPAIR_ROOT=${RUN_ROOT}/formal/repair/${REPAIR_TAG}
LOCAL_PROTOCOL=${ROOT}/MemNavData/HM3D_VINT_CONTROLLER_NATIVE_CEC_EXACT_RETRY_PROTOCOL_20260828.md

fail() { echo "ABORT: $*" >&2; exit 2; }
remote() {
  timeout 300 ssh -n -T -o BatchMode=yes -o ControlMaster=no \
    -S "${SSH_CONTROL_PATH}" "${REMOTE_HOST}" "$@" | tr -d '\r'
}

[[ "${DRY_RUN}" =~ ^[01]$ ]] || fail "DRY_RUN must be 0 or 1"
[[ -S "${SSH_CONTROL_PATH}" ]] || fail "authoritative SSH master missing"
[[ -f "${LOCAL_PROTOCOL}" ]] || fail "repair protocol missing"
timeout 15 ssh -O check -S "${SSH_CONTROL_PATH}" "${REMOTE_HOST}" \
  >/dev/null 2>&1 || fail "authoritative SSH master is not responsive"

PAIR_SBATCH=${SOURCE_ROOT}/MemNavData/slurm_hm3d_vint_controller_native_pair.sbatch
ANALYSIS_SBATCH=${SOURCE_ROOT}/MemNavData/slurm_hm3d_vint_controller_native_analysis.sbatch
common="ALL,SOURCE_ROOT=${SOURCE_ROOT},SOURCE_RECEIPT=${SOURCE_RECEIPT},EXPECTED_SOURCE_RECEIPT_SHA=${EXPECTED_SOURCE_RECEIPT_SHA},FRESH_ROOT=${FRESH_ROOT},RUN_ROOT=${RUN_ROOT},BASE_SOURCE_ROOT=${BASE_SOURCE_ROOT},BASE_SOURCE_RECEIPT_SHA=${BASE_SOURCE_RECEIPT_SHA},DEPENDENCY_RECEIPT=${DEPENDENCY_RECEIPT},EXPECTED_DEPENDENCY_RECEIPT_SHA=${EXPECTED_DEPENDENCY_RECEIPT_SHA},PORTABILITY_ENV_ROOT=${PORTABILITY_ENV_ROOT},PORTABILITY_CHECKPOINT_ROOT=${PORTABILITY_CHECKPOINT_ROOT}"

remote "set -euo pipefail
test \"\$(id -un)\" = yz11502
test \"\$(sha256sum '${SOURCE_RECEIPT}' | awk '{print \$1}')\" = '${EXPECTED_SOURCE_RECEIPT_SHA}'
cd '${SOURCE_ROOT}' && sha256sum -c --quiet '${SOURCE_RECEIPT}'
test \"\$(sha256sum '${BASE_SOURCE_ROOT}/SOURCE_BUNDLE.sha256' | awk '{print \$1}')\" = '${BASE_SOURCE_RECEIPT_SHA}'
test \"\$(sha256sum '${DEPENDENCY_RECEIPT}' | awk '{print \$1}')\" = '${EXPECTED_DEPENDENCY_RECEIPT_SHA}'
test \"\$(sha256sum '${FRESH_ROOT}/benchmarks/natural_direction/manifest.json' | awk '{print \$1}')\" = 'aada40d25d01e9385df3ffdcaf37847f471b63c7be785a704eade961346a50b0'
test \"\$(find '${RUN_ROOT}/formal/evaluation' -path '*/vint/controller_native_pair_audit.json' -type f | wc -l)\" -eq 26
test ! -e '${RUN_ROOT}/formal/formal_summary.json'
test ! -e '${RUN_ROOT}/formal/formal_independent_verification.json'
test -d '${RUN_ROOT}/formal/evaluation/023_LEFTm3JecaC_episode_0001/vint'
test ! -e '${RUN_ROOT}/formal/evaluation/023_LEFTm3JecaC_episode_0001/vint/controller_native_pair_audit.json'
test -d '${RUN_ROOT}/formal/evaluation/027_58NLZxWBSpk_episode_0001/vint'
test ! -e '${RUN_ROOT}/formal/evaluation/027_58NLZxWBSpk_episode_0001/vint/controller_native_pair_audit.json'
test ! -e '${ARCHIVE_ROOT}'
test ! -e '${REMOTE_REPAIR_ROOT}'
test \"\$(sacct -j '${ORIGINAL_ARRAY}_23' -X -n -o State | xargs)\" = FAILED
test \"\$(sacct -j '${ORIGINAL_ARRAY}_23' -X -n -o ExitCode | xargs)\" = 6:0
test \"\$(sacct -j '${ORIGINAL_ARRAY}_27' -X -n -o State | xargs)\" = FAILED
test \"\$(sacct -j '${ORIGINAL_ARRAY}_27' -X -n -o ExitCode | xargs)\" = 6:0
test \"\$(sacct -j '${SUPERSEDED_AGGREGATE}' -X -n -o State | xargs)\" = CANCELLED
test \"\$(sacct -j '${SUPERSEDED_VERIFY}' -X -n -o State | xargs)\" = CANCELLED
test -z \"\$(squeue -j '${ORIGINAL_ARRAY}' -h 2>/dev/null || true)\"
sbatch --test-only --partition=h100_tandon,a100_tandon --account=torch_pr_769_tandon_advanced --qos=gpu48 --gres=gpu:1 --time=01:00:00 --array='${REPAIR_INDICES}%2' --export='${common},PHASE=formal' '${PAIR_SBATCH}' >/dev/null
sbatch --test-only --partition=cpu_short --account=torch_pr_769_tandon_advanced --time=00:20:00 --export='${common},MODE=formal_aggregate' '${ANALYSIS_SBATCH}' >/dev/null
sbatch --test-only --partition=cpu_short --account=torch_pr_769_tandon_advanced --time=00:20:00 --export='${common},MODE=formal_verify' '${ANALYSIS_SBATCH}' >/dev/null
if [[ '${DRY_RUN}' == 1 ]]; then
  echo 'DRY_RUN_OK indices=${REPAIR_INDICES} archive=${ARCHIVE_ROOT}'
  exit 0
fi
mkdir -p '${ARCHIVE_ROOT}/023_LEFTm3JecaC_episode_0001' '${ARCHIVE_ROOT}/027_58NLZxWBSpk_episode_0001' '${REMOTE_REPAIR_ROOT}'
mv '${RUN_ROOT}/formal/evaluation/023_LEFTm3JecaC_episode_0001/vint' '${ARCHIVE_ROOT}/023_LEFTm3JecaC_episode_0001/vint'
mv '${RUN_ROOT}/formal/evaluation/027_58NLZxWBSpk_episode_0001/vint' '${ARCHIVE_ROOT}/027_58NLZxWBSpk_episode_0001/vint'
chmod -R a-w '${ARCHIVE_ROOT}'
repair_raw=\$(sbatch --parsable --job-name=vintCECpairR1 --partition=h100_tandon,a100_tandon --account=torch_pr_769_tandon_advanced --qos=gpu48 --gres=gpu:1 --time=01:00:00 --array='${REPAIR_INDICES}%2' --export='${common},PHASE=formal' '${PAIR_SBATCH}')
repair_id=\${repair_raw%%;*}
[[ \"\${repair_id}\" =~ ^[0-9]+\$ ]]
aggregate_raw=\$(sbatch --parsable --job-name=vintCECsumR1 --partition=cpu_short --account=torch_pr_769_tandon_advanced --time=00:20:00 --dependency=afterok:\${repair_id} --kill-on-invalid-dep=yes --export='${common},MODE=formal_aggregate' '${ANALYSIS_SBATCH}')
aggregate_id=\${aggregate_raw%%;*}
[[ \"\${aggregate_id}\" =~ ^[0-9]+\$ ]]
verify_raw=\$(sbatch --parsable --job-name=vintCECverR1 --partition=cpu_short --account=torch_pr_769_tandon_advanced --time=00:20:00 --dependency=afterok:\${aggregate_id} --kill-on-invalid-dep=yes --export='${common},MODE=formal_verify' '${ANALYSIS_SBATCH}')
verify_id=\${verify_raw%%;*}
[[ \"\${verify_id}\" =~ ^[0-9]+\$ ]]
printf 'REPAIR=%s AGGREGATE=%s VERIFY=%s ARCHIVE=%s REPAIR_ROOT=%s\\n' \"\${repair_id}\" \"\${aggregate_id}\" \"\${verify_id}\" '${ARCHIVE_ROOT}' '${REMOTE_REPAIR_ROOT}'"
