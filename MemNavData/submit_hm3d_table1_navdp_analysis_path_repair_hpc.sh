#!/usr/bin/env bash
# Repair analysis only; never resubmit a policy rollout.
set -euo pipefail
umask 0022

ROOT=${ROOT:-/home/asus/Research/Nav-graph-blind}
SSH_ALIAS=${SSH_ALIAS:-alantorch}
LOCAL_PY=/home/asus/miniconda3/envs/memnav/bin/python
REMOTE_BUNDLES=/scratch/yz11502/Research/Nav-axis-uturn-source-bundles
FORMAL_RUN_ROOT=/scratch/yz11502/Research/Nav-axis-uturn-results/hm3d_table1_controller_portability_20260829/formal_20260828T231109Z
CONSTRUCTION_RUN=/scratch/yz11502/Research/Nav-axis-uturn-results/hm3d_table1_fresh_query_reserve_20260829/construction_20260828T212552Z_bb757914
BENCH_ROOT=${CONSTRUCTION_RUN}/population/natural_direction
CONSTRUCTION_VERIFICATION=${CONSTRUCTION_RUN}/hm3d_table1_fresh_query_verification.json
EXPECTED_CONSTRUCTION_VERIFICATION_SHA=2a7b8f86f61a6f55762640dcbaef4b975539ec3d93cfb06649bddd6fa4c96dc8
ORIGINAL_AGGREGATE_JOB=16541367
RETAINED_VINT_VERIFY_JOB=16540208
SSH_CONTROL_PATH=${SSH_CONTROL_PATH:-$(ssh -G "${SSH_ALIAS}" 2>/dev/null | awk '$1=="controlpath"{value=$2} END{print value}')}
cd "${ROOT}"

fail() { echo "ABORT: $*" >&2; exit 2; }
remote() {
  timeout 180 ssh -n -tt -o BatchMode=yes -o ControlMaster=no \
    -S "${SSH_CONTROL_PATH}" "${SSH_ALIAS}" "$@"
}
[[ -S "${SSH_CONTROL_PATH}" ]] || fail "authoritative SSH master missing"

files=(
  MemNavData/HM3D_TABLE1_NAVDP_ANALYSIS_PATH_REPAIR_20260829.md
  MemNavData/aggregate_hm3d_table1_navdp_pair.py
  MemNavData/independent_verify_hm3d_table1_navdp_pair.py
  MemNavData/hm3d_fullmono_mixed_role.py
  MemNavData/final14_mono_factorial.py
  MemNavData/mdtec_raw_depth_gate_d.py
  MemNavData/slurm_hm3d_table1_navdp_analysis.sbatch
  MemNavData/slurm_hm3d_table1_controller_seal.sbatch
  MemNavData/test_hm3d_table1_navdp_pair.py
  MemNavData/submit_hm3d_table1_navdp_analysis_path_repair_hpc.sh
)
for path in "${files[@]}"; do
  [[ -f "${path}" && ! -L "${path}" ]] || fail "missing physical ${path}"
done
PYTHONPATH="${ROOT}" "${LOCAL_PY}" -m pytest -q \
  MemNavData/test_hm3d_table1_navdp_pair.py
PYTHONPATH="${ROOT}" "${LOCAL_PY}" -m py_compile \
  MemNavData/aggregate_hm3d_table1_navdp_pair.py \
  MemNavData/independent_verify_hm3d_table1_navdp_pair.py
bash -n MemNavData/slurm_hm3d_table1_navdp_analysis.sbatch \
  MemNavData/slurm_hm3d_table1_controller_seal.sbatch \
  MemNavData/submit_hm3d_table1_navdp_analysis_path_repair_hpc.sh

staging=$(mktemp -d)
cleanup() { rm -rf -- "${staging}"; }
trap cleanup EXIT
for path in "${files[@]}"; do
  mkdir -p "${staging}/$(dirname "${path}")"
  cp --preserve=mode,timestamps "${path}" "${staging}/${path}"
done
(
  cd "${staging}"
  find . -type f ! -name SOURCE_BUNDLE.sha256 -print0 | sort -z | \
    xargs -0 sha256sum >SOURCE_BUNDLE.sha256
  sha256sum -c SOURCE_BUNDLE.sha256 >/dev/null
)
(cd /tmp && PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="${staging}" \
  "${LOCAL_PY}" -c \
  'import MemNavData.aggregate_hm3d_table1_navdp_pair; import MemNavData.independent_verify_hm3d_table1_navdp_pair')
receipt_sha=$(sha256sum "${staging}/SOURCE_BUNDLE.sha256" | awk '{print $1}')
bundle=${REMOTE_BUNDLES}/hm3d_table1_navdp_analysis_path_repair_${receipt_sha:0:16}
stage=${bundle}.partial.$$

remote_identity=$(remote 'id -un' | tr -d '\r')
[[ "${remote_identity}" == yz11502 ]] || fail "wrong remote identity"
remote "set -euo pipefail
test \"\$(sha256sum '${CONSTRUCTION_VERIFICATION}' | awk '{print \$1}')\" = '${EXPECTED_CONSTRUCTION_VERIFICATION_SHA}'
test -r '${BENCH_ROOT}/manifest.json'
test -d '${FORMAL_RUN_ROOT}/formal/navdp/evaluation'
test -d '${FORMAL_RUN_ROOT}/formal/vint/evaluation'"
if remote "test -d '${bundle}'"; then
  remote "test \"\$(sha256sum '${bundle}/SOURCE_BUNDLE.sha256' | awk '{print \$1}')\" = '${receipt_sha}' && cd '${bundle}' && sha256sum -c --quiet SOURCE_BUNDLE.sha256"
else
  remote "test ! -e '${stage}' && mkdir -p '${stage}'"
  rsync -a --chmod=Du=rwx,Dgo=rx,Fu=rw,Fgo=r \
    -e "ssh -o BatchMode=yes -o ControlMaster=no -S ${SSH_CONTROL_PATH}" \
    "${staging}/" "${SSH_ALIAS}:${stage}/"
  remote "cd '${stage}' && sha256sum -c --quiet SOURCE_BUNDLE.sha256 && chmod -R a-w '${stage}' && mv '${stage}' '${bundle}'"
fi
remote "cd /tmp && env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH='${bundle}' /scratch/lg154/conda-envs/memnav/bin/python -c 'import MemNavData.aggregate_hm3d_table1_navdp_pair; import MemNavData.independent_verify_hm3d_table1_navdp_pair'"

task_receipt=${bundle}/SOURCE_BUNDLE.sha256
common="ALL,TASK_ROOT=${bundle},TASK_RECEIPT=${task_receipt},EXPECTED_TASK_RECEIPT_SHA=${receipt_sha},FORMAL_RUN_ROOT=${FORMAL_RUN_ROOT},BENCH_ROOT=${BENCH_ROOT},CONSTRUCTION_VERIFICATION=${CONSTRUCTION_VERIFICATION},EXPECTED_CONSTRUCTION_VERIFICATION_SHA=${EXPECTED_CONSTRUCTION_VERIFICATION_SHA}"
analysis=${bundle}/MemNavData/slurm_hm3d_table1_navdp_analysis.sbatch
seal=${bundle}/MemNavData/slurm_hm3d_table1_controller_seal.sbatch
remote "sbatch --test-only --export='${common},MODE=aggregate' '${analysis}' >/dev/null"
remote "sbatch --test-only --export='${common},MODE=verify' '${analysis}' >/dev/null"
remote "sbatch --test-only --export='${common}' '${seal}' >/dev/null"

aggregate_raw=$(remote "sbatch --parsable --job-name=h3T1NavAnaFix --dependency=afterany:${ORIGINAL_AGGREGATE_JOB} --kill-on-invalid-dep=yes --export='${common},MODE=aggregate' '${analysis}'" | tr -d '\r')
aggregate_id=${aggregate_raw%%;*}
verify_raw=$(remote "sbatch --parsable --job-name=h3T1NavVerFix --dependency=afterok:${aggregate_id} --kill-on-invalid-dep=yes --export='${common},MODE=verify' '${analysis}'" | tr -d '\r')
verify_id=${verify_raw%%;*}
seal_raw=$(remote "sbatch --parsable --job-name=h3T1SealFix --dependency=afterok:${verify_id}:${RETAINED_VINT_VERIFY_JOB} --kill-on-invalid-dep=yes --export='${common}' '${seal}'" | tr -d '\r')
seal_id=${seal_raw%%;*}
for id in "${aggregate_id}" "${verify_id}" "${seal_id}"; do
  [[ "${id}" =~ ^[0-9]+$ ]] || fail "invalid job id ${id}"
done

local_receipt=MemNavData/HM3D_TABLE1_NAVDP_ANALYSIS_PATH_REPAIR_SUBMISSION_20260829.json
[[ ! -e "${local_receipt}" ]] || fail "local receipt exists"
"${LOCAL_PY}" - "${local_receipt}" "${bundle}" "${receipt_sha}" \
  "${aggregate_id}" "${verify_id}" "${seal_id}" <<'PY'
import json,sys
path,bundle,sha,aggregate,verify,seal=sys.argv[1:]
payload={
 'schema_version':'hm3d_table1_navdp_analysis_path_repair_submission_v1_20260829',
 'failure_class':'aggregator_read_noncanonical_history_level_direction_stratum',
 'repair_scope':'analysis_only_same_raw_rollouts',
 'task_bundle':bundle,'task_receipt_sha256':sha,
 'partial_policy_outcomes_read_before_repair':False,
 'jobs':{'replacement_navdp_aggregate':int(aggregate),
         'replacement_navdp_verify':int(verify),
         'replacement_joint_seal':int(seal)},
 'scientific_guards':{'policy_rollout_resubmitted':False,
                      'population_changed':False,'statistics_changed':False},
}
open(path,'x').write(json.dumps(payload,indent=2,sort_keys=True)+'\n')
print(json.dumps(payload,indent=2,sort_keys=True))
PY
scp -q -o BatchMode=yes -o ControlMaster=no -o ControlPath="${SSH_CONTROL_PATH}" \
  "${ROOT}/${local_receipt}" "${SSH_ALIAS}:${FORMAL_RUN_ROOT}/navdp_analysis_path_repair_submission.json"
remote "sha256sum '${FORMAL_RUN_ROOT}/navdp_analysis_path_repair_submission.json' >'${FORMAL_RUN_ROOT}/navdp_analysis_path_repair_submission.json.sha256' && chmod a-w '${FORMAL_RUN_ROOT}/navdp_analysis_path_repair_submission.json' '${FORMAL_RUN_ROOT}/navdp_analysis_path_repair_submission.json.sha256'"
printf 'BUNDLE=%s\nAGGREGATE=%s\nVERIFY=%s\nSEAL=%s\n' \
  "${bundle}" "${aggregate_id}" "${verify_id}" "${seal_id}"
