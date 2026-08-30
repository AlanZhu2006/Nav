#!/usr/bin/env bash
# Re-run only Table-2 construction finalization and independent verification
# after removing an accidental simulator dependency from the CPU finalizer.
set -euo pipefail
umask 0022

ROOT=${ROOT:-/home/asus/Research/Nav-graph-blind}
SSH_ALIAS=${SSH_ALIAS:-alantorch}
LOCAL_MEMNAV_PY=/home/asus/miniconda3/envs/memnav/bin/python
REMOTE_BUNDLES=/scratch/yz11502/Research/Nav-axis-uturn-source-bundles
RUN_ROOT=/scratch/yz11502/Research/Nav-axis-uturn-results/hm3d_table2_leg3_mixed_role_20260829/construction_repair_20260829T064841Z_8e909a5b
SOURCE_RUN_ROOT=/scratch/yz11502/Research/Nav-axis-uturn-results/hm3d_fullmono_lifelong_natural_v4_20260827/formal_materialize_20260827T133704Z_d85fc50d
OLD_TASK_ROOT=/scratch/yz11502/Research/Nav-axis-uturn-source-bundles/hm3d_table2_leg3_repair_8e909a5ba81c146d
OLD_TASK_RECEIPT=${OLD_TASK_ROOT}/SOURCE_BUNDLE.sha256
EXPECTED_OLD_TASK_RECEIPT_SHA=8e909a5ba81c146dd2fe7c9e179fee553672608ee78c08ab4cbfc5b8be23baf1
EXPECTED_PROTOCOL_SHA=39b1ae703ac2984c3e9d6e20561347f918d9d5af2526f6ea4d3d9ae6699333c7
REPAIR_DOC=${REPAIR_DOC:-MemNavData/HM3D_TABLE2_LEG3_FINALIZER_IMPORT_REPAIR_20260829.md}
REPAIR_JSON=${REPAIR_JSON:-MemNavData/hm3d_table2_leg3_finalizer_import_repair_20260829.json}
BUNDLE_LABEL=${BUNDLE_LABEL:-hm3d_table2_leg3_finalize_repair}
SUBMISSION_RECEIPT=${SUBMISSION_RECEIPT:-MemNavData/HM3D_TABLE2_LEG3_FINALIZER_REPAIR_SUBMISSION_20260829.json}
SUBMISSION_SCHEMA=${SUBMISSION_SCHEMA:-hm3d_table2_leg3_finalizer_repair_submission_v1_20260829}
SUBMISSION_SCOPE=${SUBMISSION_SCOPE:-reuse immutable fragments and rerun finalizer plus verifier only}
SUBMISSION_DRIVER=${SUBMISSION_DRIVER:-MemNavData/submit_hm3d_table2_leg3_finalizer_repair_hpc.sh}
SSH_CONTROL_PATH=${SSH_CONTROL_PATH:-$(ssh -G "${SSH_ALIAS}" 2>/dev/null | awk '$1=="controlpath"{value=$2} END{print value}')}
cd "${ROOT}"

fail() { echo "ABORT: $*" >&2; exit 2; }
remote() {
  timeout 180 ssh -n -tt -o BatchMode=yes -o ControlMaster=no \
    -S "${SSH_CONTROL_PATH}" "${SSH_ALIAS}" "$@"
}
[[ -S "${SSH_CONTROL_PATH}" ]] || fail "authoritative SSH master missing"

files=(
  "${REPAIR_DOC}"
  "${REPAIR_JSON}"
  MemNavData/hm3d_table2_leg3_mixed_role_protocol_20260829.json
  MemNavData/hm3d_table2_leg3_mixed_role.py
  MemNavData/final14_role_pair_contract.py
  MemNavData/finalize_hm3d_table2_leg3_mixed_role.py
  MemNavData/independent_verify_hm3d_table2_leg3_mixed_role.py
  MemNavData/shared_online_role_pair_contract.py
  MemNavData/deterministic_eval_protocol.py
  MemNavData/slurm_hm3d_table2_leg3_analysis.sbatch
  MemNavData/test_hm3d_table2_leg3_mixed_role.py
  MemNavData/submit_hm3d_table2_leg3_finalizer_repair_hpc.sh
  "${SUBMISSION_DRIVER}"
)
for file in "${files[@]}"; do
  [[ -f "${file}" && ! -L "${file}" ]] || fail "missing physical ${file}"
done
[[ "$(sha256sum MemNavData/hm3d_table2_leg3_mixed_role_protocol_20260829.json | awk '{print $1}')" == "${EXPECTED_PROTOCOL_SHA}" ]] || fail "protocol changed"
export PYTHONPATH=${ROOT}:${ROOT}/MemNavData${PYTHONPATH:+:${PYTHONPATH}}
"${LOCAL_MEMNAV_PY}" -m pytest -q \
  MemNavData/test_hm3d_table2_leg3_mixed_role.py
"${LOCAL_MEMNAV_PY}" -m py_compile \
  MemNavData/finalize_hm3d_table2_leg3_mixed_role.py \
  MemNavData/independent_verify_hm3d_table2_leg3_mixed_role.py
"${LOCAL_MEMNAV_PY}" - <<'PY'
import sys
import finalize_hm3d_table2_leg3_mixed_role
import independent_verify_hm3d_table2_leg3_mixed_role
assert "build_final14_role_pair_scene" not in sys.modules
assert "quaternion" not in sys.modules
PY
bash -n \
  MemNavData/slurm_hm3d_table2_leg3_analysis.sbatch \
  MemNavData/submit_hm3d_table2_leg3_finalizer_repair_hpc.sh

staging=$(mktemp -d)
cleanup() { rm -rf -- "${staging}"; }
trap cleanup EXIT
for file in "${files[@]}"; do
  mkdir -p "${staging}/$(dirname "${file}")"
  cp --preserve=mode,timestamps "${file}" "${staging}/${file}"
done
(
  cd "${staging}"
  find . -type f ! -name SOURCE_BUNDLE.sha256 -print0 | sort -z | \
    xargs -0 sha256sum >SOURCE_BUNDLE.sha256
  sha256sum -c SOURCE_BUNDLE.sha256 >/dev/null
)
task_receipt_sha=$(sha256sum "${staging}/SOURCE_BUNDLE.sha256" | awk '{print $1}')
bundle_key=${task_receipt_sha:0:16}
task_root=${REMOTE_BUNDLES}/${BUNDLE_LABEL}_${bundle_key}
task_stage=${task_root}.partial.$$

remote_identity=$(remote 'id -un' | tr -d '\r')
[[ "${remote_identity}" == yz11502 ]] || fail "wrong remote identity"
remote "set -euo pipefail
test \"\$(sha256sum '${OLD_TASK_RECEIPT}' | awk '{print \$1}')\" = '${EXPECTED_OLD_TASK_RECEIPT_SHA}'
cd '${OLD_TASK_ROOT}' && sha256sum -c --quiet '${OLD_TASK_RECEIPT}'
test \"\$(find '${RUN_ROOT}/construction/fragments' -mindepth 2 -maxdepth 2 -name completion.json -type f | wc -l)\" -eq 22
test ! -e '${RUN_ROOT}/population'
test ! -e '${RUN_ROOT}/hm3d_table2_leg3_construction_verification.json'
test \"\$(find '${RUN_ROOT}' -maxdepth 1 -type d -name 'population.tmp.*' | wc -l)\" -eq 0"

if remote "test -d '${task_root}'"; then
  remote "test \"\$(sha256sum '${task_root}/SOURCE_BUNDLE.sha256' | awk '{print \$1}')\" = '${task_receipt_sha}' && cd '${task_root}' && sha256sum -c --quiet SOURCE_BUNDLE.sha256"
else
  remote "test ! -e '${task_stage}' && mkdir -p '${task_stage}'"
  timeout 180 rsync -a --timeout=60 --chmod=Du=rwx,Dgo=rx,Fu=rw,Fgo=r \
    -e "ssh -o BatchMode=yes -o ControlMaster=no -S ${SSH_CONTROL_PATH}" \
    "${staging}/" "${SSH_ALIAS}:${task_stage}/"
  remote "cd '${task_stage}' && sha256sum -c --quiet SOURCE_BUNDLE.sha256 && chmod -R a-w '${task_stage}' && mv '${task_stage}' '${task_root}'"
fi

task_receipt=${task_root}/SOURCE_BUNDLE.sha256
protocol=${task_root}/MemNavData/hm3d_table2_leg3_mixed_role_protocol_20260829.json
analysis=${task_root}/MemNavData/slurm_hm3d_table2_leg3_analysis.sbatch
remote "test \"\$(sha256sum '${protocol}' | awk '{print \$1}')\" = '${EXPECTED_PROTOCOL_SHA}'"
remote "env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH='${task_root}:${task_root}/MemNavData' /scratch/lg154/conda-envs/memnav/bin/python -c 'import sys,finalize_hm3d_table2_leg3_mixed_role,independent_verify_hm3d_table2_leg3_mixed_role; assert \"build_final14_role_pair_scene\" not in sys.modules; assert \"quaternion\" not in sys.modules'"
remote "env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH='${task_root}:${task_root}/MemNavData' /scratch/lg154/conda-envs/memnav/bin/python -m pytest -q '${task_root}/MemNavData/test_hm3d_table2_leg3_mixed_role.py'"

common="ALL,TASK_ROOT=${task_root},TASK_RECEIPT=${task_receipt},EXPECTED_TASK_RECEIPT_SHA=${task_receipt_sha},SOURCE_RUN_ROOT=${SOURCE_RUN_ROOT},RUN_ROOT=${RUN_ROOT},PROTOCOL=${protocol}"
remote "sbatch --test-only --partition=cpu_short --account=torch_pr_769_tandon_advanced --time=00:45:00 --export='${common},MODE=finalize' '${analysis}' >/dev/null"
remote "sbatch --test-only --partition=cpu_short --account=torch_pr_769_tandon_advanced --time=00:45:00 --export='${common},MODE=verify' '${analysis}' >/dev/null"

finalize_raw=$(remote "sbatch --parsable --job-name=h3T2L3fin3 --partition=cpu_short --account=torch_pr_769_tandon_advanced --time=00:45:00 --export='${common},MODE=finalize' '${analysis}'" | tr -d '\r')
finalize_id=${finalize_raw%%;*}; [[ "${finalize_id}" =~ ^[0-9]+$ ]] || fail "bad finalizer job"
verify_raw=$(remote "sbatch --parsable --job-name=h3T2L3ver3 --partition=cpu_short --account=torch_pr_769_tandon_advanced --time=00:45:00 --dependency=afterok:${finalize_id} --kill-on-invalid-dep=yes --export='${common},MODE=verify' '${analysis}'" | tr -d '\r')
verify_id=${verify_raw%%;*}; [[ "${verify_id}" =~ ^[0-9]+$ ]] || fail "bad verifier job"

receipt=${SUBMISSION_RECEIPT}
[[ ! -e "${receipt}" ]] || fail "local repair receipt already exists"
"${LOCAL_MEMNAV_PY}" - "${receipt}" "${task_root}" "${task_receipt_sha}" \
  "${finalize_id}" "${verify_id}" "${SUBMISSION_SCHEMA}" \
  "${SUBMISSION_SCOPE}" <<'PY'
import json,sys
path,bundle,bundle_sha,finalize,verify,schema,scope=sys.argv[1:]
payload={
 "schema_version":schema,"scope":scope,
 "task_bundle":bundle,"task_receipt_sha256":bundle_sha,
 "construction_cells_changed":False,"query_policy_outcomes_read":False,
 "future_policy_evaluation_submitted":False,
 "jobs":{"replacement_finalize":int(finalize),
         "replacement_independent_verify":int(verify)},
}
open(path,"x").write(json.dumps(payload,indent=2,sort_keys=True)+"\n")
print(json.dumps(payload,indent=2,sort_keys=True))
PY
printf 'TASK_ROOT=%s\nFINALIZE=%s\nVERIFY=%s\n' \
  "${task_root}" "${finalize_id}" "${verify_id}"
