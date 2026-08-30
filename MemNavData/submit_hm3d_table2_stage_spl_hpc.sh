#!/usr/bin/env bash
set -euo pipefail

LOCAL_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
SSH_ALIAS=${SSH_ALIAS:-alantorch}
REMOTE_BUNDLES=/scratch/yz11502/Research/Nav-axis-uturn-source-bundles
FORMAL_RUN_ROOT=/scratch/yz11502/Research/Nav-axis-uturn-results/hm3d_fullmono_lifelong_natural_b_expansion_execution_20260830/formal_20260830T045416Z_1f4979a7/table2_leg3_power/policy_authority_closure_repair_v3
MEETING_VERIFICATION=${FORMAL_RUN_ROOT}/meeting_result/hm3d_table2_meeting_result_independent_verification.json
PARENT_MANIFEST=/scratch/yz11502/Research/Nav-axis-uturn-results/hm3d_fresh_fullmono_mixed_role_20260820/formal_20260820T143609Z_e6dd44c6/sealed_inputs/parent_manifest.json
SOURCE_UNION_ROOT=/scratch/yz11502/Research/Nav-axis-uturn-results/hm3d_fullmono_lifelong_natural_b_expansion_execution_20260830/formal_20260830T045416Z_1f4979a7/table2_source_union
LOCAL_RECEIPT=${LOCAL_ROOT}/MemNavData/HM3D_TABLE2_STAGE_SPL_SUBMISSION_20260830.json

socket=$(ssh -G "${SSH_ALIAS}" 2>/dev/null | \
  awk '$1=="controlpath" {value=$2} END {print value}')
[[ -n "${socket}" ]]
timeout 15 ssh -O check -o ControlPath="${socket}" "${SSH_ALIAS}" >/dev/null
ssh_common=(-n -tt -o BatchMode=yes -o ControlMaster=no -S "${socket}")
identity=$(timeout 20 ssh "${ssh_common[@]}" "${SSH_ALIAS}" 'id -un' | tr -d '\r')
[[ "${identity}" == yz11502 ]]

PYTHONDONTWRITEBYTECODE=1 /home/asus/miniconda3/envs/memnav/bin/python -m pytest -q \
  "${LOCAL_ROOT}/MemNavData/test_independent_verify_hm3d_table2_stage_spl.py"

stage=$(mktemp -d)
trap 'rm -rf "${stage}"' EXIT
mkdir -p "${stage}/MemNavData"
cp "${LOCAL_ROOT}/MemNavData/independent_verify_hm3d_table2_stage_spl.py" \
   "${LOCAL_ROOT}/MemNavData/slurm_hm3d_table2_stage_spl.sbatch" \
   "${LOCAL_ROOT}/MemNavData/slurm_safe_submit.sh" \
   "${stage}/MemNavData/"
(cd "${stage}" && sha256sum \
  MemNavData/independent_verify_hm3d_table2_stage_spl.py \
  MemNavData/slurm_hm3d_table2_stage_spl.sbatch \
  MemNavData/slurm_safe_submit.sh >SOURCE_BUNDLE.sha256)
bundle_sha=$(sha256sum "${stage}/SOURCE_BUNDLE.sha256" | awk '{print $1}')
bundle_key=${bundle_sha:0:16}
task_root=${REMOTE_BUNDLES}/hm3d_table2_stage_spl_${bundle_key}

timeout 30 ssh "${ssh_common[@]}" "${SSH_ALIAS}" \
  "set -e; if test -e '${task_root}'; then (cd '${task_root}' && sha256sum -c --quiet SOURCE_BUNDLE.sha256); else mkdir -p '${task_root}'; fi"
timeout 60 rsync -a --partial \
  -e "ssh -o BatchMode=yes -o ControlMaster=no -S ${socket}" \
  "${stage}/" "${SSH_ALIAS}:${task_root}/"
timeout 30 ssh "${ssh_common[@]}" "${SSH_ALIAS}" \
  "cd '${task_root}' && sha256sum -c --quiet SOURCE_BUNDLE.sha256 && test \"\$(id -un)\" = yz11502 && /scratch/lg154/conda-envs/memnav/bin/python MemNavData/independent_verify_hm3d_table2_stage_spl.py --help >/dev/null && test -f '${MEETING_VERIFICATION}' -a -f '${MEETING_VERIFICATION}.sha256' && test -f '${PARENT_MANIFEST}' && test -f '${SOURCE_UNION_ROOT}/population/population.json' && test ! -e '${FORMAL_RUN_ROOT}/meeting_result_stage_spl_v1'"

submit=$(timeout 30 ssh "${ssh_common[@]}" "${SSH_ALIAS}" \
  "source '${task_root}/MemNavData/slurm_safe_submit.sh'; safe_sbatch --lint-fatal --parsable --partition=cpu_short --time=00:20:00 --export=ALL,TASK_ROOT='${task_root}',TASK_RECEIPT='${task_root}/SOURCE_BUNDLE.sha256',EXPECTED_TASK_RECEIPT_SHA='${bundle_sha}',MEETING_VERIFICATION='${MEETING_VERIFICATION}',PARENT_MANIFEST='${PARENT_MANIFEST}',SOURCE_UNION_ROOT='${SOURCE_UNION_ROOT}',FORMAL_RUN_ROOT='${FORMAL_RUN_ROOT}' '${task_root}/MemNavData/slurm_hm3d_table2_stage_spl.sbatch'" | tr -d '\r')
job=$(printf '%s\n' "${submit}" | awk -F';' '/^[0-9]+(;|$)/ {print $1; exit}')
[[ "${job}" =~ ^[0-9]+$ ]]

test ! -e "${LOCAL_RECEIPT}"
/home/asus/miniconda3/envs/memnav/bin/python - "${LOCAL_RECEIPT}" \
  "${job}" "${task_root}" "${bundle_sha}" "${FORMAL_RUN_ROOT}" <<'PY'
import json,sys
path,job,task,sha,formal=sys.argv[1:]
payload={
  "schema_version":"hm3d_table2_stage_spl_submission_v1_20260830",
  "job_id":int(job),
  "partition":"cpu_short",
  "time_limit":"00:20:00",
  "task_root":task,
  "source_bundle_receipt_sha256":sha,
  "formal_run_root":formal,
  "policy_rollouts_submitted":False,
  "population_or_threshold_changed":False,
}
open(path,"x").write(json.dumps(payload,indent=2,sort_keys=True)+"\n")
PY
printf 'submitted %s\n' "${job}"
