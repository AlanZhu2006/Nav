#!/usr/bin/env bash
# Submit only the outcome-blind health-receipt repair and the frozen six-item
# factual-C/B2 continuation.  The completed collect smoke is reused verbatim.
set -euo pipefail
umask 0022

ROOT=${ROOT:-/home/asus/Research/Nav-graph-blind}
SSH_ALIAS=${SSH_ALIAS:-alantorch}
EXPECTED_SSH_USER=${EXPECTED_SSH_USER:-yz11502}
SUBMIT=${SUBMIT:-0}
LOCAL_PY=${LOCAL_PY:-/home/asus/miniconda3/envs/memnav/bin/python}
REMOTE_PY=/scratch/lg154/conda-envs/memnav/bin/python
REMOTE_BUNDLES=/scratch/yz11502/Research/Nav-axis-uturn-source-bundles
RUN_ROOT=/scratch/yz11502/Research/Nav-axis-uturn-results/hm3d_fullmono_lifelong_natural_v4_20260827/formal_materialize_20260827T133704Z_d85fc50d
TASK_ROOT=/scratch/yz11502/Research/Nav-axis-uturn-source-bundles/hm3d_fullmono_lifelong_375f0b6879b2ff87
TASK_RECEIPT=${TASK_ROOT}/SOURCE_BUNDLE.sha256
EXPECTED_TASK_RECEIPT_SHA=375f0b6879b2ff87b7019dae4727880d1b03fd3185a1862e6239942a76b5bcc8
BASE_SOURCE_ROOT=/scratch/yz11502/Research/Nav-axis-uturn-source-bundles/final14_mono_factorial_5690569a4373f2d2
BASE_RECEIPT=${BASE_SOURCE_ROOT}/source_inputs.sha256
EXPECTED_BASE_RECEIPT_SHA=5690569a4373f2d2768671418f0c604c4a03aa4b0ffe01baf70b288af03ba216
DEPENDENCY_RECEIPT=/scratch/yz11502/Research/Nav-axis-uturn-results/shared_online_double_revisit_fresh_20260813/double_revisit_fresh40_20260813T200121Z/dependency_receipt.json
EXPECTED_DEPENDENCY_RECEIPT_SHA=4eb0ca6479a26f8e04f85a31d906cee4e68b1785f66cfd3ac23bf65424d36e5e
POPULATION_SHA=ec11c0dbc43a4abe585330c1ce52a8c14ad1d4b1da6fd8397e1d15592707a6d5
OLD_BUNDLE=/scratch/yz11502/Research/Nav-axis-uturn-source-bundles/hm3d_lifelong_collect_repair2_899141ad23b4ff0c
OLD_RECEIPT=${OLD_BUNDLE}/SOURCE_BUNDLE.sha256
EXPECTED_OLD_RECEIPT_SHA=899141ad23b4ff0ca3012bb68b9bc6aa0e5a8e1ee45bbfe4abcf8fa98ab89f26
SOURCE_REPAIR_ROOT=${RUN_ROOT}/underpowered_collect_repair_attempt2_20260828
OUTPUT_ROOT=${RUN_ROOT}/underpowered_collect_health_receipt_repair_20260829
COLLECT_SMOKE_ROOT=${RUN_ROOT}/underpowered_collect_repair_attempt2_smoke_20260828
B2_SMOKE_ROOT=${RUN_ROOT}/underpowered_B2_smoke_attempt2_20260828
SUBMISSION_RECEIPT=MemNavData/HM3D_LIFELONG_UNDERPOWERED_COLLECT_HEALTH_RECEIPT_REPAIR_SUBMISSION_20260829.json

cd "${ROOT}"
fail() { echo "ABORT: $*" >&2; exit 2; }
[[ "${SUBMIT}" =~ ^[01]$ ]] || fail "SUBMIT must be 0 or 1"

SSH_CONTROL_PATH=${SSH_CONTROL_PATH:-$(
  ssh -G "${SSH_ALIAS}" 2>/dev/null |
    awk '$1=="controlpath"{value=$2} END{print value}'
)}
[[ -S "${SSH_CONTROL_PATH}" ]] || fail "authoritative shared SSH socket missing"
timeout 20 ssh -O check -S "${SSH_CONTROL_PATH}" "${SSH_ALIAS}" \
  >/dev/null 2>&1 || fail "authoritative shared SSH master is not responsive"
remote_user=$(timeout 90 ssh -n -T -o BatchMode=yes -o ControlMaster=no \
  -o ServerAliveInterval=15 -S "${SSH_CONTROL_PATH}" "${SSH_ALIAS}" \
  'id -un' 2>/dev/null || true)
[[ "${remote_user}" == "${EXPECTED_SSH_USER}" ]] || \
  fail "shared SSH identity is ${remote_user:-unavailable}"
remote() {
  timeout 300 ssh -n -T -o BatchMode=yes -o ControlMaster=no \
    -o ServerAliveInterval=15 -S "${SSH_CONTROL_PATH}" "${SSH_ALIAS}" "$@"
}
job_id() { tr -d '\r' | awk -F';' '/^[0-9]+(;|$)/ {print $1; exit}'; }

files=(
  MemNavData/audit_hm3d_lifelong_underpowered_collect_repair_attempt2.py
  MemNavData/test_audit_hm3d_lifelong_underpowered_collect_repair_attempt2.py
  MemNavData/cec_hub_cli_compat.py
  MemNavData/test_cec_hub_cli_compat.py
  MemNavData/hm3d_lifelong_underpowered_collect_health_receipt_repair_20260829.json
  MemNavData/HM3D_LIFELONG_UNDERPOWERED_COLLECT_HEALTH_RECEIPT_REPAIR_20260829.md
  MemNavData/slurm_hm3d_lifelong_underpowered_collect_health_receipt_barrier.sbatch
  MemNavData/slurm_safe_submit.sh
  MemNavData/test_slurm_safe_submit.sh
  MemNavData/submit_hm3d_lifelong_underpowered_collect_health_receipt_repair_hpc.sh
)
for path in "${files[@]}"; do
  [[ -f "${path}" && ! -L "${path}" ]] || fail "missing physical ${path}"
done

"${LOCAL_PY}" -m json.tool \
  MemNavData/hm3d_lifelong_underpowered_collect_health_receipt_repair_20260829.json \
  >/dev/null
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="${ROOT}:${ROOT}/MemNavData" \
  "${LOCAL_PY}" -m pytest -q -p no:cacheprovider \
  MemNavData/test_audit_hm3d_lifelong_underpowered_collect_repair_attempt2.py \
  MemNavData/test_cec_hub_cli_compat.py
bash MemNavData/test_slurm_safe_submit.sh
"${LOCAL_PY}" -m py_compile \
  MemNavData/audit_hm3d_lifelong_underpowered_collect_repair_attempt2.py \
  MemNavData/cec_hub_cli_compat.py
bash -n \
  MemNavData/slurm_hm3d_lifelong_underpowered_collect_health_receipt_barrier.sbatch \
  MemNavData/slurm_safe_submit.sh MemNavData/test_slurm_safe_submit.sh \
  MemNavData/submit_hm3d_lifelong_underpowered_collect_health_receipt_repair_hpc.sh
(
  source MemNavData/slurm_safe_submit.sh
  lint_sbatch_template \
    MemNavData/slurm_hm3d_lifelong_underpowered_collect_health_receipt_barrier.sbatch
)

stage=$(mktemp -d /tmp/h3_health_receipt.XXXXXX)
cleanup() { rm -rf -- "${stage}"; }
trap cleanup EXIT
for path in "${files[@]}"; do
  mkdir -p "${stage}/root/$(dirname "${path}")"
  install -m 0644 "${path}" "${stage}/root/${path}"
done
(
  cd "${stage}/root"
  find . -type f ! -name SOURCE_BUNDLE.sha256 -print0 | sort -z | \
    xargs -0 sha256sum >SOURCE_BUNDLE.sha256
  sha256sum -c --quiet SOURCE_BUNDLE.sha256
)
bundle_sha=$(sha256sum "${stage}/root/SOURCE_BUNDLE.sha256" | awk '{print $1}')
bundle=${REMOTE_BUNDLES}/hm3d_lifelong_health_receipt_${bundle_sha:0:16}
partial=${bundle}.partial.$$
bundle_receipt=${bundle}/SOURCE_BUNDLE.sha256
barrier=${bundle}/MemNavData/slurm_hm3d_lifelong_underpowered_collect_health_receipt_barrier.sbatch
attempt2_protocol=${OLD_BUNDLE}/MemNavData/hm3d_lifelong_underpowered_collect_repair_attempt2_20260828.json
base_protocol=${OLD_BUNDLE}/MemNavData/hm3d_lifelong_underpowered_collect_repair_20260828.json
amendment_protocol=${OLD_BUNDLE}/MemNavData/hm3d_fullmono_lifelong_underpowered_amendment_20260828.json
arm=${OLD_BUNDLE}/MemNavData/slurm_hm3d_fullmono_shared_c_arm.sbatch
deferred=${OLD_BUNDLE}/MemNavData/slurm_hm3d_lifelong_underpowered_deferred.sbatch
finalize=${TASK_ROOT}/MemNavData/slurm_hm3d_fullmono_shared_c_finalize.sbatch

remote "set -euo pipefail; test \"\$(id -un)\" = '${EXPECTED_SSH_USER}'; \
  test \"\$(sha256sum '${TASK_RECEIPT}' | awk '{print \$1}')\" = '${EXPECTED_TASK_RECEIPT_SHA}'; \
  cd '${TASK_ROOT}' && sha256sum -c --quiet SOURCE_BUNDLE.sha256; \
  test \"\$(sha256sum '${BASE_RECEIPT}' | awk '{print \$1}')\" = '${EXPECTED_BASE_RECEIPT_SHA}'; \
  cd '${BASE_SOURCE_ROOT}' && sha256sum -c --quiet source_inputs.sha256; \
  test \"\$(sha256sum '${OLD_RECEIPT}' | awk '{print \$1}')\" = '${EXPECTED_OLD_RECEIPT_SHA}'; \
  cd '${OLD_BUNDLE}' && sha256sum -c --quiet SOURCE_BUNDLE.sha256; \
  test \"\$(sha256sum '${DEPENDENCY_RECEIPT}' | awk '{print \$1}')\" = '${EXPECTED_DEPENDENCY_RECEIPT_SHA}'; \
  test \"\$(sha256sum '${RUN_ROOT}/population/population.json' | awk '{print \$1}')\" = '${POPULATION_SHA}'; \
  test -f '${SOURCE_REPAIR_ROOT}/post_archive_audit.json'; \
  test -f '${SOURCE_REPAIR_ROOT}/post_archive_audit.json.sha256'; \
  cd '${SOURCE_REPAIR_ROOT}' && sha256sum -c --quiet post_archive_audit.json.sha256; \
  test ! -e '${RUN_ROOT}/shared_c_population'; \
  test ! -e '${RUN_ROOT}/shared_c_evaluation'; \
  test ! -e '${RUN_ROOT}/shared_c_aggregate'; \
  test ! -e '${RUN_ROOT}/shared_c_independent_verification.json'; \
  test ! -e '${B2_SMOKE_ROOT}'"

if remote "test -d '${bundle}'"; then
  remote "test \"\$(sha256sum '${bundle_receipt}' | awk '{print \$1}')\" = '${bundle_sha}' && cd '${bundle}' && sha256sum -c --quiet SOURCE_BUNDLE.sha256"
else
  remote "test ! -e '${partial}' && mkdir -p '${partial}'"
  timeout 300 rsync -a --partial \
    --chmod=Du=rwx,Dgo=rx,Fu=rw,Fgo=r \
    -e "ssh -o BatchMode=yes -o ControlMaster=no -S ${SSH_CONTROL_PATH}" \
    "${stage}/root/" "${SSH_ALIAS}:${partial}/"
  remote "cd '${partial}' && sha256sum -c --quiet SOURCE_BUNDLE.sha256 && chmod -R a-w '${partial}' && mv '${partial}' '${bundle}'"
fi

remote "PYTHONDONTWRITEBYTECODE=1 PYTHONPATH='${bundle}:${bundle}/MemNavData' '${REMOTE_PY}' -m pytest -q -p no:cacheprovider '${bundle}/MemNavData/test_audit_hm3d_lifelong_underpowered_collect_repair_attempt2.py' '${bundle}/MemNavData/test_cec_hub_cli_compat.py'"
remote "PYTHONDONTWRITEBYTECODE=1 PYTHONPATH='${bundle}:${bundle}/MemNavData' '${REMOTE_PY}' '${bundle}/MemNavData/audit_hm3d_lifelong_underpowered_collect_repair_attempt2.py' --protocol '${attempt2_protocol}' --base-protocol '${base_protocol}' --run-root '${RUN_ROOT}' --phase smoke_ready >/dev/null"

gate_env="ALL,AUDIT_ROOT=${bundle},AUDIT_RECEIPT=${bundle_receipt},EXPECTED_AUDIT_RECEIPT_SHA=${bundle_sha},PROTOCOL=${attempt2_protocol},BASE_PROTOCOL=${base_protocol},RUN_ROOT=${RUN_ROOT},SOURCE_REPAIR_ROOT=${SOURCE_REPAIR_ROOT},OUTPUT_ROOT=${OUTPUT_ROOT}"
common="ALL,AMENDMENT_ROOT=${OLD_BUNDLE},AMENDMENT_RECEIPT=${OLD_RECEIPT},EXPECTED_AMENDMENT_RECEIPT_SHA=${EXPECTED_OLD_RECEIPT_SHA},TASK_ROOT=${TASK_ROOT},TASK_RECEIPT=${TASK_RECEIPT},EXPECTED_TASK_RECEIPT_SHA=${EXPECTED_TASK_RECEIPT_SHA},BASE_SOURCE_ROOT=${BASE_SOURCE_ROOT},BASE_RECEIPT=${BASE_RECEIPT},EXPECTED_BASE_RECEIPT_SHA=${EXPECTED_BASE_RECEIPT_SHA},RUN_ROOT=${RUN_ROOT},PROTOCOL=${amendment_protocol},DEPENDENCY_RECEIPT=${DEPENDENCY_RECEIPT},EXPECTED_DEPENDENCY_RECEIPT_SHA=${EXPECTED_DEPENDENCY_RECEIPT_SHA},EXPECTED_HAB_REQUESTS_VERSION=2.32.4,EXPECTED_HAB_REQUESTS_INIT_BYTES=5057,EXPECTED_HAB_REQUESTS_INIT_SHA=1e507f1f386bcc6b5f0ff69a614c14875cd65cb67be7f6022f28adef9774573f,EXPECTED_HAB_REQUESTS_VERSION_BYTES=435,EXPECTED_HAB_REQUESTS_VERSION_SHA=143abaf3563712f063743a7952aa65319dbcb934d894cfc989bd2c015f8da577"
continuation="${common},SMOKE_ROOT=${B2_SMOKE_ROOT},SOURCE_POPULATION_SHA=${POPULATION_SHA},SOURCE_POPULATION_COUNT=22,SOURCE_SCENE_COUNT=15,EVAL_CONCURRENCY=2,GPU_TIME_LIMIT=01:00:00,UPSTREAM_POPULATION_SEAL_JOB_ID=16489720"
source_remote="source '${OLD_BUNDLE}/MemNavData/slurm_safe_submit.sh';"
remote "${source_remote} safe_sbatch --lint-fatal --test-only --partition=cpu_short --export='${gate_env},GATE_MODE=smoke' '${barrier}' >/dev/null; safe_sbatch --lint-fatal --test-only --partition=h100_tandon --nodelist=gh005 --qos=gpu48 --time=01:00:00 --array=0 --export='${common},STAGE=collect,EXPECTED_REPLAY_NODE=gh005' '${arm}' >/dev/null; safe_sbatch --lint-fatal --test-only --partition=cpu_short --export='${gate_env},GATE_MODE=ready' '${barrier}' >/dev/null; safe_sbatch --lint-fatal --test-only --partition=cpu_short --export='${common}' '${finalize}' >/dev/null; safe_sbatch --lint-fatal --test-only --partition=cpu_short --export='${continuation},DEFERRED_MODE=resume' '${deferred}' >/dev/null"

if [[ "${SUBMIT}" == 0 ]]; then
  printf 'PREPARED_ONLY=1\nBUNDLE=%s\nBUNDLE_SHA=%s\n' "${bundle}" "${bundle_sha}"
  exit 0
fi

[[ ! -e "${SUBMISSION_RECEIPT}" ]] || fail "submission receipt already exists"
remote "test ! -e '${OUTPUT_ROOT}' && mkdir -p '${OUTPUT_ROOT}'"
smoke_gate=$(remote "${source_remote} safe_sbatch --lint-fatal --parsable --partition=cpu_short --export='${gate_env},GATE_MODE=smoke' '${barrier}'" | job_id)
[[ "${smoke_gate}" =~ ^[0-9]+$ ]] || fail "bad repaired smoke gate id"

submit_repair() {
  local index=$1 node=$2 partition=$3 dependency=$4 command
  command="${source_remote} safe_sbatch --lint-fatal --parsable --partition=${partition} --nodelist=${node} --qos=gpu48 --time=01:00:00 --array=${index} --dependency=afterok:${dependency} --kill-on-invalid-dep=yes"
  command+=" --export='${common},STAGE=collect,EXPECTED_REPLAY_NODE=${node}' '${arm}'"
  remote "${command}" | job_id
}
repair_0=$(submit_repair 0 gh005 h100_tandon "${smoke_gate}")
repair_1=$(submit_repair 1 gh001 h100_tandon "${smoke_gate}")
repair_7=$(submit_repair 7 ga005 a100_tandon "${repair_0}")
repair_9=$(submit_repair 9 ga003 a100_tandon "${repair_1}")
repair_11=$(submit_repair 11 ga028 a100_tandon "${repair_7}")
repair_13=$(submit_repair 13 ga002 a100_tandon "${repair_9}")
for value in "${repair_0}" "${repair_1}" "${repair_7}" "${repair_9}" \
  "${repair_11}" "${repair_13}"; do
  [[ "${value}" =~ ^[0-9]+$ ]] || fail "bad repair job id"
done
integrity=$(remote "${source_remote} safe_sbatch --lint-fatal --parsable --partition=cpu_short --dependency=afterany:${repair_11}:${repair_13} --export='${gate_env},GATE_MODE=ready' '${barrier}'" | job_id)
seal=$(remote "${source_remote} safe_sbatch --lint-fatal --parsable --partition=cpu_short --dependency=afterok:${integrity} --kill-on-invalid-dep=yes --export='${common}' '${finalize}'" | job_id)
resume=$(remote "${source_remote} safe_sbatch --lint-fatal --parsable --partition=cpu_short --dependency=afterok:${seal} --kill-on-invalid-dep=yes --export='${continuation},DEFERRED_MODE=resume' '${deferred}'" | job_id)
[[ "${integrity}" =~ ^[0-9]+$ && "${seal}" =~ ^[0-9]+$ \
   && "${resume}" =~ ^[0-9]+$ ]] || fail "bad downstream job id"

"${LOCAL_PY}" - "${SUBMISSION_RECEIPT}" "${bundle}" "${bundle_sha}" \
  "${RUN_ROOT}" "${OUTPUT_ROOT}" "${smoke_gate}" "${repair_0}" \
  "${repair_1}" "${repair_7}" "${repair_9}" "${repair_11}" \
  "${repair_13}" "${integrity}" "${seal}" "${resume}" <<'PY'
import datetime,json,sys
(path,bundle,digest,run,output,gate,j0,j1,j7,j9,j11,j13,barrier,seal,resume)=sys.argv[1:]
payload={
 "schema_version":"hm3d_lifelong_underpowered_collect_health_receipt_submission_v1_20260829",
 "submitted_at":datetime.datetime.now(datetime.timezone.utc).isoformat(),
 "source_bundle":bundle,"source_bundle_receipt_sha256":digest,
 "run_root":run,"receipt_root":output,
 "reused_collect_smoke_job":16514058,"failed_schema_gate_job":16514066,
 "repaired_smoke_gate_job":int(gate),
 "repair_jobs":[
  {"index":0,"node":"gh005","job":int(j0)},
  {"index":1,"node":"gh001","job":int(j1)},
  {"index":7,"node":"ga005","job":int(j7)},
  {"index":9,"node":"ga003","job":int(j9)},
  {"index":11,"node":"ga028","job":int(j11)},
  {"index":13,"node":"ga002","job":int(j13)}],
 "integrity_barrier_job":int(barrier),"factual_C_seal_job":int(seal),
 "node_affine_B2_resume_launcher_job":int(resume),
 "smoke_result_enters_scientific_denominator":False,
 "successful_factual_C_navigation_outcomes_read_before_submission":False,
 "B2_navigation_outcomes_read_before_submission":False,
 "maximum_concurrent_scientific_gpu_jobs":2,"underpowered":True,
 "powered_confirmation_claim":False,"submitted":True,
}
with open(path,"x") as handle:
 json.dump(payload,handle,indent=2,sort_keys=True); handle.write("\n")
PY
sha256sum "${SUBMISSION_RECEIPT}" >"${SUBMISSION_RECEIPT}.sha256"
timeout 300 rsync -a \
  -e "ssh -o BatchMode=yes -o ControlMaster=no -S ${SSH_CONTROL_PATH}" \
  "${SUBMISSION_RECEIPT}" "${SSH_ALIAS}:${OUTPUT_ROOT}/submission.json"
remote "sha256sum '${OUTPUT_ROOT}/submission.json' >'${OUTPUT_ROOT}/submission.json.sha256' && chmod a-w '${OUTPUT_ROOT}/submission.json' '${OUTPUT_ROOT}/submission.json.sha256'"
printf 'SUBMITTED=1\nSMOKE_GATE=%s\nREPAIRS=%s,%s,%s,%s,%s,%s\nINTEGRITY=%s\nSEAL=%s\nRESUME=%s\n' \
  "${smoke_gate}" "${repair_0}" "${repair_1}" "${repair_7}" \
  "${repair_9}" "${repair_11}" "${repair_13}" "${integrity}" \
  "${seal}" "${resume}"
