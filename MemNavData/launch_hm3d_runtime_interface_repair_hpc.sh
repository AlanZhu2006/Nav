#!/usr/bin/env bash
# Submit consumed smoke -> sparse formal array -> summary -> verification.

set -euo pipefail
umask 0022

: "${TASK_ROOT:?set immutable interface-repair bundle root}"
: "${SMOKE_DATA_ROOT:?set immutable consumed-scene smoke bundle root}"
: "${RUNTIME_SOURCE_ROOT:?set verified actual-online NNR runtime root}"
: "${BASE_SOURCE_ROOT:?set dependency-only base source root}"
: "${PARENT_RUN_ROOT:?set preserved failed HM3D run root}"
: "${SMOKE_RUN_ROOT:?set new smoke output root}"
: "${FORMAL_RUN_ROOT:?set new formal output root}"
: "${EXPECTED_TASK_RECEIPT_SHA:?set task receipt SHA}"
: "${EXPECTED_SMOKE_DATA_RECEIPT_SHA:?set smoke data receipt SHA}"
: "${EXPECTED_RUNTIME_RECEIPT_SHA:?set runtime receipt SHA}"
: "${EXPECTED_BASE_SOURCE_RECEIPT_SHA:?set base receipt SHA}"

EVAL_CONCURRENCY=${EVAL_CONCURRENCY:-4}
EXPECTED_REMOTE_USER=${EXPECTED_REMOTE_USER:-yz11502}
readonly EXPECTED_MANIFEST_SHA=62bc6299203da709e65787c735a531974905f2ab8e940f72e91318914d949c89
readonly EVAL=${TASK_ROOT}/MemNavData/slurm_hm3d_runtime_interface_eval.sbatch
readonly SUMMARY=${TASK_ROOT}/MemNavData/slurm_hm3d_heldout_val10_revisit_summary.sbatch
readonly VERIFY=${TASK_ROOT}/MemNavData/slurm_hm3d_heldout_val10_revisit_verify.sbatch
readonly PARENT_MANIFEST=${PARENT_RUN_ROOT}/data_manifest.json
readonly PARENT_MANIFEST_RECEIPT=${PARENT_MANIFEST}.sha256
readonly FORMAL_MANIFEST=${FORMAL_RUN_ROOT}/data_manifest.json
readonly SUBMISSION=${FORMAL_RUN_ROOT}/runtime_interface_repair_submission.json
readonly PY=/scratch/lg154/conda-envs/memnav/bin/python

fail() { echo "ABORT: $*" >&2; exit 2; }
[[ "$(id -un)" == "${EXPECTED_REMOTE_USER}" ]] || fail "remote user differs"
[[ "${EVAL_CONCURRENCY}" =~ ^[1-9][0-9]*$ ]] || fail "bad concurrency"
[[ ! -e "${SMOKE_RUN_ROOT}" ]] || fail "smoke output already exists"
[[ ! -e "${FORMAL_RUN_ROOT}" ]] || fail "formal output already exists"
[[ "${SMOKE_RUN_ROOT}" != "${FORMAL_RUN_ROOT}" ]] || fail "output roots collide"

[[ "$(sha256sum "${TASK_ROOT}/SOURCE_BUNDLE.sha256" | awk '{print $1}')" == \
    "${EXPECTED_TASK_RECEIPT_SHA}" ]] || fail "task receipt changed"
(cd "${TASK_ROOT}" && sha256sum -c SOURCE_BUNDLE.sha256 >/dev/null) || \
  fail "task bundle changed"
[[ "$(sha256sum "${SMOKE_DATA_ROOT}/SMOKE_DATA.sha256" | awk '{print $1}')" == \
    "${EXPECTED_SMOKE_DATA_RECEIPT_SHA}" ]] || fail "smoke receipt changed"
(cd "${SMOKE_DATA_ROOT}" && sha256sum -c SMOKE_DATA.sha256 >/dev/null) || \
  fail "smoke data changed"
[[ "$(sha256sum "${RUNTIME_SOURCE_ROOT}/SOURCE_BUNDLE.sha256" | awk '{print $1}')" == \
    "${EXPECTED_RUNTIME_RECEIPT_SHA}" ]] || fail "runtime receipt changed"
(cd "${RUNTIME_SOURCE_ROOT}" && sha256sum -c SOURCE_BUNDLE.sha256 >/dev/null) || \
  fail "runtime bundle changed"
[[ "$(sha256sum "${BASE_SOURCE_ROOT}/SOURCE_BUNDLE.sha256" | awk '{print $1}')" == \
    "${EXPECTED_BASE_SOURCE_RECEIPT_SHA}" ]] || fail "base receipt changed"
(cd "${BASE_SOURCE_ROOT}" && sha256sum -c SOURCE_BUNDLE.sha256 >/dev/null) || \
  fail "dependency base bundle changed"
for path in "${PARENT_MANIFEST}" "${PARENT_MANIFEST_RECEIPT}" \
  "${EVAL}" "${SUMMARY}" "${VERIFY}" "${PY}"; do
  test -r "${path}" || fail "missing input ${path}"
done
(cd "${PARENT_RUN_ROOT}" && sha256sum -c data_manifest.json.sha256 >/dev/null) || \
  fail "parent manifest changed"
[[ "$(sha256sum "${PARENT_MANIFEST}" | awk '{print $1}')" == \
    "${EXPECTED_MANIFEST_SHA}" ]] || fail "unexpected parent manifest identity"
scontrol ping | grep -q 'is UP' || fail "Slurm controller is not UP"

mkdir -p "${SMOKE_RUN_ROOT}" "${FORMAL_RUN_ROOT}"
cp --preserve=mode,timestamps "${PARENT_MANIFEST}" "${FORMAL_MANIFEST}"
cp --preserve=mode,timestamps "${PARENT_MANIFEST_RECEIPT}" \
  "${FORMAL_MANIFEST}.sha256"
(cd "${FORMAL_RUN_ROOT}" && sha256sum -c data_manifest.json.sha256 >/dev/null) || \
  fail "isolated formal manifest copy failed"

common_exports="ALL,TASK_ROOT=${TASK_ROOT},RUNTIME_SOURCE_ROOT=${RUNTIME_SOURCE_ROOT},BASE_SOURCE_ROOT=${BASE_SOURCE_ROOT},REFERENCE_MANIFEST=${FORMAL_MANIFEST},EXPECTED_TASK_RECEIPT_SHA=${EXPECTED_TASK_RECEIPT_SHA},EXPECTED_RUNTIME_RECEIPT_SHA=${EXPECTED_RUNTIME_RECEIPT_SHA},EXPECTED_BASE_SOURCE_RECEIPT_SHA=${EXPECTED_BASE_SOURCE_RECEIPT_SHA}"
smoke_exports="${common_exports},MODE=smoke,RUN_ROOT=${SMOKE_RUN_ROOT},SMOKE_DATA_ROOT=${SMOKE_DATA_ROOT}"
formal_exports="${common_exports},MODE=formal,RUN_ROOT=${FORMAL_RUN_ROOT},FORMAL_MANIFEST=${FORMAL_MANIFEST}"
analysis_exports="ALL,TASK_ROOT=${TASK_ROOT},BASE_SOURCE_ROOT=${BASE_SOURCE_ROOT},RUN_ROOT=${FORMAL_RUN_ROOT},EXPECTED_TASK_RECEIPT_SHA=${EXPECTED_TASK_RECEIPT_SHA},EXPECTED_BASE_SOURCE_RECEIPT_SHA=${EXPECTED_BASE_SOURCE_RECEIPT_SHA}"

sbatch --test-only --time=02:00:00 --export="${smoke_exports}" "${EVAL}" >/dev/null
sbatch --test-only --array=0 --export="${formal_exports}" "${EVAL}" >/dev/null
sbatch --test-only --export="${analysis_exports}" "${SUMMARY}" >/dev/null
sbatch --test-only --export="${analysis_exports}" "${VERIFY}" >/dev/null

smoke_raw=$(sbatch --parsable --time=02:00:00 \
  --export="${smoke_exports}" "${EVAL}")
smoke_id=${smoke_raw%%;*}
[[ "${smoke_id}" =~ ^[0-9]+$ ]] || fail "bad smoke job ID"

eval_raw=$(sbatch --parsable --array="0-7,9%${EVAL_CONCURRENCY}" \
  --dependency="afterok:${smoke_id}" --kill-on-invalid-dep=yes \
  --export="${formal_exports}" "${EVAL}")
eval_id=${eval_raw%%;*}
[[ "${eval_id}" =~ ^[0-9]+$ ]] || fail "bad evaluation job ID"

summary_raw=$(sbatch --parsable --dependency="afterok:${eval_id}" \
  --kill-on-invalid-dep=yes --export="${analysis_exports}" "${SUMMARY}")
summary_id=${summary_raw%%;*}
[[ "${summary_id}" =~ ^[0-9]+$ ]] || fail "bad summary job ID"

verify_raw=$(sbatch --parsable --dependency="afterok:${summary_id}" \
  --kill-on-invalid-dep=yes --export="${analysis_exports}" "${VERIFY}")
verify_id=${verify_raw%%;*}
[[ "${verify_id}" =~ ^[0-9]+$ ]] || fail "bad verification job ID"

"${PY}" - "${SUBMISSION}" "${TASK_ROOT}" "${SMOKE_DATA_ROOT}" \
  "${RUNTIME_SOURCE_ROOT}" "${BASE_SOURCE_ROOT}" "${PARENT_RUN_ROOT}" \
  "${EXPECTED_TASK_RECEIPT_SHA}" "${EXPECTED_SMOKE_DATA_RECEIPT_SHA}" \
  "${EXPECTED_RUNTIME_RECEIPT_SHA}" "${EXPECTED_BASE_SOURCE_RECEIPT_SHA}" \
  "${EXPECTED_MANIFEST_SHA}" "${smoke_id}" "${eval_id}" \
  "${summary_id}" "${verify_id}" "${EVAL_CONCURRENCY}" <<'PY'
import json,os,sys
(path,task,smoke_data,runtime,base,parent,task_sha,smoke_sha,runtime_sha,
 base_sha,manifest_sha,smoke_job,evaluation,summary,verification,
 concurrency)=sys.argv[1:]
payload={
 "schema_version":"hm3d_runtime_interface_repair_submission_v1_20260816",
 "scope":"interface repair; no method, population, or analysis change",
 "failed_parent_run":parent,
 "frozen_manifest_sha256":manifest_sha,
 "task_bundle":task,"task_receipt_sha256":task_sha,
 "smoke_data_bundle":smoke_data,"smoke_data_receipt_sha256":smoke_sha,
 "runtime_bundle":runtime,"runtime_receipt_sha256":runtime_sha,
 "dependency_base_bundle":base,"base_receipt_sha256":base_sha,
 "evaluation_scene_indices":[0,1,2,3,4,5,6,7,9],
 "episode_count":36,"evaluation_concurrency":int(concurrency),
 "guards":{
   "new_formal_output_root":True,
   "manifest_copied_byte_identically":True,
   "heldout_depends_afterok_on_consumed_smoke":True,
   "navigation_outcomes_read_before_repair":False,
   "method_change":False,
 },
 "jobs":{
   "consumed_four_arm_smoke":int(smoke_job),
   "formal_evaluation_array":int(evaluation),
   "summary":int(summary),
   "independent_verification":int(verification),
 },
}
descriptor=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o444)
with os.fdopen(descriptor,"w",encoding="utf-8") as handle:
    json.dump(payload,handle,indent=2,sort_keys=True); handle.write("\n")
PY

echo "SMOKE_RUN_ROOT=${SMOKE_RUN_ROOT}"
echo "FORMAL_RUN_ROOT=${FORMAL_RUN_ROOT}"
echo "smoke=${smoke_id} eval=${eval_id} summary=${summary_id} verify=${verify_id}"
squeue -j "${smoke_id},${eval_id},${summary_id},${verify_id}" \
  -o '%.18i %.14P %.24j %.2t %.10M %.6D %R'
