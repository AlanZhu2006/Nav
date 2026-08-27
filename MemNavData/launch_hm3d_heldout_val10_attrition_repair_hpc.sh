#!/usr/bin/env bash
# Resume the preserved HM3D run after outcome-blind construction attrition.
set -euo pipefail
umask 0022

: "${TASK_ROOT:?set immutable repair bundle root}"
: "${BASE_SOURCE_ROOT:?set immutable CEC base source root}"
: "${RUN_ROOT:?set preserved parent run root}"
: "${DATA_ROOT:?set HM3D held-out-val10 data root}"
: "${EXPECTED_TASK_RECEIPT_SHA:?set repair task receipt SHA}"
: "${EXPECTED_BASE_SOURCE_RECEIPT_SHA:?set base receipt SHA}"

EVAL_CONCURRENCY=${EVAL_CONCURRENCY:-4}
EXPECTED_REMOTE_USER=${EXPECTED_REMOTE_USER:-yz11502}
GENERATED_ROOT=${RUN_ROOT}/data/hm3d_2leg

fail() { echo "ABORT: $*" >&2; exit 2; }
[[ "$(id -un)" == "${EXPECTED_REMOTE_USER}" ]] || fail "remote user differs"
[[ "${EVAL_CONCURRENCY}" =~ ^[1-9][0-9]*$ ]] || fail "bad eval concurrency"

readonly TASK_RECEIPT=${TASK_ROOT}/SOURCE_BUNDLE.sha256
readonly BASE_RECEIPT=${BASE_SOURCE_ROOT}/SOURCE_BUNDLE.sha256
readonly MANIFEST=${TASK_ROOT}/MemNavData/slurm_hm3d_heldout_val10_revisit_manifest_attrition_repair.sbatch
readonly EVAL=${TASK_ROOT}/MemNavData/slurm_hm3d_heldout_val10_revisit_eval.sbatch
readonly SUMMARY=${TASK_ROOT}/MemNavData/slurm_hm3d_heldout_val10_revisit_summary.sbatch
readonly VERIFY=${TASK_ROOT}/MemNavData/slurm_hm3d_heldout_val10_revisit_verify.sbatch
readonly PY=/scratch/lg154/conda-envs/memnav/bin/python
readonly REPAIR_RECEIPT=${RUN_ROOT}/construction_attrition_repair_submission.json

[[ "$(sha256sum "${TASK_RECEIPT}" | awk '{print $1}')" == \
    "${EXPECTED_TASK_RECEIPT_SHA}" ]] || fail "repair task receipt changed"
(cd "${TASK_ROOT}" && sha256sum -c SOURCE_BUNDLE.sha256 >/dev/null) || \
  fail "repair task bundle validation failed"
[[ "$(sha256sum "${BASE_RECEIPT}" | awk '{print $1}')" == \
    "${EXPECTED_BASE_SOURCE_RECEIPT_SHA}" ]] || fail "base receipt changed"
(cd "${BASE_SOURCE_ROOT}" && sha256sum -c SOURCE_BUNDLE.sha256 >/dev/null) || \
  fail "base bundle validation failed"
for path in "${RUN_ROOT}/submission.json" "${GENERATED_ROOT}" \
  "${MANIFEST}" "${EVAL}" "${SUMMARY}" "${VERIFY}" "${PY}"; do
  test -r "${path}" || fail "missing repair input ${path}"
done
[[ ! -e "${RUN_ROOT}/data_manifest.json" && ! -e "${RUN_ROOT}/scenes" && \
   ! -e "${RUN_ROOT}/hm3d_heldout_val10_revisit_summary.json" && \
   ! -e "${RUN_ROOT}/hm3d_heldout_val10_revisit_independent_verification.json" && \
   ! -e "${REPAIR_RECEIPT}" ]] || fail "downstream output already exists"
scontrol ping | grep -q 'is UP' || fail "Slurm controller is not UP"

exports="ALL,TASK_ROOT=${TASK_ROOT},BASE_SOURCE_ROOT=${BASE_SOURCE_ROOT},RUN_ROOT=${RUN_ROOT},GENERATED_ROOT=${GENERATED_ROOT},DATA_ROOT=${DATA_ROOT},EXPECTED_TASK_RECEIPT_SHA=${EXPECTED_TASK_RECEIPT_SHA},EXPECTED_BASE_SOURCE_RECEIPT_SHA=${EXPECTED_BASE_SOURCE_RECEIPT_SHA}"

sbatch --test-only --export="${exports}" "${MANIFEST}" >/dev/null
sbatch --test-only --array=0 --export="${exports}" "${EVAL}" >/dev/null
sbatch --test-only --export="${exports}" "${SUMMARY}" >/dev/null
sbatch --test-only --export="${exports}" "${VERIFY}" >/dev/null

manifest_raw=$(sbatch --parsable --export="${exports}" "${MANIFEST}")
manifest_id=${manifest_raw%%;*}
[[ "${manifest_id}" =~ ^[0-9]+$ ]] || fail "bad manifest job ID"

eval_raw=$(sbatch --parsable --array="0-7,9%${EVAL_CONCURRENCY}" \
  --dependency="afterok:${manifest_id}" --kill-on-invalid-dep=yes \
  --export="${exports}" "${EVAL}")
eval_id=${eval_raw%%;*}
[[ "${eval_id}" =~ ^[0-9]+$ ]] || fail "bad evaluation job ID"

summary_raw=$(sbatch --parsable --dependency="afterok:${eval_id}" \
  --kill-on-invalid-dep=yes --export="${exports}" "${SUMMARY}")
summary_id=${summary_raw%%;*}
[[ "${summary_id}" =~ ^[0-9]+$ ]] || fail "bad summary job ID"

verify_raw=$(sbatch --parsable --dependency="afterok:${summary_id}" \
  --kill-on-invalid-dep=yes --export="${exports}" "${VERIFY}")
verify_id=${verify_raw%%;*}
[[ "${verify_id}" =~ ^[0-9]+$ ]] || fail "bad verification job ID"

"${PY}" - "${REPAIR_RECEIPT}" "${TASK_ROOT}" \
  "${EXPECTED_TASK_RECEIPT_SHA}" "${manifest_id}" "${eval_id}" \
  "${summary_id}" "${verify_id}" "${EVAL_CONCURRENCY}" <<'PY'
import json,os,sys
(path,bundle,receipt,manifest,evaluation,summary,verification,
 concurrency)=sys.argv[1:]
payload={
 "schema_version":"hm3d_heldout_val10_attrition_repair_submission_v1_20260816",
 "scope":"outcome-blind construction attrition; no method outcome read",
 "source_bundle":bundle,"task_receipt_sha256":receipt,
 "selected_scene_count":10,"constructible_scene_count":9,
 "episode_count":36,"evaluation_scene_indices":[0,1,2,3,4,5,6,7,9],
 "guards":{"no_scene_replacement":True,"no_generation_retry":True,
           "original_scene_indices_preserved":True,
           "navigation_outcomes_read_before_repair":False},
 "evaluation_concurrency":int(concurrency),
 "jobs":{"manifest":int(manifest),"evaluation_array":int(evaluation),
         "summary":int(summary),"independent_verification":int(verification)},
}
descriptor=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o444)
with os.fdopen(descriptor,"w",encoding="utf-8") as handle:
 json.dump(payload,handle,indent=2,sort_keys=True); handle.write("\n")
PY

echo "RUN_ROOT=${RUN_ROOT}"
echo "TASK_BUNDLE=${TASK_ROOT}"
echo "manifest=${manifest_id} eval=${eval_id} summary=${summary_id} verify=${verify_id}"
squeue -j "${manifest_id},${eval_id},${summary_id},${verify_id}" \
  -o '%.18i %.14P %.24j %.2t %.10M %.6D %R'
