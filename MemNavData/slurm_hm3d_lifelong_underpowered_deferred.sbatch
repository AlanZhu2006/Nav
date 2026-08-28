#!/usr/bin/env bash
# Submit the underpowered HM3D shared-C chain one GPU array at a time.
# The launcher itself is CPU-only and should be dependency-held behind the
# preceding paper experiment's independent verifier.
#SBATCH --job-name=h3UnderDefer
#SBATCH --output=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/%x_%j.out
#SBATCH --error=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/%x_%j.err
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --partition=cpu_short
#SBATCH --account=torch_pr_769_tandon_advanced
#SBATCH --export=ALL

set -euo pipefail
umask 0022
export PYTHONDONTWRITEBYTECODE=1

: "${DEFERRED_MODE:?collect|evaluate}" "${AMENDMENT_ROOT:?}"
: "${AMENDMENT_RECEIPT:?}" "${EXPECTED_AMENDMENT_RECEIPT_SHA:?}"
: "${TASK_ROOT:?}" "${TASK_RECEIPT:?}" "${EXPECTED_TASK_RECEIPT_SHA:?}"
: "${BASE_SOURCE_ROOT:?}" "${BASE_RECEIPT:?}" "${EXPECTED_BASE_RECEIPT_SHA:?}"
: "${RUN_ROOT:?}" "${PROTOCOL:?}" "${DEPENDENCY_RECEIPT:?}"
: "${EXPECTED_DEPENDENCY_RECEIPT_SHA:?}" "${SMOKE_ROOT:?}"
: "${SOURCE_POPULATION_SHA:?}" "${SOURCE_POPULATION_COUNT:?}"
: "${SOURCE_SCENE_COUNT:?}" "${EVAL_CONCURRENCY:?}" "${GPU_TIME_LIMIT:?}"
: "${UPSTREAM_POPULATION_SEAL_JOB_ID:?}"

case "${DEFERRED_MODE}" in collect|evaluate) ;; *) exit 2 ;; esac
[[ "${EVAL_CONCURRENCY}" =~ ^[1-9][0-9]*$ ]]
[[ "${GPU_TIME_LIMIT}" =~ ^0[01]:[0-5][0-9]:[0-5][0-9]$ ]]
[[ "${SOURCE_POPULATION_COUNT}" == 22 ]]
[[ "${SOURCE_SCENE_COUNT}" == 15 ]]
[[ "${SOURCE_POPULATION_SHA}" == \
  ec11c0dbc43a4abe585330c1ce52a8c14ad1d4b1da6fd8397e1d15592707a6d5 ]]

[[ "$(sha256sum "${AMENDMENT_RECEIPT}" | awk '{print $1}')" == \
   "${EXPECTED_AMENDMENT_RECEIPT_SHA}" ]]
(cd "${AMENDMENT_ROOT}" && sha256sum -c --quiet "${AMENDMENT_RECEIPT}")
[[ "$(sha256sum "${TASK_RECEIPT}" | awk '{print $1}')" == \
   "${EXPECTED_TASK_RECEIPT_SHA}" ]]
(cd "${TASK_ROOT}" && sha256sum -c --quiet "${TASK_RECEIPT}")
[[ "$(sha256sum "${BASE_RECEIPT}" | awk '{print $1}')" == \
   "${EXPECTED_BASE_RECEIPT_SHA}" ]]
(cd "${BASE_SOURCE_ROOT}" && sha256sum -c --quiet "${BASE_RECEIPT}")
[[ "$(sha256sum "${DEPENDENCY_RECEIPT}" | awk '{print $1}')" == \
   "${EXPECTED_DEPENDENCY_RECEIPT_SHA}" ]]

audit=${AMENDMENT_ROOT}/MemNavData/audit_hm3d_lifelong_underpowered_amendment.py
safe_submit=${AMENDMENT_ROOT}/MemNavData/slurm_safe_submit.sh
arm=${AMENDMENT_ROOT}/MemNavData/slurm_hm3d_fullmono_shared_c_arm.sbatch
finalize=${TASK_ROOT}/MemNavData/slurm_hm3d_fullmono_shared_c_finalize.sbatch
analysis=${TASK_ROOT}/MemNavData/slurm_hm3d_fullmono_shared_c_analysis.sbatch
for path in "${audit}" "${safe_submit}" "${arm}" "${finalize}" "${analysis}"; do
  [[ -r "${path}" ]]
done

audit_args=(
  "${audit}" --protocol "${PROTOCOL}" --run-root "${RUN_ROOT}"
)
if [[ "${DEFERRED_MODE}" == collect ]]; then audit_args+=(--require-pristine); fi
/scratch/lg154/conda-envs/memnav/bin/python "${audit_args[@]}" >/dev/null

source "${safe_submit}"
lint_sbatch_template "${arm}"
lint_sbatch_template "${finalize}"
lint_sbatch_template "${analysis}"

common="ALL,AMENDMENT_ROOT=${AMENDMENT_ROOT},AMENDMENT_RECEIPT=${AMENDMENT_RECEIPT},EXPECTED_AMENDMENT_RECEIPT_SHA=${EXPECTED_AMENDMENT_RECEIPT_SHA},TASK_ROOT=${TASK_ROOT},BASE_SOURCE_ROOT=${BASE_SOURCE_ROOT},RUN_ROOT=${RUN_ROOT},PARENT_ROOT=/scratch/yz11502/Research/Nav-axis-uturn-results/hm3d_fresh_fullmono_mixed_role_20260820/formal_20260820T143609Z_e6dd44c6,PROTOCOL=${PROTOCOL},TASK_RECEIPT=${TASK_RECEIPT},EXPECTED_TASK_RECEIPT_SHA=${EXPECTED_TASK_RECEIPT_SHA},BASE_RECEIPT=${BASE_RECEIPT},EXPECTED_BASE_RECEIPT_SHA=${EXPECTED_BASE_RECEIPT_SHA},DEPENDENCY_RECEIPT=${DEPENDENCY_RECEIPT},EXPECTED_DEPENDENCY_RECEIPT_SHA=${EXPECTED_DEPENDENCY_RECEIPT_SHA},EXPECTED_HAB_REQUESTS_VERSION=2.32.4,EXPECTED_HAB_REQUESTS_INIT_BYTES=5057,EXPECTED_HAB_REQUESTS_INIT_SHA=1e507f1f386bcc6b5f0ff69a614c14875cd65cb67be7f6022f28adef9774573f,EXPECTED_HAB_REQUESTS_VERSION_BYTES=435,EXPECTED_HAB_REQUESTS_VERSION_SHA=143abaf3563712f063743a7952aa65319dbcb934d894cfc989bd2c015f8da577"
continuation="${common},AMENDMENT_ROOT=${AMENDMENT_ROOT},AMENDMENT_RECEIPT=${AMENDMENT_RECEIPT},EXPECTED_AMENDMENT_RECEIPT_SHA=${EXPECTED_AMENDMENT_RECEIPT_SHA},SMOKE_ROOT=${SMOKE_ROOT},SOURCE_POPULATION_SHA=${SOURCE_POPULATION_SHA},SOURCE_POPULATION_COUNT=${SOURCE_POPULATION_COUNT},SOURCE_SCENE_COUNT=${SOURCE_SCENE_COUNT},EVAL_CONCURRENCY=${EVAL_CONCURRENCY},GPU_TIME_LIMIT=${GPU_TIME_LIMIT},UPSTREAM_POPULATION_SEAL_JOB_ID=${UPSTREAM_POPULATION_SEAL_JOB_ID}"
receipt_root=${RUN_ROOT}/underpowered_deferred_submission
mkdir -p "${receipt_root}"

job_id() {
  local raw=$1 value=${1%%;*}
  [[ "${value}" =~ ^[0-9]+$ ]] || {
    echo "invalid sbatch response: ${raw}" >&2
    return 2
  }
  printf '%s' "${value}"
}

sealed_population_count() {
  local population=$1 root=${1%/*}
  [[ -f "${root}/SEALED" && -f "${root}/population.json.sha256" ]]
  (cd "${root}" && sha256sum -c --quiet population.json.sha256)
  /scratch/lg154/conda-envs/memnav/bin/python - "${population}" <<'PY'
import json,sys
p=json.load(open(sys.argv[1])); rows=p.get("accepted")
if not isinstance(rows,list) or not rows: raise SystemExit("empty population")
if len(rows)>22: raise SystemExit("population exceeds frozen source")
print(len(rows))
PY
}

write_receipt() {
  local path=$1 mode=$2 count=$3
  shift 3
  /scratch/lg154/conda-envs/memnav/bin/python - "${path}" "${mode}" \
    "${count}" "${RUN_ROOT}" "${AMENDMENT_ROOT}" "$@" <<'PY'
import json,sys
path,mode,count,run,bundle,*pairs=sys.argv[1:]
if len(pairs)%2: raise SystemExit("receipt key/value mismatch")
jobs={pairs[i]:int(pairs[i+1]) for i in range(0,len(pairs),2)}
payload={
 "schema_version":"hm3d_lifelong_underpowered_deferred_submission_v1_20260828",
 "mode":mode,"run_root":run,"amendment_bundle":bundle,
 "sealed_population_count":int(count),"jobs":jobs,
 "underpowered":True,"powered_confirmation_claim":False,
 "one_large_GPU_array_resident_per_stage":True,
 "GPU_time_limit":"01:00:00",
}
with open(path,"x") as f:
 json.dump(payload,f,indent=2,sort_keys=True); f.write("\n")
PY
  sha256sum "${path}" >"${path}.sha256"
}

case "${DEFERRED_MODE}" in
  collect)
    [[ ! -e "${RUN_ROOT}/shared_c_collection" ]]
    count=$(sealed_population_count "${RUN_ROOT}/population/population.json")
    [[ "${count}" == "${SOURCE_POPULATION_COUNT}" ]]
    array="0-$((count - 1))%${EVAL_CONCURRENCY}"
    raw=$(safe_sbatch --lint-fatal --parsable --qos=gpu48 \
      --time="${GPU_TIME_LIMIT}" --array="${array}" \
      --export="${common},STAGE=collect" "${arm}")
    collect=$(job_id "${raw}")
    raw=$(safe_sbatch --lint-fatal --parsable --dependency="afterany:${collect}" \
      --export="${common}" "${finalize}")
    seal=$(job_id "${raw}")
    raw=$(safe_sbatch --lint-fatal --parsable --qos=gpu48 \
      --time="${GPU_TIME_LIMIT}" --array=0 \
      --dependency="afterok:${seal}" --kill-on-invalid-dep=yes \
      --export="${common},STAGE=evaluate,OUTPUT_ROOT=${SMOKE_ROOT},MAX_STEPS=80" \
      "${arm}")
    smoke=$(job_id "${raw}")
    raw=$(safe_sbatch --lint-fatal --parsable --dependency="afterok:${smoke}" \
      --kill-on-invalid-dep=yes \
      --export="${continuation},DEFERRED_MODE=evaluate" "$0")
    next=$(job_id "${raw}")
    write_receipt "${receipt_root}/collect.json" collect "${count}" \
      factual_C_collection "${collect}" factual_C_seal "${seal}" \
      B2_true_stack_smoke "${smoke}" B2_launcher "${next}"
    ;;
  evaluate)
    [[ -f "${RUN_ROOT}/shared_c_population/SEALED" ]]
    [[ ! -e "${RUN_ROOT}/shared_c_evaluation" ]]
    count=$(sealed_population_count \
      "${RUN_ROOT}/shared_c_population/population.json")
    array="0-$((count - 1))%${EVAL_CONCURRENCY}"
    raw=$(safe_sbatch --lint-fatal --parsable --qos=gpu48 \
      --time="${GPU_TIME_LIMIT}" --array="${array}" \
      --export="${common},STAGE=evaluate,MAX_STEPS=600" "${arm}")
    evaluate=$(job_id "${raw}")
    raw=$(safe_sbatch --lint-fatal --parsable \
      --dependency="afterany:${evaluate}" --export="${common},MODE=aggregate" \
      "${analysis}")
    aggregate=$(job_id "${raw}")
    raw=$(safe_sbatch --lint-fatal --parsable \
      --dependency="afterok:${aggregate}" --kill-on-invalid-dep=yes \
      --export="${common},MODE=verify" "${analysis}")
    verify=$(job_id "${raw}")
    write_receipt "${receipt_root}/evaluate.json" evaluate "${count}" \
      formal_B2_evaluation "${evaluate}" aggregate "${aggregate}" \
      independent_verification "${verify}"
    ;;
esac
