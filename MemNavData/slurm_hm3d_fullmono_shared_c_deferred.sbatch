#!/usr/bin/env bash
# Submit only one large HM3D array at a time.  Torch's gpu48 QOS counts every
# pending array element against MaxSubmitJobs, including dependency-held work.
# This CPU launcher is therefore chained twice: collection first, evaluation
# only after the sealed-population true-stack smoke succeeds.
#SBATCH --job-name=h3Cdefer
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
: "${DEFERRED_MODE:?collect|evaluate}" "${DEFERRED_SCRIPT:?}"
: "${EXPECTED_DEFERRED_SCRIPT_SHA:?}" "${TASK_ROOT:?}" "${RUN_ROOT:?}"
: "${SMOKE_ROOT:?}" "${EVAL_CONCURRENCY:?}" "${UPSTREAM_SEAL_JOB_ID:?}"
: "${BASE_SOURCE_ROOT:?}" "${PARENT_ROOT:?}" "${PROTOCOL:?}"
: "${TASK_RECEIPT:?}" "${EXPECTED_TASK_RECEIPT_SHA:?}"
: "${BASE_RECEIPT:?}" "${EXPECTED_BASE_RECEIPT_SHA:?}"
: "${DEPENDENCY_RECEIPT:?}" "${EXPECTED_DEPENDENCY_RECEIPT_SHA:?}"
: "${EXPECTED_HAB_REQUESTS_VERSION:?}" "${EXPECTED_HAB_REQUESTS_INIT_BYTES:?}"
: "${EXPECTED_HAB_REQUESTS_INIT_SHA:?}"
: "${EXPECTED_HAB_REQUESTS_VERSION_BYTES:?}"
: "${EXPECTED_HAB_REQUESTS_VERSION_SHA:?}"

[[ "$(sha256sum "${DEFERRED_SCRIPT}" | awk '{print $1}')" == \
   "${EXPECTED_DEFERRED_SCRIPT_SHA}" ]]
[[ "$(sha256sum "${TASK_RECEIPT}" | awk '{print $1}')" == \
   "${EXPECTED_TASK_RECEIPT_SHA}" ]]
[[ "${EVAL_CONCURRENCY}" =~ ^[1-9][0-9]*$ ]]

arm=${TASK_ROOT}/MemNavData/slurm_hm3d_fullmono_shared_c_arm.sbatch
finalize=${TASK_ROOT}/MemNavData/slurm_hm3d_fullmono_shared_c_finalize.sbatch
analysis=${TASK_ROOT}/MemNavData/slurm_hm3d_fullmono_shared_c_analysis.sbatch
for path in "${arm}" "${finalize}" "${analysis}"; do [[ -r "${path}" ]]; done

common="ALL,TASK_ROOT=${TASK_ROOT},BASE_SOURCE_ROOT=${BASE_SOURCE_ROOT},RUN_ROOT=${RUN_ROOT},PARENT_ROOT=${PARENT_ROOT},PROTOCOL=${PROTOCOL},TASK_RECEIPT=${TASK_RECEIPT},EXPECTED_TASK_RECEIPT_SHA=${EXPECTED_TASK_RECEIPT_SHA},BASE_RECEIPT=${BASE_RECEIPT},EXPECTED_BASE_RECEIPT_SHA=${EXPECTED_BASE_RECEIPT_SHA},DEPENDENCY_RECEIPT=${DEPENDENCY_RECEIPT},EXPECTED_DEPENDENCY_RECEIPT_SHA=${EXPECTED_DEPENDENCY_RECEIPT_SHA},EXPECTED_HAB_REQUESTS_VERSION=${EXPECTED_HAB_REQUESTS_VERSION},EXPECTED_HAB_REQUESTS_INIT_BYTES=${EXPECTED_HAB_REQUESTS_INIT_BYTES},EXPECTED_HAB_REQUESTS_INIT_SHA=${EXPECTED_HAB_REQUESTS_INIT_SHA},EXPECTED_HAB_REQUESTS_VERSION_BYTES=${EXPECTED_HAB_REQUESTS_VERSION_BYTES},EXPECTED_HAB_REQUESTS_VERSION_SHA=${EXPECTED_HAB_REQUESTS_VERSION_SHA}"
deferred_common="${common},SMOKE_ROOT=${SMOKE_ROOT},EVAL_CONCURRENCY=${EVAL_CONCURRENCY},UPSTREAM_SEAL_JOB_ID=${UPSTREAM_SEAL_JOB_ID},DEFERRED_SCRIPT=${DEFERRED_SCRIPT},EXPECTED_DEFERRED_SCRIPT_SHA=${EXPECTED_DEFERRED_SCRIPT_SHA}"
receipt_root=${RUN_ROOT}/deferred_submission
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
  [[ -f "${root}/SEALED" ]]
  [[ -f "${root}/population.json.sha256" ]]
  (cd "${root}" && sha256sum -c --quiet population.json.sha256)
  /scratch/lg154/conda-envs/memnav/bin/python - "${population}" <<'PY'
import json
import sys

payload = json.load(open(sys.argv[1]))
rows = payload.get("accepted")
if not isinstance(rows, list) or not rows:
    raise SystemExit("sealed population has no accepted rows")
if len(rows) > 260:
    raise SystemExit(
        f"sealed population exceeds frozen maximum: {len(rows)} > 260")
print(len(rows))
PY
}

write_receipt() {
  local path=$1 mode=$2 population_count=$3
  shift 3
  /scratch/lg154/conda-envs/memnav/bin/python - "${path}" "${mode}" \
    "${population_count}" \
    "${RUN_ROOT}" "${TASK_ROOT}" "${UPSTREAM_SEAL_JOB_ID}" "$@" <<'PY'
import json,sys
from pathlib import Path
path,mode,population_count,run_root,task_root,upstream,*pairs=sys.argv[1:]
if len(pairs) % 2: raise SystemExit("receipt key/value mismatch")
jobs={pairs[i]:int(pairs[i+1]) for i in range(0,len(pairs),2)}
payload={
  "schema_version":"hm3d_shared_c_deferred_submission_v2_20260825",
  "mode":mode,"run_root":run_root,"task_root":task_root,
  "upstream_population_seal_job":int(upstream),"jobs":jobs,
  "sealed_population_count":int(population_count),
  "submitted_array":f"0-{int(population_count) - 1}",
  "one_large_array_resident_per_stage":True,
}
p=Path(path)
with p.open("x") as handle:
    json.dump(payload,handle,indent=2,sort_keys=True); handle.write("\n")
PY
  sha256sum "${path}" >"${path}.sha256"
}

case "${DEFERRED_MODE}" in
  collect)
    [[ ! -e "${RUN_ROOT}/shared_c_collection" ]]
    population_count=$(sealed_population_count \
      "${RUN_ROOT}/population/population.json")
    array_spec="0-$((population_count - 1))%${EVAL_CONCURRENCY}"
    raw=$(sbatch --parsable --qos=gpu48 --array="${array_spec}" \
      --export="${common},STAGE=collect" "${arm}")
    collect=$(job_id "${raw}")
    raw=$(sbatch --parsable --dependency="afterany:${collect}" \
      --export="${common}" "${finalize}")
    seal=$(job_id "${raw}")
    raw=$(sbatch --parsable --qos=gpu48 --array=0 \
      --dependency="afterok:${seal}" --kill-on-invalid-dep=yes \
      --export="${common},STAGE=evaluate,OUTPUT_ROOT=${SMOKE_ROOT},MAX_STEPS=80" \
      "${arm}")
    smoke=$(job_id "${raw}")
    raw=$(sbatch --parsable --dependency="afterok:${smoke}" \
      --kill-on-invalid-dep=yes \
      --export="${deferred_common},DEFERRED_MODE=evaluate" \
      "${DEFERRED_SCRIPT}")
    continuation=$(job_id "${raw}")
    write_receipt "${receipt_root}/collect.json" collect \
      "${population_count}" \
      shared_C_collection "${collect}" shared_C_seal "${seal}" \
      true_stack_smoke "${smoke}" evaluation_launcher "${continuation}"
    ;;
  evaluate)
    [[ -f "${RUN_ROOT}/shared_c_population/SEALED" ]]
    [[ ! -e "${RUN_ROOT}/shared_c_evaluation" ]]
    population_count=$(sealed_population_count \
      "${RUN_ROOT}/shared_c_population/population.json")
    array_spec="0-$((population_count - 1))%${EVAL_CONCURRENCY}"
    raw=$(sbatch --parsable --qos=gpu48 --array="${array_spec}" \
      --export="${common},STAGE=evaluate,MAX_STEPS=600" "${arm}")
    evaluation=$(job_id "${raw}")
    raw=$(sbatch --parsable --dependency="afterany:${evaluation}" \
      --export="${common},MODE=aggregate" "${analysis}")
    aggregate=$(job_id "${raw}")
    raw=$(sbatch --parsable --dependency="afterok:${aggregate}" \
      --kill-on-invalid-dep=yes --export="${common},MODE=verify" "${analysis}")
    verify=$(job_id "${raw}")
    write_receipt "${receipt_root}/evaluate.json" evaluate \
      "${population_count}" \
      formal_B2_evaluation "${evaluation}" aggregate "${aggregate}" \
      independent_verification "${verify}"
    ;;
  *) echo "invalid DEFERRED_MODE=${DEFERRED_MODE}" >&2; exit 2;;
esac
