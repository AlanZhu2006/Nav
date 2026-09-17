#!/usr/bin/env bash
set -euo pipefail
: "${REPAIRED_BUNDLE:?}" "${TABLE2_RUN:?}" "${TABLE2_PLAN:?}" "${TABLE2_STAGE:?}"
: "${TABLE2_WORK_ROOT:?}" "${EXPECTED_RUNTIME_SHA:?}" "${EXPECTED_PLAN_SHA:?}" "${SLURM_ARRAY_TASK_ID:?}"
[[ "$(id -un)" == yz11502 ]]
[[ "${TABLE2_STAGE}" =~ ^[ABC]$ && "${SLURM_ARRAY_TASK_ID}" =~ ^[0-9]+$ ]]
source "${REPAIRED_BUNDLE}/MemNavData/repaired_hm3d_hpc_env.sh"
TABLE2_TMP=$(mktemp -d "${TABLE2_WORK_ROOT}/temporary_XXXXXX")
export TMPDIR=${TABLE2_TMP} LIBFFI_TMPDIR=${TABLE2_TMP} PYTHONFAULTHANDLER=1
TABLE2_DURABLE=$(printf '%s/%s/task_%03d' "${TABLE2_RUN}" "${TABLE2_STAGE}" "${SLURM_ARRAY_TASK_ID}")
TABLE2_OUT=$(printf '%s/%s/task_%03d/task' "${TABLE2_WORK_ROOT}" "${TABLE2_STAGE}" "${SLURM_ARRAY_TASK_ID}")
[[ ! -e "${TABLE2_DURABLE}" ]]
mkdir -p "${TABLE2_DURABLE}/preflight"
[[ "$(sha256sum "${REPAIRED_SOURCE_RECEIPT}" | awk '{print $1}')" == "${EXPECTED_RUNTIME_SHA}" ]]
[[ "$(sha256sum "${TABLE2_PLAN}" | awk '{print $1}')" == "${EXPECTED_PLAN_SHA}" ]]
cd "${REPAIRED_BUNDLE}"
sha256sum -c --quiet SOURCE_BUNDLE.sha256
"${REPAIRED_MEM_PY}" -c 'import json,os; p=json.load(open(os.environ["TABLE2_PLAN"])); assert p["work_root"]==os.environ["TABLE2_WORK_ROOT"]'
for kind in habitat memnav navdp; do
  if [[ "${kind}" == habitat ]]; then
    "${REPAIRED_HAB_PY}" MemNavData/preflight_repaired_fullmono.py "${kind}" >"${TABLE2_DURABLE}/preflight/${kind}.log" 2>&1
  else
    env PYTHONPATH="${REPAIRED_BUNDLE}/NavDP/baselines/${kind}:${PYTHONPATH}" \
      "${REPAIRED_MEM_PY}" MemNavData/preflight_repaired_fullmono.py "${kind}" >"${TABLE2_DURABLE}/preflight/${kind}.log" 2>&1
  fi
done
source "${REPAIRED_BUNDLE}/MemNavData/slurm_port_pair.sh"
claim_slurm_tcp_port_pair table2_mixed 12000 6000
trap release_slurm_tcp_port_pair EXIT
nvidia-smi --query-gpu=name,uuid,driver_version,memory.total --format=csv >"${TABLE2_DURABLE}/preflight/gpu.csv"
set +e
timeout --signal=INT --kill-after=45s 48m "${REPAIRED_MEM_PY}" -u MemNavData/table2_mixed_hpc.py run \
  --stage "${TABLE2_STAGE}" --plan "${TABLE2_PLAN}" --run "${TABLE2_RUN}" \
  --index "${SLURM_ARRAY_TASK_ID}" --mem-port "${MEMNAV_PORT}" --nav-port "${NAVDP_PORT}"
TABLE2_EXIT=$?
set -e
if [[ -d "${TABLE2_OUT}" ]]; then
  "${REPAIRED_MEM_PY}" MemNavData/covisibility_task_archive.py --out "${TABLE2_OUT}" \
    --durable "${TABLE2_DURABLE}" --exit-code "${TABLE2_EXIT}" >"${TABLE2_DURABLE}/archive.log" 2>&1
fi
exit "${TABLE2_EXIT}"
