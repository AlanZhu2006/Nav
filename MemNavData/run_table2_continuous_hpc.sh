#!/usr/bin/env bash
set -euo pipefail
: "${REPAIRED_BUNDLE:?}" "${TABLE2_CONTINUOUS_RUN:?}" "${TABLE2_CONTINUOUS_WORK:?}"
: "${EXPECTED_RUNTIME_SHA:?}" "${EXPECTED_POPULATION_SHA:?}" "${TABLE2_TASK_INDEX:?}"
[[ "$(id -un)" == yz11502 ]]
source "${REPAIRED_BUNDLE}/MemNavData/repaired_hm3d_hpc_env.sh"
unset REPAIRED_BUFFER_ROOT REPAIRED_RUNTIME_ROOT
export PYTHONFAULTHANDLER=1
TABLE2_CONTINUOUS_TMP=$(mktemp -d "${TABLE2_CONTINUOUS_WORK}/temporary_XXXXXX")
export TMPDIR=${TABLE2_CONTINUOUS_TMP} LIBFFI_TMPDIR=${TABLE2_CONTINUOUS_TMP}
export TABLE2_POPULATION=${REPAIRED_BUNDLE}/continuous_population/population.json
cd "${REPAIRED_BUNDLE}"
[[ "$(sha256sum SOURCE_BUNDLE.sha256 | awk '{print $1}')" == "${EXPECTED_RUNTIME_SHA}" ]]
[[ "$(sha256sum "${TABLE2_POPULATION}" | awk '{print $1}')" == "${EXPECTED_POPULATION_SHA}" ]]
sha256sum -c --quiet SOURCE_BUNDLE.sha256
TABLE2_TASK_RUN=${TABLE2_CONTINUOUS_RUN}/tasks/$(printf '%03d' "${TABLE2_TASK_INDEX}")
[[ ! -e "${TABLE2_TASK_RUN}/archive_receipt.json" && ! -e "${TABLE2_TASK_RUN}/artifacts.tar.gz" ]]
TABLE2_PREFLIGHT=${TABLE2_TASK_RUN}/preflight
mkdir -p "${TABLE2_PREFLIGHT}"
for kind in habitat memnav navdp; do
  if [[ "${kind}" == habitat ]]; then
    "${REPAIRED_HAB_PY}" MemNavData/preflight_repaired_fullmono.py "${kind}" >"${TABLE2_PREFLIGHT}/${kind}.log" 2>&1
  else
    env PYTHONPATH="${REPAIRED_BUNDLE}/NavDP/baselines/${kind}:${PYTHONPATH}" \
      "${REPAIRED_MEM_PY}" MemNavData/preflight_repaired_fullmono.py "${kind}" >"${TABLE2_PREFLIGHT}/${kind}.log" 2>&1
  fi
done
"${REPAIRED_MEM_PY}" - <<'PY'
import os
from MemNavData.table2_continuous_population import task_at
from MemNavData.table2_mixed_local import sha
from MemNavData.verify_table2_common_pair import check
t=task_at(os.environ['TABLE2_POPULATION'], int(os.environ['TABLE2_TASK_INDEX']))
s=t['source']
assert s['prefix_root'] is None and s['prefix_trace_sha256'] is None
assert sha(s['asset']) == s['asset_sha256']
print('Frozen population, source asset, and paired verifier checked:', t['source_id'], t['sequence'])
PY
"${REPAIRED_MEM_PY}" -u -m MemNavData.table2_continuous_paired \
  --population "${TABLE2_POPULATION}" --task-index "${TABLE2_TASK_INDEX}" \
  --out "${TABLE2_CONTINUOUS_WORK}/cli_task" --prepare-only >"${TABLE2_PREFLIGHT}/formal_cli.log" 2>&1
if [[ "${1:-run}" == preflight ]]; then exit 0; fi
nvidia-smi --query-gpu=name,uuid,driver_version,memory.total --format=csv >"${TABLE2_PREFLIGHT}/gpu.csv"
"${REPAIRED_MEM_PY}" - <<'PY'
import subprocess
x=subprocess.check_output(['nvidia-smi','--query-gpu=memory.total','--format=csv,noheader,nounits'],text=True)
assert min(map(int,x.strip().splitlines())) >= 78000, 'Two live histories require the declared 80 GB card'
PY
source "${REPAIRED_BUNDLE}/MemNavData/slurm_port_pair.sh"
claim_slurm_tcp_port_block table2_continuous 4 12000 6000
trap release_slurm_tcp_port_block EXIT
[[ ! -e "${TABLE2_CONTINUOUS_WORK}/task" ]]
set +e
timeout --signal=INT --kill-after=45s 45m "${REPAIRED_MEM_PY}" -u -m MemNavData.table2_continuous_paired \
  --out "${TABLE2_CONTINUOUS_WORK}/task" --population "${TABLE2_POPULATION}" \
  --task-index "${TABLE2_TASK_INDEX}" --port "${CEC_PORT_BLOCK_BASE}"
TABLE2_CONTINUOUS_EXIT=$?
if [[ "${TABLE2_CONTINUOUS_EXIT}" == 0 ]]; then
  "${REPAIRED_MEM_PY}" -m MemNavData.verify_table2_common_pair --run "${TABLE2_CONTINUOUS_WORK}/task"
  TABLE2_CONTINUOUS_EXIT=$?
fi
set -e
if [[ -d "${TABLE2_CONTINUOUS_WORK}/task" ]]; then
  "${REPAIRED_MEM_PY}" MemNavData/covisibility_task_archive.py --out "${TABLE2_CONTINUOUS_WORK}/task" \
    --durable "${TABLE2_TASK_RUN}" --exit-code "${TABLE2_CONTINUOUS_EXIT}" >"${TABLE2_TASK_RUN}/archive.log" 2>&1
fi
exit "${TABLE2_CONTINUOUS_EXIT}"
