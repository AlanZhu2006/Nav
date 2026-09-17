#!/usr/bin/env bash
set -euo pipefail
: "${REPAIRED_BUNDLE:?}" "${TABLE2_COMMON_SOURCE:?}" "${TABLE2_COMMON_RUN:?}"
: "${TABLE2_COMMON_WORK:?}" "${EXPECTED_RUNTIME_SHA:?}" "${EXPECTED_SOURCE_SHA:?}"
[[ "$(id -un)" == yz11502 ]]
source "${REPAIRED_BUNDLE}/MemNavData/repaired_hm3d_hpc_env.sh"
unset REPAIRED_BUFFER_ROOT REPAIRED_RUNTIME_ROOT
export PYTHONFAULTHANDLER=1
TABLE2_COMMON_TMP=$(mktemp -d "${TABLE2_COMMON_WORK}/temporary_XXXXXX")
export TMPDIR=${TABLE2_COMMON_TMP} LIBFFI_TMPDIR=${TABLE2_COMMON_TMP}
cd "${REPAIRED_BUNDLE}"
[[ "$(sha256sum SOURCE_BUNDLE.sha256 | awk '{print $1}')" == "${EXPECTED_RUNTIME_SHA}" ]]
[[ "$(sha256sum "${TABLE2_COMMON_SOURCE}" | awk '{print $1}')" == "${EXPECTED_SOURCE_SHA}" ]]
sha256sum -c --quiet SOURCE_BUNDLE.sha256
TABLE2_PREFLIGHT=${TABLE2_COMMON_RUN}/preflight
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
import json, os
from pathlib import Path
from MemNavData.habitat_executor_audit import sha
s=json.loads(Path(os.environ['TABLE2_COMMON_SOURCE']).read_text())
assert s['prefix_root'] is None and s['prefix_trace_sha256'] is None
assert sha(Path(s['asset'])) == s['asset_sha256']
from MemNavData.table2_continuous_paired import worker, publish, await_record
from MemNavData.table2_common_goals import construct_common
from MemNavData.private_gpu_cache_server import install
from MemNavData.verify_table2_common_pair import check
print('common-chain imports and scene asset verified')
PY
if [[ "${1:-run}" == preflight ]]; then
  exit 0
fi
nvidia-smi --query-gpu=name,uuid,driver_version,memory.total --format=csv >"${TABLE2_PREFLIGHT}/gpu.csv"
"${REPAIRED_MEM_PY}" - <<'PY'
import subprocess
x=subprocess.check_output(['nvidia-smi','--query-gpu=memory.total','--format=csv,noheader,nounits'],text=True)
assert min(map(int,x.strip().splitlines())) >= 78000, 'Two live histories require the declared 80 GB card'
PY
source "${REPAIRED_BUNDLE}/MemNavData/slurm_port_pair.sh"
claim_slurm_tcp_port_block table2_common 4 12000 6000
trap release_slurm_tcp_port_block EXIT
[[ ! -e "${TABLE2_COMMON_WORK}/task" && ! -e "${TABLE2_COMMON_RUN}/artifacts.tar.gz" ]]
set +e
timeout --signal=INT --kill-after=45s 45m "${REPAIRED_MEM_PY}" -u -m MemNavData.table2_continuous_paired \
  --out "${TABLE2_COMMON_WORK}/task" --source-file "${TABLE2_COMMON_SOURCE}" --source-index 0 \
  --sequence NRR --port "${CEC_PORT_BLOCK_BASE}"
TABLE2_COMMON_EXIT=$?
if [[ "${TABLE2_COMMON_EXIT}" == 0 ]]; then
  "${REPAIRED_MEM_PY}" -m MemNavData.verify_table2_common_pair --run "${TABLE2_COMMON_WORK}/task"
  TABLE2_COMMON_EXIT=$?
fi
set -e
if [[ -d "${TABLE2_COMMON_WORK}/task" ]]; then
  "${REPAIRED_MEM_PY}" MemNavData/covisibility_task_archive.py --out "${TABLE2_COMMON_WORK}/task" \
    --durable "${TABLE2_COMMON_RUN}" --exit-code "${TABLE2_COMMON_EXIT}" >"${TABLE2_COMMON_RUN}/archive.log" 2>&1
fi
exit "${TABLE2_COMMON_EXIT}"
