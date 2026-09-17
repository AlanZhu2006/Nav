#!/usr/bin/env bash
set -euo pipefail
[[ "$(id -un)" == yz11502 ]]
source "${REPAIRED_BUNDLE}/MemNavData/repaired_hm3d_hpc_env.sh"
cd "${REPAIRED_BUNDLE}"
sha256sum -c --quiet GEM_SOURCE.sha256
: "${EXPECTED_GEM_PLAN_SHA:?}"
[[ "$(sha256sum "${GEM_NAV_PLAN}" | awk '{print $1}')" == "${EXPECTED_GEM_PLAN_SHA}" ]]
export PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1
GEM_NAV_MODES=(legacy native_interval7 connected_reciprocal)
GEM_NAV_CELL=$(( SLURM_ARRAY_TASK_ID / 3 ))
GEM_NAV_MODE=${GEM_NAV_MODES[$(( SLURM_ARRAY_TASK_ID % 3 ))]}
source "${REPAIRED_BUNDLE}/MemNavData/slurm_port_pair.sh"
claim_slurm_tcp_port_block gem_memory_nav 2 12000 6000
trap release_slurm_tcp_port_block EXIT
"${REPAIRED_MEM_PY}" -u MemNavData/run_gem_memory_navigation.py run \
  --plan "${GEM_NAV_PLAN}" --index "${GEM_NAV_CELL}" --mode "${GEM_NAV_MODE}" \
  --out "${GEM_NAV_RUN}/tasks/${SLURM_ARRAY_TASK_ID}" \
  --mem-port "${CEC_PORT_BLOCK_BASE}" --nav-port "$(( CEC_PORT_BLOCK_BASE + 1 ))"
