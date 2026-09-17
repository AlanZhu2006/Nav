#!/usr/bin/env bash
set -euo pipefail
[[ "$(id -un)" == yz11502 ]]
source "${REPAIRED_BUNDLE}/MemNavData/repaired_hm3d_hpc_env.sh"
cd "${REPAIRED_BUNDLE}"
sha256sum -c --quiet GEM_SOURCE.sha256
: "${EXPECTED_SUPPORT_PLAN_SHA:?}"
[[ "$(sha256sum "${GEM_SUPPORT_PLAN}" | awk '{print $1}')" == "${EXPECTED_SUPPORT_PLAN_SHA}" ]]
export PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1
source "${REPAIRED_BUNDLE}/MemNavData/slurm_port_pair.sh"
claim_slurm_tcp_port_block gem_support_nav 2 12000 6000
trap release_slurm_tcp_port_block EXIT
"${REPAIRED_MEM_PY}" -u MemNavData/run_gem_support_navigation.py pair \
  --plan "${GEM_SUPPORT_PLAN}" --index "${SLURM_ARRAY_TASK_ID}" \
  --out "${GEM_SUPPORT_RUN}/tasks/${SLURM_ARRAY_TASK_ID}" \
  --mem-port "${CEC_PORT_BLOCK_BASE}" --nav-port "$(( CEC_PORT_BLOCK_BASE + 1 ))"
