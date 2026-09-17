#!/usr/bin/env bash
set -euo pipefail
[[ "$(id -un)" == yz11502 ]]
source "${REPAIRED_BUNDLE}/MemNavData/repaired_hm3d_hpc_env.sh"
cd "${REPAIRED_BUNDLE}"
sha256sum -c --quiet GEM_SOURCE.sha256
[[ "$(sha256sum "${GEM_REVIEWER_PLAN}" | awk '{print $1}')" == "${EXPECTED_REVIEWER_PLAN_SHA}" ]]
source "${REPAIRED_BUNDLE}/MemNavData/slurm_port_pair.sh"
claim_slurm_tcp_port_block gem_position 2 12000 6000
trap release_slurm_tcp_port_block EXIT
"${REPAIRED_MEM_PY}" -u MemNavData/run_gem_position_readout.py run \
  --plan "${GEM_REVIEWER_PLAN}" --index "${SLURM_ARRAY_TASK_ID}" \
  --out "${GEM_REVIEWER_RUN}/tasks/${SLURM_ARRAY_TASK_ID}" \
  --mem-port "${CEC_PORT_BLOCK_BASE}" --nav-port "$(( CEC_PORT_BLOCK_BASE + 1 ))"
