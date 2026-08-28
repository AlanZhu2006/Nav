#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
source "${ROOT}/MemNavData/slurm_port_pair.sh"

SLURM_JOB_ID=991337
SLURM_ARRAY_TASK_ID=7
claim_slurm_tcp_port_pair test_contract 12000 6000
[[ "${MEMNAV_PORT}" =~ ^[0-9]+$ ]]
[[ "${NAVDP_PORT}" == "$((MEMNAV_PORT + 1))" ]]
[[ -n "${CEC_PORT_PAIR_LOCK_FD}" ]]
[[ -e "${CEC_PORT_PAIR_LOCK_PATH}" ]]
! ss -H -ltn | awk '{print $4}' | grep -Eq \
  "(^|:)(${MEMNAV_PORT}|${NAVDP_PORT})$"
release_slurm_tcp_port_pair
[[ -z "${CEC_PORT_PAIR_LOCK_FD:-}" ]]

SLURM_ARRAY_TASK_ID=8
claim_slurm_tcp_port_block test_block 5 12000 2400
[[ "${CEC_PORT_BLOCK_WIDTH}" == 5 ]]
for ((offset = 0; offset < 5; offset++)); do
  port=$((CEC_PORT_BLOCK_BASE + offset))
  ! ss -H -ltn | awk '{print $4}' | grep -Eq "(^|:)${port}$"
done
release_slurm_tcp_port_block
[[ -z "${CEC_PORT_BLOCK_LOCK_FD:-}" ]]
echo "PORT_PAIR_TEST_OK"
