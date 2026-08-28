#!/usr/bin/env bash
# Claim one consecutive TCP port pair for the lifetime of a Slurm cell.
#
# Deterministic arithmetic alone is not sufficient on shared multi-GPU nodes:
# two unrelated jobs can probe the same free port and race while their model
# servers initialize.  This helper combines a job-keyed candidate order, a
# node-local flock held by the parent shell, and a live listener check.

claim_slurm_tcp_port_block() {
  local namespace=${1:?claim_slurm_tcp_port_block requires a namespace}
  local width=${2:?claim_slurm_tcp_port_block requires a width}
  local first_port=${3:-12000}
  local block_count=${4:-6000}
  local attempt key checksum slot base lock candidate_fd offset occupied

  [[ "${namespace}" =~ ^[A-Za-z0-9_.-]+$ ]] || {
    echo "invalid port namespace: ${namespace}" >&2; return 2; }
  [[ "${width}" =~ ^[0-9]+$ && "${first_port}" =~ ^[0-9]+$ \
     && "${block_count}" =~ ^[0-9]+$ ]] || {
    echo "invalid port range" >&2; return 2; }
  (( width >= 1 && width <= 16 && first_port >= 1024 \
     && block_count >= 32 \
     && first_port + width * block_count - 1 <= 65535 )) || {
    echo "unsafe port range" >&2; return 2; }
  command -v flock >/dev/null 2>&1 || {
    echo "flock is required for port allocation" >&2; return 2; }
  command -v ss >/dev/null 2>&1 || {
    echo "ss is required for port allocation" >&2; return 2; }

  key="${UID:-0}:${SLURM_JOB_ID:-$$}:${SLURM_ARRAY_TASK_ID:-0}:${namespace}"
  checksum=$(printf '%s' "${key}" | cksum | awk '{print $1}')
  [[ "${checksum}" =~ ^[0-9]+$ ]] || return 2

  for attempt in $(seq 0 127); do
    # 7919 is coprime to 6000 and gives a well-spread deterministic retry
    # order for the default range.
    slot=$(( (checksum + attempt * 7919) % block_count ))
    base=$(( first_port + width * slot ))
    lock="/tmp/cec_port_block_${UID:-0}_${base}_${width}.lock"
    exec {candidate_fd}>"${lock}" || continue
    if ! flock -n "${candidate_fd}"; then
      exec {candidate_fd}>&-
      continue
    fi
    occupied=0
    for ((offset = 0; offset < width; offset++)); do
      if ss -H -ltn | awk '{print $4}' | grep -Eq \
          "(^|:)$((base + offset))$"; then
        occupied=1
        break
      fi
    done
    if (( occupied )); then
      flock -u "${candidate_fd}" || true
      exec {candidate_fd}>&-
      continue
    fi
    CEC_PORT_BLOCK_BASE=${base}
    CEC_PORT_BLOCK_WIDTH=${width}
    CEC_PORT_BLOCK_LOCK_FD=${candidate_fd}
    CEC_PORT_BLOCK_LOCK_PATH=${lock}
    export CEC_PORT_BLOCK_BASE CEC_PORT_BLOCK_WIDTH \
      CEC_PORT_BLOCK_LOCK_FD CEC_PORT_BLOCK_LOCK_PATH
    return 0
  done
  echo "could not claim a free TCP port block after 128 attempts" >&2
  return 2
}

release_slurm_tcp_port_block() {
  if [[ "${CEC_PORT_BLOCK_LOCK_FD:-}" =~ ^[0-9]+$ ]]; then
    flock -u "${CEC_PORT_BLOCK_LOCK_FD}" 2>/dev/null || true
    exec {CEC_PORT_BLOCK_LOCK_FD}>&-
  fi
  unset CEC_PORT_BLOCK_BASE CEC_PORT_BLOCK_WIDTH \
    CEC_PORT_BLOCK_LOCK_FD CEC_PORT_BLOCK_LOCK_PATH
}

claim_slurm_tcp_port_pair() {
  local namespace=${1:?claim_slurm_tcp_port_pair requires a namespace}
  local first_port=${2:-12000}
  local pair_count=${3:-6000}
  claim_slurm_tcp_port_block "${namespace}" 2 "${first_port}" "${pair_count}"
  MEMNAV_PORT=${CEC_PORT_BLOCK_BASE}
  NAVDP_PORT=$((CEC_PORT_BLOCK_BASE + 1))
  CEC_PORT_PAIR_LOCK_FD=${CEC_PORT_BLOCK_LOCK_FD}
  CEC_PORT_PAIR_LOCK_PATH=${CEC_PORT_BLOCK_LOCK_PATH}
  export MEMNAV_PORT NAVDP_PORT CEC_PORT_PAIR_LOCK_FD \
    CEC_PORT_PAIR_LOCK_PATH
}

release_slurm_tcp_port_pair() {
  release_slurm_tcp_port_block
  unset CEC_PORT_PAIR_LOCK_FD CEC_PORT_PAIR_LOCK_PATH
}
