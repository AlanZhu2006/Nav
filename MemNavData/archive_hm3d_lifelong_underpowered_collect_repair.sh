#!/usr/bin/env bash
# Transactionally quarantine exactly the six failed factual-C partial outputs.
set -euo pipefail
umask 0022
export PYTHONDONTWRITEBYTECODE=1

: "${RUN_ROOT:?}" "${PROTOCOL:?}" "${REPAIR_ROOT:?}" "${ARCHIVE_ROOT:?}"
: "${AMENDMENT_ROOT:?}" "${ORIGINAL_ARRAY_JOB:?}" "${REMOTE_PY:?}"
[[ "${ORIGINAL_ARRAY_JOB}" == 16505696 ]]
[[ "${ARCHIVE_ROOT}" == \
  "${RUN_ROOT}/failed_attempts/shared_c_16505696_replay_contract_20260828" ]]
[[ ! -e "${ARCHIVE_ROOT}" ]]
[[ ! -e "${ARCHIVE_ROOT}.partial.${SLURM_JOB_ID:-$$}" ]]
mkdir -p "${REPAIR_ROOT}"

audit=${AMENDMENT_ROOT}/MemNavData/audit_hm3d_lifelong_underpowered_collect_repair.py
pre=${REPAIR_ROOT}/pre_archive_audit.json
post=${REPAIR_ROOT}/post_archive_audit.json
[[ ! -e "${pre}" && ! -e "${post}" ]]

sacct -X -n -P -j "${ORIGINAL_ARRAY_JOB}" \
  --format=JobID,State,ExitCode,NodeList \
  >"${REPAIR_ROOT}/original_array_sacct.txt"
"${REMOTE_PY}" - "${REPAIR_ROOT}/original_array_sacct.txt" <<'PY'
import pathlib,sys
expected_failed={0,1,7,9,11,13}
states={}
for line in pathlib.Path(sys.argv[1]).read_text().splitlines():
 fields=line.split("|")
 if len(fields)<3 or "_" not in fields[0]: continue
 try: index=int(fields[0].rsplit("_",1)[1])
 except ValueError: continue
 states[index]=(fields[1],fields[2])
if set(states)!=set(range(22)): raise SystemExit("array task ledger changed")
for index,(state,exit_code) in states.items():
 expected="FAILED" if index in expected_failed else "COMPLETED"
 if state!=expected: raise SystemExit(f"index {index}: {state} != {expected}")
 if expected=="COMPLETED" and exit_code!="0:0":
  raise SystemExit(f"index {index}: completed exit code changed")
PY

"${REMOTE_PY}" "${audit}" --protocol "${PROTOCOL}" \
  --run-root "${RUN_ROOT}" --phase pre_archive --out "${pre}" >/dev/null
sha256sum "${pre}" >"${pre}.sha256"

mapfile -t labels < <("${REMOTE_PY}" - "${PROTOCOL}" <<'PY'
import json,sys
p=json.load(open(sys.argv[1]))
for item in p["repair_items"]: print(item["label"])
PY
)
[[ "${#labels[@]}" -eq 6 ]]
partial=${ARCHIVE_ROOT}.partial.${SLURM_JOB_ID:-$$}
mkdir -p "${partial}"

rollback() {
  set +e
  holder=${partial}
  [[ -d "${ARCHIVE_ROOT}" ]] && holder=${ARCHIVE_ROOT}
  for label in "${labels[@]}"; do
    if [[ -e "${holder}/${label}" \
       && ! -e "${RUN_ROOT}/shared_c_collection/${label}" ]]; then
      mv "${holder}/${label}" "${RUN_ROOT}/shared_c_collection/${label}"
    fi
  done
  rmdir "${holder}" 2>/dev/null || true
}
trap rollback ERR INT TERM
for label in "${labels[@]}"; do
  source=${RUN_ROOT}/shared_c_collection/${label}
  [[ -d "${source}" && ! -e "${partial}/${label}" ]]
  mv "${source}" "${partial}/${label}"
done
mv "${partial}" "${ARCHIVE_ROOT}"

"${REMOTE_PY}" "${audit}" --protocol "${PROTOCOL}" \
  --run-root "${RUN_ROOT}" --phase post_archive --out "${post}" >/dev/null
sha256sum "${post}" >"${post}.sha256"
chmod -R a-w "${ARCHIVE_ROOT}"
trap - ERR INT TERM
printf 'ARCHIVED_FAILED_PARTIALS=6\nARCHIVE_ROOT=%s\n' "${ARCHIVE_ROOT}"
