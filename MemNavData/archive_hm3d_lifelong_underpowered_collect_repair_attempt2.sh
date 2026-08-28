#!/usr/bin/env bash
# Transactionally quarantine only the three attempt-1 hub-startup partials.
set -euo pipefail
umask 0022
export PYTHONDONTWRITEBYTECODE=1

: "${RUN_ROOT:?}" "${PROTOCOL:?}" "${BASE_PROTOCOL:?}"
: "${REPAIR_ROOT:?}" "${ARCHIVE_ROOT:?}" "${AMENDMENT_ROOT:?}"
: "${REMOTE_PY:?}"
[[ ! -e "${ARCHIVE_ROOT}" ]]
[[ ! -e "${ARCHIVE_ROOT}.partial.${SLURM_JOB_ID:-$$}" ]]
[[ ! -e "${REPAIR_ROOT}" ]]
mkdir -p "${REPAIR_ROOT}"

audit=${AMENDMENT_ROOT}/MemNavData/audit_hm3d_lifelong_underpowered_collect_repair_attempt2.py
pre=${REPAIR_ROOT}/pre_archive_audit.json
post=${REPAIR_ROOT}/post_archive_audit.json
ledger=${REPAIR_ROOT}/attempt1_sacct.txt

readarray -t frozen < <("${REMOTE_PY}" - "${PROTOCOL}" <<'PY'
import json,sys
p=json.load(open(sys.argv[1]))
c=p["repair_contract"]
print(p["source_authority"]["run_root"])
print(c["attempt1_archive_root"])
print(c["repair_receipt_root"])
for index in c["attempt1_partial_indices_to_archive"]:
 item=next(row for row in p["repair_items"] if row["index"]==index)
 print(item["label"])
PY
)
[[ "${#frozen[@]}" -eq 6 ]]
[[ "${RUN_ROOT}" == "${frozen[0]}" ]]
[[ "${ARCHIVE_ROOT}" == "${frozen[1]}" ]]
[[ "${REPAIR_ROOT}" == "${frozen[2]}" ]]
labels=("${frozen[3]}" "${frozen[4]}" "${frozen[5]}")

sacct -X -n -P \
  -j 16509621,16509627,16509634,16509636,16509637,16509642,16509644,16509648,16509649 \
  --format=JobIDRaw,State,ExitCode,NodeList,Start,End >"${ledger}"
"${REMOTE_PY}" - "${ledger}" <<'PY'
import pathlib,sys
failed={16509621:"gh005",16509634:"ga005",16509637:"ga028"}
cancelled={16509627,16509636,16509642,16509644,16509648,16509649}
rows={}
for line in pathlib.Path(sys.argv[1]).read_text().splitlines():
 fields=line.split("|")
 if len(fields)<6 or not fields[0].isdigit(): continue
 rows[int(fields[0])]=fields[1:]
if set(rows)!=set(failed)|cancelled:
 raise SystemExit("attempt-1 job ledger changed")
for job,node in failed.items():
 state,code,nodelist,start,_end=rows[job]
 if state!="FAILED" or code!="2:0" or nodelist!=node or start=="None":
  raise SystemExit(f"failed job {job} provenance changed")
for job in cancelled:
 state,code,_node,start,_end=rows[job]
 if not state.startswith("CANCELLED") or code!="0:0" or start!="None":
  raise SystemExit(f"cancelled job {job} provenance changed")
PY

audit_args=(
  "${audit}" --protocol "${PROTOCOL}" --base-protocol "${BASE_PROTOCOL}"
  --run-root "${RUN_ROOT}"
)
"${REMOTE_PY}" "${audit_args[@]}" --phase pre_archive --out "${pre}" \
  >/dev/null
sha256sum "${pre}" >"${pre}.sha256"

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

"${REMOTE_PY}" "${audit_args[@]}" --phase post_archive --out "${post}" \
  >/dev/null
sha256sum "${post}" >"${post}.sha256"
chmod -R a-w "${ARCHIVE_ROOT}"
trap - ERR INT TERM
printf 'ARCHIVED_ATTEMPT1_STARTUP_PARTIALS=3\nARCHIVE_ROOT=%s\n' \
  "${ARCHIVE_ROOT}"
