#!/usr/bin/env bash
# Reuse a frozen candidate table to measure complete-pool retrieval recall.

set -euo pipefail
umask 0022

ROOT=${ROOT:?set ROOT to the committed Nav-axis-uturn checkout}
SOURCE_CSV=${SOURCE_CSV:?set SOURCE_CSV to expanded_teacher_pairs.csv}
RUN_ROOT=${RUN_ROOT:?set RUN_ROOT to a new result directory}
MEMNAV_PY=${MEMNAV_PY:?set MEMNAV_PY to the pinned Python interpreter}
SPLIT_MANIFEST=${SPLIT_MANIFEST:-${ROOT}/MemNavData/router_multiscene_split_20260805.json}
EXPECTED_INPUT_SHA=${EXPECTED_INPUT_SHA:?set EXPECTED_INPUT_SHA}

RELABELER=${ROOT}/MemNavData/relabel_router_covisibility.py
AUDITOR=${ROOT}/MemNavData/audit_router_candidate_recall.py

if [[ -e "${RUN_ROOT}" ]]; then
  echo "ABORT: output already exists: ${RUN_ROOT}" >&2
  exit 1
fi
for required in "${MEMNAV_PY}" "${SOURCE_CSV}" "${SPLIT_MANIFEST}" \
                "${RELABELER}" "${AUDITOR}"; do
  test -r "${required}" || {
    echo "ABORT: missing dependency ${required}" >&2
    exit 1
  }
done
actual_input_sha=$(sha256sum "${SOURCE_CSV}" | awk '{print $1}')
[[ "${actual_input_sha}" == "${EXPECTED_INPUT_SHA}" ]] || {
  echo "ABORT: candidate input SHA mismatch ${actual_input_sha}" >&2
  exit 1
}

mkdir -p "${RUN_ROOT}"
exec > >(tee "${RUN_ROOT}/run.log") 2>&1
echo "root=${ROOT} input=${SOURCE_CSV} input_sha=${actual_input_sha}"
"${MEMNAV_PY}" -m py_compile "${RELABELER}" "${AUDITOR}"
cd "${ROOT}"
"${MEMNAV_PY}" -m unittest MemNavData.test_router_candidate_recall -v

FULL_TEACHER=${RUN_ROOT}/complete_covisibility_teacher_pairs.csv
TEACHER_REPORT=${RUN_ROOT}/complete_covisibility_teacher_report.json
"${MEMNAV_PY}" -u "${RELABELER}" \
  --input-csv "${SOURCE_CSV}" \
  --output-csv "${FULL_TEACHER}" \
  --report "${TEACHER_REPORT}" \
  --top-k 0

CANDIDATE_REPORT=${RUN_ROOT}/candidate_recall_audit.json
"${MEMNAV_PY}" -u "${AUDITOR}" \
  --teacher-csv "${FULL_TEACHER}" \
  --report "${CANDIDATE_REPORT}" \
  --split-manifest "${SPLIT_MANIFEST}"

echo "[complete] diagnostic only; deployment_approved remains false"
echo "teacher_report=${TEACHER_REPORT}"
echo "candidate_report=${CANDIDATE_REPORT}"
