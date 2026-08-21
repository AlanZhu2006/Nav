#!/usr/bin/env bash
# Stage a task-only immutable bundle and submit the fresh160 observability audit.
set -euo pipefail
umask 0022

LOCAL_ROOT="$(git rev-parse --show-toplevel)"
REMOTE_HOST="${REMOTE_HOST:-alantorch}"
REMOTE_BUNDLE_BASE="${REMOTE_BUNDLE_BASE:-/scratch/yz11502/Research/source_bundles}"
RESULT_BASE="${RESULT_BASE:-/scratch/yz11502/Research/Nav-axis-uturn-results/revisit_fresh_online_observability_20260813}"
ORIGINAL_RUN_ROOT="${ORIGINAL_RUN_ROOT:-/scratch/yz11502/Research/Nav-axis-uturn-results/revisit_fresh_confirmation_20260811/fresh160_v3_attempt600_20260811T2000}"
RUN_CONTRACT=${RUN_CONTRACT:-fresh_confirmation}
case "${RUN_CONTRACT}" in
  fresh_confirmation)
    PROTOCOL_DOC=MemNavData/REVISIT_FRESH_ONLINE_OBSERVABILITY_AUDIT_20260813.md
    BUNDLE_PREFIX=revisit_fresh_online_obs
    JOB_NAME=revfresh_obs
    ;;
  certified_relocalization)
    PROTOCOL_DOC=MemNavData/CERTIFIED_RELOCALIZATION_ONLINE_OBSERVABILITY_AUDIT_20260813.md
    BUNDLE_PREFIX=certified_relocalization_online_obs
    JOB_NAME=certrel_obs
    ;;
  *) echo "ABORT: invalid RUN_CONTRACT=${RUN_CONTRACT}" >&2; exit 2 ;;
esac

FILES=(
  MemNavData/audit_revisit_fresh_online_observability.py
  MemNavData/deterministic_eval_protocol.py
  MemNavData/test_audit_revisit_fresh_online_observability.py
  "${PROTOCOL_DOC}"
  MemNavData/slurm_revisit_fresh_online_observability.sbatch
)
for relative in "${FILES[@]}"; do
  [[ -f "${LOCAL_ROOT}/${relative}" && ! -L "${LOCAL_ROOT}/${relative}" ]] || {
    echo "ABORT: missing physical source file ${relative}" >&2; exit 2; }
done

MEMNAV_PY=${MEMNAV_PY:-/home/asus/miniconda3/envs/memnav/bin/python}
"${MEMNAV_PY}" -m py_compile \
  "${LOCAL_ROOT}/MemNavData/audit_revisit_fresh_online_observability.py"
(cd "${LOCAL_ROOT}" && "${MEMNAV_PY}" -m pytest -q \
  MemNavData/test_audit_revisit_fresh_online_observability.py)

STAGING="$(mktemp -d)"
trap 'rm -rf -- "${STAGING}"' EXIT
mkdir -p "${STAGING}/MemNavData"
for relative in "${FILES[@]}"; do
  cp --preserve=mode,timestamps "${LOCAL_ROOT}/${relative}" \
    "${STAGING}/${relative}"
done
(
  cd "${STAGING}"
  find . -type f ! -name SOURCE_BUNDLE.sha256 -print0 | sort -z | \
    xargs -0 sha256sum > SOURCE_BUNDLE.sha256
  sha256sum -c SOURCE_BUNDLE.sha256 >/dev/null
)
SOURCE_RECEIPT_SHA=$(sha256sum "${STAGING}/SOURCE_BUNDLE.sha256" | awk '{print $1}')
BUNDLE_TAG=${SOURCE_RECEIPT_SHA:0:16}
REMOTE_BUNDLE=${REMOTE_BUNDLE_BASE}/${BUNDLE_PREFIX}_${BUNDLE_TAG}
AUDIT_ROOT=${RESULT_BASE}/online_obs_${BUNDLE_TAG}
LAUNCHER_SHA=$(sha256sum \
  "${STAGING}/MemNavData/slurm_revisit_fresh_online_observability.sbatch" | \
  awk '{print $1}')

if ssh -o BatchMode=yes "${REMOTE_HOST}" "test -e '${REMOTE_BUNDLE}'"; then
  ssh -o BatchMode=yes "${REMOTE_HOST}" \
    "test -f '${REMOTE_BUNDLE}/SOURCE_BUNDLE.sha256' && \
     test \"\$(sha256sum '${REMOTE_BUNDLE}/SOURCE_BUNDLE.sha256' | awk '{print \$1}')\" = '${SOURCE_RECEIPT_SHA}' && \
     cd '${REMOTE_BUNDLE}' && sha256sum -c SOURCE_BUNDLE.sha256 >/dev/null" || {
       echo "ABORT: existing remote bundle is not the expected immutable bundle" >&2
       exit 2
     }
else
  REMOTE_STAGING=${REMOTE_BUNDLE}.staging_$$
  ssh -o BatchMode=yes "${REMOTE_HOST}" \
    "test ! -e '${REMOTE_STAGING}' && mkdir -p '${REMOTE_STAGING}'"
  rsync -a --chmod=Du=rwx,Dgo=rx,Fu=rw,Fgo=r \
    "${STAGING}/" "${REMOTE_HOST}:${REMOTE_STAGING}/"
  ssh -o BatchMode=yes "${REMOTE_HOST}" \
    "cd '${REMOTE_STAGING}' && sha256sum -c SOURCE_BUNDLE.sha256 >/dev/null && \
     chmod -R a-w '${REMOTE_STAGING}' && mv '${REMOTE_STAGING}' '${REMOTE_BUNDLE}'"
fi

exports="ALL,SOURCE_ROOT=${REMOTE_BUNDLE},SOURCE_RECEIPT=${REMOTE_BUNDLE}/SOURCE_BUNDLE.sha256,EXPECTED_SOURCE_RECEIPT_SHA=${SOURCE_RECEIPT_SHA},EXPECTED_LAUNCHER_SHA=${LAUNCHER_SHA},ORIGINAL_RUN_ROOT=${ORIGINAL_RUN_ROOT},AUDIT_ROOT=${AUDIT_ROOT},RUN_CONTRACT=${RUN_CONTRACT}"
SBATCH=${REMOTE_BUNDLE}/MemNavData/slurm_revisit_fresh_online_observability.sbatch
ssh -o BatchMode=yes "${REMOTE_HOST}" \
  "test ! -e '${AUDIT_ROOT}' && sbatch --test-only --job-name='${JOB_NAME}' --export='${exports}' '${SBATCH}'" \
  >/dev/null
JOB_RAW=$(ssh -o BatchMode=yes "${REMOTE_HOST}" \
  "sbatch --parsable --job-name='${JOB_NAME}' --export='${exports}' '${SBATCH}'")
JOB_ID=${JOB_RAW%%;*}
[[ "${JOB_ID}" =~ ^[0-9]+$ ]] || { echo "ABORT: bad job id ${JOB_RAW}" >&2; exit 2; }

printf 'JOB_ID=%s\nRUN_CONTRACT=%s\nSOURCE_ROOT=%s\nSOURCE_RECEIPT_SHA=%s\nAUDIT_ROOT=%s\n' \
  "${JOB_ID}" "${RUN_CONTRACT}" "${REMOTE_BUNDLE}" \
  "${SOURCE_RECEIPT_SHA}" "${AUDIT_ROOT}"
