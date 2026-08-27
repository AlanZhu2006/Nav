#!/usr/bin/env bash
# Stage an immutable M2P S-1 bundle and submit the one-episode contract smoke.
set -euo pipefail
umask 0022

LOCAL_ROOT="$(git rev-parse --show-toplevel)"
REMOTE_HOST="${REMOTE_HOST:-alantorch}"
REMOTE_BUNDLE_BASE="${REMOTE_BUNDLE_BASE:-/scratch/yz11502/Research/Nav-axis-uturn-source-bundles}"
REMOTE_RESULT_BASE="${REMOTE_RESULT_BASE:-/scratch/yz11502/Research/Nav-axis-uturn-results/m2p_s1_dual_context_20260813}"
MEMNAV_PY="${MEMNAV_PY:-/home/asus/miniconda3/envs/memnav/bin/python}"

FILES=(
  MemNavData/diag_m2p_s1_gct_query.py
  MemNavData/diag_m2p_s1_train40_dual_context.py
  MemNavData/test_diag_m2p_s1_train40_dual_context.py
  MemNavData/router_multiscene_split_20260805.json
  MemNavData/UNIFIED_MEMORY_TO_POINT_PROTOCOL_20260813.md
  MemNavData/slurm_m2p_s1_train40_dual_context.sbatch
  .diagnostics/certificate_distilled_compass_20260813/static_top8_480_lightglue_open_set_rows.csv
)

for relative in "${FILES[@]}"; do
  [[ -f "${LOCAL_ROOT}/${relative}" && ! -L "${LOCAL_ROOT}/${relative}" ]] || {
    echo "ABORT: missing physical input ${relative}" >&2; exit 2; }
done

"${MEMNAV_PY}" -m py_compile \
  "${LOCAL_ROOT}/MemNavData/diag_m2p_s1_gct_query.py" \
  "${LOCAL_ROOT}/MemNavData/diag_m2p_s1_train40_dual_context.py"
"${MEMNAV_PY}" -m pytest -q \
  "${LOCAL_ROOT}/MemNavData/test_diag_m2p_s1_train40_dual_context.py"

STAGING="$(mktemp -d)"
trap 'rm -rf -- "${STAGING}"' EXIT
for relative in "${FILES[@]}"; do
  mkdir -p "${STAGING}/$(dirname "${relative}")"
  cp --preserve=mode,timestamps "${LOCAL_ROOT}/${relative}" \
    "${STAGING}/${relative}"
done
(
  cd "${STAGING}"
  find . -type f ! -name SOURCE_BUNDLE.sha256 -print0 | sort -z | \
    xargs -0 sha256sum > SOURCE_BUNDLE.sha256
)
RECEIPT_SHA="$(sha256sum "${STAGING}/SOURCE_BUNDLE.sha256" | awk '{print $1}')"
BUNDLE_TAG="${RECEIPT_SHA:0:16}"
REMOTE_BUNDLE="${REMOTE_BUNDLE_BASE}/m2p_s1_dual_context_${BUNDLE_TAG}"
RUN_TAG="smoke_${BUNDLE_TAG}_$(date +%Y%m%d_%H%M%S)"
REMOTE_RUN="${REMOTE_RESULT_BASE}/${RUN_TAG}"

ssh -o BatchMode=yes "${REMOTE_HOST}" \
  "test ! -e '${REMOTE_BUNDLE}' && mkdir -p '${REMOTE_BUNDLE}' && \
   mkdir -p '${REMOTE_RESULT_BASE}' \
            '/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs'"
rsync -a --chmod=Du=rwx,Dgo=rx,Fu=rw,Fgo=r \
  "${STAGING}/" "${REMOTE_HOST}:${REMOTE_BUNDLE}/"
REMOTE_RECEIPT_SHA="$(ssh -o BatchMode=yes "${REMOTE_HOST}" \
  "sha256sum '${REMOTE_BUNDLE}/SOURCE_BUNDLE.sha256' | awk '{print \$1}'")"
[[ "${REMOTE_RECEIPT_SHA}" == "${RECEIPT_SHA}" ]] || {
  echo "ABORT: remote receipt differs" >&2; exit 2; }
ssh -o BatchMode=yes "${REMOTE_HOST}" "chmod -R a-w '${REMOTE_BUNDLE}'"

JOB_ID="$(ssh -o BatchMode=yes "${REMOTE_HOST}" \
  "sbatch --parsable --export=ALL,SOURCE_ROOT='${REMOTE_BUNDLE}',RUN_ROOT='${REMOTE_RUN}',EXPECTED_SOURCE_RECEIPT_SHA='${RECEIPT_SHA}',MODE=smoke,MAX_EPISODES=1 '${REMOTE_BUNDLE}/MemNavData/slurm_m2p_s1_train40_dual_context.sbatch'")"
printf 'M2P_S1_SMOKE_JOB_ID=%s\n' "${JOB_ID}"
printf 'M2P_S1_SOURCE_BUNDLE=%s\n' "${REMOTE_BUNDLE}"
printf 'M2P_S1_SOURCE_RECEIPT_SHA=%s\n' "${RECEIPT_SHA}"
printf 'M2P_S1_RUN_ROOT=%s\n' "${REMOTE_RUN}"
