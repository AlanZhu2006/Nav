#!/usr/bin/env bash
# Package frozen task code plus the clean official Pi3 source, then submit a
# smoke -> full dependency chain.  The 5.4 GB model is a separately pinned asset.
set -euo pipefail
umask 0022

LOCAL_ROOT=$(git rev-parse --show-toplevel)
REMOTE_HOST=${REMOTE_HOST:-alantorch}
REMOTE_BUNDLE_BASE=${REMOTE_BUNDLE_BASE:-/scratch/yz11502/Research/Nav-axis-uturn-source-bundles}
REMOTE_RESULT_BASE=${REMOTE_RESULT_BASE:-/scratch/yz11502/Research/Nav-axis-uturn-results/pi3x_learned_relocalizer_20260817}
PI3_LOCAL_ROOT=${PI3_LOCAL_ROOT:-/home/asus/Research/Pi3}
EXPECTED_PI3_COMMIT=d07ddaf46a222acfda6bd877f72fdd099470cae8

FILES=(
  MemNavData/diag_pi3x_multiview_consistency.py
  MemNavData/test_diag_pi3x_multiview_consistency.py
  MemNavData/summarize_pi3x_multiview_shadow.py
  MemNavData/test_summarize_pi3x_multiview_shadow.py
  MemNavData/slurm_pi3x_train40_shadow.sbatch
  MemNavData/LEARNED_RELOCALIZER_NIGHT_GOAL_20260817.md
)
for relative in "${FILES[@]}"; do
  [[ -f "${LOCAL_ROOT}/${relative}" && ! -L "${LOCAL_ROOT}/${relative}" ]] || {
    echo "ABORT: missing physical input ${relative}" >&2; exit 2; }
done
[[ "$(git -C "${PI3_LOCAL_ROOT}" rev-parse HEAD)" == "${EXPECTED_PI3_COMMIT}" ]] || {
  echo "ABORT: Pi3 commit changed" >&2; exit 2; }

conda run -n lingbot-map python -m pytest -q -p no:cacheprovider \
  "${LOCAL_ROOT}/MemNavData/test_diag_pi3x_multiview_consistency.py" \
  "${LOCAL_ROOT}/MemNavData/test_summarize_pi3x_multiview_shadow.py"

STAGING=$(mktemp -d)
trap 'rm -rf -- "${STAGING}"' EXIT
for relative in "${FILES[@]}"; do
  mkdir -p "${STAGING}/$(dirname "${relative}")"
  cp --preserve=mode,timestamps "${LOCAL_ROOT}/${relative}" "${STAGING}/${relative}"
done
mkdir -p "${STAGING}/third_party/Pi3"
PI3_ARCHIVE=${STAGING}/.pi3_source.tar
git -C "${PI3_LOCAL_ROOT}" archive --format=tar --output="${PI3_ARCHIVE}" \
  "${EXPECTED_PI3_COMMIT}" pi3 LICENSE README.md
tar -xf "${PI3_ARCHIVE}" -C "${STAGING}/third_party/Pi3"
rm -f -- "${PI3_ARCHIVE}"

(
  cd "${STAGING}"
  find . -type f ! -name SOURCE_BUNDLE.sha256 -print0 | sort -z | \
    xargs -0 sha256sum > SOURCE_BUNDLE.sha256
)
RECEIPT_SHA=$(sha256sum "${STAGING}/SOURCE_BUNDLE.sha256" | awk '{print $1}')
BUNDLE_TAG=${RECEIPT_SHA:0:16}
REMOTE_BUNDLE=${REMOTE_BUNDLE_BASE}/pi3x_learned_relocalizer_${BUNDLE_TAG}
RUN_TAG=shadow_${BUNDLE_TAG}_$(date +%Y%m%d_%H%M%S)
REMOTE_RUN=${REMOTE_RESULT_BASE}/${RUN_TAG}

ssh -o BatchMode=yes "${REMOTE_HOST}" \
  "test ! -e '${REMOTE_BUNDLE}' && mkdir -p '${REMOTE_BUNDLE}' && \
   mkdir -p '${REMOTE_RUN}/smoke' '${REMOTE_RUN}/train40' \
            '/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs'"
rsync -a --chmod=Du=rwx,Dgo=rx,Fu=rw,Fgo=r \
  "${STAGING}/" "${REMOTE_HOST}:${REMOTE_BUNDLE}/"
REMOTE_RECEIPT_SHA=$(ssh -o BatchMode=yes "${REMOTE_HOST}" \
  "sha256sum '${REMOTE_BUNDLE}/SOURCE_BUNDLE.sha256' | awk '{print \$1}'")
[[ "${REMOTE_RECEIPT_SHA}" == "${RECEIPT_SHA}" ]] || {
  echo "ABORT: remote receipt differs" >&2; exit 2; }
ssh -o BatchMode=yes "${REMOTE_HOST}" "chmod -R a-w '${REMOTE_BUNDLE}'"

COMMON_EXPORTS="ALL,SOURCE_ROOT=${REMOTE_BUNDLE},SOURCE_RECEIPT=${REMOTE_BUNDLE}/SOURCE_BUNDLE.sha256,EXPECTED_SOURCE_RECEIPT_SHA=${RECEIPT_SHA}"
SMOKE_JOB=$(ssh -o BatchMode=yes "${REMOTE_HOST}" \
  "sbatch --parsable --export='${COMMON_EXPORTS},RUN_ROOT=${REMOTE_RUN}/smoke,MODE=smoke' '${REMOTE_BUNDLE}/MemNavData/slurm_pi3x_train40_shadow.sbatch'")
FULL_JOB=$(ssh -o BatchMode=yes "${REMOTE_HOST}" \
  "sbatch --parsable --dependency=afterok:${SMOKE_JOB} --export='${COMMON_EXPORTS},RUN_ROOT=${REMOTE_RUN}/train40,MODE=full' '${REMOTE_BUNDLE}/MemNavData/slurm_pi3x_train40_shadow.sbatch'")

printf 'PI3X_SMOKE_JOB_ID=%s\n' "${SMOKE_JOB}"
printf 'PI3X_FULL_JOB_ID=%s\n' "${FULL_JOB}"
printf 'PI3X_SOURCE_BUNDLE=%s\n' "${REMOTE_BUNDLE}"
printf 'PI3X_SOURCE_RECEIPT_SHA=%s\n' "${RECEIPT_SHA}"
printf 'PI3X_RUN_ROOT=%s\n' "${REMOTE_RUN}"
