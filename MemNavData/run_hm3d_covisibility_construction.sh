#!/usr/bin/env bash
set -euo pipefail
: "${COVIS_ADDON:?}" "${COVIS_RUN:?}" "${EXPECTED_COVIS_SHA:?}"
export REPAIRED_BUNDLE=/scratch/yz11502/Research/Nav-axis-uturn-source-bundles/repaired_fullmono_c8cf8c60e7efd55f
source "${REPAIRED_BUNDLE}/MemNavData/repaired_hm3d_hpc_env.sh"
COVIS_TASK_TMP=$(mktemp -d "${SLURM_TMPDIR:-/tmp}/cec_covis_${SLURM_JOB_ID:-check}_XXXXXX")
export TMPDIR=${COVIS_TASK_TMP} LIBFFI_TMPDIR=${COVIS_TASK_TMP}
export PYTHONFAULTHANDLER=1
[[ "$(id -un)" == yz11502 ]]
[[ "$(sha256sum "${REPAIRED_SOURCE_RECEIPT}" | awk '{print $1}')" == c8cf8c60e7efd55f1b360bb5a2d5fd92850dc87d6a310b05e77f0a9239cdfc08 ]]
[[ "$(sha256sum "${COVIS_ADDON}/SOURCE_BUNDLE.sha256" | awk '{print $1}')" == "${EXPECTED_COVIS_SHA}" ]]
(cd "${REPAIRED_BUNDLE}" && sha256sum -c --quiet SOURCE_BUNDLE.sha256)
(cd "${COVIS_ADDON}" && sha256sum -c --quiet SOURCE_BUNDLE.sha256)
(cd "${REPAIRED_HAB_EXTRA}" && sha256sum -c --quiet SOURCE_DEPENDENCY.sha256)
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
cd "${REPAIRED_BUNDLE}"
"${REPAIRED_HAB_PY}" "${COVIS_ADDON}/test_hm3d_covisibility_repaired.py" -v
# Construction deliberately needs neither checkpoint loading nor server startup.
"${REPAIRED_HAB_PY}" -u "${COVIS_ADDON}/build_hm3d_covisibility_repaired.py" build \
  --root "${COVIS_RUN}/construction" --index "${SLURM_ARRAY_TASK_ID:?}"
