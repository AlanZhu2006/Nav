#!/usr/bin/env bash
set -euo pipefail
[[ "$(id -un)" == yz11502 ]]
: "${SLURM_JOB_ID:?Requires a Slurm compute allocation}"
: "${REPAIRED_BUNDLE:?}"
: "${GEM_SUPPORT_RESOURCE_PLAN:?}"
: "${EXPECTED_RESOURCE_PLAN_SHA:?}"
source "${REPAIRED_BUNDLE}/MemNavData/repaired_hm3d_hpc_env.sh"
cd "${REPAIRED_BUNDLE}"
sha256sum -c --quiet GEM_SOURCE.sha256
[[ "$(sha256sum "${GEM_SUPPORT_RESOURCE_PLAN}" | awk '{print $1}')" == "${EXPECTED_RESOURCE_PLAN_SHA}" ]]
export PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1
"${REPAIRED_MEM_PY}" -u MemNavData/run_gem_support_resources_hpc.py --plan "${GEM_SUPPORT_RESOURCE_PLAN}"
