#!/usr/bin/env bash
# Source RGB reconstruction and offline query annotations; no policy inference.
set -euo pipefail
: "${FULLMONO_QUERY_ADDON:?}" "${FULLMONO_QUERY_ADDON_SHA:?}"
: "${CORE_PLAN:?}" "${CORE_PLAN_SHA:?}" "${CORE_RUN:?}" "${FULLMONO_CONSTRUCTION_ROOT:?}"
export REPAIRED_BUNDLE=/scratch/yz11502/Research/Nav-axis-uturn-source-bundles/repaired_fullmono_c8cf8c60e7efd55f
source "${REPAIRED_BUNDLE}/MemNavData/repaired_hm3d_hpc_env.sh"
FULLMONO_CONSTRUCT_TMP=$(mktemp -d "${SLURM_TMPDIR:-/tmp}/gem_role_build_${SLURM_JOB_ID:-check}_XXXXXX")
export TMPDIR=${FULLMONO_CONSTRUCT_TMP} LIBFFI_TMPDIR=${FULLMONO_CONSTRUCT_TMP}
export PYTHONFAULTHANDLER=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
[[ "$(id -un)" == yz11502 ]]
[[ "$(sha256sum "${FULLMONO_QUERY_ADDON}/SOURCE_BUNDLE.sha256" | awk '{print $1}')" == "${FULLMONO_QUERY_ADDON_SHA}" ]]
[[ "$(sha256sum "${CORE_PLAN}" | awk '{print $1}')" == "${CORE_PLAN_SHA}" ]]
(cd "${REPAIRED_BUNDLE}" && sha256sum -c --quiet SOURCE_BUNDLE.sha256)
(cd "${FULLMONO_QUERY_ADDON}" && sha256sum -c --quiet SOURCE_BUNDLE.sha256)
(cd "${REPAIRED_HAB_EXTRA}" && sha256sum -c --quiet SOURCE_DEPENDENCY.sha256)
export PYTHONPATH=${FULLMONO_QUERY_ADDON}:${PYTHONPATH}
cd "${REPAIRED_BUNDLE}"
"${REPAIRED_HAB_PY}" -m unittest test_repaired_fullmono_query_construction test_repaired_fullmono_query_seal
if [[ "${FULLMONO_PREFLIGHT_ONLY:-0}" == 1 ]]; then
  "${REPAIRED_HAB_PY}" "${FULLMONO_QUERY_ADDON}/construct_repaired_fullmono_queries.py" --help
  "${REPAIRED_HAB_PY}" "${FULLMONO_QUERY_ADDON}/seal_repaired_fullmono_queries.py" --help
  GEM_POPULATION_SHA=$(printf '%064d' 0) "${REPAIRED_HAB_PY}" \
    "${FULLMONO_QUERY_ADDON}/repaired_fullmono_query_eval.py" dry-run --out /preflight/unused_query_output
  exit 0
fi
: "${SLURM_ARRAY_TASK_ID:?}"
"${REPAIRED_HAB_PY}" -u "${FULLMONO_QUERY_ADDON}/construct_repaired_fullmono_queries.py" \
  --collection "${CORE_RUN}/scene_${SLURM_ARRAY_TASK_ID}" \
  --plan "${CORE_PLAN}" --plan-sha256 "${CORE_PLAN_SHA}" \
  --out "${FULLMONO_CONSTRUCTION_ROOT}/scene_${SLURM_ARRAY_TASK_ID}"
