#!/usr/bin/env bash
# Prepared collector wrapper. No submission or downstream query launch here.
set -euo pipefail
: "${CORE_ADDON:?}" "${CORE_PLAN:?}" "${CORE_PLAN_SHA:?}" "${CORE_RUN:?}" "${EXPECTED_CORE_ADDON_SHA:?}"
export REPAIRED_BUNDLE=/scratch/yz11502/Research/Nav-axis-uturn-source-bundles/repaired_fullmono_c8cf8c60e7efd55f
source "${REPAIRED_BUNDLE}/MemNavData/repaired_hm3d_hpc_env.sh"
CORE_TMP=$(mktemp -d "${SLURM_TMPDIR:-/tmp}/gem_actual_a_${SLURM_JOB_ID:-check}_XXXXXX")
export TMPDIR=${CORE_TMP} LIBFFI_TMPDIR=${CORE_TMP} PYTHONFAULTHANDLER=1
[[ "$(id -un)" == yz11502 ]]
[[ "$(sha256sum "${CORE_ADDON}/SOURCE_BUNDLE.sha256" | awk '{print $1}')" == "${EXPECTED_CORE_ADDON_SHA}" ]]
[[ "$(sha256sum "${CORE_PLAN}" | awk '{print $1}')" == "${CORE_PLAN_SHA}" ]]
(cd "${REPAIRED_BUNDLE}" && sha256sum -c --quiet SOURCE_BUNDLE.sha256)
(cd "${CORE_ADDON}" && sha256sum -c --quiet SOURCE_BUNDLE.sha256)
export PYTHONPATH=${CORE_ADDON}:${PYTHONPATH}
cd "${REPAIRED_BUNDLE}"
"${REPAIRED_HAB_PY}" -m unittest test_repaired_fullmono_design test_repaired_fullmono_population_plan test_repaired_fullmono_a_collector
if [[ "${CORE_PREFLIGHT_ONLY:-0}" == 1 ]]; then
  "${REPAIRED_HAB_PY}" - "${CORE_PLAN}" "${CORE_SCENE_RANK:-0}" <<'PY'
import json,sys,subprocess
from pathlib import Path
from collect_repaired_fullmono_a import scene_sources, bind_source
from MemNavData.run_repaired_fullmono_local import evaluator_command, execution_environment
sources=scene_sources(json.load(open(sys.argv[1])),int(sys.argv[2]))
for item in sources:
    source=bind_source(item)
    cmd=evaluator_command(source,Path('/preflight/no_output'),21010,21011)+['--contract_dry_run']
    subprocess.run(cmd,check=True,env=execution_environment())
    print('A_SOURCE_CLI_OK',source['scene'],source['episode'],source['seed'])
PY
  exit 0
fi
: "${SLURM_ARRAY_TASK_ID:?}"
source "${REPAIRED_BUNDLE}/MemNavData/slurm_port_pair.sh"
claim_slurm_tcp_port_pair gem_actual_a 12000 6000
CORE_DURABLE=${CORE_RUN}/scene_${SLURM_ARRAY_TASK_ID}
CORE_TASK=${CORE_TMP}/task
finish_collection() {
  local status=$?
  trap - EXIT
  release_slurm_tcp_port_pair
  if [[ -d "${CORE_TASK}" && -d "${CORE_DURABLE}" ]]; then
    "${REPAIRED_HAB_PY}" "${CORE_ADDON}/covisibility_task_archive.py" \
      --out "${CORE_TASK}" --durable "${CORE_DURABLE}" --exit-code "${status}" || status=1
  fi
  exit "${status}"
}
trap finish_collection EXIT
"${REPAIRED_MEM_PY}" -u "${CORE_ADDON}/collect_repaired_fullmono_a.py" collect \
  --plan "${CORE_PLAN}" --plan-sha256 "${CORE_PLAN_SHA}" \
  --scene-rank "${SLURM_ARRAY_TASK_ID}" --out "${CORE_TASK}" --durable "${CORE_DURABLE}" \
  --memnav-port "${MEMNAV_PORT}" --navdp-port "${NAVDP_PORT}"
