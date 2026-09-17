#!/usr/bin/env bash
# CPU preflight before allocating GPU arrays. Never change shared environments.
set -euo pipefail
: "${REPAIRED_BUNDLE:?}" "${COVERAGE_RUN:?}" "${COVERAGE_PLAN:?}" "${EXPECTED_RUNTIME_SHA:?}"
[[ "$(id -un)" == yz11502 ]]
source "${REPAIRED_BUNDLE}/MemNavData/repaired_hm3d_hpc_env.sh"
cd "${REPAIRED_BUNDLE}"
[[ "$(sha256sum SOURCE_BUNDLE.sha256 | awk '{print $1}')" == "${EXPECTED_RUNTIME_SHA}" ]]
sha256sum -c --quiet SOURCE_BUNDLE.sha256
COVERAGE_PREFLIGHT_TMP=$(mktemp -d "${COVERAGE_RUN}/preflight_tmp_XXXXXX")
export TMPDIR=${COVERAGE_PREFLIGHT_TMP} LIBFFI_TMPDIR=${COVERAGE_PREFLIGHT_TMP}
export COVIS_POPULATION=/scratch/yz11502/Research/Nav-axis-uturn-results/hm3d_covisibility_repaired_20260909/run_bfb8fc3d5a1bb081/construction/population.json
"${REPAIRED_MEM_PY}" MemNavData/coverage_ablation_hpc.py freeze-plan \
  --population "${COVIS_POPULATION}" --out "${COVERAGE_PLAN}"
export EXPECTED_PLAN_SHA=$(sha256sum "${COVERAGE_PLAN}" | awk '{print $1}')
"${REPAIRED_MEM_PY}" MemNavData/coverage_ablation_hpc.py preflight-inputs --plan "${COVERAGE_PLAN}" \
  --out "${COVERAGE_RUN}/must_not_exist"
"${REPAIRED_MEM_PY}" MemNavData/coverage_ablation_hpc.py dry-run --out "${COVERAGE_RUN}/must_not_exist"
for kind in habitat memnav navdp; do
  if [[ "${kind}" == habitat ]]; then
    "${REPAIRED_HAB_PY}" MemNavData/preflight_repaired_fullmono.py "${kind}" \
      >"${COVERAGE_RUN}/preflight_${kind}.log" 2>&1
  else
    env PYTHONPATH="${REPAIRED_BUNDLE}/NavDP/baselines/${kind}:${PYTHONPATH}" \
      "${REPAIRED_MEM_PY}" MemNavData/preflight_repaired_fullmono.py "${kind}" \
      >"${COVERAGE_RUN}/preflight_${kind}.log" 2>&1
  fi
done
"${REPAIRED_MEM_PY}" -m pytest -q MemNavData/test_coverage_ablation_hpc.py \
  MemNavData/test_repaired_covisibility_eval.py MemNavData/test_certified_relocalization_runtime.py
"${REPAIRED_MEM_PY}" -c 'import json,os,time; from pathlib import Path; p=Path(os.environ["COVERAGE_RUN"])/"preflight_success.json"; assert not p.exists(); p.write_text(json.dumps({"verified":True,"runtime_sha256":os.environ["EXPECTED_RUNTIME_SHA"],"plan_sha256":os.environ["EXPECTED_PLAN_SHA"],"checked_at":time.time()},indent=2)+"\n"); print(p)'
