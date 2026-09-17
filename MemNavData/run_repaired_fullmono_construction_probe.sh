#!/usr/bin/env bash
# Independent construction diagnostic; no model or navigation processes.
set -euo pipefail
: "${DESIGN_ADDON:?}" "${DESIGN_RUN:?}" "${EXPECTED_DESIGN_SHA:?}"
export REPAIRED_BUNDLE=/scratch/yz11502/Research/Nav-axis-uturn-source-bundles/repaired_fullmono_c8cf8c60e7efd55f
source "${REPAIRED_BUNDLE}/MemNavData/repaired_hm3d_hpc_env.sh"
DESIGN_TASK_TMP=$(mktemp -d "${SLURM_TMPDIR:-/tmp}/gem_construct_${SLURM_JOB_ID:-check}_XXXXXX")
export TMPDIR=${DESIGN_TASK_TMP} LIBFFI_TMPDIR=${DESIGN_TASK_TMP}
export PYTHONFAULTHANDLER=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
[[ "$(id -un)" == yz11502 ]]
[[ "$(sha256sum "${REPAIRED_SOURCE_RECEIPT}" | awk '{print $1}')" == c8cf8c60e7efd55f1b360bb5a2d5fd92850dc87d6a310b05e77f0a9239cdfc08 ]]
[[ "$(sha256sum "${DESIGN_ADDON}/SOURCE_BUNDLE.sha256" | awk '{print $1}')" == "${EXPECTED_DESIGN_SHA}" ]]
(cd "${REPAIRED_BUNDLE}" && sha256sum -c --quiet SOURCE_BUNDLE.sha256)
(cd "${DESIGN_ADDON}" && sha256sum -c --quiet SOURCE_BUNDLE.sha256)
(cd "${REPAIRED_HAB_EXTRA}" && sha256sum -c --quiet SOURCE_DEPENDENCY.sha256)
export PYTHONPATH=${DESIGN_ADDON}:${PYTHONPATH}
cd "${REPAIRED_BUNDLE}"
"${REPAIRED_HAB_PY}" -m unittest test_repaired_fullmono_design test_repaired_hm3d_constructibility test_repaired_fullmono_construction_probe test_hm3d_covisibility_repaired
if [[ "${DESIGN_PREFLIGHT_ONLY:-0}" == 1 ]]; then
  "${REPAIRED_HAB_PY}" - <<'PY'
import inspect
from pathlib import Path
from probe_repaired_fullmono_construction import SOURCES, load, sha
import probe_repaired_fullmono_construction as probe
print('ADDON', inspect.getfile(probe))
for scene, root, cs, vs in SOURCES:
    assert sha(root/'construction_summary.json') == cs
    assert sha(root/'independent_verification.json') == vs
    source=next(s for s in load(root/'manifest.json')['sources'] if s['scene']==scene)
    assert Path(source['asset']).is_file()
    online=root/'construction'/scene/'online_a'/scene/source['episode']
    assert (online/'receipt.json').is_file()
    assert (online/'online_a_trace.json').is_file()
    print('INPUTS_OK',scene)
print('No policy or renderer started in login-node preflight')
PY
  exit 0
fi
[[ ! -e "${DESIGN_RUN}" ]]
"${REPAIRED_HAB_PY}" -u "${DESIGN_ADDON}/probe_repaired_fullmono_construction.py" --out "${DESIGN_RUN}"
