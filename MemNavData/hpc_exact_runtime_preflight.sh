#!/usr/bin/env bash
# Outcome-blind dependency closure check for the exact HM3D Full-Mono runtime.
set -euo pipefail
umask 0022

TASK_ROOT=${TASK_ROOT:?set immutable task source root}
BASE_SOURCE_ROOT=${BASE_SOURCE_ROOT:?set immutable base source root}
MEMNAV_PY=${MEMNAV_PY:?set MemNav Python}
HAB_PY=${HAB_PY:?set Habitat Python}
OUT=${OUT:?set writable JSON receipt path}
LIGHTGLUE_REPO=${LIGHTGLUE_REPO:-${BASE_SOURCE_ROOT}/third_party/LightGlue}
DEPENDENCY_ROOT=${DEPENDENCY_ROOT:-${BASE_SOURCE_ROOT}/third_party/python}
INTERNNAV_ROOT=${INTERNNAV_ROOT:-/scratch/yz11502/Research/Nav-axis-uturn/InternNav/src/diffusion-policy}

NAVDP_SRC=${TASK_ROOT}/NavDP/baselines/navdp
MEMNAV_SRC=${TASK_ROOT}/NavDP/baselines/memnav
for path in \
  "${NAVDP_SRC}/policy_backbone.py" \
  "${NAVDP_SRC}/depth_anything/depth_anything_v2/dpt.py" \
  "${MEMNAV_SRC}/policy_agent.py" \
  "${TASK_ROOT}/MemNavData/eval_2leg_habitat.py" \
  "${LIGHTGLUE_REPO}" \
  "${DEPENDENCY_ROOT}"; do
  [[ -e "${path}" ]] || {
    echo "missing runtime dependency: ${path}" >&2
    exit 2
  }
done

runtime_tmp=${SLURM_TMPDIR:-/tmp}/hm3d_runtime_preflight_$$
mkdir -p "${runtime_tmp}/pycache"
trap 'rm -rf -- "${runtime_tmp}"' EXIT
export PYTHONPYCACHEPREFIX=${runtime_tmp}/pycache
export PYTHONDONTWRITEBYTECODE=1

HAB_SITE=$(${HAB_PY} -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
HAB_PYTHONPATH=${HAB_SITE}/pip/_vendor
COMMON_PYTHONPATH=${TASK_ROOT}:${TASK_ROOT}/MemNavData:${BASE_SOURCE_ROOT}:${BASE_SOURCE_ROOT}/MemNavData:${DEPENDENCY_ROOT}:${LIGHTGLUE_REPO}:${INTERNNAV_ROOT}:${HAB_PYTHONPATH}

(
  cd "${NAVDP_SRC}"
  PYTHONPATH="${NAVDP_SRC}:${COMMON_PYTHONPATH}" "${MEMNAV_PY}" - \
    "${TASK_ROOT}" "${runtime_tmp}/navdp.json" <<'PY'
import json
import sys
from pathlib import Path

task = Path(sys.argv[1]).resolve()
out = Path(sys.argv[2])
import policy_backbone
import policy_agent
from depth_anything.depth_anything_v2.dpt import DepthAnythingV2

paths = {
    "policy_backbone": str(Path(policy_backbone.__file__).resolve()),
    "policy_agent": str(Path(policy_agent.__file__).resolve()),
    "depth_anything_dpt": str(
        Path(sys.modules[DepthAnythingV2.__module__].__file__).resolve()
    ),
}
expected = task / "NavDP" / "baselines" / "navdp"
for name, value in paths.items():
    if not Path(value).is_relative_to(expected):
        raise SystemExit(f"{name} escaped task bundle: {value}")
out.write_text(json.dumps({"python": sys.version, "modules": paths}, sort_keys=True) + "\n")
PY
)

(
  cd "${MEMNAV_SRC}"
  PYTHONPATH="${MEMNAV_SRC}:${COMMON_PYTHONPATH}" "${MEMNAV_PY}" - \
    "${TASK_ROOT}" "${runtime_tmp}/memnav.json" "${BASE_SOURCE_ROOT}" <<'PY'
import json
import sys
from pathlib import Path

task = Path(sys.argv[1]).resolve()
base = Path(sys.argv[3]).resolve()
out = Path(sys.argv[2])
import policy_backbone
import policy_network
import policy_agent
import certified_relocalization_runtime

paths = {
    "policy_backbone": str(Path(policy_backbone.__file__).resolve()),
    "policy_network": str(Path(policy_network.__file__).resolve()),
    "policy_agent": str(Path(policy_agent.__file__).resolve()),
    "certificate_runtime": str(
        Path(certified_relocalization_runtime.__file__).resolve()
    ),
}
task_modules = {"policy_backbone", "policy_network", "policy_agent"}
for name, value in paths.items():
    path = Path(value)
    if name in task_modules and not path.is_relative_to(task):
        raise SystemExit(f"{name} escaped task bundle: {value}")
    if name not in task_modules and not (
        path.is_relative_to(task) or path.is_relative_to(base)
    ):
        raise SystemExit(f"{name} escaped verified source roots: {value}")
out.write_text(json.dumps({"python": sys.version, "modules": paths}, sort_keys=True) + "\n")
PY
)

PYTHONPATH="${COMMON_PYTHONPATH}" "${HAB_PY}" - \
  "${TASK_ROOT}" "${runtime_tmp}/habitat.json" "${runtime_tmp}" <<'PY'
import json
import py_compile
import sys
from pathlib import Path

task = Path(sys.argv[1]).resolve()
out = Path(sys.argv[2])
tmp = Path(sys.argv[3])
sources = [
    task / "MemNavData" / "eval_2leg_habitat.py",
    task / "MemNavData" / "collect_hm3d_fullmono_goal_a.py",
    task / "MemNavData" / "run_hm3d_fullmono_query_history.py",
]
compiled = []
for index, source in enumerate(sources):
    target = tmp / f"habitat_{index}.pyc"
    py_compile.compile(str(source), cfile=str(target), doraise=True)
    compiled.append(str(source))
out.write_text(json.dumps({"python": sys.version, "compiled": compiled}, sort_keys=True) + "\n")
PY

mkdir -p "$(dirname "${OUT}")"
"${MEMNAV_PY}" - "${OUT}" "${runtime_tmp}/navdp.json" \
  "${runtime_tmp}/memnav.json" "${runtime_tmp}/habitat.json" \
  "${TASK_ROOT}" "${BASE_SOURCE_ROOT}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

out, navdp, memnav, habitat, task, base = map(Path, sys.argv[1:])
payload = {
    "schema": "hm3d_exact_runtime_dependency_preflight_v1",
    "verified": True,
    "task_root": str(task),
    "base_source_root": str(base),
    "navdp": json.loads(navdp.read_text()),
    "memnav": json.loads(memnav.read_text()),
    "habitat": json.loads(habitat.read_text()),
}
encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
out.write_bytes(encoded)
out.with_suffix(out.suffix + ".sha256").write_text(
    hashlib.sha256(encoded).hexdigest() + "  " + out.name + "\n"
)
PY

echo "verified exact runtime dependency closure: ${OUT}"
