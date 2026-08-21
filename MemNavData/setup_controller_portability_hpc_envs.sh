#!/usr/bin/env bash
# Build immutable ViNT/ViPlanner runtime environments on an HPC compute node.
set -euo pipefail
umask 0022

ENV_ROOT=${ENV_ROOT:?set final immutable environment root}
MEMNAV_PY=${MEMNAV_PY:-/scratch/lg154/conda-envs/memnav/bin/python}
SETUP_SOURCE=${SETUP_SOURCE:-${BASH_SOURCE[0]}}

fail() { echo "ABORT: $*" >&2; exit 2; }
[[ -x "${MEMNAV_PY}" ]] || fail "missing MemNav Python: ${MEMNAV_PY}"
[[ -r "${SETUP_SOURCE}" ]] || fail "missing setup source"

if [[ -d "${ENV_ROOT}" ]]; then
  [[ -r "${ENV_ROOT}/environment_receipt.json" ]] || \
    fail "environment root exists without a receipt"
  "${MEMNAV_PY}" - "${ENV_ROOT}/environment_receipt.json" <<'PY'
import json,sys
p=json.load(open(sys.argv[1]))
assert p["schema"] == "cec_controller_portability_hpc_env_v1"
assert p["verified"] is True
PY
  echo "READY existing=${ENV_ROOT}"
  exit 0
fi

parent=$(dirname "${ENV_ROOT}")
mkdir -p "${parent}"
build=${ENV_ROOT}.building.${SLURM_JOB_ID:-manual}.$$
[[ ! -e "${build}" ]] || fail "build path already exists"
mkdir -p "${build}"

"${MEMNAV_PY}" -m venv --system-site-packages "${build}/vint"
"${build}/vint/bin/python" -m pip install --disable-pip-version-check \
  efficientnet_pytorch==0.7.1

"${MEMNAV_PY}" -m venv "${build}/viplanner"
"${build}/viplanner/bin/python" -m pip install --disable-pip-version-check \
  torch==2.0.1+cu118 torchvision==0.15.2+cu118 \
  --extra-index-url https://download.pytorch.org/whl/cu118
"${build}/viplanner/bin/python" -m pip install --disable-pip-version-check \
  numpy==1.26.4 opencv-python==4.10.0.84 PyYAML==6.0.3 \
  addict==2.4.0 packaging==26.3 yapf==0.43.0 mmengine==0.10.7 \
  termcolor==3.3.0 rich==15.0.0 matplotlib==3.10.9
"${build}/viplanner/bin/python" -m pip install --disable-pip-version-check \
  --no-deps mmcv==2.0.0 \
  -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.0/index.html
"${build}/viplanner/bin/python" -m pip install --disable-pip-version-check \
  imageio==2.37.0 imageio-ffmpeg==0.6.0 Flask==3.1.2 \
  mmdet==3.3.0
"${build}/viplanner/bin/python" -m pip install --disable-pip-version-check \
  git+https://github.com/cocodataset/panopticapi.git@7bb4655548f98f3fedc07bf37e9040a992b054b0

"${build}/vint/bin/python" - <<'PY'
import efficientnet_pytorch, torch
assert efficientnet_pytorch.__version__ == "0.7.1"
print("vint", torch.__version__, efficientnet_pytorch.__version__)
PY
"${build}/viplanner/bin/python" - <<'PY'
import cv2, mmcv, mmdet, mmengine, numpy, torch, torchvision
assert torch.__version__.startswith("2.0.1")
assert torchvision.__version__.startswith("0.15.2")
assert mmcv.__version__ == "2.0.0"
assert mmdet.__version__ == "3.3.0"
assert mmengine.__version__ == "0.10.7"
assert numpy.__version__ == "1.26.4"
print("viplanner", torch.__version__, torchvision.__version__, mmcv.__version__)
PY

"${MEMNAV_PY}" - "${build}/environment_receipt.json" \
  "${build}/vint/bin/python" "${build}/viplanner/bin/python" \
  "${SETUP_SOURCE}" <<'PY'
import hashlib,importlib.metadata,json,subprocess,sys
from pathlib import Path

out,vint_py,viplanner_py,setup=sys.argv[1:]
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def inspect(python,names):
    code=("import importlib.metadata,json; print(json.dumps({n: "
          "importlib.metadata.version(n) for n in " + repr(names) + "}))")
    return json.loads(subprocess.check_output([python,"-c",code],text=True))
payload={
  "schema":"cec_controller_portability_hpc_env_v1",
  "verified":True,
  "setup_source_sha256":sha(setup),
  "vint":inspect(vint_py,["efficientnet-pytorch"]),
  "viplanner":inspect(viplanner_py,[
      "torch","torchvision","numpy","opencv-python","mmcv","mmengine",
      "mmdet","Flask","panopticapi"]),
}
Path(out).write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n")
PY

find "${build}" -type d -name __pycache__ -prune -exec rm -rf -- {} +
chmod -R a-w "${build}"
mv "${build}" "${ENV_ROOT}"
echo "READY created=${ENV_ROOT}"
