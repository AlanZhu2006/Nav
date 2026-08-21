#!/usr/bin/env bash
set -euo pipefail

# Reproducible, isolated dependencies for the controller-portability baselines.
# Model weights remain outside Git under .diagnostics/.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DIAG_ROOT="${REPO_ROOT}/.diagnostics/controller_portability_20260821"
NAVDP_PYTHON="/home/asus/miniconda3/envs/navdp/bin/python"

VINT_ENV="${DIAG_ROOT}/envs/vint"
if [[ ! -x "${VINT_ENV}/bin/python" ]]; then
  "${NAVDP_PYTHON}" -m venv --system-site-packages "${VINT_ENV}"
fi
"${VINT_ENV}/bin/python" -m pip install --disable-pip-version-check \
  efficientnet_pytorch==0.7.1

# ViPlanner follows the environment recommended by the NavDP baseline release.
# A separate Torch/CUDA runtime is intentional: the local Torch 2.2/CUDA 12.1
# stack has no matching prebuilt mmcv wheel and would attempt an nvcc build.
VIPLANNER_ENV="${DIAG_ROOT}/envs/viplanner-py310-cu118"
if [[ ! -x "${VIPLANNER_ENV}/bin/python" ]]; then
  "${NAVDP_PYTHON}" -m venv "${VIPLANNER_ENV}"
fi
"${VIPLANNER_ENV}/bin/python" -m pip install --disable-pip-version-check \
  torch==2.0.1+cu118 torchvision==0.15.2+cu118 \
  --extra-index-url https://download.pytorch.org/whl/cu118
"${VIPLANNER_ENV}/bin/python" -m pip install --disable-pip-version-check \
  numpy==1.26.4 opencv-python==4.10.0.84 PyYAML==6.0.3 \
  addict==2.4.0 packaging==26.3 yapf==0.43.0 mmengine==0.10.7 \
  termcolor==3.3.0 rich==15.0.0 matplotlib==3.10.9
"${VIPLANNER_ENV}/bin/python" -m pip install --disable-pip-version-check \
  --no-deps mmcv==2.0.0 \
  -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.0/index.html
"${VIPLANNER_ENV}/bin/python" -m pip install --disable-pip-version-check \
  imageio==2.37.0 imageio-ffmpeg==0.6.0 Flask==3.1.2 \
  mmdet==3.3.0
"${VIPLANNER_ENV}/bin/python" -m pip install --disable-pip-version-check \
  git+https://github.com/cocodataset/panopticapi.git@7bb4655548f98f3fedc07bf37e9040a992b054b0
