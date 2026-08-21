#!/usr/bin/env bash
# Build the isolated Habitat 0.2.3 environment pinned by GOAT-Bench.
# This job never modifies the existing NavDP or TransportVGGT environments.
#SBATCH --job-name=goat_env023
#SBATCH --output=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/%x_%j.out
#SBATCH --error=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/%x_%j.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=cpu_short
#SBATCH --account=torch_pr_769_tandon_advanced

set -euo pipefail
umask 0022

readonly ROOT="${GOAT_HM3D_ROOT:-/scratch/yz11502/Research/datasets/goat_bench_20260814}"
readonly ENV_PREFIX="${GOAT_ENV_PREFIX:-/scratch/yz11502/conda_envs/goat-bench-habitat023-20260814}"
readonly GOAT_CODE="${GOAT_CODE_ROOT:-${ROOT}/code/goat-bench}"
readonly HABITAT_CODE="${HABITAT_CODE_ROOT:-${ROOT}/code/habitat-lab-v0.2.3}"
readonly BASE_SIF="${GOAT_BASE_SIF:-/share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif}"
readonly RECEIPT_DIR="${ROOT}/receipts"
readonly RECEIPT="${RECEIPT_DIR}/goat-environment-v023.json"
readonly EXPECTED_GOAT_COMMIT="74c41d19d4a4c3608d1575b512087b5a529aee0e"
readonly EXPECTED_HABITAT_TAG="v0.2.3"

mkdir -p "${ROOT}/code" "${ROOT}/conda_pkgs" "${RECEIPT_DIR}"
export CONDA_PKGS_DIRS="${ROOT}/conda_pkgs"

exec 9>"${ROOT}/goat-environment-v023.lock"
flock -n 9 || {
  echo "another GOAT environment setup owns the lock" >&2
  exit 3
}

test -d "${GOAT_CODE}/.git"
test -r "${BASE_SIF}"
actual_goat_commit="$(git -C "${GOAT_CODE}" rev-parse HEAD)"
test "${actual_goat_commit}" = "${EXPECTED_GOAT_COMMIT}"
test -z "$(git -C "${GOAT_CODE}" status --porcelain)"

if [[ ! -d "${HABITAT_CODE}/.git" ]]; then
  test ! -e "${HABITAT_CODE}"
  git clone --depth 1 --branch "${EXPECTED_HABITAT_TAG}" \
    https://github.com/facebookresearch/habitat-lab.git "${HABITAT_CODE}"
fi

actual_habitat_tag="$(git -C "${HABITAT_CODE}" describe --tags --exact-match HEAD)"
test "${actual_habitat_tag}" = "${EXPECTED_HABITAT_TAG}"
actual_habitat_commit="$(git -C "${HABITAT_CODE}" rev-parse HEAD)"
test -z "$(git -C "${HABITAT_CODE}" status --porcelain)"

if [[ ! -x "${ENV_PREFIX}/bin/python" ]]; then
  test ! -e "${ENV_PREFIX}" || {
    echo "refusing to reuse a partial non-environment path: ${ENV_PREFIX}" >&2
    exit 4
  }
  conda create --yes --prefix "${ENV_PREFIX}" \
    python=3.7 cmake=3.14.0
fi

# Match the versions and channels released by GOAT-Bench.  A dedicated prefix
# keeps these legacy dependencies from contaminating the frozen NavDP runtime.
conda install --yes --prefix "${ENV_PREFIX}" \
  habitat-sim=0.2.3 headless pytorch torchvision cudatoolkit=11.3 \
  -c pytorch -c nvidia -c conda-forge -c aihabitat

"${ENV_PREFIX}/bin/python" -m pip install --upgrade \
  'pip<24' 'setuptools<69' wheel
"${ENV_PREFIX}/bin/python" -m pip install \
  -e "${HABITAT_CODE}/habitat-lab" \
  -e "${HABITAT_CODE}/habitat-baselines" \
  -e "${GOAT_CODE}"

# Do not install GOAT's policy-only convenience requirements here.  The
# contract harness deliberately registers only the released dataset, task,
# simulator, and measurement modules; none imports transformers, LAVIS, VC-1,
# Open3D, or safetensors.  Keeping those packages out also avoids asking a
# 2026 resolver to build modern safetensors releases for legacy Python 3.7.

# Habitat-Lab v0.2.3 assumes the OpenCV 4 applyColorMap output shape.  An
# unconstrained 2026 resolver can select OpenCV 5 for this legacy Python,
# causing import-time failure in habitat.utils.visualizations.maps.
"${ENV_PREFIX}/bin/python" -m pip install --force-reinstall \
  'numpy==1.21.6' 'opencv-python==4.7.0.72'

# The HPC host image does not provide libOpenGL.so.0.  Use the same immutable
# system image as the established NavDP evaluations for the import audit while
# retaining all Python packages in the dedicated GOAT prefix.
singularity exec -B /scratch/yz11502 "${BASE_SIF}" env \
  PYTHONNOUSERSITE=1 \
  "${ENV_PREFIX}/bin/python" - "${RECEIPT}" "${ENV_PREFIX}" \
  "${actual_goat_commit}" "${actual_habitat_commit}" <<'PY'
import datetime
import json
import pathlib
import platform
import sys

import habitat
import habitat_sim
import hydra
import numpy
import torch
import cv2

receipt, env_prefix, goat_commit, habitat_commit = sys.argv[1:]
payload = {
    "schema_version": "goat_environment_v023_20260814",
    "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "complete": True,
    "env_prefix": str(pathlib.Path(env_prefix).resolve()),
    "python": platform.python_version(),
    "torch": torch.__version__,
    "numpy": numpy.__version__,
    "opencv": cv2.__version__,
    "habitat_sim": getattr(habitat_sim, "__version__", "unknown"),
    "habitat_lab_import": str(pathlib.Path(habitat.__file__).resolve()),
    "hydra": hydra.__version__,
    "goat_commit": goat_commit,
    "habitat_lab_commit": habitat_commit,
    "core_imports_passed": True,
    "full_goat_policy_dependencies_installed": False,
}
pathlib.Path(receipt).write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n"
)
print(json.dumps(payload, sort_keys=True))
PY
