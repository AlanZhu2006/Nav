#!/usr/bin/env bash
# Validate two official GOAT val_unseen scenes without producing a paper score.
#SBATCH --job-name=goat_smoke2
#SBATCH --output=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/%x_%j.out
#SBATCH --error=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/%x_%j.err
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
# The prior L40S and H100 failures had the same wrapper-level EGL-library
# cause, so no node family is excluded without a post-fix counterexample.
#SBATCH --partition=h100_tandon,a100_tandon,h200_public,l40s_public
#SBATCH --qos=gpu48
#SBATCH --gres=gpu:1
#SBATCH --account=torch_pr_769_tandon_advanced

set -euo pipefail
umask 0022

: "${SOURCE_ROOT:?set immutable smoke source root}"
: "${SOURCE_RECEIPT:?set source receipt}"
: "${EXPECTED_SOURCE_RECEIPT_SHA:?set source receipt SHA-256}"
: "${RUN_ROOT:?set unique GOAT smoke result root}"

readonly DATA_ROOT="${GOAT_HM3D_ROOT:-/scratch/yz11502/Research/datasets/goat_bench_20260814}"
readonly ENV_PREFIX="${GOAT_ENV_PREFIX:-/scratch/yz11502/conda_envs/goat-bench-habitat023-20260814}"
readonly GOAT_CODE="${GOAT_CODE_ROOT:-${DATA_ROOT}/code/goat-bench}"
readonly BASE_SIF="${GOAT_BASE_SIF:-/share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif}"
readonly ENV_RECEIPT="${DATA_ROOT}/receipts/goat-environment-v023.json"
readonly EXTRACT_RECEIPT="${DATA_ROOT}/receipts/goat-val-unseen-extraction.json"
readonly RUNNER="${SOURCE_ROOT}/MemNavData/goat_contract_smoke.py"
readonly ADAPTER="${SOURCE_ROOT}/MemNavData/goat_navdp_discrete_adapter.py"

fail() { echo "ABORT: $*" >&2; exit 2; }

[[ "$(sha256sum "${SOURCE_RECEIPT}" | awk '{print $1}')" == \
   "${EXPECTED_SOURCE_RECEIPT_SHA}" ]] || fail "source receipt hash changed"
(cd "${SOURCE_ROOT}" && sha256sum -c "${SOURCE_RECEIPT}") >/dev/null || \
  fail "immutable smoke bundle changed"
[[ -x "${ENV_PREFIX}/bin/python" && -f "${ENV_RECEIPT}" ]] || \
  fail "GOAT Habitat environment is incomplete"
[[ -f "${EXTRACT_RECEIPT}" && -f "${DATA_ROOT}/data/scene_datasets/hm3d/val/SEALED" ]] || \
  fail "GOAT HM3D scene extraction is incomplete"
[[ -d "${GOAT_CODE}/.git" && -f "${RUNNER}" && -f "${ADAPTER}" ]] || \
  fail "GOAT code or smoke runner is missing"
[[ -r "${BASE_SIF}" ]] || fail "GOAT system image is missing"
[[ "$(git -C "${GOAT_CODE}" rev-parse HEAD)" == \
   74c41d19d4a4c3608d1575b512087b5a529aee0e ]] || \
  fail "GOAT source commit changed"

"${ENV_PREFIX}/bin/python" - "${ENV_RECEIPT}" "${EXTRACT_RECEIPT}" <<'PY'
import json
import sys

for path in sys.argv[1:]:
    payload = json.load(open(path))
    if not payload.get("complete"):
        raise SystemExit(f"incomplete prerequisite receipt: {path}")
PY

mkdir -p "${RUN_ROOT}"
test -z "$(find "${RUN_ROOT}" -mindepth 1 -maxdepth 1 -print -quit)" || \
  fail "run root is not empty"

export PYTHONNOUSERSITE=1
# Python 3.7 predates reliable PYTHONPYCACHEPREFIX support.  The immutable
# source bundle is read-only, so disable implicit bytecode writes explicitly.
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPYCACHEPREFIX="${SLURM_TMPDIR:-/tmp}/goat_smoke_${SLURM_JOB_ID}"
export MAGNUM_LOG=quiet
export HABITAT_SIM_LOG=quiet
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
# Do not pass the outer LD_LIBRARY_PATH through ``env`` below.  Singularity
# --nv injects the NVIDIA EGL vendor libraries via /.singularity.d/libs; an
# explicit replacement silently falls back to the image's non-NVIDIA EGL and
# makes Habitat-Sim report that CUDA device 0 has no matching EGL device.

"${ENV_PREFIX}/bin/python" - "${RUNNER}" "${ADAPTER}" <<'PY'
import ast
import pathlib
import sys

for raw_path in sys.argv[1:]:
    path = pathlib.Path(raw_path)
    ast.parse(path.read_text(), filename=str(path))
PY
cd "${GOAT_CODE}"
singularity exec --nv -B /scratch/yz11502 "${BASE_SIF}" env \
  PYTHONNOUSERSITE=1 \
  PYTHONDONTWRITEBYTECODE=1 \
  PYTHONPYCACHEPREFIX="${PYTHONPYCACHEPREFIX}" \
  MAGNUM_LOG="${MAGNUM_LOG}" \
  HABITAT_SIM_LOG="${HABITAT_SIM_LOG}" \
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
  "${ENV_PREFIX}/bin/python" "${RUNNER}" \
    --goat-code "${GOAT_CODE}" \
    --data-root "${DATA_ROOT}" \
    --output-dir "${RUN_ROOT}"

test -s "${RUN_ROOT}/goat_two_scene_contract_smoke.json"
sha256sum "${RUN_ROOT}/goat_two_scene_contract_smoke.json" \
  > "${RUN_ROOT}/goat_two_scene_contract_smoke.json.sha256"
: > "${RUN_ROOT}/SEALED"
