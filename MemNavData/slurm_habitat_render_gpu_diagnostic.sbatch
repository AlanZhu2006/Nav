#!/usr/bin/env bash
# Same frozen MP3D scene/pose on one allocated GPU; partition is set at submit.
#SBATCH --job-name=hab_render_diag
#SBATCH --output=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/%x_%j.out
#SBATCH --error=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/%x_%j.err
#SBATCH --time=00:20:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu48
#SBATCH --account=torch_pr_769_tandon_advanced
#SBATCH --export=ALL

set -euo pipefail
umask 0022

: "${DIAG_SOURCE_ROOT:?set immutable diagnostic source bundle}"
: "${DIAG_SOURCE_RECEIPT_SHA:?set diagnostic receipt SHA}"
: "${METHOD_SOURCE_ROOT:?set immutable Attempt 6 method source bundle}"
: "${METHOD_SOURCE_RECEIPT_SHA:?set Attempt 6 source receipt SHA}"
: "${OUT_JSONL:?set unique output path}"

HAB_PY=/scratch/lg154/conda-envs/habitat/bin/python
BASE_SIF=/share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif
SCENE=/scratch/lg154/Research/datasets/mp3d/mp3d/8WUmhLawc2A/8WUmhLawc2A.glb
SCENE_SHA=d652bd05699b2ab481f4c474b0dbdc0ba8422cdc8d1310182fb25a9497b153ff
ENTRYPOINT=${DIAG_SOURCE_ROOT}/MemNavData/diagnose_habitat_render_stability.py

fail() { echo "ABORT: $*" >&2; exit 2; }
[[ "$(sha256sum "${DIAG_SOURCE_ROOT}/SOURCE_BUNDLE.sha256" | awk '{print $1}')" == \
  "${DIAG_SOURCE_RECEIPT_SHA}" ]] || fail "diagnostic receipt changed"
(cd "${DIAG_SOURCE_ROOT}" && sha256sum -c SOURCE_BUNDLE.sha256 --quiet) || \
  fail "diagnostic bundle changed"
[[ "$(sha256sum "${METHOD_SOURCE_ROOT}/SOURCE_BUNDLE.sha256" | awk '{print $1}')" == \
  "${METHOD_SOURCE_RECEIPT_SHA}" ]] || fail "method receipt changed"
(cd "${METHOD_SOURCE_ROOT}" && sha256sum -c SOURCE_BUNDLE.sha256 --quiet) || \
  fail "method bundle changed"
[[ -x "${HAB_PY}" && -f "${BASE_SIF}" && -r "${SCENE}" ]] || \
  fail "runtime dependency missing"
[[ ! -e "${OUT_JSONL}" ]] || fail "diagnostic output already exists"
mkdir -p "$(dirname "${OUT_JSONL}")"

echo "host=$(hostname) job=${SLURM_JOB_ID}"
nvidia-smi --query-gpu=name,uuid,driver_version,memory.total \
  --format=csv,noheader
ulimit -c unlimited || true
export PYTHONFAULTHANDLER=1
export PYTHONPYCACHEPREFIX=${SLURM_TMPDIR:-/tmp}/hab_render_diag_${SLURM_JOB_ID}
mkdir -p "${PYTHONPYCACHEPREFIX}"

singularity exec --nv -B /scratch/lg154 -B /scratch/yz11502 \
  "${BASE_SIF}" env \
  PYTHONFAULTHANDLER=1 PYTHONPYCACHEPREFIX="${PYTHONPYCACHEPREFIX}" \
  PYTHONPATH="${METHOD_SOURCE_ROOT}:${METHOD_SOURCE_ROOT}/MemNavData" \
  "${HAB_PY}" -X faulthandler "${ENTRYPOINT}" \
    --scene "${SCENE}" --scene-sha256 "${SCENE_SHA}" \
    --position -0.9123131036758423 0.10065580904483795 0.5004647374153137 \
    --yaw 4.57714258977832 --renders 16 --yaw-step-deg 1.0 \
  | tee "${OUT_JSONL}"

test "$(grep -c '"event": "render"' "${OUT_JSONL}")" -eq 16 || \
  fail "render count incomplete"
grep -q '"event": "complete"' "${OUT_JSONL}" || \
  fail "completion record missing"
