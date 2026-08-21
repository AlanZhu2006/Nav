#!/usr/bin/env bash
# Frozen-feature M2P S-1 contract/timing smoke or formal train40 collection.
# A partial smoke never evaluates or authorizes an effectiveness gate.
#SBATCH --job-name=m2p_s1_gct
#SBATCH --output=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/%x_%j.out
#SBATCH --error=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/%x_%j.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --partition=l40s_public,h100_tandon,a100_tandon
#SBATCH --gres=gpu:1
#SBATCH --account=torch_pr_769_tandon_advanced
#SBATCH --export=ALL

set -euo pipefail
umask 0022

: "${SOURCE_ROOT:?set immutable task source root}"
: "${RUN_ROOT:?set unique result root}"
: "${EXPECTED_SOURCE_RECEIPT_SHA:?pin source receipt}"

MODE="${MODE:-smoke}"
MAX_EPISODES="${MAX_EPISODES:-1}"
[[ "${MODE}" == "smoke" || "${MODE}" == "formal" ]] || {
  echo "ABORT: MODE must be smoke or formal" >&2; exit 2; }
[[ "${MAX_EPISODES}" =~ ^[0-9]+$ ]] || {
  echo "ABORT: invalid MAX_EPISODES" >&2; exit 2; }
if [[ "${MODE}" == "smoke" ]]; then
  (( MAX_EPISODES >= 1 )) || {
    echo "ABORT: smoke needs at least one episode" >&2; exit 2; }
else
  (( MAX_EPISODES == 0 )) || {
    echo "ABORT: formal collection must use all episodes" >&2; exit 2; }
fi

RECEIPT="${SOURCE_ROOT}/SOURCE_BUNDLE.sha256"
ENTRYPOINT="${SOURCE_ROOT}/MemNavData/diag_m2p_s1_train40_dual_context.py"
SELECTION="${SOURCE_ROOT}/.diagnostics/certificate_distilled_compass_20260813/static_top8_480_lightglue_open_set_rows.csv"
SPLIT="${SOURCE_ROOT}/MemNavData/router_multiscene_split_20260805.json"
MEMNAV_PY=/scratch/lg154/conda-envs/memnav/bin/python
BASE_SIF=/share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif
OVERLAY=/scratch/lg154/Research/datasets/_overlays/mp3d_revisit_v0_pt1.sqf
LINGBOT_REPO=/scratch/lg154/Research/Nav/NavDP/baselines/memnav/lingbot-map
WEIGHTS="${LINGBOT_REPO}/weights/lingbot-map-long.pt"

EXPECTED_SELECTION_SHA=85f9064bff15ce59106ad2a1aa8e5dc4720ee1b1ad894aac1bcedf8581a1d127
EXPECTED_SPLIT_SHA=97309c183e25cb3dd65472908748d55a94798a636db6157ab6fe120fca05cf7a
EXPECTED_WEIGHTS_SHA=832bc82cbae0bc9bbe946ef5ee1f7226abd8c0e183ccf8beddbb3d133576f409
EXPECTED_LINGBOT_REVISION=7ff6f3ed0913d4d326f8f13bbb429c4ffc0195c2
EXPECTED_OVERLAY_BYTES=128854888448
EXPECTED_OVERLAY_HEAD_SHA=43d803c2b39735e764f0c6df8f7c40a7035a1c23789e22fcc9831d29928c801b

fail() { echo "ABORT: $*" >&2; exit 2; }
for command_name in singularity sha256sum nvidia-smi git; do
  command -v "${command_name}" >/dev/null || fail "${command_name} unavailable"
done
[[ -d "${SOURCE_ROOT}" && ! -L "${SOURCE_ROOT}" ]] || fail "bad source root"
[[ -f "${RECEIPT}" && ! -L "${RECEIPT}" ]] || fail "missing source receipt"
[[ "$(sha256sum "${RECEIPT}" | awk '{print $1}')" == \
   "${EXPECTED_SOURCE_RECEIPT_SHA}" ]] || fail "source receipt SHA changed"
(cd "${SOURCE_ROOT}" && sha256sum -c SOURCE_BUNDLE.sha256)
[[ "$(sha256sum "${SELECTION}" | awk '{print $1}')" == \
   "${EXPECTED_SELECTION_SHA}" ]] || fail "selection CSV changed"
[[ "$(sha256sum "${SPLIT}" | awk '{print $1}')" == \
   "${EXPECTED_SPLIT_SHA}" ]] || fail "split manifest changed"
[[ -f "${WEIGHTS}" && ! -L "${WEIGHTS}" ]] || fail "weights missing/symlinked"
[[ -f "${BASE_SIF}" && -f "${OVERLAY}" ]] || fail "container inputs missing"
[[ "$(stat -c '%s' "${OVERLAY}")" == "${EXPECTED_OVERLAY_BYTES}" ]] || \
  fail "overlay size changed"
overlay_head_sha="$(dd if="${OVERLAY}" bs=1048576 count=1 status=none | \
  sha256sum | awk '{print $1}')"
[[ "${overlay_head_sha}" == "${EXPECTED_OVERLAY_HEAD_SHA}" ]] || \
  fail "overlay header changed"
lingbot_revision="$(git -c safe.directory="${LINGBOT_REPO}" \
  -C "${LINGBOT_REPO}" rev-parse HEAD)"
[[ "${lingbot_revision}" == "${EXPECTED_LINGBOT_REVISION}" ]] || \
  fail "LingBot revision changed"

mkdir -p "${RUN_ROOT}"
[[ -z "$(find "${RUN_ROOT}" -mindepth 1 -maxdepth 1 -print -quit)" ]] || \
  fail "result root is not empty"
nvidia-smi

export PYTHONPATH="${SOURCE_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export APPTAINERENV_PYTHONPATH="${PYTHONPATH}"
export SINGULARITYENV_PYTHONPATH="${PYTHONPATH}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export GIT_CONFIG_COUNT=1
export GIT_CONFIG_KEY_0=safe.directory
export GIT_CONFIG_VALUE_0="${LINGBOT_REPO}"

CONTAINER=(singularity exec --nv --overlay "${OVERLAY}:ro"
  -B /scratch/lg154
  -B /scratch/yz11502
  "${BASE_SIF}")

ARGS=(
  --selection-csv "${SELECTION}"
  --split-manifest "${SPLIT}"
  --lingbot-repo "${LINGBOT_REPO}"
  --weights "${WEIGHTS}"
  --out "${RUN_ROOT}/collection"
  --expected-selection-sha256 "${EXPECTED_SELECTION_SHA}"
  --expected-split-sha256 "${EXPECTED_SPLIT_SHA}"
  --expected-weight-sha256 "${EXPECTED_WEIGHTS_SHA}"
  --expected-lingbot-revision "${EXPECTED_LINGBOT_REVISION}"
)
if [[ "${MODE}" == "smoke" ]]; then
  ARGS+=(--smoke-mode --max-episodes "${MAX_EPISODES}" --bootstrap-resamples 100)
fi

started="$(date +%s)"
"${CONTAINER[@]}" "${MEMNAV_PY}" "${ENTRYPOINT}" "${ARGS[@]}"
finished="$(date +%s)"
printf '{"job_id":"%s","mode":"%s","max_episodes":%s,"elapsed_s":%s}\n' \
  "${SLURM_JOB_ID}" "${MODE}" "${MAX_EPISODES}" "$((finished - started))" \
  > "${RUN_ROOT}/runtime_receipt.json"
echo "M2P_S1_RUN_ROOT=${RUN_ROOT}"
