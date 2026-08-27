#!/usr/bin/env bash
# Train-only NavDP arrival-consensus audit.  This does not authorize a STOP rule.
#SBATCH --job-name=navdp_arrival
#SBATCH --output=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/%x_%j.out
#SBATCH --error=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/%x_%j.err
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --partition=h100_tandon,a100_tandon,h200_public,l40s_public
#SBATCH --qos=gpu48
#SBATCH --gres=gpu:1
#SBATCH --account=torch_pr_769_tandon_advanced
#SBATCH --export=ALL

set -euo pipefail
umask 0022

: "${SOURCE_ROOT:?set immutable audit source root}"
: "${SOURCE_RECEIPT:?set source receipt path}"
: "${EXPECTED_SOURCE_RECEIPT_SHA:?set source receipt SHA-256}"
: "${BASE_SOURCE_ROOT:?set pinned NavDP source root}"
: "${EXPECTED_BASE_SOURCE_RECEIPT_SHA:?set base source receipt SHA-256}"
: "${RUN_ROOT:?set unique result root}"

MODE=${MODE:-smoke}
MAX_EPISODES=${MAX_EPISODES:-1}
EPISODE_START_INDEX=${EPISODE_START_INDEX:-0}
[[ "${MODE}" == "smoke" || "${MODE}" == "formal" || \
   "${MODE}" == "repair" ]] || {
  echo "ABORT: MODE must be smoke, formal, or repair" >&2; exit 2; }
[[ "${MAX_EPISODES}" =~ ^[0-9]+$ ]] || {
  echo "ABORT: MAX_EPISODES must be non-negative" >&2; exit 2; }
[[ "${EPISODE_START_INDEX}" =~ ^[0-9]+$ ]] || {
  echo "ABORT: EPISODE_START_INDEX must be non-negative" >&2; exit 2; }
if [[ "${MODE}" == "smoke" ]]; then
  (( MAX_EPISODES >= 1 && EPISODE_START_INDEX == 0 )) || {
    echo "ABORT: smoke requires at least one episode" >&2; exit 2; }
elif [[ "${MODE}" == "formal" ]]; then
  (( MAX_EPISODES == 0 && EPISODE_START_INDEX == 0 )) || {
    echo "ABORT: formal must consume the full frozen train set" >&2; exit 2; }
else
  (( MAX_EPISODES == 0 && EPISODE_START_INDEX == 74 )) || {
    echo "ABORT: repair must consume the frozen suffix beginning at 74" >&2
    exit 2
  }
  : "${PREFIX_SAMPLES:?set frozen failed-run prefix samples}"
  : "${EXPECTED_PREFIX_SAMPLES_SHA:?set prefix samples SHA-256}"
fi

readonly ENTRYPOINT="${SOURCE_ROOT}/MemNavData/audit_navdp_arrival_consensus.py"
readonly MERGER="${SOURCE_ROOT}/MemNavData/merge_navdp_arrival_consensus_repair.py"
readonly SELECTION="${SOURCE_ROOT}/.diagnostics/certificate_distilled_compass_20260813/static_top8_480_lightglue_open_set_rows.csv"
readonly SPLIT="${SOURCE_ROOT}/MemNavData/router_multiscene_split_20260805.json"
readonly NAVDP_SERVER="${BASE_SOURCE_ROOT}/NavDP/baselines/navdp/navdp_server.py"
readonly NAVDP_CKPT="/scratch/yz11502/Research/Nav-axis-uturn/.diagnostics/unseen_scene_eval_20260803/checkpoints/navdp_checkpoint.ckpt"
readonly MEMNAV_PY="/scratch/lg154/conda-envs/memnav/bin/python"
readonly BASE_SIF="/share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif"
readonly OVERLAY="/scratch/lg154/Research/datasets/_overlays/mp3d_revisit_v0_pt1.sqf"
readonly EPISODE_ROOT="/mp3d_revisit_v0/vln_n1/traj_data/mp3d_3leg"
readonly RAW_GAP_SCENE="/scratch/yz11502/Research/Nav-axis-uturn-results/nlsr_gapfill_20260807/nlsr_routes_41c1773_20260807_v1/raw/mp3d_3leg/YmJkqBEsHnH"
readonly RAW_GAP_AUDIT="/scratch/yz11502/Research/Nav-axis-uturn-results/nlsr_gapfill_20260807/nlsr_routes_41c1773_20260807_v1/raw_audit.json"
readonly EXPECTED_SELECTION_SHA="85f9064bff15ce59106ad2a1aa8e5dc4720ee1b1ad894aac1bcedf8581a1d127"
readonly EXPECTED_SPLIT_SHA="97309c183e25cb3dd65472908748d55a94798a636db6157ab6fe120fca05cf7a"
readonly EXPECTED_NAVDP_CKPT_SHA="3bb3ad4ab241e857bb57a4021cc6aab76d5263e81fbf80298d579053ef011947"
readonly EXPECTED_OVERLAY_BYTES="128854888448"
readonly EXPECTED_OVERLAY_HEAD_SHA="43d803c2b39735e764f0c6df8f7c40a7035a1c23789e22fcc9831d29928c801b"
readonly EXPECTED_RAW_GAP_AUDIT_SHA="d5a9e7548aa897f04be4e75cf27ad0634b4573f8c8b2ad79a7dbbf79997f771d"

fail() { echo "ABORT: $*" >&2; exit 2; }
for command_name in singularity sha256sum nvidia-smi ss; do
  command -v "${command_name}" >/dev/null || fail "${command_name} unavailable"
done
[[ -d "${SOURCE_ROOT}" && ! -L "${SOURCE_ROOT}" ]] || fail "bad source root"
[[ -f "${SOURCE_RECEIPT}" && ! -L "${SOURCE_RECEIPT}" ]] || \
  fail "missing source receipt"
[[ "$(sha256sum "${SOURCE_RECEIPT}" | awk '{print $1}')" == \
   "${EXPECTED_SOURCE_RECEIPT_SHA}" ]] || fail "source receipt changed"
(cd "${SOURCE_ROOT}" && sha256sum -c "${SOURCE_RECEIPT}") >/dev/null || \
  fail "source bundle changed"
[[ "$(sha256sum "${BASE_SOURCE_ROOT}/SOURCE_BUNDLE.sha256" | awk '{print $1}')" == \
   "${EXPECTED_BASE_SOURCE_RECEIPT_SHA}" ]] || fail "base source receipt changed"
(cd "${BASE_SOURCE_ROOT}" && sha256sum -c SOURCE_BUNDLE.sha256) >/dev/null || \
  fail "base source bundle changed"
[[ "$(sha256sum "${SELECTION}" | awk '{print $1}')" == \
   "${EXPECTED_SELECTION_SHA}" ]] || fail "selection table changed"
[[ "$(sha256sum "${SPLIT}" | awk '{print $1}')" == \
   "${EXPECTED_SPLIT_SHA}" ]] || fail "split manifest changed"
[[ "$(sha256sum "${NAVDP_CKPT}" | awk '{print $1}')" == \
   "${EXPECTED_NAVDP_CKPT_SHA}" ]] || fail "NavDP checkpoint changed"
[[ -f "${ENTRYPOINT}" && -f "${MERGER}" && -f "${NAVDP_SERVER}" && \
   -x "${MEMNAV_PY}" && -f "${BASE_SIF}" && -f "${OVERLAY}" && \
   -d "${RAW_GAP_SCENE}" && -f "${RAW_GAP_AUDIT}" ]] || \
  fail "runtime input missing"
[[ "$(stat -c '%s' "${OVERLAY}")" == "${EXPECTED_OVERLAY_BYTES}" ]] || \
  fail "dataset overlay size changed"
overlay_head_sha=$(dd if="${OVERLAY}" bs=1048576 count=1 status=none | \
  sha256sum | awk '{print $1}')
[[ "${overlay_head_sha}" == "${EXPECTED_OVERLAY_HEAD_SHA}" ]] || \
  fail "dataset overlay header changed"
[[ "$(sha256sum "${RAW_GAP_AUDIT}" | awk '{print $1}')" == \
   "${EXPECTED_RAW_GAP_AUDIT_SHA}" ]] || fail "raw gap audit changed"
if [[ "${MODE}" == "repair" ]]; then
  [[ -f "${PREFIX_SAMPLES}" && ! -L "${PREFIX_SAMPLES}" ]] || \
    fail "prefix samples are missing"
  [[ "$(sha256sum "${PREFIX_SAMPLES}" | awk '{print $1}')" == \
     "${EXPECTED_PREFIX_SAMPLES_SHA}" ]] || fail "prefix samples changed"
fi
[[ -d "${RUN_ROOT}" && -f "${RUN_ROOT}/submission.json" ]] || \
  fail "precommitted run root missing"
[[ "$(find "${RUN_ROOT}" -mindepth 1 -maxdepth 1 -printf '%f\n' | sort)" == \
   "submission.json" ]] || fail "run root contains pre-run artifacts"

export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
export PYTHONPYCACHEPREFIX="${SLURM_TMPDIR:-/tmp}/navdp_arrival_${SLURM_JOB_ID}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-12}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-12}"
mkdir -p "${PYTHONPYCACHEPREFIX}" "${RUN_ROOT}/logs"
cp "${SOURCE_RECEIPT}" "${RUN_ROOT}/source_bundle.sha256"
chmod a-w "${RUN_ROOT}/source_bundle.sha256"

port_key=$(( SLURM_JOB_ID % 18000 ))
NAVDP_PORT=$((20000 + port_key))
ss -ltn | awk '{print $4}' | grep -Eq "(^|:)${NAVDP_PORT}$" && \
  fail "NavDP port ${NAVDP_PORT} is in use"
readonly NAVDP_URL="http://127.0.0.1:${NAVDP_PORT}"
runtime_root="${SLURM_TMPDIR:-/tmp}/navdp_arrival_${SLURM_JOB_ID}"
mkdir -p "${runtime_root}/navdp"

NAVDP_PID=
MONITOR_PID=
cleanup() {
  for pid in "${MONITOR_PID}" "${NAVDP_PID}"; do
    if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
      kill "${pid}" 2>/dev/null || true
      wait "${pid}" 2>/dev/null || true
    fi
  done
}
trap cleanup EXIT INT TERM

(
  cd "${runtime_root}/navdp"
  exec env NAVDP_DISABLE_VIDEO=1 PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPYCACHEPREFIX="${PYTHONPYCACHEPREFIX}" \
    PYTHONPATH="${BASE_SOURCE_ROOT}:${BASE_SOURCE_ROOT}/NavDP/baselines/navdp:${BASE_SOURCE_ROOT}/InternNav/src/diffusion-policy" \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "${MEMNAV_PY}" -u "${NAVDP_SERVER}" \
      --port "${NAVDP_PORT}" --checkpoint "${NAVDP_CKPT}"
) > "${RUN_ROOT}/logs/navdp_server.log" 2>&1 &
NAVDP_PID=$!

ready=0
for _ in $(seq 1 240); do
  kill -0 "${NAVDP_PID}" 2>/dev/null || {
    tail -n 160 "${RUN_ROOT}/logs/navdp_server.log" >&2
    fail "NavDP server exited before binding"
  }
  if ss -ltn | awk '{print $4}' | grep -Eq "(^|:)${NAVDP_PORT}$"; then
    ready=1
    break
  fi
  sleep 2
done
[[ "${ready}" -eq 1 ]] || fail "NavDP server startup timed out"

(
  echo "timestamp_utc,gpu_index,gpu_name,memory_used_mib,utilization_gpu_percent"
  while true; do
    timestamp=$(date -u +%Y-%m-%dT%H:%M:%SZ)
    nvidia-smi --id="${CUDA_VISIBLE_DEVICES%%,*}" \
      --query-gpu=index,name,memory.used,utilization.gpu \
      --format=csv,noheader,nounits | sed "s/^/${timestamp},/"
    sleep 5
  done
) > "${RUN_ROOT}/gpu_samples.csv" 2> "${RUN_ROOT}/logs/gpu_monitor.err" &
MONITOR_PID=$!

{
  echo "host=$(hostname)"
  echo "job_id=${SLURM_JOB_ID}"
  echo "mode=${MODE}"
  echo "max_episodes=${MAX_EPISODES}"
  echo "navdp_port=${NAVDP_PORT}"
  echo "source_receipt_sha256=${EXPECTED_SOURCE_RECEIPT_SHA}"
  echo "base_source_receipt_sha256=${EXPECTED_BASE_SOURCE_RECEIPT_SHA}"
  echo "selection_sha256=${EXPECTED_SELECTION_SHA}"
  echo "split_sha256=${EXPECTED_SPLIT_SHA}"
  echo "navdp_checkpoint_sha256=${EXPECTED_NAVDP_CKPT_SHA}"
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
} > "${RUN_ROOT}/runtime_environment.txt"

CONTAINER=(singularity exec --nv --overlay "${OVERLAY}:ro"
  -B /scratch/lg154 -B /scratch/yz11502
  -B "${RAW_GAP_SCENE}:${EPISODE_ROOT}/YmJkqBEsHnH:ro"
  "${BASE_SIF}")
ARGS=(
  --selection-csv "${SELECTION}"
  --split-manifest "${SPLIT}"
  --allowed-role train
  --navdp-url "${NAVDP_URL}"
  --out "${RUN_ROOT}/collection"
  --samples-per-state 4
  --context-frames 7
  --base-seed 2026081501
  --stop-threshold -0.5
  --request-timeout-s 240
  --episode-start-index "${EPISODE_START_INDEX}"
)
if (( MAX_EPISODES > 0 )); then
  ARGS+=(--max-episodes "${MAX_EPISODES}")
fi

started=$(date +%s)
"${CONTAINER[@]}" env \
  PYTHONNOUSERSITE=1 \
  PYTHONDONTWRITEBYTECODE=1 \
  PYTHONPYCACHEPREFIX="${PYTHONPYCACHEPREFIX}" \
  PYTHONPATH="${SOURCE_ROOT}" \
  "${MEMNAV_PY}" -u "${ENTRYPOINT}" "${ARGS[@]}" \
  > "${RUN_ROOT}/logs/collector.log" 2>&1
finished=$(date +%s)

test -s "${RUN_ROOT}/collection/report.json"
"${MEMNAV_PY}" - "${RUN_ROOT}/collection/report.json" "${MODE}" <<'PY'
import json
import sys
report = json.load(open(sys.argv[1]))
assert report["complete"] is True
assert report["scope"] == "train-only offline mechanism audit; not a deployment rule"
assert report["method_or_threshold_authorized"] is False
assert report["goat_validation_read"] is False
assert report["sample_count"] == 4 * report["state_count"]
if sys.argv[2] == "formal":
    assert report["scene_count"] == 40
    assert report["episode_count"] == 80
    assert report["source_episode_count"] == 80
    assert report["episode_start_index"] == 0
elif sys.argv[2] == "repair":
    assert report["scene_count"] == 3
    assert report["episode_count"] == 6
    assert report["source_episode_count"] == 80
    assert report["episode_start_index"] == 74
PY
if [[ "${MODE}" == "repair" ]]; then
  "${CONTAINER[@]}" env \
    PYTHONNOUSERSITE=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPYCACHEPREFIX="${PYTHONPYCACHEPREFIX}" \
    PYTHONPATH="${SOURCE_ROOT}" \
    "${MEMNAV_PY}" -u "${MERGER}" \
      --prefix-samples "${PREFIX_SAMPLES}" \
      --expected-prefix-sha256 "${EXPECTED_PREFIX_SAMPLES_SHA}" \
      --prefix-episode-count 74 \
      --repair-root "${RUN_ROOT}" \
      --selection-csv "${SELECTION}" \
      --split-manifest "${SPLIT}" \
      --allowed-role train \
      --out "${RUN_ROOT}/merged_collection" \
      > "${RUN_ROOT}/logs/merger.log" 2>&1
  "${MEMNAV_PY}" - "${RUN_ROOT}/merged_collection/report.json" <<'PY'
import json
import sys
report = json.load(open(sys.argv[1]))
assert report["complete"] is True
assert report["scene_count"] == 40
assert report["episode_count"] == 80
assert report["source_episode_count"] == 80
assert report["episode_start_index"] == 0
assert report["method_or_threshold_authorized"] is False
assert report["goat_validation_read"] is False
PY
  sha256sum "${RUN_ROOT}/merged_collection/report.json" \
    > "${RUN_ROOT}/merged_collection/report.json.sha256"
fi
printf '{"job_id":%s,"mode":"%s","max_episodes":%s,"elapsed_s":%s}\n' \
  "${SLURM_JOB_ID}" "${MODE}" "${MAX_EPISODES}" "$((finished - started))" \
  > "${RUN_ROOT}/runtime_receipt.json"
sha256sum "${RUN_ROOT}/collection/report.json" \
  > "${RUN_ROOT}/collection/report.json.sha256"
: > "${RUN_ROOT}/SEALED"
echo "NAVDP_ARRIVAL_RUN_ROOT=${RUN_ROOT}"
