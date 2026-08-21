#!/usr/bin/env bash
# One frozen GOAT first-ImageGoal episode per GPU task.
# This confirms semantic arrival only; it is not a full GOAT score.
#SBATCH --job-name=goat_arrive
#SBATCH --output=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/%x_%A_%a.out
#SBATCH --error=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/%x_%A_%a.err
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --partition=h100_tandon,a100_tandon,h200_public
#SBATCH --qos=gpu48
#SBATCH --gres=gpu:1
#SBATCH --account=torch_pr_769_tandon_advanced

set -euo pipefail
umask 0022

: "${SOURCE_ROOT:?set immutable confirmation source root}"
: "${SOURCE_RECEIPT:?set confirmation source receipt}"
: "${EXPECTED_SOURCE_RECEIPT_SHA:?set confirmation receipt SHA-256}"
: "${BASE_SOURCE_ROOT:?set pinned base source root}"
: "${EXPECTED_BASE_SOURCE_RECEIPT_SHA:?set base receipt SHA-256}"
: "${MANIFEST:?set frozen episode manifest}"
: "${EXPECTED_MANIFEST_SHA:?set episode manifest SHA-256}"
: "${RUN_ROOT:?set immutable result root}"
: "${RUN_MODE:?set smoke or formal}"
: "${SLURM_ARRAY_TASK_ID:?submit as an array}"

readonly DATA_ROOT="${GOAT_HM3D_ROOT:-/scratch/yz11502/Research/datasets/goat_bench_20260814}"
readonly ENV_PREFIX="${GOAT_ENV_PREFIX:-/scratch/yz11502/conda_envs/goat-bench-habitat023-20260814}"
readonly GOAT_CODE="${GOAT_CODE_ROOT:-${DATA_ROOT}/code/goat-bench}"
readonly BASE_SIF="${GOAT_BASE_SIF:-/share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif}"
readonly ENV_RECEIPT="${DATA_ROOT}/receipts/goat-environment-v023.json"
readonly EXTRACT_RECEIPT="${DATA_ROOT}/receipts/goat-val-unseen-extraction.json"
readonly MEMNAV_PY="${MEMNAV_PY:-/scratch/lg154/conda-envs/memnav/bin/python}"
readonly INTERNNAV_ROOT="${BASE_SOURCE_ROOT}/InternNav"
readonly NAVDP_SERVER="${BASE_SOURCE_ROOT}/NavDP/baselines/navdp/navdp_server.py"
readonly MEMNAV_SERVER="${SOURCE_ROOT}/NavDP/baselines/memnav/memnav_server.py"
readonly RUNNER="${SOURCE_ROOT}/MemNavData/goat_certified_arrival_confirmation.py"
readonly LIGHTGLUE_REPO="${BASE_SOURCE_ROOT}/third_party/LightGlue"
readonly DEPENDENCY_ROOT="${BASE_SOURCE_ROOT}/third_party/python"
readonly TORCH_HOME="${BASE_SOURCE_ROOT}/torch_home"
readonly LINGBOT_REPO="/scratch/lg154/Research/Nav/NavDP/baselines/memnav/lingbot-map"
readonly LINGBOT_WEIGHTS="${LINGBOT_REPO}/weights/lingbot-map-long.pt"
readonly MEMNAV_CKPT="/scratch/yz11502/Research/Nav-axis-uturn/.diagnostics/unseen_scene_eval_20260803/checkpoints/gatecurr600.memnav.ckpt"
readonly NAVDP_CKPT="/scratch/yz11502/Research/Nav-axis-uturn/.diagnostics/unseen_scene_eval_20260803/checkpoints/navdp_checkpoint.ckpt"
readonly EXPECTED_MEMNAV_CKPT_SHA="9b7a5811ff0aea212503f58b45258ba4f66b06420f87c350946aead39db6fdb7"
readonly EXPECTED_NAVDP_CKPT_SHA="3bb3ad4ab241e857bb57a4021cc6aab76d5263e81fbf80298d579053ef011947"
readonly EXPECTED_LINGBOT_WEIGHTS_SHA="832bc82cbae0bc9bbe946ef5ee1f7226abd8c0e183ccf8beddbb3d133576f409"
readonly EXPECTED_LINGBOT_COMMIT="7ff6f3ed0913d4d326f8f13bbb429c4ffc0195c2"
readonly EXPECTED_GOAT_COMMIT="74c41d19d4a4c3608d1575b512087b5a529aee0e"
readonly DEPENDENCY_RECEIPT="${SOURCE_ROOT}/dependency_receipt.json"

fail() { echo "ABORT: $*" >&2; exit 2; }
[[ "${RUN_MODE}" == smoke || "${RUN_MODE}" == formal ]] || \
  fail "RUN_MODE must be smoke or formal"
for command_name in singularity sha256sum nvidia-smi ss; do
  command -v "${command_name}" >/dev/null || fail "${command_name} unavailable"
done
[[ "$(sha256sum "${SOURCE_RECEIPT}" | awk '{print $1}')" == \
   "${EXPECTED_SOURCE_RECEIPT_SHA}" ]] || fail "source receipt changed"
(cd "${SOURCE_ROOT}" && sha256sum -c "${SOURCE_RECEIPT}") >/dev/null || \
  fail "confirmation source bundle changed"
[[ "$(sha256sum "${BASE_SOURCE_ROOT}/SOURCE_BUNDLE.sha256" | awk '{print $1}')" == \
   "${EXPECTED_BASE_SOURCE_RECEIPT_SHA}" ]] || fail "base receipt changed"
(cd "${BASE_SOURCE_ROOT}" && sha256sum -c SOURCE_BUNDLE.sha256) >/dev/null || \
  fail "base source bundle changed"
[[ "$(sha256sum "${MANIFEST}" | awk '{print $1}')" == \
   "${EXPECTED_MANIFEST_SHA}" ]] || fail "frozen manifest changed"
[[ -f "${DEPENDENCY_RECEIPT}" ]] || fail "dependency receipt is missing"
"${MEMNAV_PY}" - "${DEPENDENCY_RECEIPT}" \
  "${EXPECTED_MEMNAV_CKPT_SHA}" "${EXPECTED_NAVDP_CKPT_SHA}" \
  "${EXPECTED_LINGBOT_WEIGHTS_SHA}" "${MEMNAV_CKPT}" "${NAVDP_CKPT}" \
  "${LINGBOT_WEIGHTS}" <<'PY'
import json
import os
import sys
receipt = json.load(open(sys.argv[1]))
expected_hashes = dict(zip(
    ("gatecurr600", "navdp_checkpoint", "lingbot_map_long"), sys.argv[2:]))
expected_paths = dict(zip(
    ("gatecurr600", "navdp_checkpoint", "lingbot_map_long"), sys.argv[5:]))
for name, expected_hash in expected_hashes.items():
    item = receipt["dependencies"][name]
    if item["sha256"] != expected_hash:
        raise SystemExit("dependency receipt hash changed: {}".format(name))
    if item["path"] != expected_paths[name]:
        raise SystemExit("dependency receipt path changed: {}".format(name))
    if os.stat(item["path"]).st_size != int(item["bytes"]):
        raise SystemExit("dependency size changed: {}".format(name))
PY
[[ "$(git -c safe.directory="${LINGBOT_REPO}" -C "${LINGBOT_REPO}" rev-parse HEAD)" == \
   "${EXPECTED_LINGBOT_COMMIT}" ]] || fail "LingBot source commit changed"
[[ "$(git -C "${GOAT_CODE}" rev-parse HEAD)" == "${EXPECTED_GOAT_COMMIT}" ]] || \
  fail "GOAT source commit changed"
[[ -x "${ENV_PREFIX}/bin/python" && -x "${MEMNAV_PY}" ]] || \
  fail "required Python environment is missing"
[[ -f "${ENV_RECEIPT}" && -f "${EXTRACT_RECEIPT}" && \
   -f "${DATA_ROOT}/data/scene_datasets/hm3d/val/SEALED" ]] || \
  fail "GOAT environment or scene extraction is incomplete"
[[ -f "${RUN_ROOT}/submission.json" ]] || fail "submission receipt is missing"

"${ENV_PREFIX}/bin/python" - "${ENV_RECEIPT}" "${EXTRACT_RECEIPT}" <<'PY'
import json
import sys
for path in sys.argv[1:]:
    if not json.load(open(path)).get("complete"):
        raise SystemExit("incomplete prerequisite receipt: {}".format(path))
PY

export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
export PYTHONPYCACHEPREFIX="${SLURM_TMPDIR:-/tmp}/goat_arrive_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-12}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-12}"
export MAGNUM_LOG=quiet
export HABITAT_SIM_LOG=quiet
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_HOME
export PYTHONPATH="${SOURCE_ROOT}:${BASE_SOURCE_ROOT}:${DEPENDENCY_ROOT}:${LIGHTGLUE_REPO}:${INTERNNAV_ROOT}/src/diffusion-policy${PYTHONPATH:+:${PYTHONPATH}}"
mkdir -p "${PYTHONPYCACHEPREFIX}" "${RUN_ROOT}/episodes" "${RUN_ROOT}/logs"

# GOAT is pinned to Python 3.7. Parse every module it imports before model
# startup so syntax drift fails as an engineering smoke, not midway in eval.
"${ENV_PREFIX}/bin/python" - \
  "${RUNNER}" \
  "${SOURCE_ROOT}/MemNavData/goat_certified_arrival_contract.py" \
  "${SOURCE_ROOT}/MemNavData/goat_navdp_discrete_adapter.py" \
  "${SOURCE_ROOT}/MemNavData/goat_contract_smoke.py" \
  "${SOURCE_ROOT}/MemNavData/goat_navdp_runtime_pilot.py" <<'PY'
import ast
import pathlib
import sys
for raw in sys.argv[1:]:
    path = pathlib.Path(raw)
    ast.parse(path.read_text(), filename=str(path))
PY

port_key=$(( (SLURM_JOB_ID + SLURM_ARRAY_TASK_ID * 37) % 10000 ))
MEMNAV_PORT=$((18000 + port_key * 2))
NAVDP_PORT=$((MEMNAV_PORT + 1))
for port in "${MEMNAV_PORT}" "${NAVDP_PORT}"; do
  ss -ltn | awk '{print $4}' | grep -Eq "(^|:)${port}$" && \
    fail "port ${port} is already in use"
done
readonly MEMNAV_URL="http://127.0.0.1:${MEMNAV_PORT}"
readonly NAVDP_URL="http://127.0.0.1:${NAVDP_PORT}"
runtime_root="${SLURM_TMPDIR:-/tmp}/goat_arrive_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "${runtime_root}/memnav" "${runtime_root}/navdp"
printf -v task_label '%02d' "${SLURM_ARRAY_TASK_ID}"

MEMNAV_PID=
NAVDP_PID=
MONITOR_PID=
cleanup() {
  for pid in "${MONITOR_PID}" "${NAVDP_PID}" "${MEMNAV_PID}"; do
    if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
      kill "${pid}" 2>/dev/null || true
      wait "${pid}" 2>/dev/null || true
    fi
  done
}
trap cleanup EXIT INT TERM

(
  cd "${runtime_root}/memnav"
  exec env PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPYCACHEPREFIX="${PYTHONPYCACHEPREFIX}" PYTHONPATH="${PYTHONPATH}" \
    TORCH_HOME="${TORCH_HOME}" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    LINGBOT_REPO="${LINGBOT_REPO}" LINGBOT_WEIGHTS="${LINGBOT_WEIGHTS}" \
    MEMNAV_WINDOW=32 MEMNAV_NUM_SCALE=8 MEMNAV_MAX_FRAME_NUM=2048 \
    MEMNAV_GROUND_SCALE_MAX=6.0 MEMNAV_GATE_FUSION=complementary \
    MEMNAV_AUX_POSE_CALIBRATION=empirical MEMNAV_COLLISION_SELECT=1 \
    MEMNAV_REPORT_TO=none \
    "${MEMNAV_PY}" -u "${MEMNAV_SERVER}" \
      --port "${MEMNAV_PORT}" --checkpoint "${MEMNAV_CKPT}" \
      --internnav_root "${INTERNNAV_ROOT}" --num_samples 16 \
      --exclude_recent 32 --retrieval raw \
      --retrieval_candidate_top_k 32 --retrieval_candidate_min_gap 16 \
      --graph_subgoal_spacing_m 0.0 --graph_subgoal_arrival_m 0.60 \
      --flow_gate auto --buffer_root "${RUN_ROOT}/buffer/${task_label}" \
      --certified_relocalization --lightglue_repo "${LIGHTGLUE_REPO}" \
      --lightglue_dependency_root "${DEPENDENCY_ROOT}" \
      --lightglue_max_keypoints 2048
) > "${RUN_ROOT}/logs/memnav_${task_label}.log" 2>&1 &
MEMNAV_PID=$!

ready=0
for _ in $(seq 1 300); do
  kill -0 "${MEMNAV_PID}" 2>/dev/null || {
    tail -n 200 "${RUN_ROOT}/logs/memnav_${task_label}.log" >&2
    fail "arrival sidecar exited before binding"
  }
  if ss -ltn | awk '{print $4}' | grep -Eq "(^|:)${MEMNAV_PORT}$"; then
    ready=1; break
  fi
  sleep 2
done
[[ "${ready}" -eq 1 ]] || fail "arrival sidecar startup timed out"

(
  cd "${runtime_root}/navdp"
  exec env NAVDP_DISABLE_VIDEO=1 PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 PYTHONPYCACHEPREFIX="${PYTHONPYCACHEPREFIX}" \
    PYTHONPATH="${BASE_SOURCE_ROOT}:${BASE_SOURCE_ROOT}/NavDP/baselines/navdp:${INTERNNAV_ROOT}/src/diffusion-policy" \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "${MEMNAV_PY}" -u "${NAVDP_SERVER}" \
      --port "${NAVDP_PORT}" --checkpoint "${NAVDP_CKPT}"
) > "${RUN_ROOT}/logs/navdp_${task_label}.log" 2>&1 &
NAVDP_PID=$!

ready=0
for _ in $(seq 1 300); do
  kill -0 "${NAVDP_PID}" 2>/dev/null || {
    tail -n 200 "${RUN_ROOT}/logs/navdp_${task_label}.log" >&2
    fail "NavDP server exited before binding"
  }
  if ss -ltn | awk '{print $4}' | grep -Eq "(^|:)${NAVDP_PORT}$"; then
    ready=1; break
  fi
  sleep 2
done
[[ "${ready}" -eq 1 ]] || fail "NavDP server startup timed out"

(
  echo "timestamp_utc,gpu_name,memory_used_mib,utilization_gpu_percent"
  while true; do
    timestamp=$(date -u +%Y-%m-%dT%H:%M:%SZ)
    nvidia-smi --query-gpu=name,memory.used,utilization.gpu \
      --format=csv,noheader,nounits | head -1 | sed "s/^/${timestamp},/"
    sleep 10
  done
) > "${RUN_ROOT}/logs/gpu_${task_label}.csv" 2>/dev/null &
MONITOR_PID=$!

{
  echo "mode=${RUN_MODE}"
  echo "host=$(hostname)"
  echo "job_id=${SLURM_JOB_ID}"
  echo "array_task_id=${SLURM_ARRAY_TASK_ID}"
  echo "manifest_sha256=${EXPECTED_MANIFEST_SHA}"
  echo "source_receipt_sha256=${EXPECTED_SOURCE_RECEIPT_SHA}"
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
} > "${RUN_ROOT}/logs/environment_${task_label}.txt"

cd "${GOAT_CODE}"
singularity exec --nv -B /scratch/lg154 -B /scratch/yz11502 \
  "${BASE_SIF}" env \
  PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 \
  PYTHONPYCACHEPREFIX="${PYTHONPYCACHEPREFIX}" \
  PYTHONPATH="${SOURCE_ROOT}" MAGNUM_LOG="${MAGNUM_LOG}" \
  HABITAT_SIM_LOG="${HABITAT_SIM_LOG}" \
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
  "${ENV_PREFIX}/bin/python" -u "${RUNNER}" \
    --goat-code "${GOAT_CODE}" --data-root "${DATA_ROOT}" \
    --manifest "${MANIFEST}" --output-dir "${RUN_ROOT}/episodes" \
    --index "${SLURM_ARRAY_TASK_ID}" --navdp-url "${NAVDP_URL}" \
    --arrival-url "${MEMNAV_URL}" \
  > "${RUN_ROOT}/logs/episode_${task_label}.log" 2>&1

record="${RUN_ROOT}/episodes/episode_${task_label}.json"
test -s "${record}" || fail "episode output is absent"
"${ENV_PREFIX}/bin/python" - "${record}" "${SLURM_ARRAY_TASK_ID}" <<'PY'
import json
import sys
payload = json.load(open(sys.argv[1]))
assert payload["complete"] is True
assert payload["ground_truth_used_by_decision"] is False
assert payload["is_full_goat_benchmark_score"] is False
assert int(payload["episode_index"]) == int(sys.argv[2])
assert payload["record"]["status"] == "complete"
PY
sha256sum "${record}" > "${record}.sha256"
