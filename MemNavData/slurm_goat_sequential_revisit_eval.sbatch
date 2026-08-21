#!/usr/bin/env bash
# One frozen paired official-GOAT vs role-free CEC target per GPU task.
# This is a targeted sequential-Revisit evaluation, not a full GOAT score.
#SBATCH --job-name=goat_cec_seq
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

: "${SOURCE_ROOT:?set immutable source root}"
: "${SOURCE_RECEIPT:?set source receipt}"
: "${EXPECTED_SOURCE_RECEIPT_SHA:?set source receipt SHA-256}"
: "${BASE_SOURCE_ROOT:?set pinned base source root}"
: "${EXPECTED_BASE_SOURCE_RECEIPT_SHA:?set base receipt SHA-256}"
: "${MANIFEST:?set frozen episode manifest}"
: "${EXPECTED_MANIFEST_SHA:?set manifest SHA-256}"
: "${RUN_ROOT:?set immutable result root}"
: "${RUN_MODE:?set smoke or formal}"
: "${SLURM_ARRAY_TASK_ID:?submit as an array}"

readonly DATA_ROOT="${GOAT_HM3D_ROOT:-/scratch/yz11502/Research/datasets/goat_bench_20260814}"
readonly ENV_PREFIX="${GOAT_ENV_PREFIX:-/scratch/yz11502/conda_envs/goat-bench-habitat023-20260814}"
readonly GOAT_CODE="${GOAT_CODE_ROOT:-${DATA_ROOT}/code/goat-bench}"
readonly GOAT_POLICY_CKPT="${DATA_ROOT}/data/goat-assets/checkpoints/sense_act_nn_monolithic/ckpt_best.pth"
readonly BASE_SIF="${GOAT_BASE_SIF:-/share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif}"
readonly ENV_RECEIPT="${DATA_ROOT}/receipts/goat-environment-v023.json"
readonly EXTRACT_RECEIPT="${DATA_ROOT}/receipts/goat-val-unseen-extraction.json"
readonly MEMNAV_PY="${MEMNAV_PY:-/scratch/lg154/conda-envs/memnav/bin/python}"
readonly INTERNNAV_ROOT="${BASE_SOURCE_ROOT}/InternNav"
readonly NAVDP_SERVER="${BASE_SOURCE_ROOT}/NavDP/baselines/navdp/navdp_server.py"
readonly MEMNAV_SERVER="${SOURCE_ROOT}/NavDP/baselines/memnav/memnav_server.py"
readonly RUNNER="${SOURCE_ROOT}/MemNavData/goat_sequential_revisit_pilot.py"
readonly LIGHTGLUE_REPO="${BASE_SOURCE_ROOT}/third_party/LightGlue"
readonly DEPENDENCY_ROOT="${BASE_SOURCE_ROOT}/third_party/python"
readonly OPENAI_CLIP_ROOT="${SOURCE_ROOT}/third_party/openai_clip"
readonly GOAT_RUNTIME_DEPS="${SOURCE_ROOT}/third_party/goat_runtime"
readonly EXPECTED_OPENAI_CLIP_COMMIT="d05afc436d78f1c48dc0dbf8e5980a9d471f35f6"
readonly EXPECTED_FTFY_VERSION="6.1.1"
readonly EXPECTED_REGEX_DISTRIBUTION_VERSION="2024.4.16"
readonly EXPECTED_REGEX_MODULE_VERSION="2.5.141"
readonly EXPECTED_WCWIDTH_VERSION="0.2.14"
readonly OPENAI_CLIP_RN50="/scratch/yz11502/Research/datasets/goat_bench_20260814/data/goat-assets/checkpoints/openai_clip/RN50.pt"
readonly OPENAI_CLIP_DEFAULT_CACHE="/home/yz11502/.cache/clip/RN50.pt"
readonly EXPECTED_OPENAI_CLIP_RN50_SHA="afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762"
readonly TORCH_HOME="${BASE_SOURCE_ROOT}/torch_home"
readonly LINGBOT_REPO="/scratch/lg154/Research/Nav/NavDP/baselines/memnav/lingbot-map"
readonly LINGBOT_WEIGHTS="${LINGBOT_REPO}/weights/lingbot-map-long.pt"
readonly MEMNAV_CKPT="/scratch/yz11502/Research/Nav-axis-uturn/.diagnostics/unseen_scene_eval_20260803/checkpoints/gatecurr600.memnav.ckpt"
readonly NAVDP_CKPT="/scratch/yz11502/Research/Nav-axis-uturn/.diagnostics/unseen_scene_eval_20260803/checkpoints/navdp_checkpoint.ckpt"
readonly EXPECTED_GOAT_POLICY_SHA="55e89c3d083198d4add4e9e70164b54ff892900963a2925471362e2d4761b3eb"
readonly EXPECTED_MEMNAV_CKPT_SHA="9b7a5811ff0aea212503f58b45258ba4f66b06420f87c350946aead39db6fdb7"
readonly EXPECTED_NAVDP_CKPT_SHA="3bb3ad4ab241e857bb57a4021cc6aab76d5263e81fbf80298d579053ef011947"
readonly EXPECTED_LINGBOT_WEIGHTS_SHA="832bc82cbae0bc9bbe946ef5ee1f7226abd8c0e183ccf8beddbb3d133576f409"
readonly EXPECTED_LINGBOT_COMMIT="7ff6f3ed0913d4d326f8f13bbb429c4ffc0195c2"
readonly EXPECTED_GOAT_COMMIT="74c41d19d4a4c3608d1575b512087b5a529aee0e"
readonly DEPENDENCY_RECEIPT="${SOURCE_ROOT}/dependency_receipt.json"

fail() { echo "ABORT: $*" >&2; exit 2; }
[[ "${RUN_MODE}" == smoke || "${RUN_MODE}" == formal ]] || \
  fail "RUN_MODE must be smoke or formal"
for command_name in singularity sha256sum nvidia-smi readlink ss; do
  command -v "${command_name}" >/dev/null || fail "${command_name} unavailable"
done
[[ "$(sha256sum "${SOURCE_RECEIPT}" | awk '{print $1}')" == \
   "${EXPECTED_SOURCE_RECEIPT_SHA}" ]] || fail "source receipt changed"
(cd "${SOURCE_ROOT}" && sha256sum -c "${SOURCE_RECEIPT}") >/dev/null || \
  fail "source bundle changed"
[[ "$(sha256sum "${BASE_SOURCE_ROOT}/SOURCE_BUNDLE.sha256" | awk '{print $1}')" == \
   "${EXPECTED_BASE_SOURCE_RECEIPT_SHA}" ]] || fail "base receipt changed"
(cd "${BASE_SOURCE_ROOT}" && sha256sum -c SOURCE_BUNDLE.sha256) >/dev/null || \
  fail "base source bundle changed"
[[ "$(sha256sum "${MANIFEST}" | awk '{print $1}')" == \
   "${EXPECTED_MANIFEST_SHA}" ]] || fail "frozen manifest changed"
[[ "$(sha256sum "${GOAT_POLICY_CKPT}" | awk '{print $1}')" == \
   "${EXPECTED_GOAT_POLICY_SHA}" ]] || fail "GOAT policy checkpoint changed"
[[ "$(git -C "${GOAT_CODE}" rev-parse HEAD)" == "${EXPECTED_GOAT_COMMIT}" ]] || \
  fail "GOAT source commit changed"
[[ "$(git -c safe.directory="${LINGBOT_REPO}" -C "${LINGBOT_REPO}" rev-parse HEAD)" == \
   "${EXPECTED_LINGBOT_COMMIT}" ]] || fail "LingBot source commit changed"
[[ -x "${ENV_PREFIX}/bin/python" && -x "${MEMNAV_PY}" ]] || \
  fail "required Python environment is missing"
[[ -f "${ENV_RECEIPT}" && -f "${EXTRACT_RECEIPT}" && \
   -f "${DATA_ROOT}/data/scene_datasets/hm3d/val/SEALED" ]] || \
  fail "GOAT environment or scene extraction is incomplete"
[[ -f "${RUN_ROOT}/submission.json" ]] || fail "submission receipt is missing"
[[ -f "${OPENAI_CLIP_RN50}" ]] || fail "pinned OpenAI CLIP RN50 is missing"
[[ -L "${OPENAI_CLIP_DEFAULT_CACHE}" ]] || \
  fail "OpenAI CLIP default cache is not a symlink"
[[ "$(readlink -f "${OPENAI_CLIP_DEFAULT_CACHE}")" == \
   "${OPENAI_CLIP_RN50}" ]] || fail "OpenAI CLIP cache target changed"
[[ "$(sha256sum "${OPENAI_CLIP_RN50}" | awk '{print $1}')" == \
   "${EXPECTED_OPENAI_CLIP_RN50_SHA}" ]] || fail "OpenAI CLIP RN50 changed"

"${ENV_PREFIX}/bin/python" - "${OPENAI_CLIP_ROOT}/upstream.json" \
  "${EXPECTED_OPENAI_CLIP_COMMIT}" <<'PY'
import json
import pathlib
import sys
receipt_path = pathlib.Path(sys.argv[1])
expected_commit = sys.argv[2]
receipt = json.loads(receipt_path.read_text())
if receipt != {
    "package": "openai/CLIP",
    "upstream_commit": expected_commit,
}:
    raise SystemExit("vendored OpenAI CLIP receipt changed")
required = (
    "clip/__init__.py",
    "clip/clip.py",
    "clip/model.py",
    "clip/simple_tokenizer.py",
    "clip/bpe_simple_vocab_16e6.txt.gz",
)
for relative in required:
    if not (receipt_path.parent / relative).is_file():
        raise SystemExit("vendored OpenAI CLIP file is absent: {}".format(relative))
PY
"${ENV_PREFIX}/bin/python" - \
  "${GOAT_RUNTIME_DEPS}/runtime_dependencies.json" \
  "${EXPECTED_FTFY_VERSION}" "${EXPECTED_REGEX_DISTRIBUTION_VERSION}" \
  "${EXPECTED_REGEX_MODULE_VERSION}" "${EXPECTED_WCWIDTH_VERSION}" <<'PY'
import json
import pathlib
import sys
receipt = json.loads(pathlib.Path(sys.argv[1]).read_text())
expected = {
    "ftfy": {"distribution_version": sys.argv[2],
             "module_version": sys.argv[2]},
    "regex": {"distribution_version": sys.argv[3],
              "module_version": sys.argv[4]},
    "wcwidth": {"distribution_version": sys.argv[5],
                "module_version": sys.argv[5]},
}
if receipt.get("packages") != expected:
    raise SystemExit("vendored GOAT runtime dependency receipt changed")
if receipt.get("python_abi") != ".cpython-37m-x86_64-linux-gnu.so":
    raise SystemExit("vendored regex Python ABI changed")
for package in expected:
    if not (pathlib.Path(sys.argv[1]).parent / package / "__init__.py").is_file():
        raise SystemExit("vendored runtime package is absent: {}".format(package))
for dist_info in ("ftfy-6.1.1.dist-info", "regex-2024.4.16.dist-info",
                  "wcwidth-0.2.14.dist-info"):
    if not (pathlib.Path(sys.argv[1]).parent / dist_info / "METADATA").is_file():
        raise SystemExit("vendored runtime metadata is absent: {}".format(dist_info))
PY

"${MEMNAV_PY}" - "${DEPENDENCY_RECEIPT}" \
  "${EXPECTED_MEMNAV_CKPT_SHA}" "${EXPECTED_NAVDP_CKPT_SHA}" \
  "${EXPECTED_LINGBOT_WEIGHTS_SHA}" "${EXPECTED_OPENAI_CLIP_RN50_SHA}" \
  "${MEMNAV_CKPT}" "${NAVDP_CKPT}" "${LINGBOT_WEIGHTS}" \
  "${OPENAI_CLIP_RN50}" <<'PY'
import json
import os
import sys
receipt = json.load(open(sys.argv[1]))
names = ("gatecurr600", "navdp_checkpoint", "lingbot_map_long",
         "openai_clip_rn50")
expected_hashes = dict(zip(names, sys.argv[2:6]))
expected_paths = dict(zip(names, sys.argv[6:10]))
for name in names:
    item = receipt["dependencies"][name]
    if item["sha256"] != expected_hashes[name]:
        raise SystemExit("dependency receipt hash changed: {}".format(name))
    if item["path"] != expected_paths[name]:
        raise SystemExit("dependency receipt path changed: {}".format(name))
    if os.stat(item["path"]).st_size != int(item["bytes"]):
        raise SystemExit("dependency size changed: {}".format(name))
PY

"${ENV_PREFIX}/bin/python" - "${ENV_RECEIPT}" "${EXTRACT_RECEIPT}" <<'PY'
import json
import sys
for path in sys.argv[1:]:
    if not json.load(open(path)).get("complete"):
        raise SystemExit("incomplete prerequisite receipt: {}".format(path))
PY

export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
export PYTHONPYCACHEPREFIX="${SLURM_TMPDIR:-/tmp}/goat_cec_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-12}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-12}"
export MAGNUM_LOG=quiet
export HABITAT_SIM_LOG=quiet
export TORCH_HOME
export PYTHONPATH="${SOURCE_ROOT}:${OPENAI_CLIP_ROOT}:${GOAT_RUNTIME_DEPS}:${BASE_SOURCE_ROOT}:${DEPENDENCY_ROOT}:${LIGHTGLUE_REPO}:${INTERNNAV_ROOT}/src/diffusion-policy${PYTHONPATH:+:${PYTHONPATH}}"
mkdir -p "${PYTHONPYCACHEPREFIX}" "${RUN_ROOT}/episodes" "${RUN_ROOT}/logs"

"${ENV_PREFIX}/bin/python" - \
  "${RUNNER}" \
  "${SOURCE_ROOT}/MemNavData/goat_contract_smoke.py" \
  "${SOURCE_ROOT}/MemNavData/goat_navdp_discrete_adapter.py" \
  "${SOURCE_ROOT}/MemNavData/goat_navdp_runtime_pilot.py" \
  "${SOURCE_ROOT}/MemNavData/goat_certified_arrival_contract.py" \
  "${SOURCE_ROOT}/MemNavData/goat_terminal_alignment.py" \
  "${SOURCE_ROOT}/MemNavData/revisit_bearing_adapter.py" \
  "${SOURCE_ROOT}/MemNavData/xnavdp_revisit_contract.py" \
  "${SOURCE_ROOT}/MemNavData/lingbot_colored_registration.py" \
  "${SOURCE_ROOT}/MemNavData/lingbot_pnp_localization.py" <<'PY'
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
runtime_root="${SLURM_TMPDIR:-/tmp}/goat_cec_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "${runtime_root}/memnav" "${runtime_root}/navdp" "${runtime_root}/buffer"
printf -v task_label '%03d' "${SLURM_ARRAY_TASK_ID}"
attempt_label="${task_label}_${SLURM_JOB_ID}"
task_output="${RUN_ROOT}/episodes/${task_label}"
[[ ! -e "${task_output}" ]] || fail "task output already exists"

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
    MEMNAV_WINDOW=32 MEMNAV_NUM_SCALE=8 MEMNAV_MAX_FRAME_NUM=6000 \
    MEMNAV_GROUND_SCALE_MAX=6.0 MEMNAV_GATE_FUSION=complementary \
    MEMNAV_AUX_POSE_CALIBRATION=empirical MEMNAV_COLLISION_SELECT=1 \
    MEMNAV_REPORT_TO=none \
    "${MEMNAV_PY}" -u "${MEMNAV_SERVER}" \
      --port "${MEMNAV_PORT}" --checkpoint "${MEMNAV_CKPT}" \
      --internnav_root "${INTERNNAV_ROOT}" --num_samples 16 \
      --exclude_recent 32 --retrieval raw \
      --retrieval_candidate_top_k 32 --retrieval_candidate_min_gap 16 \
      --graph_subgoal_spacing_m 0.0 --graph_subgoal_arrival_m 0.60 \
      --flow_gate auto --buffer_root "${runtime_root}/buffer" \
      --certified_relocalization --lightglue_repo "${LIGHTGLUE_REPO}" \
      --lightglue_dependency_root "${DEPENDENCY_ROOT}" \
      --lightglue_max_keypoints 2048
) > "${RUN_ROOT}/logs/memnav_${attempt_label}.log" 2>&1 &
MEMNAV_PID=$!

ready=0
for _ in $(seq 1 300); do
  kill -0 "${MEMNAV_PID}" 2>/dev/null || {
    tail -n 200 "${RUN_ROOT}/logs/memnav_${attempt_label}.log" >&2
    fail "CEC sidecar exited before binding"
  }
  if ss -ltn | awk '{print $4}' | grep -Eq "(^|:)${MEMNAV_PORT}$"; then
    ready=1; break
  fi
  sleep 2
done
[[ "${ready}" -eq 1 ]] || fail "CEC sidecar startup timed out"

(
  cd "${runtime_root}/navdp"
  exec env NAVDP_DISABLE_VIDEO=1 PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 PYTHONPYCACHEPREFIX="${PYTHONPYCACHEPREFIX}" \
    PYTHONPATH="${BASE_SOURCE_ROOT}:${BASE_SOURCE_ROOT}/NavDP/baselines/navdp:${INTERNNAV_ROOT}/src/diffusion-policy" \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "${MEMNAV_PY}" -u "${NAVDP_SERVER}" \
      --port "${NAVDP_PORT}" --checkpoint "${NAVDP_CKPT}"
) > "${RUN_ROOT}/logs/navdp_${attempt_label}.log" 2>&1 &
NAVDP_PID=$!

ready=0
for _ in $(seq 1 300); do
  kill -0 "${NAVDP_PID}" 2>/dev/null || {
    tail -n 200 "${RUN_ROOT}/logs/navdp_${attempt_label}.log" >&2
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
) > "${RUN_ROOT}/logs/gpu_${attempt_label}.csv" 2>/dev/null &
MONITOR_PID=$!

{
  echo "mode=${RUN_MODE}"
  echo "host=$(hostname)"
  echo "job_id=${SLURM_JOB_ID}"
  echo "array_task_id=${SLURM_ARRAY_TASK_ID}"
  echo "manifest_sha256=${EXPECTED_MANIFEST_SHA}"
  echo "source_receipt_sha256=${EXPECTED_SOURCE_RECEIPT_SHA}"
  echo "openai_clip_commit=${EXPECTED_OPENAI_CLIP_COMMIT}"
  echo "openai_clip_rn50_sha256=${EXPECTED_OPENAI_CLIP_RN50_SHA}"
  echo "ftfy_version=${EXPECTED_FTFY_VERSION}"
  echo "regex_distribution_version=${EXPECTED_REGEX_DISTRIBUTION_VERSION}"
  echo "wcwidth_version=${EXPECTED_WCWIDTH_VERSION}"
  echo "goat_pytorch_cuda_alloc_conf=unset"
  echo "service_pytorch_cuda_alloc_conf=expandable_segments:True"
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
} > "${RUN_ROOT}/logs/environment_${attempt_label}.txt"

# The frozen GOAT environment uses PyTorch 1.13.1, whose caching allocator
# rejects the newer ``expandable_segments`` option used by the independently
# hosted MemNav/NavDP services.  Those service subprocesses receive the option
# explicitly above; the official GOAT policy must not inherit it.
unset PYTORCH_CUDA_ALLOC_CONF
cd "${GOAT_CODE}"
singularity exec --nv -B /scratch/lg154 -B /scratch/yz11502 \
  "${BASE_SIF}" env -u PYTORCH_CUDA_ALLOC_CONF \
  PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 \
  PYTHONPYCACHEPREFIX="${PYTHONPYCACHEPREFIX}" \
  PYTHONPATH="${SOURCE_ROOT}:${OPENAI_CLIP_ROOT}:${GOAT_RUNTIME_DEPS}" \
  MAGNUM_LOG="${MAGNUM_LOG}" \
  HABITAT_SIM_LOG="${HABITAT_SIM_LOG}" \
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
  "${ENV_PREFIX}/bin/python" -u "${RUNNER}" \
    --goat-code "${GOAT_CODE}" --data-root "${DATA_ROOT}" \
    --checkpoint "${GOAT_POLICY_CKPT}" --manifest "${MANIFEST}" \
    --index "${SLURM_ARRAY_TASK_ID}" --output-dir "${task_output}" \
    --memnav-url "${MEMNAV_URL}" --navdp-url "${NAVDP_URL}" \
    --policy-device cuda --base-seed 100 --max-steps 5000 \
  > "${RUN_ROOT}/logs/episode_${attempt_label}.log" 2>&1

record="${task_output}/goat_sequential_revisit_pilot.json"
[[ -s "${record}" ]] || fail "paired output is absent"
"${ENV_PREFIX}/bin/python" - "${record}" "${SLURM_ARRAY_TASK_ID}" \
  "${EXPECTED_MANIFEST_SHA}" <<'PY'
import json
import sys
payload = json.load(open(sys.argv[1]))
assert payload["complete"] is True
assert payload["role_label_read_by_controller"] is False
assert payload["official_goat_stop_authority_retained"] is True
assert payload["official_goat_subtask_stop_preserved_exactly"] is True
assert payload["official_policy_uses_released_stochastic_eval_semantics"] is True
assert payload["policy_device"].startswith("cuda")
assert int(payload["manifest_entry"]["index"]) == int(sys.argv[2])
assert payload["manifest_sha256"] == sys.argv[3]
assert len(payload["pairs"]) == 1
assert payload["pairs"][0]["prefix_audit"][
    "prefix_paired_before_first_override"] is True
PY
sha256sum "${record}" > "${record}.sha256"
