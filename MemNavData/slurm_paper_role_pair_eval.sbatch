#!/usr/bin/env bash
# One frozen history/protocol per task; paired parent or learned-amendment arms.
#SBATCH --job-name=paper_pair
#SBATCH --output=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/%x_%A_%a.out
#SBATCH --error=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/%x_%A_%a.err
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --partition=h100_tandon,h200_public,a100_tandon
#SBATCH --qos=gpu48
#SBATCH --gres=gpu:1
#SBATCH --account=torch_pr_769_tandon_advanced
#SBATCH --export=ALL

set -euo pipefail
umask 0022
: "${SLURM_ARRAY_TASK_ID:?submit a two-protocol array}"
: "${SOURCE_ROOT:?set immutable source root}"
: "${SOURCE_RECEIPT:?set source receipt}"
: "${EXPECTED_SOURCE_RECEIPT_SHA:?set receipt SHA}"
: "${BASE_SOURCE_ROOT:?set base dependency source root}"
: "${BASE_SOURCE_RECEIPT_SHA:?set base receipt SHA}"
: "${DEPENDENCY_RECEIPT:?set model dependency receipt}"
: "${EXPECTED_DEPENDENCY_RECEIPT_SHA:?set dependency receipt SHA}"
: "${RUN_ROOT:?set paper run root}"
: "${SOURCE_OVERLAY:?set immutable episode overlay}"
: "${EXPECTED_SOURCE_OVERLAY_BYTES:?set episode overlay size}"
: "${OVERLAY_EPISODE_ROOT:?set episode root inside overlay}"

INCLUDE_LEARNED_PI3X=${INCLUDE_LEARNED_PI3X:-0}
FINAL14_POPULATION_MODE=${FINAL14_POPULATION_MODE:-0}
EXPECTED_PI3X_MODEL_SHA=${EXPECTED_PI3X_MODEL_SHA:-}
EXPECTED_PI3X_PROOF_SHA=${EXPECTED_PI3X_PROOF_SHA:-}
PI3X_ROOT=${PI3X_ROOT:-${SOURCE_ROOT}/third_party/Pi3}
PI3X_SNAPSHOT=${PI3X_SNAPSHOT:-}
PI3X_PROOF_MANIFEST=${PI3X_PROOF_MANIFEST:-${SOURCE_ROOT}/pi3x_deployment/deployment_manifest.json}
PORTABLE_CHECKSUM_VERIFIER=${SOURCE_ROOT}/MemNavData/verify_portable_checksum_manifest.py

HAB_PY=/scratch/lg154/conda-envs/habitat/bin/python
MEMNAV_PY=/scratch/lg154/conda-envs/memnav/bin/python
BASE_SIF=/share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif
MEMNAV_SERVER=${SOURCE_ROOT}/NavDP/baselines/memnav/memnav_server.py
NAVDP_SERVER=${SOURCE_ROOT}/NavDP/baselines/navdp/navdp_server.py
SERVER_LAUNCHER=${SOURCE_ROOT}/MemNavData/retrying_server_launcher.py
RUNNER=${SOURCE_ROOT}/MemNavData/run_paper_role_pair_episode.sh
INTERNNAV_ROOT=${BASE_SOURCE_ROOT}/InternNav
LINGBOT_REPO=/scratch/lg154/Research/Nav/NavDP/baselines/memnav/lingbot-map
LIGHTGLUE_REPO=${BASE_SOURCE_ROOT}/third_party/LightGlue
DEPENDENCY_ROOT=${BASE_SOURCE_ROOT}/third_party/python
TORCH_HOME=${BASE_SOURCE_ROOT}/torch_home
BASE_RECEIPT=${BASE_SOURCE_ROOT}/SOURCE_BUNDLE.sha256
fail() { echo "ABORT: $*" >&2; exit 2; }

[[ "${INCLUDE_LEARNED_PI3X}" == 0 || "${INCLUDE_LEARNED_PI3X}" == 1 ]] || \
  fail "INCLUDE_LEARNED_PI3X must be 0 or 1"
[[ "${FINAL14_POPULATION_MODE}" == 0 || "${FINAL14_POPULATION_MODE}" == 1 ]] || \
  fail "FINAL14_POPULATION_MODE must be 0 or 1"
pi3x_server_args=()
if [[ "${INCLUDE_LEARNED_PI3X}" == 1 ]]; then
  [[ "${EXPECTED_PI3X_MODEL_SHA}" =~ ^[0-9a-f]{64}$ ]] || \
    fail "invalid Pi3X model SHA"
  [[ "${EXPECTED_PI3X_PROOF_SHA}" =~ ^[0-9a-f]{64}$ ]] || \
    fail "invalid Pi3X proof SHA"
  for required in \
    "${PI3X_ROOT}/pi3/models/pi3x.py" \
    "${PI3X_SNAPSHOT}/model.safetensors" \
    "${PI3X_PROOF_MANIFEST}" \
    "${PORTABLE_CHECKSUM_VERIFIER}"; do
    [[ -r "${required}" ]] || fail "missing learned dependency ${required}"
  done
  [[ "$(sha256sum "${PI3X_SNAPSHOT}/model.safetensors" | awk '{print $1}')" == \
    "${EXPECTED_PI3X_MODEL_SHA}" ]] || fail "Pi3X model changed"
  [[ "$(sha256sum "${PI3X_PROOF_MANIFEST}" | awk '{print $1}')" == \
    "${EXPECTED_PI3X_PROOF_SHA}" ]] || fail "Pi3X proof manifest changed"
  "${MEMNAV_PY}" "${PORTABLE_CHECKSUM_VERIFIER}" \
    "$(dirname "${PI3X_PROOF_MANIFEST}")/OUTPUTS.sha256" --quiet || \
    fail "Pi3X proof deployment is invalid or nonportable"
  pi3x_server_args=(
    --pi3x_learned_relocalizer
    --pi3x_root "${PI3X_ROOT}"
    --pi3x_snapshot "${PI3X_SNAPSHOT}"
    --pi3x_model_sha256 "${EXPECTED_PI3X_MODEL_SHA}"
    --pi3x_spatial_proof_manifest "${PI3X_PROOF_MANIFEST}"
    --pi3x_inference_dtype auto
    --synchronize_cuda_http_handoff
  )
fi

if [[ "${FINAL14_POPULATION_MODE}" == 1 ]]; then
  default_population_capacity=42
else
  default_population_capacity=32
fi
MAX_POPULATION_PER_PROTOCOL=${MAX_POPULATION_PER_PROTOCOL:-${default_population_capacity}}
[[ "${MAX_POPULATION_PER_PROTOCOL}" =~ ^[1-9][0-9]*$ ]] || \
  fail "bad per-protocol population capacity"
(( MAX_POPULATION_PER_PROTOCOL <= 128 )) || \
  fail "per-protocol population capacity exceeds safety ceiling"
[[ "${SLURM_ARRAY_TASK_ID}" =~ ^[0-9]+$ ]] || fail "bad array index"
TOTAL_ARRAY_TASKS=$((2 * MAX_POPULATION_PER_PROTOCOL))
(( SLURM_ARRAY_TASK_ID < TOTAL_ARRAY_TASKS )) || \
  fail "array index outside frozen two-protocol capacity"
[[ "$(sha256sum "${SOURCE_RECEIPT}" | awk '{print $1}')" == \
   "${EXPECTED_SOURCE_RECEIPT_SHA}" ]] || fail "source receipt changed"
(cd "${SOURCE_ROOT}" && sha256sum -c "${SOURCE_RECEIPT}") >/dev/null || \
  fail "source bundle changed"
[[ "$(sha256sum "${BASE_RECEIPT}" | awk '{print $1}')" == \
   "${BASE_SOURCE_RECEIPT_SHA}" ]] || fail "base source receipt changed"
[[ "$(sha256sum "${DEPENDENCY_RECEIPT}" | awk '{print $1}')" == \
   "${EXPECTED_DEPENDENCY_RECEIPT_SHA}" ]] || fail "dependency receipt changed"
[[ -f "${BASE_SIF}" && -x "${HAB_PY}" && -x "${MEMNAV_PY}" ]] || \
  fail "runtime dependency missing"
[[ "${EXPECTED_SOURCE_OVERLAY_BYTES}" =~ ^[1-9][0-9]*$ ]] || \
  fail "invalid overlay size contract"
[[ -r "${SOURCE_OVERLAY}" ]] || fail "episode overlay missing"
[[ "$(stat -c %s "${SOURCE_OVERLAY}")" -eq \
   "${EXPECTED_SOURCE_OVERLAY_BYTES}" ]] || fail "episode overlay size changed"
[[ "${OVERLAY_EPISODE_ROOT}" == \
   /mp3d_revisit_v0/vln_n1/traj_data/mp3d_2leg ]] || \
  fail "episode root contract changed"
[[ -f "${RUN_ROOT}/benchmarks/SEALED" ]] || fail "benchmark is not sealed"
[[ "$(sha256sum "${RUN_ROOT}/benchmarks/BENCHMARK_FILES.sha256" | awk '{print $1}')" == \
   "$(awk '{print $1}' "${RUN_ROOT}/benchmarks/BENCHMARK_FILES.sha256.sha256")" ]] || \
  fail "benchmark aggregate receipt changed"
(cd "${RUN_ROOT}/benchmarks" && sha256sum -c BENCHMARK_FILES.sha256) >/dev/null || \
  fail "sealed benchmark files changed"

if [[ "${FINAL14_POPULATION_MODE}" == 1 ]]; then
  FIRST_PROTOCOL=natural_direction
  SECOND_PROTOCOL=hard_support
else
  FIRST_PROTOCOL=support_controlled
  SECOND_PROTOCOL=natural_direction
fi
if (( SLURM_ARRAY_TASK_ID < MAX_POPULATION_PER_PROTOCOL )); then
  PROTOCOL=${FIRST_PROTOCOL}
  EPISODE_INDEX=${SLURM_ARRAY_TASK_ID}
else
  PROTOCOL=${SECOND_PROTOCOL}
  EPISODE_INDEX=$((SLURM_ARRAY_TASK_ID - MAX_POPULATION_PER_PROTOCOL))
fi
BENCH_ROOT=${RUN_ROOT}/benchmarks/${PROTOCOL}
EXPECTED_MANIFEST_SHA=$(awk '{print $1}' "${BENCH_ROOT}/manifest.json.sha256")
accepted_count=$("${MEMNAV_PY}" - "${BENCH_ROOT}/manifest.json" <<'PY'
import json,sys
print(len(json.load(open(sys.argv[1]))["episodes"]))
PY
)
if (( EPISODE_INDEX >= accepted_count )); then
  mkdir -p "${RUN_ROOT}/evaluation/${PROTOCOL}/skipped"
  printf 'protocol=%s\npopulation=%s\narray_index=%s\n' \
    "${PROTOCOL}" "${accepted_count}" "${EPISODE_INDEX}" \
    > "${RUN_ROOT}/evaluation/${PROTOCOL}/skipped/$(printf '%03d' "${EPISODE_INDEX}").txt"
  echo "SKIP protocol=${PROTOCOL} index=${EPISODE_INDEX} population=${accepted_count}"
  exit 0
fi

readarray -t dependencies < <("${MEMNAV_PY}" - "${DEPENDENCY_RECEIPT}" <<'PY'
import json,os,sys
receipt=json.load(open(sys.argv[1]))
for name in ("gatecurr600","navdp_checkpoint","lingbot_map_long"):
    row=receipt["dependencies"][name]
    if os.stat(row["path"]).st_size != int(row["bytes"]):
        raise SystemExit(f"dependency size changed: {name}")
    print(row["path"])
PY
)
[[ "${#dependencies[@]}" -eq 3 ]] || fail "dependency receipt invalid"
MEMNAV_CKPT=${dependencies[0]}; NAVDP_CKPT=${dependencies[1]}
LINGBOT_WEIGHTS=${dependencies[2]}
actual_lingbot_commit=$(git -c safe.directory="${LINGBOT_REPO}" \
  -C "${LINGBOT_REPO}" rev-parse HEAD)
[[ "${actual_lingbot_commit}" == 7ff6f3ed0913d4d326f8f13bbb429c4ffc0195c2 ]] || \
  fail "LingBot source commit changed"

echo "host=$(hostname) job=${SLURM_JOB_ID} task=${SLURM_ARRAY_TASK_ID} protocol=${PROTOCOL} episode_index=${EPISODE_INDEX}"
nvidia-smi --query-gpu=name,uuid,driver_version,memory.total \
  --format=csv,noheader

export PYTHONPYCACHEPREFIX=${SLURM_TMPDIR:-/tmp}/paper_pair_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}
export PYTHONPATH=${SOURCE_ROOT}:${SOURCE_ROOT}/MemNavData:${DEPENDENCY_ROOT}:${LIGHTGLUE_REPO}:${BASE_SOURCE_ROOT}/NavDP/baselines/navdp:${BASE_SOURCE_ROOT}/NavDP/baselines/memnav:${INTERNNAV_ROOT}/src/diffusion-policy${PYTHONPATH:+:${PYTHONPATH}}
export TORCH_HOME
mkdir -p "${PYTHONPYCACHEPREFIX}" "${RUN_ROOT}/logs" "${RUN_ROOT}/buffer"
"${MEMNAV_PY}" -m py_compile "${MEMNAV_SERVER}" "${NAVDP_SERVER}" \
  "${SOURCE_ROOT}/MemNavData/eval_shared_online_role_pairs.py" \
  "${SOURCE_ROOT}/MemNavData/bearing_diagnostics.py" \
  "${SERVER_LAUNCHER}"

port_key=$(( (SLURM_JOB_ID + SLURM_ARRAY_TASK_ID * 59) % 15000 ))
MEMNAV_BASE_PORT=$((20000 + port_key * 2))
NAVDP_BASE_PORT=$((MEMNAV_BASE_PORT + 1))
runtime_root=${SLURM_TMPDIR:-/tmp}/paper_pair_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}
mkdir -p "${runtime_root}/memnav" "${runtime_root}/navdp"
label=$(printf '%02d' "${SLURM_ARRAY_TASK_ID}")
MEMNAV_PORT_FILE=${runtime_root}/memnav.port
NAVDP_PORT_FILE=${runtime_root}/navdp.port
MEMNAV_LAUNCHER_LOG=${RUN_ROOT}/logs/server_memnav_${PROTOCOL}_${label}.log
NAVDP_LAUNCHER_LOG=${RUN_ROOT}/logs/server_navdp_${PROTOCOL}_${label}.log
MEMNAV_PID=; NAVDP_PID=
cleanup() {
  for pid in "${NAVDP_PID}" "${MEMNAV_PID}"; do
    if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
      kill "${pid}" 2>/dev/null || true; wait "${pid}" 2>/dev/null || true
    fi
  done
}
trap cleanup EXIT INT TERM

(
  exec env PYTHONUNBUFFERED=1 PYTHONPYCACHEPREFIX="${PYTHONPYCACHEPREFIX}" \
    PYTHONPATH="${PYTHONPATH}" TORCH_HOME="${TORCH_HOME}" \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    LINGBOT_REPO="${LINGBOT_REPO}" LINGBOT_WEIGHTS="${LINGBOT_WEIGHTS}" \
    MEMNAV_WINDOW=32 MEMNAV_NUM_SCALE=8 MEMNAV_MAX_FRAME_NUM=2048 \
    MEMNAV_GROUND_SCALE_MAX=6.0 MEMNAV_GATE_FUSION=complementary \
    MEMNAV_AUX_POSE_CALIBRATION=empirical MEMNAV_COLLISION_SELECT=1 \
    MEMNAV_REPORT_TO=none \
    "${MEMNAV_PY}" -u "${SERVER_LAUNCHER}" \
      --base-port "${MEMNAV_BASE_PORT}" --range-start 20000 --range-end 49999 \
      --stride 1994 --max-attempts 12 --ready-timeout 720 \
      --port-file "${MEMNAV_PORT_FILE}" \
      --receipt-file "${RUN_ROOT}/logs/server_memnav_${PROTOCOL}_${label}.launcher.json" \
      --log-prefix "${MEMNAV_LAUNCHER_LOG}.child" \
      --cwd "${runtime_root}/memnav" -- \
      "${MEMNAV_PY}" -u "${MEMNAV_SERVER}" --port '{port}' \
        --checkpoint "${MEMNAV_CKPT}" --internnav_root "${INTERNNAV_ROOT}" \
        --num_samples 16 --exclude_recent 32 --retrieval raw \
        --retrieval_candidate_top_k 32 --retrieval_candidate_min_gap 16 \
        --graph_subgoal_spacing_m 0.0 --graph_subgoal_arrival_m 0.60 \
        --flow_gate auto --buffer_root "${RUN_ROOT}/buffer/${PROTOCOL}_${label}" \
        --certified_relocalization --lightglue_repo "${LIGHTGLUE_REPO}" \
        --lightglue_dependency_root "${DEPENDENCY_ROOT}" \
        --lightglue_max_keypoints 2048 \
        "${pi3x_server_args[@]}"
) > "${MEMNAV_LAUNCHER_LOG}" 2>&1 &
MEMNAV_PID=$!
(
  exec env NAVDP_DISABLE_VIDEO=1 PYTHONUNBUFFERED=1 \
    PYTHONPYCACHEPREFIX="${PYTHONPYCACHEPREFIX}" PYTHONPATH="${PYTHONPATH}" \
    "${MEMNAV_PY}" -u "${SERVER_LAUNCHER}" \
      --base-port "${NAVDP_BASE_PORT}" --range-start 20000 --range-end 49999 \
      --stride 1994 --max-attempts 12 --ready-timeout 720 \
      --port-file "${NAVDP_PORT_FILE}" \
      --receipt-file "${RUN_ROOT}/logs/server_navdp_${PROTOCOL}_${label}.launcher.json" \
      --log-prefix "${NAVDP_LAUNCHER_LOG}.child" \
      --cwd "${runtime_root}/navdp" -- \
      "${MEMNAV_PY}" -u "${NAVDP_SERVER}" --port '{port}' \
        --checkpoint "${NAVDP_CKPT}"
) > "${NAVDP_LAUNCHER_LOG}" 2>&1 &
NAVDP_PID=$!

ready=0
for _ in $(seq 1 420); do
  kill -0 "${MEMNAV_PID}" 2>/dev/null || {
    tail -n 200 "${MEMNAV_LAUNCHER_LOG}" >&2
    fail "MemNav launcher died"
  }
  kill -0 "${NAVDP_PID}" 2>/dev/null || {
    tail -n 200 "${NAVDP_LAUNCHER_LOG}" >&2
    fail "NavDP launcher died"
  }
  if [[ -s "${MEMNAV_PORT_FILE}" && -s "${NAVDP_PORT_FILE}" ]]; then
    ready=1; break
  fi
  sleep 2
done
[[ "${ready}" -eq 1 ]] || {
  tail -n 200 "${MEMNAV_LAUNCHER_LOG}" "${NAVDP_LAUNCHER_LOG}" >&2
  fail "server launchers timed out"
}
MEMNAV_PORT=$(<"${MEMNAV_PORT_FILE}")
NAVDP_PORT=$(<"${NAVDP_PORT_FILE}")
[[ "${MEMNAV_PORT}" =~ ^[0-9]+$ && "${NAVDP_PORT}" =~ ^[0-9]+$ ]] || \
  fail "invalid launcher port"
(( MEMNAV_PORT >= 20000 && MEMNAV_PORT <= 49999 )) || fail "bad MemNav port"
(( NAVDP_PORT >= 20000 && NAVDP_PORT <= 49999 )) || fail "bad NavDP port"
[[ "${MEMNAV_PORT}" != "${NAVDP_PORT}" ]] || fail "launcher ports collide"

singularity exec --nv --overlay "${SOURCE_OVERLAY}:ro" \
  -B /scratch/lg154 -B /scratch/yz11502 \
  "${BASE_SIF}" env \
  SOURCE_ROOT="${SOURCE_ROOT}" RUN_ROOT="${RUN_ROOT}" \
  BENCH_ROOT="${BENCH_ROOT}" EXPECTED_MANIFEST_SHA="${EXPECTED_MANIFEST_SHA}" \
  PROTOCOL="${PROTOCOL}" EPISODE_INDEX="${EPISODE_INDEX}" \
  HAB_PY="${HAB_PY}" MEMNAV_PORT="${MEMNAV_PORT}" NAVDP_PORT="${NAVDP_PORT}" \
  INCLUDE_LEARNED_PI3X="${INCLUDE_LEARNED_PI3X}" \
  EXPECTED_PI3X_MODEL_SHA="${EXPECTED_PI3X_MODEL_SHA}" \
  EXPECTED_PI3X_PROOF_SHA="${EXPECTED_PI3X_PROOF_SHA}" \
  MAX_STEPS=600 PYTHONPYCACHEPREFIX="${PYTHONPYCACHEPREFIX}" \
  PYTHONPATH="${PYTHONPATH}" bash "${RUNNER}"
