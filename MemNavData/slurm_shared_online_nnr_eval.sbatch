#!/usr/bin/env bash
# Evaluate one constructible episode, all five arms, on one persistent GPU.
#SBATCH --job-name=nnr_eval
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
: "${SLURM_ARRAY_TASK_ID:?submit 0-21}"
: "${SOURCE_ROOT:?set immutable source root}"
: "${SOURCE_RECEIPT:?set source receipt}"
: "${EXPECTED_SOURCE_RECEIPT_SHA:?set receipt SHA}"
: "${BASE_SOURCE_ROOT:?set base dependency source root}"
: "${BASE_SOURCE_RECEIPT_SHA:?set base receipt SHA}"
: "${DEPENDENCY_RECEIPT:?set model dependency receipt}"
: "${EXPECTED_DEPENDENCY_RECEIPT_SHA:?set dependency receipt SHA}"
: "${RUN_ROOT:?set run root}"

HAB_PY=/scratch/lg154/conda-envs/habitat/bin/python
MEMNAV_PY=/scratch/lg154/conda-envs/memnav/bin/python
BASE_SIF=/share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif
BENCH_ROOT=${RUN_ROOT}/prepared/benchmark
TRACE_ROOT=${RUN_ROOT}/prepared/traces
MEMNAV_SERVER=${SOURCE_ROOT}/NavDP/baselines/memnav/memnav_server.py
NAVDP_SERVER=${SOURCE_ROOT}/NavDP/baselines/navdp/navdp_server.py
RUNNER=${SOURCE_ROOT}/MemNavData/run_shared_online_novel_revisit_formal_episode.sh
INTERNNAV_ROOT=${BASE_SOURCE_ROOT}/InternNav
LINGBOT_REPO=/scratch/lg154/Research/Nav/NavDP/baselines/memnav/lingbot-map
LIGHTGLUE_REPO=${BASE_SOURCE_ROOT}/third_party/LightGlue
DEPENDENCY_ROOT=${BASE_SOURCE_ROOT}/third_party/python
TORCH_HOME=${BASE_SOURCE_ROOT}/torch_home
BASE_RECEIPT=${BASE_SOURCE_ROOT}/SOURCE_BUNDLE.sha256
fail() { echo "ABORT: $*" >&2; exit 2; }

[[ "${SLURM_ARRAY_TASK_ID}" =~ ^([0-9]|1[0-9]|2[01])$ ]] || \
  fail "array index outside 0-21"
[[ "$(sha256sum "${SOURCE_RECEIPT}" | awk '{print $1}')" == \
   "${EXPECTED_SOURCE_RECEIPT_SHA}" ]] || fail "source receipt changed"
(cd "${SOURCE_ROOT}" && sha256sum -c "${SOURCE_RECEIPT}") >/dev/null || \
  fail "source bundle changed"
[[ "$(sha256sum "${BASE_RECEIPT}" | awk '{print $1}')" == \
   "${BASE_SOURCE_RECEIPT_SHA}" ]] || fail "base source receipt changed"
[[ "$(sha256sum "${DEPENDENCY_RECEIPT}" | awk '{print $1}')" == \
   "${EXPECTED_DEPENDENCY_RECEIPT_SHA}" ]] || fail "dependency receipt changed"
(cd "${BENCH_ROOT}" && sha256sum -c manifest.json.sha256) >/dev/null || \
  fail "benchmark manifest changed"
(cd "${TRACE_ROOT}" && sha256sum -c manifest.json.sha256) >/dev/null || \
  fail "trace manifest changed"
[[ -f "${BENCH_ROOT}/SEALED" && -f "${BASE_SIF}" ]] || \
  fail "sealed benchmark/container missing"

accepted_count=$("${MEMNAV_PY}" - "${BENCH_ROOT}/manifest.json" <<'PY'
import json,sys
print(len(json.load(open(sys.argv[1]))["accepted"]))
PY
)
if (( SLURM_ARRAY_TASK_ID >= accepted_count )); then
  mkdir -p "${RUN_ROOT}/skipped"
  printf 'accepted_population=%s\narray_index=%s\n' \
    "${accepted_count}" "${SLURM_ARRAY_TASK_ID}" \
    > "${RUN_ROOT}/skipped/$(printf '%03d' "${SLURM_ARRAY_TASK_ID}").txt"
  echo "SKIP index=${SLURM_ARRAY_TASK_ID} accepted=${accepted_count}"
  exit 0
fi

readarray -t dependencies < <("${MEMNAV_PY}" - "${DEPENDENCY_RECEIPT}" <<'PY'
import json,os,sys
r=json.load(open(sys.argv[1]))
for name in ("gatecurr600","navdp_checkpoint","lingbot_map_long"):
    row=r["dependencies"][name]
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

export PYTHONPYCACHEPREFIX=${SLURM_TMPDIR:-/tmp}/nnr_eval_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}
export PYTHONPATH=${SOURCE_ROOT}:${SOURCE_ROOT}/MemNavData:${DEPENDENCY_ROOT}:${LIGHTGLUE_REPO}:${BASE_SOURCE_ROOT}/NavDP/baselines/navdp:${BASE_SOURCE_ROOT}/NavDP/baselines/memnav:${INTERNNAV_ROOT}/src/diffusion-policy${PYTHONPATH:+:${PYTHONPATH}}
export TORCH_HOME
mkdir -p "${PYTHONPYCACHEPREFIX}" "${RUN_ROOT}/logs" "${RUN_ROOT}/buffer"
"${MEMNAV_PY}" -m py_compile "${MEMNAV_SERVER}" "${NAVDP_SERVER}" \
  "${SOURCE_ROOT}/MemNavData/eval_shared_online_novel_revisit.py"
singularity exec --nv -B /scratch/lg154 -B /scratch/yz11502 \
  -B /scratch/qw2440 "${BASE_SIF}" env \
  PYTHONPYCACHEPREFIX="${PYTHONPYCACHEPREFIX}" PYTHONPATH="${PYTHONPATH}" \
  "${HAB_PY}" -c 'import quaternion,numba,habitat_sim; print("habitat ok")'

port_key=$(( (SLURM_JOB_ID + SLURM_ARRAY_TASK_ID * 59) % 15000 ))
MEMNAV_PORT=$((20000 + port_key * 2)); NAVDP_PORT=$((MEMNAV_PORT + 1))
for port in "${MEMNAV_PORT}" "${NAVDP_PORT}"; do
  ss -ltn | awk '{print $4}' | grep -Eq "(^|:)${port}$" && fail "port in use"
done
runtime_root=${SLURM_TMPDIR:-/tmp}/nnr_eval_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}
mkdir -p "${runtime_root}/memnav" "${runtime_root}/navdp"
label=$(printf '%03d' "${SLURM_ARRAY_TASK_ID}")
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
  cd "${runtime_root}/memnav"
  exec env PYTHONUNBUFFERED=1 PYTHONPYCACHEPREFIX="${PYTHONPYCACHEPREFIX}" \
    PYTHONPATH="${PYTHONPATH}" TORCH_HOME="${TORCH_HOME}" \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    LINGBOT_REPO="${LINGBOT_REPO}" LINGBOT_WEIGHTS="${LINGBOT_WEIGHTS}" \
    MEMNAV_WINDOW=32 MEMNAV_NUM_SCALE=8 MEMNAV_MAX_FRAME_NUM=2048 \
    MEMNAV_GROUND_SCALE_MAX=6.0 MEMNAV_GATE_FUSION=complementary \
    MEMNAV_AUX_POSE_CALIBRATION=empirical MEMNAV_COLLISION_SELECT=1 \
    MEMNAV_REPORT_TO=none \
    "${MEMNAV_PY}" -u "${MEMNAV_SERVER}" --port "${MEMNAV_PORT}" \
      --checkpoint "${MEMNAV_CKPT}" --internnav_root "${INTERNNAV_ROOT}" \
      --num_samples 16 --exclude_recent 32 --retrieval raw \
      --retrieval_candidate_top_k 32 --retrieval_candidate_min_gap 16 \
      --graph_subgoal_spacing_m 1.25 --graph_subgoal_arrival_m 0.60 \
      --flow_gate auto --buffer_root "${RUN_ROOT}/buffer/${label}" \
      --certified_relocalization --lightglue_repo "${LIGHTGLUE_REPO}" \
      --lightglue_dependency_root "${DEPENDENCY_ROOT}" \
      --lightglue_max_keypoints 2048
) > "${RUN_ROOT}/logs/server_memnav_${label}.log" 2>&1 &
MEMNAV_PID=$!
(
  cd "${runtime_root}/navdp"
  exec env NAVDP_DISABLE_VIDEO=1 PYTHONUNBUFFERED=1 \
    PYTHONPYCACHEPREFIX="${PYTHONPYCACHEPREFIX}" PYTHONPATH="${PYTHONPATH}" \
    "${MEMNAV_PY}" -u "${NAVDP_SERVER}" --port "${NAVDP_PORT}" \
      --checkpoint "${NAVDP_CKPT}"
) > "${RUN_ROOT}/logs/server_navdp_${label}.log" 2>&1 &
NAVDP_PID=$!

for spec in \
  "memnav:${MEMNAV_PID}:${MEMNAV_PORT}:${RUN_ROOT}/logs/server_memnav_${label}.log" \
  "navdp:${NAVDP_PID}:${NAVDP_PORT}:${RUN_ROOT}/logs/server_navdp_${label}.log"; do
  IFS=: read -r name pid port log <<<"${spec}"; ready=0
  for _ in $(seq 1 240); do
    kill -0 "${pid}" 2>/dev/null || { tail -n 160 "${log}" >&2; fail "${name} died"; }
    if ss -ltn | awk '{print $4}' | grep -Eq "(^|:)${port}$"; then ready=1; break; fi
    sleep 2
  done
  [[ "${ready}" -eq 1 ]] || { tail -n 160 "${log}" >&2; fail "${name} timeout"; }
done

EXPECTED_MANIFEST_SHA=$(awk '{print $1}' "${BENCH_ROOT}/manifest.json.sha256")
EXPECTED_TRACE_MANIFEST_SHA=$(awk '{print $1}' "${TRACE_ROOT}/manifest.json.sha256")
singularity exec --nv -B /scratch/lg154 -B /scratch/yz11502 \
  -B /scratch/qw2440 "${BASE_SIF}" env \
  SOURCE_ROOT="${SOURCE_ROOT}" RUN_ROOT="${RUN_ROOT}" \
  BENCH_ROOT="${BENCH_ROOT}" TRACE_ROOT="${TRACE_ROOT}" \
  EXPECTED_MANIFEST_SHA="${EXPECTED_MANIFEST_SHA}" \
  EXPECTED_TRACE_MANIFEST_SHA="${EXPECTED_TRACE_MANIFEST_SHA}" \
  EPISODE_INDEX="${SLURM_ARRAY_TASK_ID}" HAB_PY="${HAB_PY}" \
  MEMNAV_PORT="${MEMNAV_PORT}" NAVDP_PORT="${NAVDP_PORT}" MAX_STEPS=600 \
  PYTHONPYCACHEPREFIX="${PYTHONPYCACHEPREFIX}" PYTHONPATH="${PYTHONPATH}" \
  bash "${RUNNER}"
