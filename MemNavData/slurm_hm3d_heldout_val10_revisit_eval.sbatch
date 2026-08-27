#!/usr/bin/env bash
# One GPU owns one HM3D held-out val10 scene, one server pair, and all paired arms.
#SBATCH --job-name=hm3dh10_eval
#SBATCH --output=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/%x_%A_%a.out
#SBATCH --error=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/%x_%A_%a.err
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --partition=h100_tandon,a100_tandon,h200_public,l40s_public
#SBATCH --gres=gpu:1
#SBATCH --account=torch_pr_769_tandon_advanced

set -euo pipefail
umask 0022

: "${SLURM_ARRAY_TASK_ID:?submit as array 0-9}"
: "${TASK_ROOT:?set immutable task bundle root}"
: "${BASE_SOURCE_ROOT:?set immutable base source root}"
: "${RUN_ROOT:?set run root}"
: "${EXPECTED_TASK_RECEIPT_SHA:?set task receipt SHA}"
: "${EXPECTED_BASE_SOURCE_RECEIPT_SHA:?set base receipt SHA}"

readonly TASK_RECEIPT=${TASK_ROOT}/SOURCE_BUNDLE.sha256
readonly BASE_RECEIPT=${BASE_SOURCE_ROOT}/SOURCE_BUNDLE.sha256
readonly MANIFEST=${RUN_ROOT}/data_manifest.json
readonly MANIFEST_RECEIPT=${MANIFEST}.sha256
readonly RUNNER=${TASK_ROOT}/MemNavData/run_hm3d_heldout_val10_revisit_scene.sh
readonly MEMNAV_SERVER=${BASE_SOURCE_ROOT}/NavDP/baselines/memnav/memnav_server.py
readonly NAVDP_SERVER=${BASE_SOURCE_ROOT}/NavDP/baselines/navdp/navdp_server.py
readonly INTERNNAV_ROOT=${BASE_SOURCE_ROOT}/InternNav
readonly LIGHTGLUE_REPO=${BASE_SOURCE_ROOT}/third_party/LightGlue
readonly DEPENDENCY_ROOT=${BASE_SOURCE_ROOT}/third_party/python
readonly TORCH_HOME=${BASE_SOURCE_ROOT}/torch_home
readonly MEMNAV_PY=/scratch/lg154/conda-envs/memnav/bin/python
readonly HAB_PY=/scratch/lg154/conda-envs/habitat/bin/python
readonly LINGBOT_REPO=/scratch/lg154/Research/Nav/NavDP/baselines/memnav/lingbot-map
readonly BASE_SIF=/share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif

fail() { echo "ABORT: $*" >&2; exit 2; }
[[ "${SLURM_ARRAY_TASK_ID}" =~ ^[0-9]$ ]] || fail "array index must be 0..9"
[[ "$(sha256sum "${TASK_RECEIPT}" | awk '{print $1}')" == \
    "${EXPECTED_TASK_RECEIPT_SHA}" ]] || fail "task receipt changed"
(cd "${TASK_ROOT}" && sha256sum -c SOURCE_BUNDLE.sha256 >/dev/null) || \
  fail "task bundle changed"
[[ "$(sha256sum "${BASE_RECEIPT}" | awk '{print $1}')" == \
    "${EXPECTED_BASE_SOURCE_RECEIPT_SHA}" ]] || fail "base receipt changed"
for path in "${MANIFEST}" "${MANIFEST_RECEIPT}" "${RUNNER}" \
  "${MEMNAV_SERVER}" "${NAVDP_SERVER}" "${MEMNAV_PY}" "${HAB_PY}" \
  "${LINGBOT_REPO}" "${LIGHTGLUE_REPO}" "${DEPENDENCY_ROOT}" \
  "${BASE_SIF}"; do
  test -r "${path}" || fail "missing runtime input ${path}"
done
sha256sum -c "${MANIFEST_RECEIPT}" >/dev/null || fail "manifest changed"
EXPECTED_MANIFEST_SHA=$(awk '{print $1}' "${MANIFEST_RECEIPT}")

readarray -t dependencies < <("${MEMNAV_PY}" - "${MANIFEST}" <<'PY'
import hashlib,json,os,sys
m=json.load(open(sys.argv[1]))
for name in ("gatecurr600","navdp_checkpoint","lingbot_map_long"):
    row=m["dependencies"][name]
    if os.stat(row["path"]).st_size != row["bytes"]:
        raise SystemExit(f"dependency size changed: {name}")
    digest=hashlib.sha256()
    with open(row["path"],"rb") as handle:
        for block in iter(lambda:handle.read(8<<20),b""):
            digest.update(block)
    if digest.hexdigest() != row["sha256"]:
        raise SystemExit(f"dependency hash changed: {name}")
    print(row["path"])
PY
)
[[ "${#dependencies[@]}" -eq 3 ]] || fail "dependency manifest invalid"
MEMNAV_CKPT=${dependencies[0]}; NAVDP_CKPT=${dependencies[1]}
LINGBOT_WEIGHTS=${dependencies[2]}

mkdir -p "${RUN_ROOT}/logs" "${RUN_ROOT}/buffer"
export PYTHONPYCACHEPREFIX=${SLURM_TMPDIR:-/tmp}/hm3dh10_eval_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}
export PYTHONPATH=${BASE_SOURCE_ROOT}:${TASK_ROOT}:${DEPENDENCY_ROOT}:${LIGHTGLUE_REPO}:${INTERNNAV_ROOT}/src/diffusion-policy${PYTHONPATH:+:${PYTHONPATH}}
export TORCH_HOME
mkdir -p "${PYTHONPYCACHEPREFIX}"

port_key=$(( (SLURM_JOB_ID + SLURM_ARRAY_TASK_ID * 37) % 15000 ))
MEMNAV_PORT=$((20000 + port_key * 2)); NAVDP_PORT=$((MEMNAV_PORT + 1))
for port in "${MEMNAV_PORT}" "${NAVDP_PORT}"; do
  ! ss -ltn | awk '{print $4}' | grep -Eq "(^|:)${port}$" || \
    fail "port ${port} already in use"
done
runtime_root=${SLURM_TMPDIR:-/tmp}/hm3dh10_eval_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}
mkdir -p "${runtime_root}/memnav" "${runtime_root}/navdp"
MEMNAV_PID=; NAVDP_PID=
cleanup() {
  for process_id in "${NAVDP_PID}" "${MEMNAV_PID}"; do
    if [[ -n "${process_id}" ]] && kill -0 "${process_id}" 2>/dev/null; then
      kill "${process_id}" 2>/dev/null || true
      wait "${process_id}" 2>/dev/null || true
    fi
  done
}
trap cleanup EXIT INT TERM

scene_label=$(printf '%02d' "${SLURM_ARRAY_TASK_ID}")
(
  cd "${runtime_root}/memnav"
  exec env PYTHONUNBUFFERED=1 PYTHONPYCACHEPREFIX="${PYTHONPYCACHEPREFIX}" \
    PYTHONPATH="${PYTHONPATH}" TORCH_HOME="${TORCH_HOME}" \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    LINGBOT_REPO="${LINGBOT_REPO}" LINGBOT_WEIGHTS="${LINGBOT_WEIGHTS}" \
    MEMNAV_WINDOW=32 MEMNAV_NUM_SCALE=8 MEMNAV_MAX_FRAME_NUM=2048 \
    MEMNAV_GROUND_SCALE_MAX=6.0 MEMNAV_GATE_FUSION=complementary \
    MEMNAV_AUX_POSE_CALIBRATION=empirical MEMNAV_COLLISION_SELECT=1 \
    MEMNAV_REPORT_TO=none "${MEMNAV_PY}" -u "${MEMNAV_SERVER}" \
      --port "${MEMNAV_PORT}" --checkpoint "${MEMNAV_CKPT}" \
      --internnav_root "${INTERNNAV_ROOT}" --num_samples 16 \
      --exclude_recent 32 --retrieval raw --retrieval_candidate_top_k 32 \
      --retrieval_candidate_min_gap 16 --graph_subgoal_spacing_m 0.0 \
      --graph_subgoal_arrival_m 0.60 --flow_gate auto \
      --buffer_root "${RUN_ROOT}/buffer/${scene_label}" \
      --certified_relocalization --lightglue_repo "${LIGHTGLUE_REPO}" \
      --lightglue_dependency_root "${DEPENDENCY_ROOT}" \
      --lightglue_max_keypoints 2048
) >"${RUN_ROOT}/logs/server_memnav_${scene_label}.log" 2>&1 &
MEMNAV_PID=$!
(
  cd "${runtime_root}/navdp"
  exec env NAVDP_DISABLE_VIDEO=1 PYTHONUNBUFFERED=1 \
    PYTHONPYCACHEPREFIX="${PYTHONPYCACHEPREFIX}" PYTHONPATH="${PYTHONPATH}" \
    "${MEMNAV_PY}" -u "${NAVDP_SERVER}" \
      --port "${NAVDP_PORT}" --checkpoint "${NAVDP_CKPT}"
) >"${RUN_ROOT}/logs/server_navdp_${scene_label}.log" 2>&1 &
NAVDP_PID=$!

for spec in \
  "memnav:${MEMNAV_PID}:${MEMNAV_PORT}:${RUN_ROOT}/logs/server_memnav_${scene_label}.log" \
  "navdp:${NAVDP_PID}:${NAVDP_PORT}:${RUN_ROOT}/logs/server_navdp_${scene_label}.log"; do
  IFS=: read -r label process_id port log_path <<<"${spec}"
  ready=0
  for _ in $(seq 1 240); do
    kill -0 "${process_id}" 2>/dev/null || {
      tail -n 160 "${log_path}" >&2; fail "${label} server exited"; }
    if ss -ltn | awk '{print $4}' | grep -Eq "(^|:)${port}$"; then
      ready=1; break
    fi
    sleep 2
  done
  [[ "${ready}" -eq 1 ]] || fail "${label} server startup timeout"
done

echo "host=$(hostname) job=${SLURM_JOB_ID} task=${SLURM_ARRAY_TASK_ID} manifest=${EXPECTED_MANIFEST_SHA}"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
singularity exec --nv -B /scratch/lg154 -B /scratch/yz11502 \
  "${BASE_SIF}" env ROOT="${BASE_SOURCE_ROOT}" TASK_ROOT="${TASK_ROOT}" \
  RUN_ROOT="${RUN_ROOT}" MANIFEST="${MANIFEST}" \
  EXPECTED_MANIFEST_SHA="${EXPECTED_MANIFEST_SHA}" \
  SCENE_INDEX="${SLURM_ARRAY_TASK_ID}" HAB_PY="${HAB_PY}" \
  MEMNAV_PORT="${MEMNAV_PORT}" NAVDP_PORT="${NAVDP_PORT}" \
  PYTHONPYCACHEPREFIX="${PYTHONPYCACHEPREFIX}" PYTHONPATH="${PYTHONPATH}" \
  bash "${RUNNER}"
