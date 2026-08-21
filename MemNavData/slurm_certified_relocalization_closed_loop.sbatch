#!/usr/bin/env bash
# Four-arm, 20-scene certified relocalization closed-loop evaluation.
# One task owns one GPU, one scene, one MemNav process and one NavDP process.
#SBATCH --job-name=certrel_cl
#SBATCH --output=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/%x_%A_%a.out
#SBATCH --error=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/%x_%A_%a.err
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --partition=a100_tandon,h100_tandon,h200_public,l40s_public
#SBATCH --qos=gpu48
#SBATCH --gres=gpu:1
#SBATCH --account=torch_pr_769_tandon_advanced
#SBATCH --export=ALL

set -euo pipefail
umask 0022
: "${SLURM_ARRAY_TASK_ID:?submit as array 0-19}"
: "${SOURCE_ROOT:?set immutable SOURCE_ROOT}"
: "${RUN_ROOT:?set RUN_ROOT}"
: "${SOURCE_RECEIPT:?set SOURCE_RECEIPT}"
: "${EXPECTED_SOURCE_RECEIPT_SHA:?set EXPECTED_SOURCE_RECEIPT_SHA}"

HAB_PY=/scratch/lg154/conda-envs/habitat/bin/python
MEMNAV_PY=/scratch/lg154/conda-envs/memnav/bin/python
MANIFEST=${RUN_ROOT}/data_manifest.json
MANIFEST_RECEIPT=${MANIFEST}.sha256
DEPENDENCY_RECEIPT=${RUN_ROOT}/dependency_receipt.json
DEPENDENCY_RECEIPT_SHA=${DEPENDENCY_RECEIPT}.sha256
RUNNER=${SOURCE_ROOT}/MemNavData/run_certified_relocalization_closed_loop_scene.sh
MEMNAV_SERVER=${SOURCE_ROOT}/NavDP/baselines/memnav/memnav_server.py
NAVDP_SERVER=${SOURCE_ROOT}/NavDP/baselines/navdp/navdp_server.py
INTERNNAV_ROOT=${SOURCE_ROOT}/InternNav
LINGBOT_REPO=/scratch/lg154/Research/Nav/NavDP/baselines/memnav/lingbot-map
LIGHTGLUE_REPO=${SOURCE_ROOT}/third_party/LightGlue
DEPENDENCY_ROOT=${SOURCE_ROOT}/third_party/python
TORCH_HOME=${SOURCE_ROOT}/torch_home
export PYTHONPYCACHEPREFIX=${SLURM_TMPDIR:-/tmp}/certrel_pycache_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}
mkdir -p "${PYTHONPYCACHEPREFIX}"

fail() { echo "ABORT: $*" >&2; exit 2; }
[[ "$(sha256sum "${SOURCE_RECEIPT}" | awk '{print $1}')" == \
    "${EXPECTED_SOURCE_RECEIPT_SHA}" ]] || fail "source receipt SHA mismatch"
(cd "${SOURCE_ROOT}" && sha256sum -c "${SOURCE_RECEIPT}") >/dev/null || \
  fail "source bundle changed"
for path in "${MANIFEST}" "${MANIFEST_RECEIPT}" \
            "${DEPENDENCY_RECEIPT}" "${DEPENDENCY_RECEIPT_SHA}"; do
  test -r "${path}" || fail "missing run receipt ${path}"
done
(cd "${RUN_ROOT}" && sha256sum -c "$(basename "${MANIFEST_RECEIPT}")") \
  >/dev/null || fail "manifest changed"
(cd "${RUN_ROOT}" && sha256sum -c "$(basename "${DEPENDENCY_RECEIPT_SHA}")") \
  >/dev/null || fail "dependency receipt changed"
EXPECTED_MANIFEST_SHA=$(awk '{print $1}' "${MANIFEST_RECEIPT}")

readarray -t dependencies < <("${MEMNAV_PY}" - \
  "${MANIFEST}" "${DEPENDENCY_RECEIPT}" <<'PY'
import json, os, sys
manifest=json.load(open(sys.argv[1]))
receipt=json.load(open(sys.argv[2]))
for name in ("gatecurr600", "navdp_checkpoint", "lingbot_map_long"):
    expected=manifest["dependencies"][name]
    audited=receipt["dependencies"][name]
    if audited.get("path") != expected.get("path"):
        raise SystemExit(f"dependency path receipt changed: {name}")
    if audited.get("bytes") != expected.get("bytes"):
        raise SystemExit(f"dependency size receipt changed: {name}")
    if audited.get("sha256") != expected.get("sha256"):
        raise SystemExit(f"dependency SHA receipt changed: {name}")
    if os.stat(expected["path"]).st_size != expected["bytes"]:
        raise SystemExit(f"dependency size changed on node: {name}")
    print(expected["path"])
PY
)
[[ "${#dependencies[@]}" -eq 3 ]] || fail "dependency manifest invalid"
MEMNAV_CKPT=${dependencies[0]}
NAVDP_CKPT=${dependencies[1]}
LINGBOT_WEIGHTS=${dependencies[2]}
actual_lingbot_commit=$(git -c safe.directory="${LINGBOT_REPO}" \
  -C "${LINGBOT_REPO}" rev-parse HEAD)
[[ "${actual_lingbot_commit}" == 7ff6f3ed0913d4d326f8f13bbb429c4ffc0195c2 ]] || \
  fail "LingBot source commit changed"

export PYTHONPATH="${SOURCE_ROOT}:${DEPENDENCY_ROOT}:${LIGHTGLUE_REPO}:${INTERNNAV_ROOT}/src/diffusion-policy${PYTHONPATH:+:${PYTHONPATH}}"
export TORCH_HOME
"${MEMNAV_PY}" -m py_compile \
  "${SOURCE_ROOT}/MemNavData/certified_relocalization_runtime.py" \
  "${SOURCE_ROOT}/MemNavData/revisit_bearing_adapter.py" \
  "${SOURCE_ROOT}/MemNavData/eval_2leg_habitat.py" \
  "${MEMNAV_SERVER}" \
  "${SOURCE_ROOT}/NavDP/baselines/memnav/policy_agent.py"
"${MEMNAV_PY}" -m pytest -q \
  "${SOURCE_ROOT}/MemNavData/test_certified_relocalization_runtime.py" \
  "${SOURCE_ROOT}/MemNavData/test_revisit_bearing_adapter.py" \
  "${SOURCE_ROOT}/MemNavData/test_lingbot_pnp_localization.py" \
  "${SOURCE_ROOT}/MemNavData/test_lingbot_goal_loop_closure.py" \
  "${SOURCE_ROOT}/MemNavData/test_policy_agent_graph.py" \
  "${SOURCE_ROOT}/MemNavData/test_prepare_certified_relocalization_closed_loop.py" \
  "${SOURCE_ROOT}/MemNavData/test_summarize_certified_relocalization_closed_loop.py"

mkdir -p "${RUN_ROOT}/logs" "${RUN_ROOT}/buffer"
port_key=$(( (${SLURM_JOB_ID} + SLURM_ARRAY_TASK_ID * 37) % 15000 ))
MEMNAV_PORT=$((20000 + port_key * 2))
NAVDP_PORT=$((MEMNAV_PORT + 1))
for port in "${MEMNAV_PORT}" "${NAVDP_PORT}"; do
  ss -ltn | awk '{print $4}' | grep -Eq "(^|:)${port}$" && \
    fail "port ${port} already in use"
done

runtime_root=${SLURM_TMPDIR:-/tmp}/certrel_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}
mkdir -p "${runtime_root}/memnav" "${runtime_root}/navdp"
MEMNAV_PID=; NAVDP_PID=
cleanup() {
  for pid in "${NAVDP_PID}" "${MEMNAV_PID}"; do
    if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
      kill "${pid}" 2>/dev/null || true
      wait "${pid}" 2>/dev/null || true
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
    MEMNAV_REPORT_TO=none \
    "${MEMNAV_PY}" -u "${MEMNAV_SERVER}" \
      --port "${MEMNAV_PORT}" --checkpoint "${MEMNAV_CKPT}" \
      --internnav_root "${INTERNNAV_ROOT}" --num_samples 16 \
      --exclude_recent 32 --retrieval raw \
      --retrieval_candidate_top_k 32 --retrieval_candidate_min_gap 16 \
      --graph_subgoal_spacing_m 0.0 --graph_subgoal_arrival_m 0.60 \
      --flow_gate auto --buffer_root "${RUN_ROOT}/buffer/${scene_label}" \
      --certified_relocalization \
      --lightglue_repo "${LIGHTGLUE_REPO}" \
      --lightglue_dependency_root "${DEPENDENCY_ROOT}" \
      --lightglue_max_keypoints 2048
) > "${RUN_ROOT}/logs/server_memnav_${scene_label}.log" 2>&1 &
MEMNAV_PID=$!

(
  cd "${runtime_root}/navdp"
  exec env NAVDP_DISABLE_VIDEO=1 PYTHONUNBUFFERED=1 \
    PYTHONPYCACHEPREFIX="${PYTHONPYCACHEPREFIX}" PYTHONPATH="${PYTHONPATH}" \
    "${MEMNAV_PY}" -u "${NAVDP_SERVER}" \
      --port "${NAVDP_PORT}" --checkpoint "${NAVDP_CKPT}"
) > "${RUN_ROOT}/logs/server_navdp_${scene_label}.log" 2>&1 &
NAVDP_PID=$!

for spec in \
  "memnav:${MEMNAV_PID}:${MEMNAV_PORT}:${RUN_ROOT}/logs/server_memnav_${scene_label}.log" \
  "navdp:${NAVDP_PID}:${NAVDP_PORT}:${RUN_ROOT}/logs/server_navdp_${scene_label}.log"; do
  IFS=: read -r label pid port log <<<"${spec}"
  ready=0
  for _ in $(seq 1 240); do
    kill -0 "${pid}" 2>/dev/null || {
      tail -n 160 "${log}" >&2; fail "${label} server exited"; }
    if ss -ltn | awk '{print $4}' | grep -Eq "(^|:)${port}$"; then
      ready=1
      break
    fi
    sleep 2
  done
  [[ "${ready}" -eq 1 ]] || {
    tail -n 160 "${log}" >&2; fail "${label} server did not bind"; }
done

ROOT="${SOURCE_ROOT}" RUN_ROOT="${RUN_ROOT}" MANIFEST="${MANIFEST}" \
EXPECTED_MANIFEST_SHA="${EXPECTED_MANIFEST_SHA}" \
SOURCE_RECEIPT="${SOURCE_RECEIPT}" \
EXPECTED_SOURCE_RECEIPT_SHA="${EXPECTED_SOURCE_RECEIPT_SHA}" \
SCENE_INDEX="${SLURM_ARRAY_TASK_ID}" HAB_PY="${HAB_PY}" \
MEMNAV_PORT="${MEMNAV_PORT}" NAVDP_PORT="${NAVDP_PORT}" \
  bash "${RUNNER}"
