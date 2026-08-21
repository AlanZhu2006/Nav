#!/usr/bin/env bash
#SBATCH --job-name=cdec_cl
#SBATCH --output=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/%x_%A_%a.out
#SBATCH --error=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/%x_%A_%a.err
#SBATCH --time=06:00:00
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
: "${SOURCE_ROOT:?set immutable overlay source}"
: "${BASE_SOURCE_ROOT:?set immutable certified base source}"
: "${RUN_ROOT:?set result root}"
: "${TRACE_RUN_ROOT:?set frozen trace source run}"
: "${SOURCE_RECEIPT:?set overlay source receipt}"
: "${EXPECTED_SOURCE_RECEIPT_SHA:?set overlay receipt SHA}"
: "${BASE_SOURCE_RECEIPT:?set base source receipt}"
: "${EXPECTED_BASE_SOURCE_RECEIPT_SHA:?set base source receipt SHA}"
: "${GATE_VERIFICATION:?set independent gate report}"
: "${EXPECTED_GATE_VERIFICATION_SHA:?set gate report SHA}"
: "${CDEC_ARTIFACT:?set bundled CDEC artifact}"
: "${EXPECTED_CDEC_ARTIFACT_SHA:?set CDEC artifact SHA}"

HAB_PY=/scratch/lg154/conda-envs/habitat/bin/python
MEMNAV_PY=/scratch/lg154/conda-envs/memnav/bin/python
MANIFEST=${RUN_ROOT}/data_manifest.json
TRACE_RECEIPT=${RUN_ROOT}/trace_receipt.json
DEPENDENCY_RECEIPT=${RUN_ROOT}/dependency_receipt.json
RUNNER=${SOURCE_ROOT}/MemNavData/run_cdec_consumed_closed_loop_scene.sh
MEMNAV_SERVER=${SOURCE_ROOT}/NavDP/baselines/memnav/memnav_server.py
NAVDP_SERVER=${BASE_SOURCE_ROOT}/NavDP/baselines/navdp/navdp_server.py
INTERNNAV_ROOT=${BASE_SOURCE_ROOT}/InternNav
LIGHTGLUE_REPO=${BASE_SOURCE_ROOT}/third_party/LightGlue
DEPENDENCY_ROOT=${BASE_SOURCE_ROOT}/third_party/python
TORCH_HOME=${BASE_SOURCE_ROOT}/torch_home
export PYTHONPYCACHEPREFIX=${SLURM_TMPDIR:-/tmp}/cdec_cl_pycache_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}
mkdir -p "${PYTHONPYCACHEPREFIX}"

fail() { echo "ABORT: $*" >&2; exit 2; }
for pair in \
  "${SOURCE_RECEIPT}:${EXPECTED_SOURCE_RECEIPT_SHA}" \
  "${BASE_SOURCE_RECEIPT}:${EXPECTED_BASE_SOURCE_RECEIPT_SHA}" \
  "${GATE_VERIFICATION}:${EXPECTED_GATE_VERIFICATION_SHA}" \
  "${CDEC_ARTIFACT}:${EXPECTED_CDEC_ARTIFACT_SHA}"; do
  path=${pair%%:*}; expected=${pair#*:}
  [[ "$(sha256sum "${path}" | awk '{print $1}')" == "${expected}" ]] || \
    fail "SHA mismatch: ${path}"
done
(cd "${SOURCE_ROOT}" && sha256sum -c "${SOURCE_RECEIPT}") >/dev/null || \
  fail "overlay source bundle changed"
(cd "${BASE_SOURCE_ROOT}" && sha256sum -c "${BASE_SOURCE_RECEIPT}") >/dev/null || \
  fail "base source bundle changed"
for receipt in data_manifest.json trace_receipt.json dependency_receipt.json \
               preparation.json; do
  test -f "${RUN_ROOT}/${receipt}" -a -f "${RUN_ROOT}/${receipt}.sha256" || \
    fail "missing run receipt ${receipt}"
  (cd "${RUN_ROOT}" && sha256sum -c "${receipt}.sha256") >/dev/null || \
    fail "run receipt changed: ${receipt}"
done
EXPECTED_MANIFEST_SHA=$(awk '{print $1}' "${MANIFEST}.sha256")
EXPECTED_TRACE_RECEIPT_SHA=$(awk '{print $1}' "${TRACE_RECEIPT}.sha256")

"${MEMNAV_PY}" - "${GATE_VERIFICATION}" <<'PY'
import json,sys
report=json.load(open(sys.argv[1]))
if report.get("verified") is not True:
    raise SystemExit("independent CDEC verification did not pass")
scope=report.get("scope") or {}
if (scope.get("train40_only") is not True
        or scope.get("development_or_blind_read") is not False
        or scope.get("closed_loop") is not False):
    raise SystemExit("CDEC gate scope changed")
gate=((report.get("reconstructed") or {}).get("method_gate") or {})
if gate.get("pass") is not True:
    raise SystemExit("CDEC proposal gate did not authorize closed loop")
required=gate.get("requirements") or {}
if not required or not all(value is True for value in required.values()):
    raise SystemExit("CDEC method-gate requirements are incomplete")
PY

readarray -t dependencies < <("${MEMNAV_PY}" - "${DEPENDENCY_RECEIPT}" <<'PY'
import hashlib,json,os,sys
receipt=json.load(open(sys.argv[1]))
for name in ("gatecurr600","navdp_checkpoint","lingbot_map_long"):
    row=receipt["dependencies"][name]
    path=row["path"]
    if os.stat(path).st_size!=row["bytes"]:
        raise SystemExit(f"dependency size changed: {name}")
    digest=hashlib.sha256()
    with open(path,"rb") as handle:
        for block in iter(lambda:handle.read(8<<20),b""):
            digest.update(block)
    if digest.hexdigest()!=row["sha256"]:
        raise SystemExit(f"dependency SHA changed: {name}")
    print(path)
PY
)
[[ "${#dependencies[@]}" -eq 3 ]] || fail "dependency receipt invalid"
MEMNAV_CKPT=${dependencies[0]}
NAVDP_CKPT=${dependencies[1]}
LINGBOT_WEIGHTS=${dependencies[2]}
LINGBOT_REPO=/scratch/lg154/Research/Nav/NavDP/baselines/memnav/lingbot-map
[[ "$(git -c safe.directory="${LINGBOT_REPO}" -C "${LINGBOT_REPO}" rev-parse HEAD)" == \
  7ff6f3ed0913d4d326f8f13bbb429c4ffc0195c2 ]] || fail "LingBot commit changed"

export PYTHONPATH="${SOURCE_ROOT}:${BASE_SOURCE_ROOT}:${DEPENDENCY_ROOT}:${LIGHTGLUE_REPO}:${INTERNNAV_ROOT}/src/diffusion-policy${PYTHONPATH:+:${PYTHONPATH}}"
export TORCH_HOME
"${MEMNAV_PY}" -m py_compile \
  "${SOURCE_ROOT}/MemNavData/cdec_pairwise_runtime.py" \
  "${SOURCE_ROOT}/MemNavData/eval_2leg_habitat.py" \
  "${MEMNAV_SERVER}" \
  "${SOURCE_ROOT}/NavDP/baselines/memnav/policy_agent.py"
"${MEMNAV_PY}" -m pytest -q \
  "${SOURCE_ROOT}/MemNavData/test_cdec_pairwise_runtime.py" \
  "${SOURCE_ROOT}/MemNavData/test_certified_relocalization_runtime.py" \
  "${SOURCE_ROOT}/MemNavData/test_revisit_bearing_adapter.py" \
  "${SOURCE_ROOT}/MemNavData/test_policy_agent_graph.py"

mkdir -p "${RUN_ROOT}/logs" "${RUN_ROOT}/buffer"
port_key=$(( (SLURM_JOB_ID + SLURM_ARRAY_TASK_ID * 37) % 15000 ))
MEMNAV_PORT=$((20000 + port_key * 2))
NAVDP_PORT=$((MEMNAV_PORT + 1))
for port in "${MEMNAV_PORT}" "${NAVDP_PORT}"; do
  ss -ltn | awk '{print $4}' | grep -Eq "(^|:)${port}$" && \
    fail "port ${port} in use"
done

runtime_root=${SLURM_TMPDIR:-/tmp}/cdec_cl_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}
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
      --lightglue_max_keypoints 2048 \
      --cdec_pairwise_artifact "${CDEC_ARTIFACT}" \
      --cdec_pairwise_allow_unapproved
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
    kill -0 "${pid}" 2>/dev/null || { tail -n 160 "${log}" >&2; fail "${label} exited"; }
    if ss -ltn | awk '{print $4}' | grep -Eq "(^|:)${port}$"; then
      ready=1; break
    fi
    sleep 2
  done
  [[ "${ready}" -eq 1 ]] || { tail -n 160 "${log}" >&2; fail "${label} did not bind"; }
done

ROOT="${SOURCE_ROOT}" RUN_ROOT="${RUN_ROOT}" MANIFEST="${MANIFEST}" \
EXPECTED_MANIFEST_SHA="${EXPECTED_MANIFEST_SHA}" \
TRACE_RECEIPT="${TRACE_RECEIPT}" \
EXPECTED_TRACE_RECEIPT_SHA="${EXPECTED_TRACE_RECEIPT_SHA}" \
SOURCE_RECEIPT="${SOURCE_RECEIPT}" \
EXPECTED_SOURCE_RECEIPT_SHA="${EXPECTED_SOURCE_RECEIPT_SHA}" \
GATE_VERIFICATION="${GATE_VERIFICATION}" \
EXPECTED_GATE_VERIFICATION_SHA="${EXPECTED_GATE_VERIFICATION_SHA}" \
CDEC_ARTIFACT_SHA="${EXPECTED_CDEC_ARTIFACT_SHA}" \
SCENE_INDEX="${SLURM_ARRAY_TASK_ID}" HAB_PY="${HAB_PY}" \
MEMNAV_PORT="${MEMNAV_PORT}" NAVDP_PORT="${NAVDP_PORT}" \
TRACE_RUN_ROOT="${TRACE_RUN_ROOT}" bash "${RUNNER}"
