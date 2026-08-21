#!/usr/bin/env bash
# Render the frozen pose while the formal MemNav/NavDP servers share the GPU.
# Submit twice with INCLUDE_PI3X=0/1 on the same GPU partition.
#SBATCH --job-name=f14cores_diag
#SBATCH --output=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/%x_%j.out
#SBATCH --error=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/%x_%j.err
#SBATCH --time=00:20:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu48
#SBATCH --account=torch_pr_769_tandon_advanced
#SBATCH --export=ALL

set -euo pipefail
umask 0022

: "${DIAG_SOURCE_ROOT:?set immutable diagnostic bundle}"
: "${DIAG_SOURCE_RECEIPT_SHA:?set diagnostic receipt SHA}"
: "${METHOD_SOURCE_ROOT:?set immutable Attempt 6 method bundle}"
: "${METHOD_SOURCE_RECEIPT_SHA:?set Attempt 6 receipt SHA}"
: "${OUT_ROOT:?set unique output directory}"
INCLUDE_PI3X=${INCLUDE_PI3X:?set 0 or 1}
PROBE_MODE=${PROBE_MODE:-render}
DIAG_MAX_STEPS=${DIAG_MAX_STEPS:-16}
MEMNAV_CUDA_LAUNCH_BLOCKING=${MEMNAV_CUDA_LAUNCH_BLOCKING:-0}
MEMNAV_HTTP_HANDOFF_SYNC=${MEMNAV_HTTP_HANDOFF_SYNC:-0}
[[ "${INCLUDE_PI3X}" == 0 || "${INCLUDE_PI3X}" == 1 ]] || {
  echo "ABORT: INCLUDE_PI3X must be 0 or 1" >&2; exit 2;
}
[[ "${PROBE_MODE}" == render || "${PROBE_MODE}" == rollout ]] || {
  echo "ABORT: PROBE_MODE must be render or rollout" >&2; exit 2;
}
[[ "${DIAG_MAX_STEPS}" =~ ^[1-9][0-9]*$ ]] || {
  echo "ABORT: DIAG_MAX_STEPS must be positive" >&2; exit 2;
}
[[ "${MEMNAV_CUDA_LAUNCH_BLOCKING}" == 0 || \
   "${MEMNAV_CUDA_LAUNCH_BLOCKING}" == 1 ]] || {
  echo "ABORT: MEMNAV_CUDA_LAUNCH_BLOCKING must be 0 or 1" >&2; exit 2;
}
[[ "${MEMNAV_HTTP_HANDOFF_SYNC}" == 0 || \
   "${MEMNAV_HTTP_HANDOFF_SYNC}" == 1 ]] || {
  echo "ABORT: MEMNAV_HTTP_HANDOFF_SYNC must be 0 or 1" >&2; exit 2;
}

HAB_PY=/scratch/lg154/conda-envs/habitat/bin/python
MEMNAV_PY=/scratch/lg154/conda-envs/memnav/bin/python
BASE_SIF=/share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif
BASE_SOURCE_ROOT=/scratch/yz11502/Research/Nav-axis-uturn-source-bundles/certified_relocalization_closed_loop_d3bd281fc374cc80
BASE_SOURCE_RECEIPT_SHA=74001a9e0150c38c599a206fa0f4dd5e1279b9bed5d167119f4d14cb77995e98
DEPENDENCY_RECEIPT=/scratch/yz11502/Research/Nav-axis-uturn-results/shared_online_double_revisit_fresh_20260813/double_revisit_fresh40_20260813T200121Z/dependency_receipt.json
DEPENDENCY_RECEIPT_SHA=4eb0ca6479a26f8e04f85a31d906cee4e68b1785f66cfd3ac23bf65424d36e5e
PI3X_SNAPSHOT=/scratch/yz11502/Research/model_assets/pi3x_69972d6e1c4492c
PI3X_MODEL_SHA=69972d6e1c4492cb4d737a84fe940e357087d81c52f5c9b7c160b49c1f41669a
PI3X_PROOF=${METHOD_SOURCE_ROOT}/pi3x_deployment/deployment_manifest.json
PI3X_PROOF_SHA=1a05aaa7cf75296cb68e32f9ea57fba6bcce2b9f57313a8cede05b7c7b0cffdd
PI3X_ROOT=${METHOD_SOURCE_ROOT}/third_party/Pi3
PORTABLE_VERIFIER=${METHOD_SOURCE_ROOT}/MemNavData/verify_portable_checksum_manifest.py
MEMNAV_SERVER=${METHOD_SOURCE_ROOT}/NavDP/baselines/memnav/memnav_server.py
NAVDP_SERVER=${METHOD_SOURCE_ROOT}/NavDP/baselines/navdp/navdp_server.py
SERVER_LAUNCHER=${METHOD_SOURCE_ROOT}/MemNavData/retrying_server_launcher.py
ENTRYPOINT=${DIAG_SOURCE_ROOT}/MemNavData/diagnose_habitat_render_stability.py
ROLLOUT_ENTRYPOINT=${DIAG_SOURCE_ROOT}/MemNavData/diagnose_role_pair_runtime.py
INTERNNAV_ROOT=${BASE_SOURCE_ROOT}/InternNav
LIGHTGLUE_REPO=${BASE_SOURCE_ROOT}/third_party/LightGlue
DEPENDENCY_ROOT=${BASE_SOURCE_ROOT}/third_party/python
TORCH_HOME=${BASE_SOURCE_ROOT}/torch_home
LINGBOT_REPO=/scratch/lg154/Research/Nav/NavDP/baselines/memnav/lingbot-map
SOURCE_OVERLAY=/scratch/lg154/Research/datasets/_overlays/mp3d_revisit_v0_pt1.sqf
SOURCE_OVERLAY_BYTES=128854888448
BENCHMARK_RUN=/scratch/yz11502/Research/Nav-axis-uturn-results/final14_cec_learned_20260817/final14_learned_20260817T0915Z_attempt6_overlay
BENCH_ROOT=${BENCHMARK_RUN}/benchmarks/natural_direction
BENCH_MANIFEST_SHA=7468703a9efbb10e801ffdd226911f696a30fa9432ef9ab486d3134f6e40fe6a

fail() { echo "ABORT: $*" >&2; exit 2; }
[[ ! -e "${OUT_ROOT}" ]] || fail "output exists"
[[ "$(sha256sum "${DIAG_SOURCE_ROOT}/SOURCE_BUNDLE.sha256" | awk '{print $1}')" == \
  "${DIAG_SOURCE_RECEIPT_SHA}" ]] || fail "diagnostic receipt changed"
(cd "${DIAG_SOURCE_ROOT}" && sha256sum -c SOURCE_BUNDLE.sha256 --quiet) || \
  fail "diagnostic bundle changed"
[[ "$(sha256sum "${METHOD_SOURCE_ROOT}/SOURCE_BUNDLE.sha256" | awk '{print $1}')" == \
  "${METHOD_SOURCE_RECEIPT_SHA}" ]] || fail "method receipt changed"
(cd "${METHOD_SOURCE_ROOT}" && sha256sum -c SOURCE_BUNDLE.sha256 --quiet) || \
  fail "method bundle changed"
[[ "$(sha256sum "${BASE_SOURCE_ROOT}/SOURCE_BUNDLE.sha256" | awk '{print $1}')" == \
  "${BASE_SOURCE_RECEIPT_SHA}" ]] || fail "base source receipt changed"
[[ "$(sha256sum "${DEPENDENCY_RECEIPT}" | awk '{print $1}')" == \
  "${DEPENDENCY_RECEIPT_SHA}" ]] || fail "dependency receipt changed"
[[ "$(sha256sum "${PI3X_SNAPSHOT}/model.safetensors" | awk '{print $1}')" == \
  "${PI3X_MODEL_SHA}" ]] || fail "Pi3X model changed"
[[ "$(sha256sum "${PI3X_PROOF}" | awk '{print $1}')" == \
  "${PI3X_PROOF_SHA}" ]] || fail "Pi3X proof changed"
[[ "$(stat -c %s "${SOURCE_OVERLAY}")" -eq "${SOURCE_OVERLAY_BYTES}" ]] || \
  fail "episode overlay changed"
[[ "$(sha256sum "${BENCH_ROOT}/manifest.json" | awk '{print $1}')" == \
  "${BENCH_MANIFEST_SHA}" ]] || fail "benchmark manifest changed"

readarray -t dependencies < <("${MEMNAV_PY}" - "${DEPENDENCY_RECEIPT}" <<'PY'
import json,os,sys
x=json.load(open(sys.argv[1]))
for name in ("gatecurr600","navdp_checkpoint","lingbot_map_long"):
    row=x["dependencies"][name]
    if os.stat(row["path"]).st_size != int(row["bytes"]):
        raise SystemExit(f"dependency size changed: {name}")
    print(row["path"])
PY
)
[[ "${#dependencies[@]}" -eq 3 ]] || fail "dependency receipt invalid"
MEMNAV_CKPT=${dependencies[0]}; NAVDP_CKPT=${dependencies[1]}
LINGBOT_WEIGHTS=${dependencies[2]}

pi3x_args=()
handoff_args=()
if [[ "${MEMNAV_HTTP_HANDOFF_SYNC}" == 1 ]]; then
  handoff_args=(--synchronize_cuda_http_handoff)
fi
if [[ "${INCLUDE_PI3X}" == 1 ]]; then
  "${MEMNAV_PY}" "${PORTABLE_VERIFIER}" \
    "$(dirname "${PI3X_PROOF}")/OUTPUTS.sha256" --quiet || \
    fail "Pi3X deployment receipt invalid"
  pi3x_args=(
    --pi3x_learned_relocalizer
    --pi3x_root "${PI3X_ROOT}"
    --pi3x_snapshot "${PI3X_SNAPSHOT}"
    --pi3x_model_sha256 "${PI3X_MODEL_SHA}"
    --pi3x_spatial_proof_manifest "${PI3X_PROOF}"
    --pi3x_inference_dtype auto
  )
fi

mkdir -p "${OUT_ROOT}/logs" "${OUT_ROOT}/buffer"
echo "host=$(hostname) job=${SLURM_JOB_ID} include_pi3x=${INCLUDE_PI3X} probe_mode=${PROBE_MODE}"
nvidia-smi --query-gpu=name,uuid,driver_version,memory.total \
  --format=csv,noheader | tee "${OUT_ROOT}/gpu.txt"

export PYTHONPYCACHEPREFIX=${SLURM_TMPDIR:-/tmp}/f14cores_${SLURM_JOB_ID}
export PYTHONPATH=${METHOD_SOURCE_ROOT}:${METHOD_SOURCE_ROOT}/MemNavData:${DEPENDENCY_ROOT}:${LIGHTGLUE_REPO}:${BASE_SOURCE_ROOT}/NavDP/baselines/navdp:${BASE_SOURCE_ROOT}/NavDP/baselines/memnav:${INTERNNAV_ROOT}/src/diffusion-policy${PYTHONPATH:+:${PYTHONPATH}}
export TORCH_HOME PYTHONFAULTHANDLER=1
HAB_SITE_PACKAGES=$("${HAB_PY}" -c \
  'import sysconfig; print(sysconfig.get_paths()["purelib"])')
HAB_PYTHONPATH=${METHOD_SOURCE_ROOT}:${METHOD_SOURCE_ROOT}/MemNavData:${HAB_SITE_PACKAGES}/pip/_vendor${PYTHONPATH:+:${PYTHONPATH}}
runtime_root=${SLURM_TMPDIR:-/tmp}/f14cores_${SLURM_JOB_ID}
mkdir -p "${runtime_root}/memnav" "${runtime_root}/navdp" "${PYTHONPYCACHEPREFIX}"

port_key=$((SLURM_JOB_ID % 12000))
MEMNAV_BASE_PORT=$((20000 + port_key * 2))
NAVDP_BASE_PORT=$((MEMNAV_BASE_PORT + 1))
MEMNAV_PORT_FILE=${runtime_root}/memnav.port
NAVDP_PORT_FILE=${runtime_root}/navdp.port
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

(
  exec env PYTHONUNBUFFERED=1 PYTHONPYCACHEPREFIX="${PYTHONPYCACHEPREFIX}" \
    PYTHONPATH="${PYTHONPATH}" TORCH_HOME="${TORCH_HOME}" \
    CUDA_LAUNCH_BLOCKING="${MEMNAV_CUDA_LAUNCH_BLOCKING}" \
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
      --receipt-file "${OUT_ROOT}/logs/memnav_launcher.json" \
      --log-prefix "${OUT_ROOT}/logs/memnav" \
      --cwd "${runtime_root}/memnav" -- \
      "${MEMNAV_PY}" -u "${MEMNAV_SERVER}" --port '{port}' \
        --checkpoint "${MEMNAV_CKPT}" --internnav_root "${INTERNNAV_ROOT}" \
        --num_samples 16 --exclude_recent 32 --retrieval raw \
        --retrieval_candidate_top_k 32 --retrieval_candidate_min_gap 16 \
        --graph_subgoal_spacing_m 0.0 --graph_subgoal_arrival_m 0.60 \
        --flow_gate auto --buffer_root "${OUT_ROOT}/buffer" \
        --certified_relocalization --lightglue_repo "${LIGHTGLUE_REPO}" \
        --lightglue_dependency_root "${DEPENDENCY_ROOT}" \
        --lightglue_max_keypoints 2048 "${pi3x_args[@]}" \
        "${handoff_args[@]}"
) > "${OUT_ROOT}/logs/memnav_launcher.log" 2>&1 &
MEMNAV_PID=$!

(
  exec env NAVDP_DISABLE_VIDEO=1 PYTHONUNBUFFERED=1 \
    PYTHONPYCACHEPREFIX="${PYTHONPYCACHEPREFIX}" PYTHONPATH="${PYTHONPATH}" \
    "${MEMNAV_PY}" -u "${SERVER_LAUNCHER}" \
      --base-port "${NAVDP_BASE_PORT}" --range-start 20000 --range-end 49999 \
      --stride 1994 --max-attempts 12 --ready-timeout 720 \
      --port-file "${NAVDP_PORT_FILE}" \
      --receipt-file "${OUT_ROOT}/logs/navdp_launcher.json" \
      --log-prefix "${OUT_ROOT}/logs/navdp" \
      --cwd "${runtime_root}/navdp" -- \
      "${MEMNAV_PY}" -u "${NAVDP_SERVER}" --port '{port}' \
        --checkpoint "${NAVDP_CKPT}"
) > "${OUT_ROOT}/logs/navdp_launcher.log" 2>&1 &
NAVDP_PID=$!

ready=0
for _ in $(seq 1 420); do
  kill -0 "${MEMNAV_PID}" 2>/dev/null || fail "MemNav launcher died"
  kill -0 "${NAVDP_PID}" 2>/dev/null || fail "NavDP launcher died"
  if [[ -s "${MEMNAV_PORT_FILE}" && -s "${NAVDP_PORT_FILE}" ]]; then
    ready=1; break
  fi
  sleep 2
done
[[ "${ready}" -eq 1 ]] || fail "servers did not become ready"

nvidia-smi --query-compute-apps=pid,process_name,used_memory \
  --format=csv,noheader > "${OUT_ROOT}/gpu_processes_before_render.txt" || true
ulimit -c unlimited || true
MEMNAV_PORT=$(<"${MEMNAV_PORT_FILE}")
NAVDP_PORT=$(<"${NAVDP_PORT_FILE}")
if [[ "${PROBE_MODE}" == render ]]; then
  singularity exec --nv -B /scratch/lg154 -B /scratch/yz11502 \
    "${BASE_SIF}" env PYTHONFAULTHANDLER=1 \
    PYTHONPYCACHEPREFIX="${PYTHONPYCACHEPREFIX}" \
    PYTHONPATH="${METHOD_SOURCE_ROOT}:${METHOD_SOURCE_ROOT}/MemNavData" \
    "${HAB_PY}" -X faulthandler "${ENTRYPOINT}" \
      --scene /scratch/lg154/Research/datasets/mp3d/mp3d/8WUmhLawc2A/8WUmhLawc2A.glb \
      --scene-sha256 d652bd05699b2ab481f4c474b0dbdc0ba8422cdc8d1310182fb25a9497b153ff \
      --position -0.9123131036758423 0.10065580904483795 0.5004647374153137 \
      --yaw 4.57714258977832 --renders 16 --yaw-step-deg 1.0 \
    | tee "${OUT_ROOT}/render.jsonl"
  test "$(grep -c '"event": "render"' "${OUT_ROOT}/render.jsonl")" -eq 16 || \
    fail "render count incomplete"
  grep -q '"event": "complete"' "${OUT_ROOT}/render.jsonl" || \
    fail "completion record missing"
else
  mkdir "${OUT_ROOT}/eval"
  singularity exec --nv --overlay "${SOURCE_OVERLAY}:ro" \
    -B /scratch/lg154 -B /scratch/yz11502 "${BASE_SIF}" env \
    PYTHONFAULTHANDLER=1 PYTHONPYCACHEPREFIX="${PYTHONPYCACHEPREFIX}" \
    PYTHONPATH="${HAB_PYTHONPATH}" \
    "${HAB_PY}" -X faulthandler "${ROLLOUT_ENTRYPOINT}" \
      --episode_root "${BENCH_ROOT}/8WUmhLawc2A" \
      --episode_ids episode_0000 \
      --scene /scratch/lg154/Research/datasets/mp3d/mp3d/8WUmhLawc2A/8WUmhLawc2A.glb \
      --scene_identity 8WUmhLawc2A --host 127.0.0.1 \
      --port "${MEMNAV_PORT}" --novel_port "${NAVDP_PORT}" \
      --server_backend hybrid_pose --hybrid_route memory_geometry \
      --revisit_adapter verified_bearing_v1 \
      --router_visual_floor 0.88 --router_min_matches 20 \
      --router_min_inliers 12 --router_min_inlier_ratio 0.50 \
      --router_confirm_plans 2 --router_verify_top_k 8 \
      --success_dist 1.0 --max_steps "${DIAG_MAX_STEPS}" --exec_horizon 8 \
      --trajectory_selector server --trajectory_selector_scope all \
      --leg1_mode shared_trace --leg1_goal_source own --seed 0 \
      --terminal_uturn off --terminal_visual_refine off \
      --deterministic_plan_seeds --retrieval_override off \
      --certified_cdec_rescue off --certified_stagnation_graph off \
      --revisit_controller navdp_mixed --role_pair_scope paper_heldout \
      --out "${OUT_ROOT}/eval" \
    > "${OUT_ROOT}/logs/eval_stdout.log" \
    2> "${OUT_ROOT}/logs/eval_stderr.log"
  [[ -s "${OUT_ROOT}/eval/summary.json" ]] || fail "rollout summary missing"
  grep -q '"event": "runtime_timing"' "${OUT_ROOT}/logs/eval_stderr.log" || \
    fail "rollout timing events missing"
fi
"${MEMNAV_PY}" - "${OUT_ROOT}/completion.json" "${INCLUDE_PI3X}" \
  "${PROBE_MODE}" "${DIAG_MAX_STEPS}" <<'PY'
import hashlib,json,sys
p=sys.argv[1]
x={"schema_version":"final14_model_co_residency_diagnostic_v1_20260817",
   "include_pi3x":bool(int(sys.argv[2])),"probe_mode":sys.argv[3],
   "diagnostic_max_steps":int(sys.argv[4]),"complete":True}
b=(json.dumps(x,indent=2,sort_keys=True)+"\n").encode()
open(p,"xb").write(b)
open(p+".sha256","x").write(hashlib.sha256(b).hexdigest()+"  completion.json\n")
PY
