#!/usr/bin/env bash
# Local P0 protocol: freeze one Goal-A trace, then compare geometry ordering,
# Phase-B ordering + identical geometry gate, and native NavDP on that trace.

set -euo pipefail
umask 0022

ROOT=${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
DATA_ROOT=${DATA_ROOT:-/home/asus/Research/datasets/mp3d_20scene}
MANIFEST=${MANIFEST:-${ROOT}/MemNavData/expanded_navdp_router_eval_20260805.json}
OUT_ROOT=${OUT_ROOT:-${ROOT}/.diagnostics/p0_hybrid_full_20260808}
HAB_PY=${HAB_PY:-/home/asus/miniconda3/envs/habitat/bin/python}
MEMNAV_PORT=${MEMNAV_PORT:-21001}
NAVDP_PORT=${NAVDP_PORT:-21000}
MAX_STEPS=${MAX_STEPS:-500}
BASE_SEED=${BASE_SEED:-20260803}
SCENE_INDICES=${SCENE_INDICES:-}
RUN_GEOMETRY=${RUN_GEOMETRY:-1}
RUN_LEARNED=${RUN_LEARNED:-1}
RUN_NATIVE=${RUN_NATIVE:-1}
RUN_KNOWN_REVISIT_DIRECT=${RUN_KNOWN_REVISIT_DIRECT:-0}
RUN_KNOWN_REVISIT_FRONT_SUPPORT=${RUN_KNOWN_REVISIT_FRONT_SUPPORT:-0}
EXPECTED_PHASE_B_SHA=${EXPECTED_PHASE_B_SHA:-1232a426458cedf36869304116a2dd5c779bbcdaca587f76abd5ed3572164f2c}

for required in "${HAB_PY}" "${MANIFEST}" \
                "${ROOT}/MemNavData/eval_2leg_habitat.py"; do
  test -r "${required}" || {
    echo "ABORT: missing input ${required}" >&2
    exit 1
  }
done
[[ "${MAX_STEPS}" =~ ^[1-9][0-9]*$ ]] || {
  echo "ABORT: MAX_STEPS must be positive" >&2; exit 1; }
[[ "${BASE_SEED}" =~ ^[0-9]+$ ]] || {
  echo "ABORT: BASE_SEED must be non-negative" >&2; exit 1; }
for flag in RUN_GEOMETRY RUN_LEARNED RUN_NATIVE RUN_KNOWN_REVISIT_DIRECT \
            RUN_KNOWN_REVISIT_FRONT_SUPPORT; do
  [[ "${!flag}" =~ ^[01]$ ]] || {
    echo "ABORT: ${flag} must be 0 or 1" >&2; exit 1; }
done
[[ "${RUN_GEOMETRY}" -eq 1 ]] || {
  echo "ABORT: shared-prefix P0 requires the geometry trace source" >&2
  exit 1
}
if [[ "${RUN_KNOWN_REVISIT_FRONT_SUPPORT}" -eq 1 \
      && "${RUN_KNOWN_REVISIT_DIRECT}" -ne 1 ]]; then
  echo "ABORT: front-support pairing requires known-Revisit direct" >&2
  exit 1
fi

HAB_SITE_PACKAGES=$("${HAB_PY}" -c \
  'import sysconfig; print(sysconfig.get_paths()["purelib"])')
HAB_PYTHONPATH=${HAB_SITE_PACKAGES}/pip/_vendor${PYTHONPATH:+:${PYTHONPATH}}
hab_python() {
  env PYTHONPATH="${HAB_PYTHONPATH}" "${HAB_PY}" "$@"
}

server_status=$(curl --fail --silent --show-error \
  --max-time 10 -X POST "http://127.0.0.1:${MEMNAV_PORT}/navigator_reset" \
  -H 'Content-Type: application/json' -d '{}')
hab_python - "${server_status}" "${EXPECTED_PHASE_B_SHA}" <<'PY'
import json, sys
status = json.loads(sys.argv[1])
ranker = status.get("phase_b_ranker") or {}
if ranker.get("enabled") is not True:
    raise SystemExit("MemNav server has no Phase-B ranker")
if ranker.get("checkpoint_sha256") != sys.argv[2]:
    raise SystemExit("MemNav server Phase-B checkpoint SHA mismatch")
if ranker.get("activation_semantics") != "diagnostic_only_geometry_gate_unchanged":
    raise SystemExit("MemNav server does not guarantee ranking-only semantics")
if ranker.get("deployment_approved") is not False:
    raise SystemExit("P0 expects the explicitly unapproved experimental checkpoint")
if ranker.get("allow_unapproved") is not True:
    raise SystemExit("P0 experimental checkpoint was not explicitly acknowledged")
if status.get("retrieval") != "raw" or status.get("retrieval_candidate_min_gap") != 16:
    raise SystemExit("MemNav server violates raw-DINO/gap-16 protocol")
PY
# Flask returns 404 at root; any HTTP response proves the server is listening.
status=$(curl --silent --output /dev/null --write-out '%{http_code}' \
  --max-time 10 "http://127.0.0.1:${NAVDP_PORT}/")
[[ "${status}" =~ ^[1-5][0-9][0-9]$ ]] || {
  echo "ABORT: NavDP server is unavailable" >&2; exit 1; }

mkdir -p "${OUT_ROOT}/scenes" "${OUT_ROOT}/logs"
contract_tmp=$(mktemp)
trap 'rm -f "${contract_tmp}"' EXIT
sha256sum \
  "${ROOT}/MemNavData/eval_2leg_habitat.py" \
  "${ROOT}/MemNavData/revisit_bearing_adapter.py" \
  "${ROOT}/MemNavData/phase_b_feature_schema.py" \
  "${ROOT}/MemNavData/phase_b_model.py" \
  "${ROOT}/MemNavData/phase_b_runtime.py" \
  "${ROOT}/NavDP/baselines/memnav/memnav_server.py" \
  "${ROOT}/NavDP/baselines/memnav/policy_agent.py" \
  "${MANIFEST}" > "${contract_tmp}"
if [[ -e "${OUT_ROOT}/source_inputs.sha256" ]]; then
  cmp --silent "${contract_tmp}" "${OUT_ROOT}/source_inputs.sha256" || {
    echo "ABORT: source inputs changed within the P0 run" >&2
    exit 1
  }
else
  mv "${contract_tmp}" "${OUT_ROOT}/source_inputs.sha256"
  contract_tmp=$(mktemp)
fi

mapfile -t SCENES < <(hab_python - "${MANIFEST}" <<'PY'
import json, sys
print(*json.load(open(sys.argv[1]))["selection"]["selected_scenes"], sep="\n")
PY
)
[[ "${#SCENES[@]}" -eq 20 ]] || {
  echo "ABORT: expected the frozen 20-scene selection" >&2; exit 1; }

if [[ -n "${SCENE_INDICES}" ]]; then
  IFS=',' read -r -a INDICES <<<"${SCENE_INDICES}"
else
  INDICES=($(seq 0 19))
fi
for index in "${INDICES[@]}"; do
  [[ "${index}" =~ ^([0-9]|1[0-9])$ ]] || {
    echo "ABORT: invalid scene index ${index}" >&2; exit 1; }
done

arm_complete() {
  local arm_root=$1
  [[ -s "${arm_root}/summary.json" && -s "${arm_root}/metric.csv" ]] || return 1
  local rows
  rows=$(($(wc -l < "${arm_root}/metric.csv") - 1))
  [[ "${rows}" -eq 2 ]]
}

prepare_arm() {
  local arm_root=$1
  if arm_complete "${arm_root}"; then
    return 1
  fi
  if [[ -e "${arm_root}" ]]; then
    local quarantine=${arm_root}.partial.$(date +%Y%m%dT%H%M%S)
    mv "${arm_root}" "${quarantine}"
    echo "[resume] moved incomplete output to ${quarantine}"
  fi
  mkdir -p "${arm_root}"
  return 0
}

validate_arm() {
  local summary=$1
  local expected_route=$2
  local expected_adapter=${3:-}
  hab_python - "${summary}" "${expected_route}" \
      "${EXPECTED_PHASE_B_SHA}" "${expected_adapter}" <<'PY'
import json, sys
summary = json.load(open(sys.argv[1]))
if summary.get("episodes") != 2:
    raise SystemExit("arm did not produce exactly two episodes")
if summary.get("hybrid_route") != sys.argv[2]:
    raise SystemExit("arm route differs from protocol")
if sys.argv[2] == "learned_rank_geometry":
    ranker = summary.get("phase_b_ranker") or {}
    if ranker.get("checkpoint_sha256") != sys.argv[3]:
        raise SystemExit("learned arm checkpoint SHA changed")
    if summary.get("phase_b_p0_transport_valid") is not True:
        raise SystemExit("learned arm had fallback or activation violation")
if sys.argv[4] and summary.get("revisit_adapter") != sys.argv[4]:
    raise SystemExit("arm revisit adapter differs from protocol")
PY
}

for index in "${INDICES[@]}"; do
  scene=${SCENES[10#${index}]}
  scene_file=${DATA_ROOT}/assets/${scene}/${scene}.glb
  episode_root=${DATA_ROOT}/episodes/${scene}
  test -r "${scene_file}" || {
    echo "ABORT: missing scene ${scene_file}" >&2; exit 1; }
  test -d "${episode_root}" || {
    echo "ABORT: missing episodes ${episode_root}" >&2; exit 1; }
  scene_root=${OUT_ROOT}/scenes/$(printf '%02d' "${index}")_${scene}
  mkdir -p "${scene_root}/logs"

  common=(
    --episode_root "${episode_root}"
    --scene "${scene_file}"
    --host 127.0.0.1
    --success_dist 1.0
    --max_steps "${MAX_STEPS}"
    --exec_horizon 8
    --trajectory_selector server
    --navdp_goal_switch_reset carry
    --leg1_goal_source own
    --seed "${BASE_SEED}"
    --episode_ids episode_0000,episode_0001
    --terminal_uturn off
    --terminal_visual_refine off
    --deterministic_plan_seeds
  )

  geometry_root=${scene_root}/geometry_router
  if prepare_arm "${geometry_root}"; then
    echo "[p0] scene=${index}/${scene} arm=geometry_router"
    hab_python -u "${ROOT}/MemNavData/eval_2leg_habitat.py" \
      "${common[@]}" \
      --port "${MEMNAV_PORT}" --novel_port "${NAVDP_PORT}" \
      --out "${geometry_root}" \
      --server_backend hybrid_pose \
      --leg1_mode policy --write_leg1_trace \
      --hybrid_route memory_geometry \
      --router_visual_floor 0.88 \
      --router_min_matches 20 --router_min_inliers 12 \
      --router_min_inlier_ratio 0.50 --router_confirm_plans 2 \
      --router_verify_top_k 8 \
      > "${scene_root}/logs/eval_geometry_router.log" 2>&1
  else
    echo "[skip] scene=${index}/${scene} arm=geometry_router complete"
  fi
  validate_arm "${geometry_root}/summary.json" memory_geometry

  # The benchmark contract already declares Goal A Novel and Goal B Revisit.
  # This arm therefore removes only the automatic RANSAC activation gate at
  # the known A->B boundary.  Retrieval, LingBot pose recovery, the mixed
  # NavDP controller, Goal-A trace, and per-plan diffusion seeds are unchanged.
  if [[ "${RUN_KNOWN_REVISIT_DIRECT}" -eq 1 ]]; then
    direct_root=${scene_root}/known_revisit_direct
    if prepare_arm "${direct_root}"; then
      echo "[p0] scene=${index}/${scene} arm=known_revisit_direct"
      hab_python -u "${ROOT}/MemNavData/eval_2leg_habitat.py" \
        "${common[@]}" \
        --port "${MEMNAV_PORT}" --novel_port "${NAVDP_PORT}" \
        --out "${direct_root}" \
        --server_backend hybrid_pose \
        --leg1_mode shared_trace \
        --shared_leg1_trace_root "${geometry_root}" \
        --hybrid_route phase \
        --revisit_adapter legacy_metric \
        > "${scene_root}/logs/eval_known_revisit_direct.log" 2>&1
    else
      echo "[skip] scene=${index}/${scene} arm=known_revisit_direct complete"
    fi
    validate_arm "${direct_root}/summary.json" phase legacy_metric
  fi

  if [[ "${RUN_KNOWN_REVISIT_FRONT_SUPPORT}" -eq 1 ]]; then
    support_root=${scene_root}/front_support_residual
    if prepare_arm "${support_root}"; then
      echo "[p0] scene=${index}/${scene} arm=front_support_residual"
      hab_python -u "${ROOT}/MemNavData/eval_2leg_habitat.py" \
        "${common[@]}" \
        --port "${MEMNAV_PORT}" --novel_port "${NAVDP_PORT}" \
        --out "${support_root}" \
        --server_backend hybrid_pose \
        --leg1_mode shared_trace \
        --shared_leg1_trace_root "${geometry_root}" \
        --hybrid_route phase \
        --revisit_adapter navdp_front_support_v1 \
        > "${scene_root}/logs/eval_front_support_residual.log" 2>&1
    else
      echo "[skip] scene=${index}/${scene} arm=front_support_residual complete"
    fi
    validate_arm "${support_root}/summary.json" phase \
      navdp_front_support_v1
  fi

  if [[ "${RUN_LEARNED}" -eq 1 ]]; then
    learned_root=${scene_root}/learned_rank_geometry
    if prepare_arm "${learned_root}"; then
      echo "[p0] scene=${index}/${scene} arm=learned_rank_geometry"
      hab_python -u "${ROOT}/MemNavData/eval_2leg_habitat.py" \
        "${common[@]}" \
        --port "${MEMNAV_PORT}" --novel_port "${NAVDP_PORT}" \
        --out "${learned_root}" \
        --server_backend hybrid_pose \
        --leg1_mode shared_trace \
        --shared_leg1_trace_root "${geometry_root}" \
        --hybrid_route learned_rank_geometry \
        --router_visual_floor 0.88 \
        --router_min_matches 20 --router_min_inliers 12 \
        --router_min_inlier_ratio 0.50 --router_confirm_plans 2 \
        --router_verify_top_k 8 \
        > "${scene_root}/logs/eval_learned_rank_geometry.log" 2>&1
    else
      echo "[skip] scene=${index}/${scene} arm=learned_rank_geometry complete"
    fi
    validate_arm "${learned_root}/summary.json" learned_rank_geometry
  fi

  if [[ "${RUN_NATIVE}" -eq 1 ]]; then
    native_root=${scene_root}/navdp_native
    if prepare_arm "${native_root}"; then
      echo "[p0] scene=${index}/${scene} arm=navdp_native"
      hab_python -u "${ROOT}/MemNavData/eval_2leg_habitat.py" \
        "${common[@]}" \
        --port "${NAVDP_PORT}" \
        --out "${native_root}" \
        --server_backend navdp \
        --leg1_mode shared_trace \
        --shared_leg1_trace_root "${geometry_root}" \
        > "${scene_root}/logs/eval_navdp_native.log" 2>&1
    else
      echo "[skip] scene=${index}/${scene} arm=navdp_native complete"
    fi
    validate_arm "${native_root}/summary.json" phase
  fi
done

echo "[complete] local Phase-B P0 outputs: ${OUT_ROOT}"
