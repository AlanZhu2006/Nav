#!/usr/bin/env bash
# Same-process four-arm causal gate for the shared-online double-Revisit pilot.

set -euo pipefail
umask 0022

ROOT=${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
OUT_ROOT=${OUT_ROOT:-${ROOT}/.diagnostics/shared_online_double_revisit_gate_local_20260813}
BENCH_ROOT=${BENCH_ROOT:-${ROOT}/.diagnostics/shared_online_double_revisit_v2_route_negative_pilot_20260812}
ASSET_ROOT=${ASSET_ROOT:-/home/asus/Research/datasets/mp3d_20scene/assets}
MEMNAV_PORT=${MEMNAV_PORT:-21430}
NAVDP_PORT=${NAVDP_PORT:-21431}
MEMNAV_PY=${MEMNAV_PY:-/home/asus/miniconda3/envs/memnav/bin/python}
HAB_PY=${HAB_PY:-/home/asus/miniconda3/envs/habitat/bin/python}
MEMNAV_CKPT=${MEMNAV_CKPT:-/home/asus/Research/Nav-axis-uturn/.diagnostics/unseen_scene_eval_20260803/checkpoints/gatecurr600.memnav.ckpt}
NAVDP_CKPT=${NAVDP_CKPT:-/home/asus/Research/Nav/NavDP/baselines/navdp/checkpoints/navdp_checkpoint.ckpt}
LINGBOT_REPO=${LINGBOT_REPO:-/home/asus/Research/Nav/NavDP/baselines/memnav/lingbot-map}
LINGBOT_WEIGHTS=${LINGBOT_WEIGHTS:-${LINGBOT_REPO}/weights/lingbot-map-long.pt}
LIGHTGLUE_REPO=${LIGHTGLUE_REPO:-${ROOT}/.diagnostics/dependencies/LightGlue}
DEPENDENCY_ROOT=${DEPENDENCY_ROOT:-${ROOT}/.diagnostics/dependencies/python}
INTERNNAV_ROOT=${INTERNNAV_ROOT:-${ROOT}/InternNav}
SCENE_INDICES=${SCENE_INDICES:-0,1,2,3}
MAX_STEPS=${MAX_STEPS:-600}

EXPECTED_BENCHMARK_SHA=95f5cbb311c10f3f6604eca47632cefea4b77b80d9f1e0e6ec93c1056c30786f
EXPECTED_MEMNAV_SHA=9b7a5811ff0aea212503f58b45258ba4f66b06420f87c350946aead39db6fdb7
EXPECTED_NAVDP_SHA=3bb3ad4ab241e857bb57a4021cc6aab76d5263e81fbf80298d579053ef011947

fail() { echo "ABORT: $*" >&2; exit 2; }

for required in \
  "${MEMNAV_PY}" "${HAB_PY}" "${MEMNAV_CKPT}" "${NAVDP_CKPT}" \
  "${LINGBOT_WEIGHTS}" "${LIGHTGLUE_REPO}" "${DEPENDENCY_ROOT}" \
  "${BENCH_ROOT}/manifest.json" "${BENCH_ROOT}/manifest.json.sha256" \
  "${ROOT}/MemNavData/eval_shared_online_double_revisit.py"; do
  test -r "${required}" || fail "missing input ${required}"
done
[[ "$(awk '{print $1}' "${BENCH_ROOT}/manifest.json.sha256")" == \
  "${EXPECTED_BENCHMARK_SHA}" ]] || fail "benchmark manifest receipt changed"
[[ "$(sha256sum "${BENCH_ROOT}/manifest.json" | awk '{print $1}')" == \
  "${EXPECTED_BENCHMARK_SHA}" ]] || fail "benchmark manifest changed"
[[ "$(sha256sum "${MEMNAV_CKPT}" | awk '{print $1}')" == \
  "${EXPECTED_MEMNAV_SHA}" ]] || fail "MemNav checkpoint changed"
[[ "$(sha256sum "${NAVDP_CKPT}" | awk '{print $1}')" == \
  "${EXPECTED_NAVDP_SHA}" ]] || fail "NavDP checkpoint changed"
[[ "${MAX_STEPS}" =~ ^[1-9][0-9]*$ ]] || fail "MAX_STEPS must be positive"

scene_names=(gxdoqLR6rwA pLe4wQe7qrG yqstnuAEVhm mJXqzFtmKg4)
episode_names=(episode_0000 episode_0000 episode_0001 episode_0001)
IFS=',' read -r -a selected_indices <<<"${SCENE_INDICES}"
for index in "${selected_indices[@]}"; do
  [[ "${index}" =~ ^[0-3]$ ]] || fail "SCENE_INDICES must contain only 0..3"
done
for port in "${MEMNAV_PORT}" "${NAVDP_PORT}"; do
  if ss -ltn | awk '{print $4}' | grep -Eq "(^|:)${port}$"; then
    fail "port ${port} is already in use"
  fi
done

mkdir -p "${OUT_ROOT}/logs" "${OUT_ROOT}/buffer" "${OUT_ROOT}/scenes"
source_receipt_tmp=$(mktemp)
runtime_root=$(mktemp -d /tmp/shared_online_gate.XXXXXX)
MEMNAV_PID=
NAVDP_PID=
cleanup() {
  for pid in "${NAVDP_PID}" "${MEMNAV_PID}"; do
    if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
      kill "${pid}" 2>/dev/null || true
      wait "${pid}" 2>/dev/null || true
    fi
  done
  rm -f -- "${source_receipt_tmp}"
  rm -rf -- "${runtime_root}"
}
trap cleanup EXIT INT TERM

sha256sum \
  "${ROOT}/MemNavData/run_shared_online_double_revisit_gate_local.sh" \
  "${ROOT}/MemNavData/eval_shared_online_double_revisit.py" \
  "${ROOT}/MemNavData/shared_online_double_revisit_runtime.py" \
  "${ROOT}/MemNavData/multigoal_policy_contract.py" \
  "${ROOT}/MemNavData/eval_2leg_habitat.py" \
  "${ROOT}/MemNavData/revisit_bearing_adapter.py" \
  "${ROOT}/MemNavData/certified_relocalization_runtime.py" \
  "${ROOT}/MemNavData/lingbot_pnp_localization.py" \
  "${ROOT}/NavDP/baselines/memnav/memnav_server.py" \
  "${ROOT}/NavDP/baselines/memnav/policy_agent.py" \
  "${ROOT}/NavDP/baselines/navdp/navdp_server.py" \
  "${ROOT}/NavDP/baselines/navdp/policy_agent.py" \
  "${BENCH_ROOT}/manifest.json" "${MEMNAV_CKPT}" "${NAVDP_CKPT}" \
  "${LINGBOT_WEIGHTS}" > "${source_receipt_tmp}"
if [[ -e "${OUT_ROOT}/source_inputs.sha256" ]]; then
  cmp --silent "${source_receipt_tmp}" "${OUT_ROOT}/source_inputs.sha256" || \
    fail "source inputs changed during resume"
else
  mv "${source_receipt_tmp}" "${OUT_ROOT}/source_inputs.sha256"
  source_receipt_tmp=$(mktemp)
fi

"${HAB_PY}" -m py_compile \
  "${ROOT}/MemNavData/eval_2leg_habitat.py" \
  "${ROOT}/MemNavData/eval_shared_online_double_revisit.py" \
  "${ROOT}/MemNavData/multigoal_policy_contract.py"
"${HAB_PY}" -m unittest \
  MemNavData.test_multigoal_policy_contract \
  MemNavData.test_shared_online_double_revisit_runtime \
  MemNavData.test_navdp_goal_switch

mkdir -p "${runtime_root}/memnav" "${runtime_root}/navdp"
server_pythonpath=${ROOT}:${DEPENDENCY_ROOT}:${LIGHTGLUE_REPO}:${INTERNNAV_ROOT}/src/diffusion-policy${PYTHONPATH:+:${PYTHONPATH}}
(
  cd "${runtime_root}/memnav"
  exec env PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    PYTHONPATH="${server_pythonpath}" \
    LINGBOT_REPO="${LINGBOT_REPO}" LINGBOT_WEIGHTS="${LINGBOT_WEIGHTS}" \
    MEMNAV_WINDOW=32 MEMNAV_NUM_SCALE=8 MEMNAV_MAX_FRAME_NUM=2048 \
    MEMNAV_GROUND_SCALE_MAX=6.0 MEMNAV_GATE_FUSION=complementary \
    MEMNAV_AUX_POSE_CALIBRATION=empirical MEMNAV_COLLISION_SELECT=1 \
    MEMNAV_REPORT_TO=none \
    "${MEMNAV_PY}" -u "${ROOT}/NavDP/baselines/memnav/memnav_server.py" \
      --port "${MEMNAV_PORT}" --checkpoint "${MEMNAV_CKPT}" \
      --internnav_root "${INTERNNAV_ROOT}" --num_samples 16 \
      --exclude_recent 32 --retrieval raw \
      --retrieval_candidate_top_k 32 --retrieval_candidate_min_gap 16 \
      --graph_subgoal_spacing_m 0.0 --graph_subgoal_arrival_m 0.60 \
      --flow_gate auto --buffer_root "${OUT_ROOT}/buffer" \
      --certified_relocalization \
      --lightglue_repo "${LIGHTGLUE_REPO}" \
      --lightglue_dependency_root "${DEPENDENCY_ROOT}" \
      --lightglue_max_keypoints 2048
) > "${OUT_ROOT}/logs/server_memnav.log" 2>&1 &
MEMNAV_PID=$!

(
  cd "${runtime_root}/navdp"
  exec env NAVDP_DISABLE_VIDEO=1 PYTHONUNBUFFERED=1 \
    PYTHONPATH="${server_pythonpath}" \
    "${MEMNAV_PY}" -u "${ROOT}/NavDP/baselines/navdp/navdp_server.py" \
      --port "${NAVDP_PORT}" --checkpoint "${NAVDP_CKPT}"
) > "${OUT_ROOT}/logs/server_navdp.log" 2>&1 &
NAVDP_PID=$!

for spec in \
  "memnav:${MEMNAV_PID}:${MEMNAV_PORT}:${OUT_ROOT}/logs/server_memnav.log" \
  "navdp:${NAVDP_PID}:${NAVDP_PORT}:${OUT_ROOT}/logs/server_navdp.log"; do
  IFS=: read -r label pid port log <<<"${spec}"
  ready=0
  for _ in $(seq 1 240); do
    if ! kill -0 "${pid}" 2>/dev/null; then
      tail -n 160 "${log}" >&2
      fail "${label} server exited during startup"
    fi
    if ss -ltn | awk '{print $4}' | grep -Eq "(^|:)${port}$"; then
      ready=1
      break
    fi
    sleep 2
  done
  [[ "${ready}" -eq 1 ]] || {
    tail -n 160 "${log}" >&2
    fail "${label} server did not bind port ${port}"
  }
done

hab_site_packages=$("${HAB_PY}" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
hab_pythonpath=${ROOT}:${hab_site_packages}/pip/_vendor${PYTHONPATH:+:${PYTHONPATH}}

for index in "${selected_indices[@]}"; do
  scene=${scene_names[${index}]}
  episode=${episode_names[${index}]}
  scene_file=${ASSET_ROOT}/${scene}/${scene}.glb
  episode_root=${BENCH_ROOT}/${scene}
  test -r "${scene_file}" || fail "missing scene asset ${scene_file}"
  test -r "${episode_root}/${episode}/benchmark.json" || \
    fail "missing benchmark episode ${scene}/${episode}"
  episode_seed=$("${HAB_PY}" - "${episode_root}/${episode}/benchmark.json" <<'PY'
import json, sys
from pathlib import Path
p=json.load(open(sys.argv[1]))
trace=json.load(open(Path(p["source_online_episode"])/"online_a_trace.json"))
print(int(trace["episode_seed"]))
PY
)
  scene_root=${OUT_ROOT}/scenes/$(printf '%02d' "${index}")_${scene}
  mkdir -p "${scene_root}/logs"
  case "${index}" in
    0) arm_order=(certified full_memory native memory_b_native_c) ;;
    1) arm_order=(full_memory memory_b_native_c certified native) ;;
    2) arm_order=(memory_b_native_c native full_memory certified) ;;
    3) arm_order=(native certified memory_b_native_c full_memory) ;;
  esac
  printf '%s\n' "${arm_order[@]}" > "${scene_root}/arm_order.txt"
  common=(
    --episode_root "${episode_root}"
    --episode_ids "${episode}"
    --scene "${scene_file}"
    --host 127.0.0.1
    --success_dist 1.0
    --max_steps "${MAX_STEPS}"
    --exec_horizon 8
    --trajectory_selector server
    --trajectory_selector_scope all
    --navdp_goal_switch_reset before_c
    --leg1_mode shared_trace
    --leg1_goal_source own
    --seed "${episode_seed}"
    --terminal_uturn off
    --terminal_visual_refine off
    --deterministic_plan_seeds
    --double_revisit_c_history initial_leg_only
    --shared_online_variant v1_controlled_pose_perturbation
    --shared_online_c_tail_max_covis 0.10
  )
  for arm in "${arm_order[@]}"; do
    arm_root=${scene_root}/${arm}
    if [[ -s "${arm_root}/summary.json" && -s "${arm_root}/metric.csv" \
          && -s "${arm_root}/${episode}_plans.json" ]]; then
      echo "[skip] ${scene}/${episode}/${arm} complete"
      continue
    fi
    if [[ -e "${arm_root}" ]]; then
      mv "${arm_root}" "${arm_root}.partial.$(date +%Y%m%dT%H%M%S)"
    fi
    mkdir -p "${arm_root}"
    echo "[run] ${scene}/${episode}/${arm}"
    case "${arm}" in
      full_memory)
        env PYTHONPATH="${hab_pythonpath}" "${HAB_PY}" -u \
          "${ROOT}/MemNavData/eval_shared_online_double_revisit.py" \
          "${common[@]}" --out "${arm_root}" \
          --port "${MEMNAV_PORT}" --novel_port "${NAVDP_PORT}" \
          --server_backend hybrid_pose --hybrid_route phase \
          --revisit_adapter legacy_metric \
          --shared_online_known_revisit_scope both \
          > "${scene_root}/logs/eval_${arm}.log" 2>&1
        ;;
      memory_b_native_c)
        env PYTHONPATH="${hab_pythonpath}" "${HAB_PY}" -u \
          "${ROOT}/MemNavData/eval_shared_online_double_revisit.py" \
          "${common[@]}" --out "${arm_root}" \
          --port "${MEMNAV_PORT}" --novel_port "${NAVDP_PORT}" \
          --server_backend hybrid_pose --hybrid_route phase \
          --revisit_adapter legacy_metric \
          --shared_online_known_revisit_scope b_only \
          > "${scene_root}/logs/eval_${arm}.log" 2>&1
        ;;
      certified)
        env PYTHONPATH="${hab_pythonpath}" "${HAB_PY}" -u \
          "${ROOT}/MemNavData/eval_shared_online_double_revisit.py" \
          "${common[@]}" --out "${arm_root}" \
          --port "${MEMNAV_PORT}" --novel_port "${NAVDP_PORT}" \
          --server_backend hybrid_pose \
          --hybrid_route certified_relocalization \
          --revisit_controller navdp_mixed \
          --revisit_adapter verified_bearing_v1 \
          > "${scene_root}/logs/eval_${arm}.log" 2>&1
        ;;
      native)
        env PYTHONPATH="${hab_pythonpath}" "${HAB_PY}" -u \
          "${ROOT}/MemNavData/eval_shared_online_double_revisit.py" \
          "${common[@]}" --out "${arm_root}" \
          --port "${NAVDP_PORT}" --server_backend navdp --hybrid_route phase \
          > "${scene_root}/logs/eval_${arm}.log" 2>&1
        ;;
      *) fail "unknown arm ${arm}" ;;
    esac
    "${HAB_PY}" - "${arm_root}/summary.json" "${arm}" <<'PY'
import json, sys
s=json.load(open(sys.argv[1]))
assert s["episodes"] == 1
assert s["variant"] == "v1_controlled_pose_perturbation"
assert s["navdp_goal_switch_reset"] == "before_c"
assert s["shared_A_all_hashes_ok"] is True
assert s["shared_A_total_diffusion_samples"] == 0
arm=sys.argv[2]
if arm == "memory_b_native_c":
    assert s["policy_backends"] == {"B": "navdp_mix", "C": "navdp"}
    assert s["C_long_memory_enabled"] is False
elif arm == "full_memory":
    assert s["policy_backends"] == {"B": "navdp_mix", "C": "navdp_mix"}
    assert s["C_long_memory_enabled"] is True
elif arm == "certified":
    assert s["policy_backends"] == {"B": "navdp_auto", "C": "navdp_auto"}
    assert s["C_long_memory_enabled"] is True
else:
    assert s["policy_backends"] == {"B": None, "C": None}
    assert s["C_long_memory_enabled"] is False
PY
  done
done

echo "[complete] selected scenes finished; root=${OUT_ROOT}"
