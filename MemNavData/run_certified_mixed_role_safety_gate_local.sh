#!/usr/bin/env bash
# Same-process strict-v4 safety gate for the role-free certified route.

set -euo pipefail
umask 0022

ROOT=${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
OUT_ROOT=${OUT_ROOT:-${ROOT}/.diagnostics/certified_mixed_role_safety_gate_local_20260813}
BENCH_ROOT=${BENCH_ROOT:-${ROOT}/.diagnostics/multigoal_v4_quickpilot_r1_20260812}
ASSET_ROOT=${ASSET_ROOT:-/home/asus/Research/datasets/mp3d_20scene/assets}
MEMNAV_PORT=${MEMNAV_PORT:-21440}
NAVDP_PORT=${NAVDP_PORT:-21441}
MEMNAV_PY=${MEMNAV_PY:-/home/asus/miniconda3/envs/memnav/bin/python}
HAB_PY=${HAB_PY:-/home/asus/miniconda3/envs/habitat/bin/python}
MEMNAV_CKPT=${MEMNAV_CKPT:-/home/asus/Research/Nav-axis-uturn/.diagnostics/unseen_scene_eval_20260803/checkpoints/gatecurr600.memnav.ckpt}
NAVDP_CKPT=${NAVDP_CKPT:-/home/asus/Research/Nav/NavDP/baselines/navdp/checkpoints/navdp_checkpoint.ckpt}
LINGBOT_REPO=${LINGBOT_REPO:-/home/asus/Research/Nav/NavDP/baselines/memnav/lingbot-map}
LINGBOT_WEIGHTS=${LINGBOT_WEIGHTS:-${LINGBOT_REPO}/weights/lingbot-map-long.pt}
LIGHTGLUE_REPO=${LIGHTGLUE_REPO:-${ROOT}/.diagnostics/dependencies/LightGlue}
DEPENDENCY_ROOT=${DEPENDENCY_ROOT:-${ROOT}/.diagnostics/dependencies/python}
INTERNNAV_ROOT=${INTERNNAV_ROOT:-${ROOT}/InternNav}
# Default is the two-scene smoke: one Novel-B failure and the only Revisit-C
# positive control.  Set 0,1,2,3 only after the smoke passes.
SCENE_INDICES=${SCENE_INDICES:-0,3}
MAX_STEPS=${MAX_STEPS:-600}
AUDIT_ALLOW_NO_POSITIVE_CONTROL=${AUDIT_ALLOW_NO_POSITIVE_CONTROL:-0}

EXPECTED_MEMNAV_SHA=9b7a5811ff0aea212503f58b45258ba4f66b06420f87c350946aead39db6fdb7
EXPECTED_NAVDP_SHA=3bb3ad4ab241e857bb57a4021cc6aab76d5263e81fbf80298d579053ef011947
EXPECTED_LINGBOT_SHA=832bc82cbae0bc9bbe946ef5ee1f7226abd8c0e183ccf8beddbb3d133576f409
EXPECTED_ROLE_RECEIPT_SHA=0a255ca09d601cc6b8a49a5b1ca9c77ef760c8a44fbfd2b9894ee21c5844dc5d

fail() { echo "ABORT: $*" >&2; exit 2; }

for required in \
  "${MEMNAV_PY}" "${HAB_PY}" "${MEMNAV_CKPT}" "${NAVDP_CKPT}" \
  "${LINGBOT_WEIGHTS}" "${LIGHTGLUE_REPO}" "${DEPENDENCY_ROOT}" \
  "${ROOT}/MemNavData/eval_3leg_habitat.py" \
  "${ROOT}/MemNavData/audit_certified_mixed_role_safety_gate.py" \
  "${ROOT}/MemNavData/THREE_LEG_SCENE_ROLE_RECEIPT_20260813.json"; do
  test -r "${required}" || fail "missing input ${required}"
done
[[ "$(sha256sum "${MEMNAV_CKPT}" | awk '{print $1}')" == \
  "${EXPECTED_MEMNAV_SHA}" ]] || fail "MemNav checkpoint changed"
[[ "$(sha256sum "${NAVDP_CKPT}" | awk '{print $1}')" == \
  "${EXPECTED_NAVDP_SHA}" ]] || fail "NavDP checkpoint changed"
[[ "$(sha256sum "${LINGBOT_WEIGHTS}" | awk '{print $1}')" == \
  "${EXPECTED_LINGBOT_SHA}" ]] || fail "LingBot weights changed"
[[ "$(sha256sum "${ROOT}/MemNavData/THREE_LEG_SCENE_ROLE_RECEIPT_20260813.json" | awk '{print $1}')" == \
  "${EXPECTED_ROLE_RECEIPT_SHA}" ]] || fail "scene-role receipt changed"
[[ "${MAX_STEPS}" =~ ^[1-9][0-9]*$ ]] || fail "MAX_STEPS must be positive"
[[ "${AUDIT_ALLOW_NO_POSITIVE_CONTROL}" =~ ^[01]$ ]] || \
  fail "AUDIT_ALLOW_NO_POSITIVE_CONTROL must be 0 or 1"

scene_names=(e9zR4mvMWw7 gxdoqLR6rwA dhjEzFoUFzH gTV8FGcVJC9)
episode_names=(episode_0000 episode_0000 episode_0000 episode_0000)
expected_meta_hashes=(
  6cffffd60a0d5901c03b64c5759aac30a4be265ed89b1984c1220483dfd4a503
  89085dc1077de4b876b43b2f457671cca747a9ad1b4a20caab5bc8f7da93cca5
  5a807c2b6a16c4d39d194a80ec8a7921924e65a061eb9ecc2587a25afdd286ad
  92cb3b9cff328b5374b319fa980cf7f78a78c54d6732eee35d367d3255c71648
)
IFS=',' read -r -a selected_indices <<<"${SCENE_INDICES}"
[[ "${#selected_indices[@]}" -gt 0 ]] || fail "SCENE_INDICES is empty"
for index in "${selected_indices[@]}"; do
  [[ "${index}" =~ ^[0-3]$ ]] || fail "SCENE_INDICES must contain only 0..3"
  scene=${scene_names[${index}]}
  episode=${episode_names[${index}]}
  metadata=${BENCH_ROOT}/${scene}/${episode}/meta/gen_meta.json
  test -r "${metadata}" || fail "missing strict-v4 metadata ${metadata}"
  [[ "$(sha256sum "${metadata}" | awk '{print $1}')" == \
    "${expected_meta_hashes[${index}]}" ]] || fail "metadata changed for ${scene}"
done
for port in "${MEMNAV_PORT}" "${NAVDP_PORT}"; do
  if ss -ltn | awk '{print $4}' | grep -Eq "(^|:)${port}$"; then
    fail "port ${port} is already in use"
  fi
done

mkdir -p "${OUT_ROOT}/logs" "${OUT_ROOT}/buffer" "${OUT_ROOT}/scenes"
runtime_root=$(mktemp -d /tmp/certified_mixed_role_gate.XXXXXX)
receipt_tmp=$(mktemp)
contract_tmp=$(mktemp)
MEMNAV_PID=
NAVDP_PID=
cleanup() {
  for pid in "${NAVDP_PID}" "${MEMNAV_PID}"; do
    if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
      kill "${pid}" 2>/dev/null || true
      wait "${pid}" 2>/dev/null || true
    fi
  done
  rm -f -- "${receipt_tmp}" "${contract_tmp}"
  rm -rf -- "${runtime_root}"
}
trap cleanup EXIT INT TERM

sha256sum \
  "${ROOT}/MemNavData/run_certified_mixed_role_safety_gate_local.sh" \
  "${ROOT}/MemNavData/CERTIFIED_MIXED_ROLE_SAFETY_GATE_PROTOCOL_20260813.md" \
  "${ROOT}/MemNavData/audit_certified_mixed_role_safety_gate.py" \
  "${ROOT}/MemNavData/eval_3leg_habitat.py" \
  "${ROOT}/MemNavData/eval_2leg_habitat.py" \
  "${ROOT}/MemNavData/multigoal_policy_contract.py" \
  "${ROOT}/MemNavData/revisit_bearing_adapter.py" \
  "${ROOT}/MemNavData/certified_relocalization_runtime.py" \
  "${ROOT}/MemNavData/lingbot_pnp_localization.py" \
  "${ROOT}/NavDP/baselines/memnav/memnav_server.py" \
  "${ROOT}/NavDP/baselines/memnav/policy_agent.py" \
  "${ROOT}/NavDP/baselines/navdp/navdp_server.py" \
  "${ROOT}/NavDP/baselines/navdp/policy_agent.py" \
  "${ROOT}/MemNavData/THREE_LEG_SCENE_ROLE_RECEIPT_20260813.json" \
  "${MEMNAV_CKPT}" "${NAVDP_CKPT}" "${LINGBOT_WEIGHTS}" \
  > "${receipt_tmp}"
if [[ -e "${OUT_ROOT}/source_inputs.sha256" ]]; then
  cmp --silent "${receipt_tmp}" "${OUT_ROOT}/source_inputs.sha256" || \
    fail "source inputs changed during resume"
else
  mv "${receipt_tmp}" "${OUT_ROOT}/source_inputs.sha256"
  receipt_tmp=$(mktemp)
fi

"${HAB_PY}" - "${SCENE_INDICES}" "${MAX_STEPS}" \
  "${AUDIT_ALLOW_NO_POSITIVE_CONTROL}" \
  "${EXPECTED_MEMNAV_SHA}" "${EXPECTED_NAVDP_SHA}" "${EXPECTED_LINGBOT_SHA}" \
  > "${contract_tmp}" <<'PY'
import json, sys
indices=[int(value) for value in sys.argv[1].split(',')]
scenes=['e9zR4mvMWw7','gxdoqLR6rwA','dhjEzFoUFzH','gTV8FGcVJC9']
orders={0:['certified','known_c_reference'],
        1:['known_c_reference','certified'],
        2:['certified','known_c_reference'],
        3:['known_c_reference','certified']}
print(json.dumps({
  'protocol':'certified_mixed_role_safety_gate_v1_20260813',
  'scope':'strict-v4 implementation/causal safety gate; not an SR estimate',
  'selected_indices':indices,
  'selected_scenes':[scenes[index] for index in indices],
  'arm_order':{scenes[index]:orders[index] for index in indices},
  'episode':'episode_0000',
  'seed':20260830,
  'max_steps_per_leg':int(sys.argv[2]),
  'audit_allow_no_positive_control':bool(int(sys.argv[3])),
  'exec_horizon':8,
  'trajectory_selector':'server',
  'deterministic_plan_seeds':True,
  'blind_data_read':False,
  'checkpoint_sha256':{
    'memnav':sys.argv[4], 'navdp':sys.argv[5], 'lingbot':sys.argv[6]},
}, indent=2, sort_keys=True))
PY
if [[ -e "${OUT_ROOT}/run_contract.json" ]]; then
  cmp --silent "${contract_tmp}" "${OUT_ROOT}/run_contract.json" || \
    fail "run contract changed during resume"
else
  mv "${contract_tmp}" "${OUT_ROOT}/run_contract.json"
  contract_tmp=$(mktemp)
fi

"${HAB_PY}" -m py_compile \
  "${ROOT}/MemNavData/eval_2leg_habitat.py" \
  "${ROOT}/MemNavData/eval_3leg_habitat.py" \
  "${ROOT}/MemNavData/multigoal_policy_contract.py" \
  "${ROOT}/MemNavData/audit_certified_mixed_role_safety_gate.py"
env PYTHONPATH="${ROOT}:${ROOT}/MemNavData${PYTHONPATH:+:${PYTHONPATH}}" \
  "${HAB_PY}" -m unittest \
  MemNavData.test_multigoal_policy_contract \
  MemNavData.test_revisit_bearing_adapter \
  MemNavData.test_certified_relocalization_runtime \
  MemNavData.test_audit_certified_mixed_role_safety_gate

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
  scene_root=${OUT_ROOT}/scenes/$(printf '%02d' "${index}")_${scene}
  mkdir -p "${scene_root}/logs"
  case "${index}" in
    0|2) arm_order=(certified known_c_reference) ;;
    1|3) arm_order=(known_c_reference certified) ;;
  esac
  "${HAB_PY}" - "${scene_root}/arm_order.json" "${arm_order[@]}" <<'PY'
import json, sys
open(sys.argv[1], 'w').write(json.dumps(sys.argv[2:], indent=2) + '\n')
PY
  common=(
    --episode_root "${episode_root}"
    --episode_ids "${episode}"
    --scene "${scene_file}"
    --host 127.0.0.1
    --port "${MEMNAV_PORT}"
    --novel_port "${NAVDP_PORT}"
    --server_backend hybrid_pose
    --success_dist 1.0
    --max_steps "${MAX_STEPS}"
    --exec_horizon 8
    --trajectory_selector server
    --trajectory_selector_scope all
    --navdp_goal_switch_reset carry
    --leg1_mode policy
    --leg1_goal_source own
    --seed 20260830
    --terminal_uturn off
    --terminal_visual_refine off
    --deterministic_plan_seeds
    --retrieval_override off
    --double_revisit_c_history all_prior
    --certified_cdec_rescue off
    --certified_stagnation_graph off
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
      known_c_reference)
        env PYTHONPATH="${hab_pythonpath}" "${HAB_PY}" -u \
          "${ROOT}/MemNavData/eval_3leg_habitat.py" \
          "${common[@]}" --out "${arm_root}" \
          --hybrid_route phase --revisit_controller navdp_mixed \
          --revisit_adapter legacy_metric \
          > "${scene_root}/logs/eval_${arm}.log" 2>&1
        ;;
      certified)
        env PYTHONPATH="${hab_pythonpath}" "${HAB_PY}" -u \
          "${ROOT}/MemNavData/eval_3leg_habitat.py" \
          "${common[@]}" --out "${arm_root}" \
          --hybrid_route certified_relocalization \
          --revisit_controller navdp_mixed \
          --revisit_adapter verified_bearing_v1 \
          > "${scene_root}/logs/eval_${arm}.log" 2>&1
        ;;
      *) fail "unknown arm ${arm}" ;;
    esac
    "${HAB_PY}" - "${arm_root}/summary.json" "${arm}" <<'PY'
import json, sys
s=json.load(open(sys.argv[1]))
assert s['episodes'] == 1
assert s['contract_valid_episodes'] == 1
assert s['multigoal_contract'] == 'multileg_v4_role_paired_20260812'
assert s['role_labels'] == {
    'A':'initial_imagegoal', 'B':'novel', 'C':'revisit'}
if sys.argv[2] == 'known_c_reference':
    assert s['hybrid_route'] == 'phase'
    assert s['policy_backends'] == {
        'A':'navdp', 'B':'navdp', 'C':'navdp_mix'}
else:
    assert s['hybrid_route'] == 'certified_relocalization'
    assert s['policy_backends'] == {
        'A':'navdp_auto', 'B':'navdp_auto', 'C':'navdp_auto'}
PY
  done
done

audit_args=(--run-root "${OUT_ROOT}" --out "${OUT_ROOT}/audit.json")
if [[ "${AUDIT_ALLOW_NO_POSITIVE_CONTROL}" -eq 1 ]]; then
  audit_args+=(--allow-no-positive-control)
fi
env PYTHONPATH="${ROOT}" "${HAB_PY}" \
  "${ROOT}/MemNavData/audit_certified_mixed_role_safety_gate.py" \
  "${audit_args[@]}" > "${OUT_ROOT}/audit.stdout.json"
echo "[complete] mixed-role safety gate passed; root=${OUT_ROOT}"
