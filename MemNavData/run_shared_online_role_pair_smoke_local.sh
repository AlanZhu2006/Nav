#!/usr/bin/env bash
# Consumed-scene integration smoke for native/raw-direct/certified role pairs.

set -euo pipefail
umask 0022

ROOT=${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
BENCH_ROOT=${BENCH_ROOT:-${ROOT}/.diagnostics/shared_online_role_pair_heading30_v3_smoke_20260814}
OUT_ROOT=${OUT_ROOT:-${ROOT}/.diagnostics/shared_online_role_pair_closed_loop_smoke_v2_20260814}
SCENE_INDICES=${SCENE_INDICES:-0}
MAX_STEPS=${MAX_STEPS:-120}
RUN_SCOPE=${RUN_SCOPE:-consumed-scene integration only; no SR claim}
INCLUDE_LEARNED_PI3X=${INCLUDE_LEARNED_PI3X:-0}
MEMNAV_PORT=${MEMNAV_PORT:-21540}
NAVDP_PORT=${NAVDP_PORT:-21541}
MEMNAV_PY=${MEMNAV_PY:-/home/asus/miniconda3/envs/memnav/bin/python}
HAB_PY=${HAB_PY:-/home/asus/miniconda3/envs/habitat/bin/python}
MEMNAV_CKPT=${MEMNAV_CKPT:-/home/asus/Research/Nav-axis-uturn/.diagnostics/unseen_scene_eval_20260803/checkpoints/gatecurr600.memnav.ckpt}
NAVDP_CKPT=${NAVDP_CKPT:-/home/asus/Research/Nav/NavDP/baselines/navdp/checkpoints/navdp_checkpoint.ckpt}
LINGBOT_REPO=${LINGBOT_REPO:-/home/asus/Research/Nav/NavDP/baselines/memnav/lingbot-map}
LINGBOT_WEIGHTS=${LINGBOT_WEIGHTS:-${LINGBOT_REPO}/weights/lingbot-map-long.pt}
LIGHTGLUE_REPO=${LIGHTGLUE_REPO:-${ROOT}/.diagnostics/dependencies/LightGlue}
DEPENDENCY_ROOT=${DEPENDENCY_ROOT:-${ROOT}/.diagnostics/dependencies/python}
INTERNNAV_ROOT=${INTERNNAV_ROOT:-${ROOT}/InternNav}
PI3X_ROOT=${PI3X_ROOT:-}
PI3X_SNAPSHOT=${PI3X_SNAPSHOT:-}
PI3X_PROOF_MANIFEST=${PI3X_PROOF_MANIFEST:-}

EXPECTED_BENCH_SHA=${EXPECTED_BENCH_SHA:-123ddfcb047653d0fceed1be51aacba32a58b6f8e2f5656dbac47f672993de88}
EXPECTED_MEMNAV_SHA=9b7a5811ff0aea212503f58b45258ba4f66b06420f87c350946aead39db6fdb7
EXPECTED_NAVDP_SHA=3bb3ad4ab241e857bb57a4021cc6aab76d5263e81fbf80298d579053ef011947
EXPECTED_LINGBOT_SHA=832bc82cbae0bc9bbe946ef5ee1f7226abd8c0e183ccf8beddbb3d133576f409
EXPECTED_PI3X_MODEL_SHA=69972d6e1c4492cb4d737a84fe940e357087d81c52f5c9b7c160b49c1f41669a
EXPECTED_PI3X_PROOF_SHA=1a05aaa7cf75296cb68e32f9ea57fba6bcce2b9f57313a8cede05b7c7b0cffdd

fail() { echo "ABORT: $*" >&2; exit 2; }
for required in \
  "${BENCH_ROOT}/manifest.json" "${MEMNAV_PY}" "${HAB_PY}" \
  "${MEMNAV_CKPT}" "${NAVDP_CKPT}" "${LINGBOT_WEIGHTS}" \
  "${LIGHTGLUE_REPO}" "${DEPENDENCY_ROOT}" "${INTERNNAV_ROOT}" \
  "${ROOT}/MemNavData/eval_shared_online_role_pairs.py" \
  "${ROOT}/MemNavData/audit_shared_online_role_pairs.py" \
  "${ROOT}/MemNavData/audit_shared_online_role_pair_smoke.py" \
  "${ROOT}/MemNavData/bearing_diagnostics.py" \
  "${ROOT}/MemNavData/lingbot_pnp_localization.py" \
  "${ROOT}/MemNavData/lingbot_colored_registration.py" \
  "${ROOT}/MemNavData/goat_terminal_alignment.py" \
  "${ROOT}/NavDP/baselines/memnav/policy_agent.py" \
  "${ROOT}/NavDP/baselines/memnav/pose_alignment.py" \
  "${ROOT}/NavDP/baselines/memnav/reverse_memory_graph.py" \
  "${ROOT}/NavDP/baselines/memnav/router_candidates.py" \
  "${ROOT}/NavDP/baselines/navdp/policy_agent.py" \
  "${ROOT}/NavDP/baselines/navdp/policy_network.py" \
  "${ROOT}/NavDP/baselines/navdp/policy_backbone.py" \
  "${ROOT}/NavDP/baselines/navdp/depth_anything/depth_anything_v2/dpt.py" \
  "${ROOT}/NavDP/baselines/navdp/deterministic_seed.py"; do
  [[ -r "${required}" ]] || fail "missing input ${required}"
done
[[ "$(sha256sum "${BENCH_ROOT}/manifest.json" | awk '{print $1}')" == \
  "${EXPECTED_BENCH_SHA}" ]] || fail "benchmark manifest changed"
[[ "$(sha256sum "${MEMNAV_CKPT}" | awk '{print $1}')" == \
  "${EXPECTED_MEMNAV_SHA}" ]] || fail "MemNav checkpoint changed"
[[ "$(sha256sum "${NAVDP_CKPT}" | awk '{print $1}')" == \
  "${EXPECTED_NAVDP_SHA}" ]] || fail "NavDP checkpoint changed"
[[ "$(sha256sum "${LINGBOT_WEIGHTS}" | awk '{print $1}')" == \
  "${EXPECTED_LINGBOT_SHA}" ]] || fail "LingBot weights changed"
[[ "${MAX_STEPS}" =~ ^[1-9][0-9]*$ ]] || fail "MAX_STEPS must be positive"
[[ "${INCLUDE_LEARNED_PI3X}" == 0 || "${INCLUDE_LEARNED_PI3X}" == 1 ]] || \
  fail "INCLUDE_LEARNED_PI3X must be 0 or 1"
case "${RUN_SCOPE}" in
  "consumed-scene integration only; no SR claim"|\
  "Replica cross-dataset integration only; no SR claim") ;;
  *) fail "unsupported RUN_SCOPE" ;;
esac
[[ ! -e "${OUT_ROOT}" ]] || fail "output root already exists: ${OUT_ROOT}"
export PYTHONPYCACHEPREFIX=${OUT_ROOT}/pycache
pi3x_server_args=()
pi3x_source_inputs=()
if [[ "${INCLUDE_LEARNED_PI3X}" == 1 ]]; then
  for required in \
    "${ROOT}/MemNavData/pi3x_online_relocalizer.py" \
    "${ROOT}/MemNavData/pi3x_spatial_proof_runtime.py" \
    "${ROOT}/MemNavData/pi3x_spatial_reliability_model.py" \
    "${PI3X_ROOT}/pi3/models/pi3x.py" \
    "${PI3X_SNAPSHOT}/model.safetensors" \
    "${PI3X_PROOF_MANIFEST}"; do
    [[ -r "${required}" ]] || fail "missing learned input ${required}"
  done
  [[ "$(sha256sum "${PI3X_SNAPSHOT}/model.safetensors" | awk '{print $1}')" == \
    "${EXPECTED_PI3X_MODEL_SHA}" ]] || fail "Pi3X model changed"
  [[ "$(sha256sum "${PI3X_PROOF_MANIFEST}" | awk '{print $1}')" == \
    "${EXPECTED_PI3X_PROOF_SHA}" ]] || fail "Pi3X proof manifest changed"
  (
    cd "$(dirname "${PI3X_PROOF_MANIFEST}")"
    sha256sum -c OUTPUTS.sha256 >/dev/null
  ) || fail "Pi3X proof deployment changed"
  pi3x_server_args=(
    --pi3x_learned_relocalizer
    --pi3x_root "${PI3X_ROOT}"
    --pi3x_snapshot "${PI3X_SNAPSHOT}"
    --pi3x_model_sha256 "${EXPECTED_PI3X_MODEL_SHA}"
    --pi3x_spatial_proof_manifest "${PI3X_PROOF_MANIFEST}"
    --pi3x_inference_dtype auto
  )
  pi3x_source_inputs=(
    "${ROOT}/MemNavData/pi3x_online_relocalizer.py"
    "${ROOT}/MemNavData/pi3x_spatial_proof_runtime.py"
    "${ROOT}/MemNavData/pi3x_spatial_reliability_model.py"
    "${PI3X_ROOT}/pi3/models/pi3x.py"
    "${PI3X_SNAPSHOT}/model.safetensors"
    "${PI3X_PROOF_MANIFEST}"
  )
fi
for port in "${MEMNAV_PORT}" "${NAVDP_PORT}"; do
  if ss -ltn | awk '{print $4}' | grep -Eq "(^|:)${port}$"; then
    fail "port ${port} is already in use"
  fi
done

hab_site_packages=$("${HAB_PY}" -c \
  'import sysconfig; print(sysconfig.get_paths()["purelib"])')
hab_pythonpath=${ROOT}:${ROOT}/MemNavData:${hab_site_packages}/pip/_vendor${PYTHONPATH:+:${PYTHONPATH}}
hab_python() { env PYTHONPATH="${hab_pythonpath}" "${HAB_PY}" "$@"; }

hab_python -m py_compile \
  "${ROOT}/MemNavData/eval_shared_online_role_pairs.py" \
  "${ROOT}/MemNavData/audit_shared_online_role_pairs.py" \
  "${ROOT}/MemNavData/audit_shared_online_role_pair_smoke.py" \
  "${ROOT}/MemNavData/bearing_diagnostics.py" \
  "${ROOT}/MemNavData/shared_online_role_pair_contract.py"
env PYTHONPATH="${ROOT}/MemNavData${PYTHONPATH:+:${PYTHONPATH}}" \
  "${MEMNAV_PY}" -m pytest -q -p no:cacheprovider \
  "${ROOT}/MemNavData/test_shared_online_role_pair_contract.py"
hab_python "${ROOT}/MemNavData/test_build_shared_online_role_pairs.py"
hab_python "${ROOT}/MemNavData/audit_shared_online_role_pairs.py" \
  --root "${BENCH_ROOT}" >/dev/null

mkdir -p "${OUT_ROOT}/logs" "${OUT_ROOT}/buffer" "${OUT_ROOT}/scenes"
runtime_root=$(mktemp -d /tmp/shared_online_role_pair.XXXXXX)
MEMNAV_PID=
NAVDP_PID=
cleanup() {
  for pid in "${NAVDP_PID}" "${MEMNAV_PID}"; do
    if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
      kill "${pid}" 2>/dev/null || true
      wait "${pid}" 2>/dev/null || true
    fi
  done
  rm -rf -- "${runtime_root}"
}
trap cleanup EXIT INT TERM

source_inputs=( \
  "${ROOT}/MemNavData/run_shared_online_role_pair_smoke_local.sh" \
  "${ROOT}/MemNavData/SHARED_ONLINE_ROLE_PAIR_PROTOCOL_20260814.md" \
  "${ROOT}/MemNavData/eval_shared_online_role_pairs.py" \
  "${ROOT}/MemNavData/audit_shared_online_role_pairs.py" \
  "${ROOT}/MemNavData/shared_online_role_pair_contract.py" \
  "${ROOT}/MemNavData/eval_2leg_habitat.py" \
  "${ROOT}/MemNavData/bearing_diagnostics.py" \
  "${ROOT}/MemNavData/revisit_bearing_adapter.py" \
  "${ROOT}/MemNavData/certified_relocalization_runtime.py" \
  "${ROOT}/MemNavData/lingbot_pnp_localization.py" \
  "${ROOT}/MemNavData/lingbot_colored_registration.py" \
  "${ROOT}/MemNavData/goat_terminal_alignment.py" \
  "${ROOT}/NavDP/baselines/memnav/memnav_server.py" \
  "${ROOT}/NavDP/baselines/memnav/policy_agent.py" \
  "${ROOT}/NavDP/baselines/memnav/pose_alignment.py" \
  "${ROOT}/NavDP/baselines/memnav/reverse_memory_graph.py" \
  "${ROOT}/NavDP/baselines/memnav/router_candidates.py" \
  "${ROOT}/NavDP/baselines/navdp/navdp_server.py" \
  "${ROOT}/NavDP/baselines/navdp/policy_agent.py" \
  "${ROOT}/NavDP/baselines/navdp/policy_network.py" \
  "${ROOT}/NavDP/baselines/navdp/policy_backbone.py" \
  "${ROOT}/NavDP/baselines/navdp/deterministic_seed.py" \
  "${BENCH_ROOT}/manifest.json" \
  "${MEMNAV_CKPT}" "${NAVDP_CKPT}" "${LINGBOT_WEIGHTS}"
)
mapfile -d '' navdp_depth_sources < <(
  find "${ROOT}/NavDP/baselines/navdp/depth_anything/depth_anything_v2" \
    -type f -name '*.py' -print0 | sort -z
)
[[ "${#navdp_depth_sources[@]}" -gt 0 ]] || \
  fail "NavDP DepthAnythingV2 source closure is empty"
source_inputs+=("${navdp_depth_sources[@]}")
source_inputs+=("${pi3x_source_inputs[@]}")
sha256sum "${source_inputs[@]}" > "${OUT_ROOT}/source_inputs.sha256"

hab_python - "${BENCH_ROOT}/manifest.json" "${OUT_ROOT}/run_contract.json" \
  "${SCENE_INDICES}" "${MAX_STEPS}" "${EXPECTED_BENCH_SHA}" \
  "${RUN_SCOPE}" "${INCLUDE_LEARNED_PI3X}" \
  "${EXPECTED_PI3X_MODEL_SHA}" "${EXPECTED_PI3X_PROOF_SHA}" <<'PY'
import json, sys
manifest=json.load(open(sys.argv[1]))
indices=[int(value) for value in sys.argv[3].split(',')]
episodes=manifest['episodes']
if not indices or any(index < 0 or index >= len(episodes) for index in indices):
    raise SystemExit('invalid scene indices')
include_learned=bool(int(sys.argv[7]))
arms=(
  ['native','raw_fixed_bearing','geometry_fixed','certified',
   'learned_pi3x_spatial']
  if include_learned else
  ['native','raw_direct','raw_fixed_bearing','certified']
)
payload={
  'schema_version':'shared_online_role_pair_local_smoke_v1_20260814',
  'scope':sys.argv[6],
  'benchmark_manifest_sha256':sys.argv[5],
  'selected_indices':indices,
  'selected_identities':[
    [episodes[index]['scene'], episodes[index]['episode']] for index in indices],
  'arms':arms,
  'role_visible_to_runtime':False,
  'max_steps':int(sys.argv[4]),
  'exec_horizon':8,
  'deterministic_plan_seeds':True,
  'cdec_rescue':'off',
  'graph_rescue':'off',
  'blind_data_read':False,
  'learned_pi3x':({
    'model_sha256':sys.argv[8],
    'proof_manifest_sha256':sys.argv[9],
    'consensus':'2/4',
    'bridge_frames':16,
    'controller_residual_m':2.5,
  } if include_learned else None),
}
open(sys.argv[2],'x').write(json.dumps(payload,indent=2,sort_keys=True)+'\n')
PY

server_pythonpath=${ROOT}:${DEPENDENCY_ROOT}:${LIGHTGLUE_REPO}:${INTERNNAV_ROOT}/src/diffusion-policy${PYTHONPATH:+:${PYTHONPATH}}
memnav_pythonpath=${ROOT}/NavDP/baselines/memnav:${server_pythonpath}
navdp_pythonpath=${ROOT}/NavDP/baselines/navdp:${server_pythonpath}
mkdir -p "${runtime_root}/memnav" "${runtime_root}/navdp"
(
  cd "${runtime_root}/memnav"
  exec env PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    PYTHONPATH="${memnav_pythonpath}" \
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
      --lightglue_max_keypoints 2048 \
      "${pi3x_server_args[@]}"
) > "${OUT_ROOT}/logs/server_memnav.log" 2>&1 &
MEMNAV_PID=$!
(
  cd "${runtime_root}/navdp"
  exec env NAVDP_DISABLE_VIDEO=1 PYTHONUNBUFFERED=1 \
    PYTHONPATH="${navdp_pythonpath}" \
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
  [[ "${ready}" -eq 1 ]] || fail "${label} server did not bind port ${port}"
done

IFS=',' read -r -a selected_indices <<<"${SCENE_INDICES}"
for index in "${selected_indices[@]}"; do
  readarray -t identity < <(hab_python - "${BENCH_ROOT}/manifest.json" "${index}" <<'PY'
import json, sys
row=json.load(open(sys.argv[1]))['episodes'][int(sys.argv[2])]
receipt=json.load(open(row['online_a_episode'] + '/receipt.json'))
print(row['scene'])
print(row['episode'])
print(receipt['source_asset'])
print(row['online_a_episode'])
PY
  )
  [[ "${#identity[@]}" -eq 4 ]] || fail "identity reader failed"
  scene=${identity[0]}
  episode=${identity[1]}
  scene_file=${identity[2]}
  scene_root=${OUT_ROOT}/scenes/$(printf '%02d' "${index}")_${scene}
  mkdir -p "${scene_root}/logs"
  if [[ "${INCLUDE_LEARNED_PI3X}" == 1 ]]; then
    canonical_arms=(
      native raw_fixed_bearing geometry_fixed certified learned_pi3x_spatial
    )
  else
    canonical_arms=(native raw_direct raw_fixed_bearing certified)
  fi
  arm_offset=$((10#${index} % ${#canonical_arms[@]}))
  arm_order=()
  for ((arm_index=0; arm_index<${#canonical_arms[@]}; arm_index++)); do
    arm_order+=(
      "${canonical_arms[$(((arm_offset+arm_index)%${#canonical_arms[@]}))]}"
    )
  done
  common=(
    --episode_root "${BENCH_ROOT}/${scene}"
    --episode_ids "${episode}"
    --scene "${scene_file}"
    --scene_identity "${scene}"
    --host 127.0.0.1
    --success_dist 1.0
    --max_steps "${MAX_STEPS}"
    --exec_horizon 8
    --trajectory_selector server
    --trajectory_selector_scope all
    --leg1_mode shared_trace
    --leg1_goal_source own
    --terminal_uturn off
    --terminal_visual_refine off
    --deterministic_plan_seeds
    --retrieval_override off
    --certified_cdec_rescue off
    --certified_stagnation_graph off
    --revisit_controller navdp_mixed
  )
  for arm in "${arm_order[@]}"; do
    arm_root=${scene_root}/${arm}
    mkdir -p "${arm_root}"
    case "${arm}" in
      native)
        extra=(--port "${NAVDP_PORT}" --server_backend navdp \
          --hybrid_route phase --revisit_adapter legacy_metric)
        ;;
      raw_direct)
        extra=(--port "${MEMNAV_PORT}" --novel_port "${NAVDP_PORT}" \
          --server_backend hybrid_pose --hybrid_route phase \
          --revisit_adapter legacy_metric)
        ;;
      raw_fixed_bearing)
        extra=(--port "${MEMNAV_PORT}" --novel_port "${NAVDP_PORT}" \
          --server_backend hybrid_pose --hybrid_route phase \
          --revisit_adapter raw_fixed_bearing_v1)
        ;;
      geometry_fixed)
        extra=(--port "${MEMNAV_PORT}" --novel_port "${NAVDP_PORT}" \
          --server_backend hybrid_pose --hybrid_route memory_geometry \
          --revisit_adapter verified_bearing_v1 \
          --router_visual_floor 0.88 --router_min_matches 20 \
          --router_min_inliers 12 --router_min_inlier_ratio 0.50 \
          --router_confirm_plans 2 --router_verify_top_k 8)
        ;;
      certified)
        extra=(--port "${MEMNAV_PORT}" --novel_port "${NAVDP_PORT}" \
          --server_backend hybrid_pose --hybrid_route certified_relocalization \
          --revisit_adapter verified_bearing_v1)
        ;;
      learned_pi3x_spatial)
        extra=(--port "${MEMNAV_PORT}" --novel_port "${NAVDP_PORT}" \
          --server_backend hybrid_pose \
          --hybrid_route learned_pi3x_relocalization \
          --revisit_adapter verified_bearing_v1 \
          --expected_pi3x_model_sha256 "${EXPECTED_PI3X_MODEL_SHA}" \
          --expected_pi3x_proof_manifest_sha256 \
            "${EXPECTED_PI3X_PROOF_SHA}")
        ;;
      *) fail "unknown smoke arm ${arm}" ;;
    esac
    hab_python -u "${ROOT}/MemNavData/eval_shared_online_role_pairs.py" \
      "${common[@]}" "${extra[@]}" --out "${arm_root}" \
      > "${scene_root}/logs/eval_${arm}.log" 2>&1
  done
done

hab_python "${ROOT}/MemNavData/audit_shared_online_role_pair_smoke.py" \
  --root "${OUT_ROOT}" --out "${OUT_ROOT}/independent_audit.json" \
  > "${OUT_ROOT}/logs/independent_audit.log"
echo "[shared-online-role-pair-smoke] complete: ${OUT_ROOT}"
