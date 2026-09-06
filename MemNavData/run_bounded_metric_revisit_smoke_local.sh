#!/usr/bin/env bash
# One consumed MP3D Revisit query: canonical fixed radius vs bounded first40 metric.
set -euo pipefail
umask 0022

ROOT=${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
BENCH_ROOT=${BENCH_ROOT:-${ROOT}/.diagnostics/shared_online_role_pair_heading30_v3_smoke_20260814}
OUT_ROOT=${OUT_ROOT:-${ROOT}/.diagnostics/cec_bounded_metric_smoke_20260831_attempt1}
HISTORY_INDEX=${HISTORY_INDEX:-0}
MAX_STEPS=${MAX_STEPS:-8}
MEMNAV_PORT=${MEMNAV_PORT:-22640}
NAVDP_PORT=${NAVDP_PORT:-22641}
MEMNAV_PY=${MEMNAV_PY:-/home/asus/miniconda3/envs/memnav/bin/python}
HAB_PY=${HAB_PY:-/home/asus/miniconda3/envs/habitat/bin/python}
MEMNAV_CKPT=${MEMNAV_CKPT:-/home/asus/Research/Nav-axis-uturn/.diagnostics/unseen_scene_eval_20260803/checkpoints/gatecurr600.memnav.ckpt}
NAVDP_CKPT=${NAVDP_CKPT:-/home/asus/Research/Nav/NavDP/baselines/navdp/checkpoints/navdp_checkpoint.ckpt}
LINGBOT_REPO=${LINGBOT_REPO:-/home/asus/Research/Nav/NavDP/baselines/memnav/lingbot-map}
LINGBOT_WEIGHTS=${LINGBOT_WEIGHTS:-${LINGBOT_REPO}/weights/lingbot-map-long.pt}
LIGHTGLUE_REPO=${LIGHTGLUE_REPO:-${ROOT}/.diagnostics/dependencies/LightGlue}
DEPENDENCY_ROOT=${DEPENDENCY_ROOT:-${ROOT}/.diagnostics/dependencies/python}
INTERNNAV_ROOT=${INTERNNAV_ROOT:-${ROOT}/InternNav}

fail() { echo "ABORT: $*" >&2; exit 2; }
[[ "${HISTORY_INDEX}" =~ ^[0-9]+$ ]] || fail "bad HISTORY_INDEX"
[[ "${MAX_STEPS}" =~ ^[1-9][0-9]*$ ]] || fail "bad MAX_STEPS"
[[ ! -e "${OUT_ROOT}" ]] || fail "output exists: ${OUT_ROOT}"
for path in \
  "${BENCH_ROOT}/manifest.json" "${MEMNAV_PY}" "${HAB_PY}" \
  "${MEMNAV_CKPT}" "${NAVDP_CKPT}" "${LINGBOT_WEIGHTS}" \
  "${LIGHTGLUE_REPO}" "${DEPENDENCY_ROOT}" "${INTERNNAV_ROOT}"; do
  [[ -e "${path}" ]] || fail "missing ${path}"
done
for port in "${MEMNAV_PORT}" "${NAVDP_PORT}"; do
  if ss -ltn | awk '{print $4}' | grep -Eq "(^|:)${port}$"; then
    fail "port ${port} is in use"
  fi
done

mkdir -p "${OUT_ROOT}/logs" "${OUT_ROOT}/buffer"
runtime_root=$(mktemp -d /tmp/cec_bounded_metric_smoke.XXXXXX)
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

readarray -t identity < <("${HAB_PY}" - "${BENCH_ROOT}/manifest.json" \
  "${HISTORY_INDEX}" <<'PY'
import json,sys
row=json.load(open(sys.argv[1]))['episodes'][int(sys.argv[2])]
receipt=json.load(open(row['online_a_episode'] + '/receipt.json'))
print(row['scene'])
print(row['episode'])
print(receipt['source_asset'])
PY
)
[[ "${#identity[@]}" -eq 3 ]] || fail "failed to resolve history"
scene=${identity[0]}
episode=${identity[1]}
scene_file=${identity[2]}
[[ -r "${scene_file}" ]] || fail "missing scene ${scene_file}"

hab_site=$(${HAB_PY} -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
hab_path=${ROOT}:${ROOT}/MemNavData:${hab_site}/pip/_vendor
server_path=${ROOT}:${DEPENDENCY_ROOT}:${LIGHTGLUE_REPO}:${INTERNNAV_ROOT}/src/diffusion-policy

PYTHONPATH="${ROOT}/MemNavData:${ROOT}" "${MEMNAV_PY}" -m pytest -q \
  -p no:cacheprovider \
  "${ROOT}/MemNavData/test_revisit_bearing_adapter.py" \
  "${ROOT}/MemNavData/test_first40_local_pose_scale.py"

(
  cd "${runtime_root}"
  exec env PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    PYTHONPATH="${ROOT}/NavDP/baselines/memnav:${server_path}" \
    LINGBOT_REPO="${LINGBOT_REPO}" LINGBOT_WEIGHTS="${LINGBOT_WEIGHTS}" \
    MEMNAV_WINDOW=32 MEMNAV_NUM_SCALE=8 MEMNAV_MAX_FRAME_NUM=2048 \
    MEMNAV_GROUND_SCALE_MAX=6.0 MEMNAV_GATE_FUSION=complementary \
    MEMNAV_AUX_POSE_CALIBRATION=empirical MEMNAV_COLLISION_SELECT=1 \
    MEMNAV_REPORT_TO=none "${MEMNAV_PY}" -u \
    "${ROOT}/NavDP/baselines/memnav/memnav_server.py" \
      --port "${MEMNAV_PORT}" --checkpoint "${MEMNAV_CKPT}" \
      --internnav_root "${INTERNNAV_ROOT}" --num_samples 16 \
      --exclude_recent 32 --retrieval raw \
      --retrieval_candidate_top_k 32 --retrieval_candidate_min_gap 16 \
      --graph_subgoal_spacing_m 0.0 --graph_subgoal_arrival_m 0.60 \
      --flow_gate auto --buffer_root "${OUT_ROOT}/buffer" \
      --certified_relocalization --lightglue_repo "${LIGHTGLUE_REPO}" \
      --lightglue_dependency_root "${DEPENDENCY_ROOT}" \
      --lightglue_max_keypoints 2048
) >"${OUT_ROOT}/logs/server_memnav.log" 2>&1 &
MEMNAV_PID=$!
(
  cd "${runtime_root}"
  exec env NAVDP_DISABLE_VIDEO=1 PYTHONUNBUFFERED=1 \
    PYTHONPATH="${ROOT}/NavDP/baselines/navdp:${ROOT}" \
    "${MEMNAV_PY}" -u "${ROOT}/NavDP/baselines/navdp/navdp_server.py" \
      --port "${NAVDP_PORT}" --checkpoint "${NAVDP_CKPT}" \
      --depth_source metric_request --allow_depth_source_override \
      --monocular_depth_url \
        "http://127.0.0.1:${MEMNAV_PORT}/monocular_depth_query" \
      --require_monocular_depth_transaction
) >"${OUT_ROOT}/logs/server_navdp.log" 2>&1 &
NAVDP_PID=$!

for spec in \
  "memnav:${MEMNAV_PID}:${MEMNAV_PORT}" \
  "navdp:${NAVDP_PID}:${NAVDP_PORT}"; do
  IFS=: read -r label pid port <<<"${spec}"
  ready=0
  for _ in $(seq 1 240); do
    if ! kill -0 "${pid}" 2>/dev/null; then
      tail -n 120 "${OUT_ROOT}/logs/server_${label}.log" >&2
      fail "${label} exited during startup"
    fi
    if ss -ltn | awk '{print $4}' | grep -Eq "(^|:)${port}$"; then
      ready=1; break
    fi
    sleep 2
  done
  [[ "${ready}" -eq 1 ]] || fail "${label} startup timeout"
done

common=(
  --episode_root "${BENCH_ROOT}/${scene}"
  --episode_ids "${episode}"
  --scene "${scene_file}"
  --scene_identity "${scene}"
  --host 127.0.0.1 --port "${MEMNAV_PORT}" --novel_port "${NAVDP_PORT}"
  --server_backend hybrid_pose --hybrid_route certified_relocalization
  --navdp_depth_source monocular_sidecar --success_dist 1.0
  --max_steps "${MAX_STEPS}" --exec_horizon 8
  --trajectory_selector server --trajectory_selector_scope all
  --leg1_mode shared_trace --leg1_goal_source own --seed 0
  --terminal_uturn off --terminal_visual_refine off
  --deterministic_plan_seeds --retrieval_override off
  --certified_cdec_rescue off --certified_stagnation_graph off
  --revisit_controller navdp_mixed
  --role_pair_scope consumed_integration --role_pair_query_role revisit
)
for adapter in verified_bearing_v1 verified_bounded_metric_v1; do
  arm=${adapter%_v1}
  mkdir -p "${OUT_ROOT}/${arm}"
  env PYTHONPATH="${hab_path}" "${HAB_PY}" -u \
    "${ROOT}/MemNavData/eval_shared_online_role_pairs.py" \
      "${common[@]}" --revisit_adapter "${adapter}" \
      --out "${OUT_ROOT}/${arm}" \
      >"${OUT_ROOT}/logs/eval_${arm}.log" 2>&1
done

env PYTHONPATH="${ROOT}/MemNavData:${ROOT}" "${MEMNAV_PY}" - \
  "${OUT_ROOT}" "${scene}" "${episode}" "${MAX_STEPS}" <<'PY'
import hashlib,json,math,sys
from pathlib import Path
root=Path(sys.argv[1])
arms={}
for arm in ('verified_bearing','verified_bounded_metric'):
    files=list((root/arm).glob('*_plans.json'))
    if len(files) != 1:
        raise SystemExit(f'{arm}: expected one plans file, got {len(files)}')
    payload=json.loads(files[0].read_text())
    plans=payload['query_leg']
    accepted=[p for p in plans if p.get('revisit_adapter_takeover') is True]
    if not accepted:
        raise SystemExit(f'{arm}: smoke produced no certified takeover')
    first=accepted[0]
    arms[arm]={
        'plans':len(plans),
        'takeovers':len(accepted),
        'first_controller_distance_m':first.get(
            'memory_controller_pointgoal_distance_m'),
        'first_raw_metric_distance_m':first.get(
            'memory_unbounded_pointgoal_distance_m'),
        'first_bearing':first.get('memory_bearing_unit'),
        'first_anchor':first.get('router_selected_anchor'),
        'first_scale_m_per_raw':first.get('memory_metric_scale_m_per_raw'),
        'radius_cap_m':first.get('memory_pointgoal_radius_cap_m'),
    }
fixed=arms['verified_bearing']
bounded=arms['verified_bounded_metric']
if abs(float(fixed['first_controller_distance_m'])-2.5) > 1e-9:
    raise SystemExit('canonical fixed arm no longer emits 2.5 m')
if not 0 < float(bounded['first_controller_distance_m']) <= 2.5:
    raise SystemExit('bounded arm escaped (0,2.5] authority')
if bounded['first_scale_m_per_raw'] is None:
    raise SystemExit('bounded arm did not audit its frozen scale')
if fixed['first_bearing'] != bounded['first_bearing']:
    raise SystemExit('paired arms changed first certified bearing')
receipt={
    'schema':'cec_bounded_metric_local_smoke_v1_20260831',
    'scope':'consumed integration only; no SR claim',
    'scene':sys.argv[2], 'episode':sys.argv[3],
    'max_steps':int(sys.argv[4]), 'same_server_pair':True,
    'arms':arms,
}
encoded=(json.dumps(receipt,indent=2,sort_keys=True)+'\n').encode()
(root/'smoke_receipt.json').write_bytes(encoded)
(root/'smoke_receipt.json.sha256').write_text(
    hashlib.sha256(encoded).hexdigest()+'  smoke_receipt.json\n')
print(json.dumps(receipt,indent=2,sort_keys=True))
PY

echo "[complete] ${OUT_ROOT}"
