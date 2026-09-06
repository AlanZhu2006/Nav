#!/usr/bin/env bash
# One consumed two-leg Revisit query: fixed 2.5 m CEC vs bounded metric CEC.
# Reuses one immutable online Goal-A trace, so only the certified residual
# radius changes.  This is an engineering/mechanism test, not an SR estimate.
set -euo pipefail
umask 0022

ROOT=${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
EPISODE_ROOT=${EPISODE_ROOT:?set EPISODE_ROOT}
EPISODE_ID=${EPISODE_ID:-episode_0001}
SHARED_TRACE_ROOT=${SHARED_TRACE_ROOT:?set SHARED_TRACE_ROOT}
SCENE_FILE=${SCENE_FILE:?set SCENE_FILE}
OUT_ROOT=${OUT_ROOT:?set OUT_ROOT}
MAX_STEPS=${MAX_STEPS:-120}
MEMNAV_PORT=${MEMNAV_PORT:-22650}
NAVDP_PORT=${NAVDP_PORT:-22651}

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
[[ "${MAX_STEPS}" =~ ^[1-9][0-9]*$ ]] || fail "MAX_STEPS must be positive"
[[ ! -e "${OUT_ROOT}" ]] || fail "output already exists: ${OUT_ROOT}"
for path in \
  "${EPISODE_ROOT}/${EPISODE_ID}/meta/gen_meta.json" \
  "${EPISODE_ROOT}/${EPISODE_ID}/data/chunk-000/episode_000000.parquet" \
  "${EPISODE_ROOT}/${EPISODE_ID}/goal_image.jpg" \
  "${SHARED_TRACE_ROOT}/${EPISODE_ID}_leg1_trace.json" \
  "${SCENE_FILE}" "${MEMNAV_PY}" "${HAB_PY}" "${MEMNAV_CKPT}" \
  "${NAVDP_CKPT}" "${LINGBOT_WEIGHTS}" "${LIGHTGLUE_REPO}" \
  "${DEPENDENCY_ROOT}" "${INTERNNAV_ROOT}"; do
  [[ -e "${path}" ]] || fail "missing ${path}"
done
for port in "${MEMNAV_PORT}" "${NAVDP_PORT}"; do
  if ss -ltn | awk '{print $4}' | grep -Eq "(^|:)${port}$"; then
    fail "port ${port} is in use"
  fi
done

mkdir -p "${OUT_ROOT}/logs" "${OUT_ROOT}/buffer"
runtime_root=$(mktemp -d /tmp/cec_bounded_metric_2leg.XXXXXX)
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
  --episode_root "${EPISODE_ROOT}" --episode_ids "${EPISODE_ID}"
  --scene "${SCENE_FILE}" --host 127.0.0.1
  --port "${MEMNAV_PORT}" --novel_port "${NAVDP_PORT}"
  --server_backend hybrid_pose --hybrid_route certified_relocalization
  --navdp_depth_source monocular_sidecar --success_dist 1.0
  --max_steps "${MAX_STEPS}" --exec_horizon 8
  --trajectory_selector server --trajectory_selector_scope all
  --navdp_goal_switch_reset carry --leg1_goal_source own
  --leg1_mode shared_trace --shared_leg1_trace_root "${SHARED_TRACE_ROOT}"
  --seed 2026081602 --terminal_uturn off --terminal_visual_refine off
  --deterministic_plan_seeds --retrieval_override off
  --certified_cdec_rescue off --certified_stagnation_graph off
  --revisit_controller navdp_mixed
)
for adapter in verified_bearing_v1 verified_bounded_metric_v1; do
  arm=${adapter%_v1}
  mkdir -p "${OUT_ROOT}/${arm}"
  env PYTHONPATH="${hab_path}" "${HAB_PY}" -u \
    "${ROOT}/MemNavData/eval_2leg_habitat.py" "${common[@]}" \
      --revisit_adapter "${adapter}" --out "${OUT_ROOT}/${arm}" \
      >"${OUT_ROOT}/logs/eval_${arm}.log" 2>&1
done

env PYTHONPATH="${ROOT}/MemNavData:${ROOT}" "${MEMNAV_PY}" - \
  "${OUT_ROOT}" "${EPISODE_ID}" "${MAX_STEPS}" <<'PY'
import csv,json,sys
from pathlib import Path

root=Path(sys.argv[1])
arms={}
for arm in ('verified_bearing','verified_bounded_metric'):
    plan_path=root/arm/f'{sys.argv[2]}_plans.json'
    payload=json.loads(plan_path.read_text())
    plans=payload['legB']
    accepted=[p for p in plans if p.get('revisit_adapter_takeover') is True]
    with (root/arm/'metric.csv').open(newline='') as handle:
        rows=list(csv.DictReader(handle))
    if len(rows) != 1 or not accepted:
        raise SystemExit(f'{arm}: incomplete paired result')
    arms[arm]={
        'reached':float(rows[0]['reached_B']) > 0.5,
        'steps':int(float(rows[0]['steps_B'])),
        'final_goal_dist_m':float(rows[0]['final_dist_B']),
        'path_len_m':float(rows[0]['len_B']),
        'takeovers':len(accepted),
        'controller_radii_m':[p.get(
            'memory_controller_pointgoal_distance_m') for p in accepted],
        'raw_metric_radii_m':[p.get(
            'memory_unbounded_pointgoal_distance_m') for p in accepted],
        'scale_m_per_raw':accepted[0].get('memory_metric_scale_m_per_raw'),
    }
fixed=arms['verified_bearing']['controller_radii_m']
if any(abs(float(value)-2.5) > 1e-9 for value in fixed):
    raise SystemExit('canonical arm changed its fixed 2.5 m contract')
receipt={
    'schema':'cec_bounded_metric_2leg_local_v1_20260901',
    'scope':'one consumed HM3D mechanism test; no population SR claim',
    'episode':sys.argv[2], 'max_steps':int(sys.argv[3]),
    'same_server_pair':True, 'shared_goal_a_trace':True, 'arms':arms,
}
(root/'result.json').write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n')
print(json.dumps(receipt,indent=2,sort_keys=True))
PY

echo "[complete] ${OUT_ROOT}"
