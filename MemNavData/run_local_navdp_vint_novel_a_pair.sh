#!/usr/bin/env bash
# One-episode exploratory comparison of two frozen native ImageGoal policies.
#
# This is deliberately a controller smoke, not a paper-level statistical
# comparison. Both arms share the same Habitat scene, generated episode,
# current/goal RGB bytes, success radius, step budget, and pure-pursuit
# trajectory executor. NavDP uses its native metric-depth request while ViNT
# is RGB-only; the receipt keeps that sensor difference explicit.

set -euo pipefail
umask 0022

ROOT=${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
SCENE=${SCENE:-gxdoqLR6rwA}
EPISODE=${EPISODE:-episode_0000}
EPISODE_ROOT=${EPISODE_ROOT:-${ROOT}/.diagnostics/revisit_fresh160_minimal_mirror_20260813/data/mp3d_2leg/${SCENE}}
SCENE_FILE=${SCENE_FILE:-/home/asus/Research/datasets/mp3d_20scene/assets/${SCENE}/${SCENE}.glb}
MAX_STEPS=${MAX_STEPS:-400}
EVAL_SEED=${EVAL_SEED:-20260803}

MEMNAV_PY=${MEMNAV_PY:-/home/asus/miniconda3/envs/memnav/bin/python}
HAB_PY=${HAB_PY:-/home/asus/miniconda3/envs/habitat/bin/python}
VINT_PY=${VINT_PY:-${ROOT}/.diagnostics/controller_portability_20260821/envs/vint/bin/python}
NAVDP_CKPT=${NAVDP_CKPT:-/home/asus/Research/Nav/NavDP/baselines/navdp/checkpoints/navdp_checkpoint.ckpt}
VINT_CKPT=${VINT_CKPT:-${ROOT}/.diagnostics/controller_portability_20260821/checkpoints/vint.pth}

NAVDP_PORT=${NAVDP_PORT:-22960}
VINT_PORT=${VINT_PORT:-22961}
PROXY_PORT=${PROXY_PORT:-22962}
RUN_ROOT=${RUN_ROOT:-${ROOT}/.diagnostics/navdp_vint_novel_a_pair_20260828/$(date -u +%Y%m%dT%H%M%SZ)}

fail() { printf 'ABORT: %s\n' "$*" >&2; exit 2; }

required=(
  "${MEMNAV_PY}"
  "${HAB_PY}"
  "${VINT_PY}"
  "${NAVDP_CKPT}"
  "${VINT_CKPT}"
  "${SCENE_FILE}"
  "${EPISODE_ROOT}/${EPISODE}/meta/gen_meta.json"
  "${ROOT}/MemNavData/eval_2leg_habitat.py"
  "${ROOT}/MemNavData/controller_portability_proxy.py"
)
for item in "${required[@]}"; do
  [[ -r "${item}" ]] || fail "missing input ${item}"
done
[[ "${MAX_STEPS}" =~ ^[1-9][0-9]*$ ]] || fail "MAX_STEPS must be positive"
[[ ! -e "${RUN_ROOT}" ]] || fail "output already exists: ${RUN_ROOT}"

for port in "${NAVDP_PORT}" "${VINT_PORT}" "${PROXY_PORT}"; do
  if ss -ltn | awk '{print $4}' | grep -Eq "(^|:)${port}$"; then
    fail "port ${port} is already in use"
  fi
done

mkdir -p "${RUN_ROOT}/logs" "${RUN_ROOT}/navdp" "${RUN_ROOT}/vint"
runtime_root=$(mktemp -d /tmp/navdp_vint_novel_a.XXXXXX)
navdp_pid=
vint_pid=
proxy_pid=

stop_process() {
  local pid=${1:-}
  if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
    kill "${pid}" 2>/dev/null || true
    wait "${pid}" 2>/dev/null || true
  fi
}

cleanup() {
  stop_process "${proxy_pid}"
  stop_process "${vint_pid}"
  stop_process "${navdp_pid}"
}
trap cleanup EXIT INT TERM

wait_for_port() {
  local label=$1 pid=$2 port=$3 log=$4
  local attempt
  for attempt in $(seq 1 240); do
    if ! kill -0 "${pid}" 2>/dev/null; then
      tail -80 "${log}" >&2 || true
      fail "${label} exited before opening port ${port}"
    fi
    if ss -ltn | awk '{print $4}' | grep -Eq "(^|:)${port}$"; then
      return 0
    fi
    sleep 1
  done
  tail -80 "${log}" >&2 || true
  fail "${label} did not open port ${port}"
}

common_eval=(
  --episode_root "${EPISODE_ROOT}"
  --episode_ids "${EPISODE}"
  --episodes 1
  --scene "${SCENE_FILE}"
  --scene_identity "${SCENE}"
  --stop_after_leg1
  --leg1_mode policy
  --success_dist 1.0
  --leg1_success_dist 1.0
  --max_steps "${MAX_STEPS}"
  --trajectory_selector server
  --seed "${EVAL_SEED}"
)

printf 'Starting frozen NavDP Novel-A arm\n'
(
  cd "${runtime_root}"
  exec env NAVDP_DISABLE_VIDEO=1 PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="${ROOT}" \
    "${MEMNAV_PY}" -u "${ROOT}/NavDP/baselines/navdp/navdp_server.py" \
      --port "${NAVDP_PORT}" --checkpoint "${NAVDP_CKPT}" \
      --depth_source metric_request
) >"${RUN_ROOT}/logs/navdp_server.log" 2>&1 &
navdp_pid=$!
wait_for_port navdp "${navdp_pid}" "${NAVDP_PORT}" \
  "${RUN_ROOT}/logs/navdp_server.log"
"${HAB_PY}" -u "${ROOT}/MemNavData/eval_2leg_habitat.py" \
  "${common_eval[@]}" --server_backend navdp --port "${NAVDP_PORT}" \
  --navdp_depth_source metric_request --out "${RUN_ROOT}/navdp" \
  >"${RUN_ROOT}/logs/navdp_eval.log" 2>&1
stop_process "${navdp_pid}"
navdp_pid=

printf 'Starting frozen ViNT Novel-A arm\n'
(
  cd "${ROOT}/NavDP/baselines/vint"
  exec env PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
    "${VINT_PY}" -u vint_server.py --port "${VINT_PORT}" \
      --robot_config configs/robot_config.yaml \
      --vint_config configs/vint.yaml --vint_checkpoint "${VINT_CKPT}"
) >"${RUN_ROOT}/logs/vint_server.log" 2>&1 &
vint_pid=$!
wait_for_port vint "${vint_pid}" "${VINT_PORT}" \
  "${RUN_ROOT}/logs/vint_server.log"
(
  cd "${runtime_root}"
  exec env PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="${ROOT}" \
    "${MEMNAV_PY}" -u "${ROOT}/MemNavData/controller_portability_proxy.py" \
      --controller vint --protocol native_imagegoal --depth-source none \
      --query-population any --reject-policy not_applicable \
      --repo-root "${ROOT}" --upstream-base "http://127.0.0.1:${VINT_PORT}" \
      --checkpoint "vint=${VINT_CKPT}" --host 127.0.0.1 \
      --port "${PROXY_PORT}"
) >"${RUN_ROOT}/logs/vint_proxy.log" 2>&1 &
proxy_pid=$!
wait_for_port vint_proxy "${proxy_pid}" "${PROXY_PORT}" \
  "${RUN_ROOT}/logs/vint_proxy.log"
"${HAB_PY}" -u "${ROOT}/MemNavData/eval_2leg_habitat.py" \
  "${common_eval[@]}" --server_backend rgb_imagegoal --port "${PROXY_PORT}" \
  --out "${RUN_ROOT}/vint" >"${RUN_ROOT}/logs/vint_eval.log" 2>&1

"${HAB_PY}" - "${RUN_ROOT}" "${SCENE}" "${EPISODE}" <<'PY'
import csv
import hashlib
import json
from pathlib import Path
import sys

root = Path(sys.argv[1])
scene, episode = sys.argv[2:]
arms = {
    "navdp": {"sensor": "RGB-D metric_request"},
    "vint": {"sensor": "RGB only"},
}
for name, receipt in arms.items():
    path = root / name / "metric.csv"
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 1 or rows[0]["episode"] != episode:
        raise RuntimeError(f"{name} did not produce exactly one paired row")
    row = rows[0]
    receipt.update({
        "reached_A": bool(float(row["reached_A"])),
        "spl_A": float(row["spl_A"]),
        "geodesic_A_m": float(row["geo_A"]),
        "path_A_m": float(row["len_A"]),
        "final_distance_A_m": float(row["final_dist_A"]),
        "steps_A": int(float(row["steps_A"])),
        "termination_reason_A": row["termination_reason_A"] or None,
        "metric_csv_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    })
payload = {
    "schema_version": "local_navdp_vint_novel_a_pair_v1_20260828",
    "scope": "single-episode exploratory controller smoke",
    "paper_level_statistical_result": False,
    "scene": scene,
    "episode": episode,
    "shared_start_goal_scoring_and_executor": True,
    "sensor_matched": False,
    "arms": arms,
}
summary = root / "summary.json"
summary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
print(json.dumps(payload, indent=2, sort_keys=True))
PY

printf 'Result root: %s\n' "${RUN_ROOT}"
