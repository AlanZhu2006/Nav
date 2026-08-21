#!/usr/bin/env bash
# Collect one real online NavDP Goal-A trace on Replica room_0.

set -euo pipefail
umask 0022

ROOT=${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
CROSS_ROOT=${CROSS_ROOT:-${ROOT}/.diagnostics/replica_cross_dataset_v1_20260814}
SCENE_IDENTITY=${SCENE_IDENTITY:-room_0}
SOURCE_ROOT=${SOURCE_ROOT:-${CROSS_ROOT}/source_episodes/${SCENE_IDENTITY}}
INPUT_RECEIPT=${INPUT_RECEIPT:-${CROSS_ROOT}/source_input_receipt.json}
EPISODE_IDS=${EPISODE_IDS:-episode_0000}
OUT_ROOT=${OUT_ROOT:-${CROSS_ROOT}/online_a_smoke_v2}
REPLICA_ROOT=${REPLICA_ROOT:-/home/asus/Research/Pi3/data/replica_v1_full_20260814}
STAGE=${STAGE:-${REPLICA_ROOT}/${SCENE_IDENTITY}/habitat/replica_stage.stage_config.json}
NAVMESH=${NAVMESH:-${REPLICA_ROOT}/${SCENE_IDENTITY}/habitat/mesh_semantic.navmesh}
NAVDP_PORT=${NAVDP_PORT:-21941}
RUN_SEED=${RUN_SEED:-2026081401}
MEMNAV_PY=${MEMNAV_PY:-/home/asus/miniconda3/envs/memnav/bin/python}
HAB_PY=${HAB_PY:-/home/asus/miniconda3/envs/habitat/bin/python}
NAVDP_CKPT=${NAVDP_CKPT:-/home/asus/Research/Nav/NavDP/baselines/navdp/checkpoints/navdp_checkpoint.ckpt}
DEPENDENCY_ROOT=${DEPENDENCY_ROOT:-${ROOT}/.diagnostics/dependencies/python}
INTERNNAV_ROOT=${INTERNNAV_ROOT:-${ROOT}/InternNav}

fail() { echo "ABORT: $*" >&2; exit 2; }
[[ ! -e "${OUT_ROOT}" ]] || fail "output exists: ${OUT_ROOT}"
! ss -ltn | awk '{print $4}' | grep -Eq "(^|:)${NAVDP_PORT}$" || \
  fail "port ${NAVDP_PORT} is already in use"
[[ "$(sha256sum "${NAVDP_CKPT}" | awk '{print $1}')" == \
  3bb3ad4ab241e857bb57a4021cc6aab76d5263e81fbf80298d579053ef011947 ]] || \
  fail "NavDP checkpoint changed"

"${MEMNAV_PY}" - "${INPUT_RECEIPT}" "${SOURCE_ROOT}" "${STAGE}" \
  "${NAVMESH}" "${EPISODE_IDS}" "${SCENE_IDENTITY}" <<'PY'
import hashlib,json,sys
from pathlib import Path
receipt=json.load(open(sys.argv[1])); root=Path(sys.argv[2])
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
requested=[value.strip() for value in sys.argv[5].split(",") if value.strip()]
if receipt["scene"] != sys.argv[6]:
    raise SystemExit("source receipt scene differs from requested scene")
episodes=receipt.get("episodes")
if episodes is None:
    episodes={receipt["episode"]: receipt["input_files"]}
if requested != list(episodes):
    raise SystemExit(
        f"requested episodes {requested} do not equal sealed receipt order "
        f"{list(episodes)}")
for episode,files in episodes.items():
    for relative,expected in files.items():
        path=root/episode/relative
        if not path.is_file() or sha(path)!=expected:
            raise SystemExit(f"source input changed: {episode}/{relative}")
if sha(sys.argv[3])!=receipt["source_stage_sha256"]:
    raise SystemExit("Replica stage changed")
if sha(sys.argv[4])!=receipt["source_navmesh_sha256"]:
    raise SystemExit("Replica navmesh changed")
PY

mkdir -p "${OUT_ROOT}/logs" "${OUT_ROOT}/native_a"
runtime_root=$(mktemp -d /tmp/replica_online_a.XXXXXX)
SERVER_PID=
cleanup() {
  if [[ -n "${SERVER_PID}" ]] && kill -0 "${SERVER_PID}" 2>/dev/null; then
    kill "${SERVER_PID}" 2>/dev/null || true
    wait "${SERVER_PID}" 2>/dev/null || true
  fi
  rm -rf -- "${runtime_root}"
}
trap cleanup EXIT INT TERM

server_pythonpath=${ROOT}:${DEPENDENCY_ROOT}:${INTERNNAV_ROOT}/src/diffusion-policy${PYTHONPATH:+:${PYTHONPATH}}
(
  cd "${runtime_root}"
  exec env NAVDP_DISABLE_VIDEO=1 PYTHONUNBUFFERED=1 \
    PYTHONPATH="${server_pythonpath}" "${MEMNAV_PY}" -u \
    "${ROOT}/NavDP/baselines/navdp/navdp_server.py" \
    --port "${NAVDP_PORT}" --checkpoint "${NAVDP_CKPT}"
) >"${OUT_ROOT}/logs/navdp.log" 2>&1 &
SERVER_PID=$!
ready=0
for _ in $(seq 1 240); do
  kill -0 "${SERVER_PID}" 2>/dev/null || {
    tail -n 120 "${OUT_ROOT}/logs/navdp.log" >&2
    fail "NavDP server exited"
  }
  if ss -ltn | awk '{print $4}' | grep -Eq "(^|:)${NAVDP_PORT}$"; then
    ready=1; break
  fi
  sleep 2
done
[[ "${ready}" -eq 1 ]] || fail "NavDP server did not bind"

hab_site_packages=$("${HAB_PY}" -c \
  'import sysconfig; print(sysconfig.get_paths()["purelib"])')
hab_pythonpath=${ROOT}:${ROOT}/MemNavData:${hab_site_packages}/pip/_vendor${PYTHONPATH:+:${PYTHONPATH}}
env PYTHONPATH="${hab_pythonpath}" "${HAB_PY}" -u \
  "${ROOT}/MemNavData/eval_2leg_habitat.py" \
  --episode_root "${SOURCE_ROOT}" --episode_ids "${EPISODE_IDS}" \
  --scene "${STAGE}" --scene_identity "${SCENE_IDENTITY}" --host 127.0.0.1 \
  --port "${NAVDP_PORT}" --out "${OUT_ROOT}/native_a" \
  --server_backend navdp --hybrid_route phase \
  --revisit_adapter legacy_metric --leg1_mode policy \
  --write_leg1_trace --stop_after_leg1 --leg1_goal_source own \
  --native_trace_navdp_checkpoint_sha256 \
  3bb3ad4ab241e857bb57a4021cc6aab76d5263e81fbf80298d579053ef011947 \
  --navdp_goal_switch_reset carry --success_dist 1.0 --max_steps 600 \
  --exec_horizon 8 --trajectory_selector server \
  --trajectory_selector_scope all --seed "${RUN_SEED}" \
  --terminal_uturn off --terminal_visual_refine off \
  --deterministic_plan_seeds >"${OUT_ROOT}/logs/eval.log" 2>&1

env PYTHONPATH="${hab_pythonpath}" "${HAB_PY}" - \
  "${OUT_ROOT}/native_a" "${INPUT_RECEIPT}" "${OUT_ROOT}/receipt.json" \
  "${EPISODE_IDS}" "${SCENE_IDENTITY}" <<'PY'
import csv,hashlib,json,sys
from pathlib import Path
root=Path(sys.argv[1]); source=Path(sys.argv[2]); out=Path(sys.argv[3])
summary=json.load(open(root/"summary.json"))
episode_ids=[value.strip() for value in sys.argv[4].split(",") if value.strip()]
scene_identity=sys.argv[5]
with open(root/"metric.csv",newline="") as handle: rows=list(csv.DictReader(handle))
assert len(rows)==len(episode_ids) and summary["episodes"]==len(episode_ids)
assert summary["leg1_policy_backend"]=="navdp" and summary["stop_after_leg1"]
traces=[]
for episode_id in episode_ids:
    trace_path=root/f"{episode_id}_leg1_trace.json"
    trace=json.load(open(trace_path))
    assert trace["source_scene"]==scene_identity and trace["source_backend"]=="navdp"
    traces.append({
      "episode":episode_id,
      "trace_sha256":hashlib.sha256(trace_path.read_bytes()).hexdigest(),
      "online_a_reached":bool(trace["reached"]),
      "online_a_steps":trace["steps"],
      "final_goal_dist_m":trace["final_goal_dist_m"],
    })
receipt={
 "schema_version":"replica_online_a_smoke_v1_20260814",
 "scope":f"Replica {scene_identity} online-A implementation smoke; no Revisit SR",
 "source_input_receipt_sha256":hashlib.sha256(source.read_bytes()).hexdigest(),
 "traces":traces,
 "all_online_a_reached":all(row["online_a_reached"] for row in traces),
 "policy":"frozen_native_navdp_imagegoal","scene_identity":scene_identity,
 "query_outcomes_read":False,
}
out.write_text(json.dumps(receipt,indent=2,sort_keys=True)+"\n")
print(json.dumps(receipt,sort_keys=True))
PY
sha256sum "${OUT_ROOT}/receipt.json"
