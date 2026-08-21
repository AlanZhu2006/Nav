#!/usr/bin/env bash
# Evaluate one scene from the frozen fresh-episode Revisit confirmation.
# The caller owns one MemNav server and one NavDP server for this whole script.

set -euo pipefail
umask 0022

ROOT=${ROOT:?set immutable source ROOT}
RUN_ROOT=${RUN_ROOT:?set RUN_ROOT}
MANIFEST=${MANIFEST:?set frozen MANIFEST}
EXPECTED_MANIFEST_SHA=${EXPECTED_MANIFEST_SHA:?set EXPECTED_MANIFEST_SHA}
SOURCE_RECEIPT=${SOURCE_RECEIPT:?set SOURCE_RECEIPT}
EXPECTED_SOURCE_RECEIPT_SHA=${EXPECTED_SOURCE_RECEIPT_SHA:?set EXPECTED_SOURCE_RECEIPT_SHA}
SCENE_INDEX=${SCENE_INDEX:?set SCENE_INDEX}
HAB_PY=${HAB_PY:?set HAB_PY}
MEMNAV_PORT=${MEMNAV_PORT:?set MEMNAV_PORT}
NAVDP_PORT=${NAVDP_PORT:?set NAVDP_PORT}

fail() { echo "ABORT: $*" >&2; exit 2; }
[[ "${SCENE_INDEX}" =~ ^([0-9]|1[0-9])$ ]] || fail "invalid scene index"
[[ "$(sha256sum "${MANIFEST}" | awk '{print $1}')" == \
    "${EXPECTED_MANIFEST_SHA}" ]] || fail "data manifest SHA mismatch"
[[ "$(sha256sum "${SOURCE_RECEIPT}" | awk '{print $1}')" == \
    "${EXPECTED_SOURCE_RECEIPT_SHA}" ]] || fail "source receipt SHA mismatch"

HAB_SITE_PACKAGES=$("${HAB_PY}" -c \
  'import sysconfig; print(sysconfig.get_paths()["purelib"])')
HAB_PYTHONPATH=${HAB_SITE_PACKAGES}/pip/_vendor${PYTHONPATH:+:${PYTHONPATH}}
hab_python() { env PYTHONPATH="${ROOT}:${HAB_PYTHONPATH}" "${HAB_PY}" "$@"; }

readarray -t identity < <(hab_python - "${MANIFEST}" "${SCENE_INDEX}" <<'PY'
import hashlib, json, sys
from pathlib import Path

def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

manifest = json.load(open(sys.argv[1]))
index = int(sys.argv[2])
scene = manifest["scenes"][index]
episode_root = Path(manifest["paths"]["episode_root"]) / scene
asset = Path(manifest["paths"]["asset_root"]) / scene / f"{scene}.glb"
if sha(asset) != manifest["assets"][scene]["sha256"]:
    raise SystemExit("asset identity changed")
ids = []
for row in manifest["episodes"][scene]:
    episode = episode_root / row["episode"]
    paths = {
        "metadata": episode / "meta" / "gen_meta.json",
        "parquet": episode / "data" / "chunk-000" / "episode_000000.parquet",
        "goal": episode / "goal_image.jpg",
    }
    for kind, path in paths.items():
        if sha(path) != row["files"][kind]["sha256"]:
            raise SystemExit(f"episode identity changed: {scene}/{row['episode']}/{kind}")
    ids.append(row["episode"])
print(scene)
print(asset)
print(episode_root)
print(",".join(ids))
print(manifest["evaluation"]["base_seed"])
print(manifest["evaluation"]["max_steps_per_leg"])
PY
)
[[ "${#identity[@]}" -eq 6 ]] || fail "manifest identity reader failed"
scene=${identity[0]}
scene_file=${identity[1]}
episode_root=${identity[2]}
episode_csv=${identity[3]}
base_seed=${identity[4]}
max_steps=${identity[5]}

scene_root=${RUN_ROOT}/scenes/$(printf '%02d' "${SCENE_INDEX}")_${scene}
[[ ! -e "${scene_root}" ]] || fail "scene output already exists: ${scene_root}"
mkdir -p "${scene_root}/logs"

case $((10#${SCENE_INDEX} % 6)) in
  0) arm_order=(geometry_router known_revisit_direct native) ;;
  1) arm_order=(geometry_router native known_revisit_direct) ;;
  2) arm_order=(known_revisit_direct geometry_router native) ;;
  3) arm_order=(known_revisit_direct native geometry_router) ;;
  4) arm_order=(native geometry_router known_revisit_direct) ;;
  5) arm_order=(native known_revisit_direct geometry_router) ;;
esac

hab_python - "${scene_root}/scene_contract.json" "${scene}" \
  "${SCENE_INDEX}" "${EXPECTED_MANIFEST_SHA}" "${arm_order[@]}" <<'PY'
import json, sys
path, scene, index, manifest_sha, *order = sys.argv[1:]
with open(path, "x", encoding="utf-8") as handle:
    json.dump({
        "scene": scene,
        "scene_index": int(index),
        "manifest_sha256": manifest_sha,
        "arm_order": order,
    }, handle, indent=2, sort_keys=True)
    handle.write("\n")
PY

common=(
  --episode_root "${episode_root}"
  --episode_ids "${episode_csv}"
  --scene "${scene_file}"
  --host 127.0.0.1
  --success_dist 1.0
  --max_steps "${max_steps}"
  --exec_horizon 8
  --trajectory_selector server
  --trajectory_selector_scope all
  --navdp_goal_switch_reset carry
  --leg1_goal_source own
  --seed "${base_seed}"
  --terminal_uturn off
  --terminal_visual_refine off
  --deterministic_plan_seeds
)

trace_root=${scene_root}/trace_source
mkdir -p "${trace_root}"
hab_python -u "${ROOT}/MemNavData/eval_2leg_habitat.py" \
  "${common[@]}" \
  --port "${MEMNAV_PORT}" --novel_port "${NAVDP_PORT}" \
  --out "${trace_root}" \
  --server_backend hybrid_pose \
  --leg1_mode policy --write_leg1_trace --stop_after_leg1 \
  --hybrid_route phase --revisit_adapter legacy_metric \
  > "${scene_root}/logs/eval_trace_source.log" 2>&1

for arm in "${arm_order[@]}"; do
  arm_root=${scene_root}/${arm}
  mkdir -p "${arm_root}"
  case "${arm}" in
    geometry_router)
      hab_python -u "${ROOT}/MemNavData/eval_2leg_habitat.py" \
        "${common[@]}" \
        --port "${MEMNAV_PORT}" --novel_port "${NAVDP_PORT}" \
        --out "${arm_root}" --server_backend hybrid_pose \
        --leg1_mode shared_trace --shared_leg1_trace_root "${trace_root}" \
        --hybrid_route memory_geometry --revisit_adapter legacy_metric \
        --router_visual_floor 0.88 --router_min_matches 20 \
        --router_min_inliers 12 --router_min_inlier_ratio 0.50 \
        --router_confirm_plans 2 --router_verify_top_k 8 \
        > "${scene_root}/logs/eval_${arm}.log" 2>&1
      ;;
    known_revisit_direct)
      hab_python -u "${ROOT}/MemNavData/eval_2leg_habitat.py" \
        "${common[@]}" \
        --port "${MEMNAV_PORT}" --novel_port "${NAVDP_PORT}" \
        --out "${arm_root}" --server_backend hybrid_pose \
        --leg1_mode shared_trace --shared_leg1_trace_root "${trace_root}" \
        --hybrid_route phase --revisit_adapter legacy_metric \
        > "${scene_root}/logs/eval_${arm}.log" 2>&1
      ;;
    native)
      hab_python -u "${ROOT}/MemNavData/eval_2leg_habitat.py" \
        "${common[@]}" \
        --port "${NAVDP_PORT}" \
        --out "${arm_root}" --server_backend navdp \
        --leg1_mode shared_trace --shared_leg1_trace_root "${trace_root}" \
        --hybrid_route phase \
        > "${scene_root}/logs/eval_${arm}.log" 2>&1
      ;;
    *) fail "unknown arm ${arm}" ;;
  esac
done

hab_python - "${scene_root}" "${episode_csv}" <<'PY'
import csv, json, sys
from pathlib import Path
root = Path(sys.argv[1])
expected = sys.argv[2].split(",")
for arm in ("trace_source", "geometry_router", "known_revisit_direct", "native"):
    summary = json.loads((root / arm / "summary.json").read_text())
    if summary.get("episodes") != len(expected):
        raise SystemExit(f"{arm}: incomplete summary")
    with (root / arm / "metric.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if [row["episode"] for row in rows] != expected:
        raise SystemExit(f"{arm}: metric episode order/identity mismatch")
print(json.dumps({"scene": root.name, "status": "complete", "episodes": len(expected)}))
PY
