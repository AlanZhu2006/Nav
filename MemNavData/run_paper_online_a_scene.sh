#!/usr/bin/env bash
# Collect native online Goal-A traces for one frozen MP3D paper scene.

set -euo pipefail
umask 0022

ROOT=${ROOT:?set immutable source ROOT}
RUN_ROOT=${RUN_ROOT:?set RUN_ROOT}
MANIFEST=${MANIFEST:?set frozen MANIFEST}
EXPECTED_MANIFEST_SHA=${EXPECTED_MANIFEST_SHA:?set EXPECTED_MANIFEST_SHA}
SOURCE_RECEIPT=${SOURCE_RECEIPT:?set SOURCE_RECEIPT}
EXPECTED_SOURCE_RECEIPT_SHA=${EXPECTED_SOURCE_RECEIPT_SHA:?set receipt SHA}
SCENE_INDEX=${SCENE_INDEX:?set SCENE_INDEX}
HAB_PY=${HAB_PY:?set HAB_PY}
NAVDP_PORT=${NAVDP_PORT:?set NAVDP_PORT}
NAVDP_CHECKPOINT=${NAVDP_CHECKPOINT:?set NAVDP_CHECKPOINT}
FINAL14_POPULATION_MODE=${FINAL14_POPULATION_MODE:-0}

fail() { echo "ABORT: $*" >&2; exit 2; }
[[ "${SCENE_INDEX}" =~ ^([0-9]|1[0-5])$ ]] || fail "invalid scene index"
[[ "${FINAL14_POPULATION_MODE}" == 0 || "${FINAL14_POPULATION_MODE}" == 1 ]] || \
  fail "FINAL14_POPULATION_MODE must be 0 or 1"
[[ "$(sha256sum "${SOURCE_RECEIPT}" | awk '{print $1}')" == \
    "${EXPECTED_SOURCE_RECEIPT_SHA}" ]] || fail "source receipt changed"
(cd "${ROOT}" && sha256sum -c "${SOURCE_RECEIPT}") >/dev/null || \
  fail "immutable source bundle changed"
[[ "$(sha256sum "${MANIFEST}" | awk '{print $1}')" == \
    "${EXPECTED_MANIFEST_SHA}" ]] || fail "paper manifest changed"

HAB_SITE_PACKAGES=$("${HAB_PY}" -c \
  'import sysconfig; print(sysconfig.get_paths()["purelib"])')
HAB_PYTHONPATH=${ROOT}:${ROOT}/MemNavData:${HAB_SITE_PACKAGES}/pip/_vendor${PYTHONPATH:+:${PYTHONPATH}}
hab_python() { env PYTHONPATH="${HAB_PYTHONPATH}" "${HAB_PY}" "$@"; }

mkdir -p "${RUN_ROOT}/preflight" "${RUN_ROOT}/traces"
preflight=${RUN_ROOT}/preflight/scene_$(printf '%02d' "${SCENE_INDEX}").json
[[ ! -e "${preflight}" ]] || fail "preflight output already exists"
expected_scene_count=16
if [[ "${FINAL14_POPULATION_MODE}" == 1 ]]; then
  expected_scene_count=14
fi
hab_python "${ROOT}/MemNavData/validate_paper_online_a_scene.py" \
  --manifest "${MANIFEST}" \
  --expected-manifest-sha "${EXPECTED_MANIFEST_SHA}" \
  --scene-index "${SCENE_INDEX}" \
  --expected-scene-count "${expected_scene_count}" \
  --navdp-checkpoint "${NAVDP_CHECKPOINT}" --out "${preflight}" >/dev/null

readarray -t identity < <(hab_python - "${preflight}" "${MANIFEST}" <<'PY'
import json, sys
audit=json.load(open(sys.argv[1]))
manifest=json.load(open(sys.argv[2]))
print(audit["scene"])
print(audit["asset"])
print(audit["episode_root"])
print(",".join(audit["episodes"]))
print(manifest["evaluation"]["base_seed"])
print(manifest["paths"]["asset_root"])
print(manifest["paths"]["expanded_episode_root"])
print(audit["navdp_checkpoint_sha256"])
PY
)
[[ "${#identity[@]}" -eq 8 ]] || fail "preflight identity read failed"
scene=${identity[0]}
scene_file=${identity[1]}
episode_root=${identity[2]}
episode_csv=${identity[3]}
base_seed=${identity[4]}
asset_root=${identity[5]}
source_episode_root=${identity[6]}
navdp_checkpoint_sha=${identity[7]}
scene_root=${RUN_ROOT}/traces/$(printf '%02d' "${SCENE_INDEX}")_${scene}
[[ ! -e "${scene_root}" ]] || fail "scene trace output already exists"
mkdir -p "${scene_root}/native_a"

if [[ -n "${episode_csv}" ]]; then
  hab_python -u "${ROOT}/MemNavData/eval_2leg_habitat.py" \
    --episode_root "${episode_root}" --episode_ids "${episode_csv}" \
    --scene "${scene_file}" --host 127.0.0.1 --port "${NAVDP_PORT}" \
    --out "${scene_root}/native_a" --server_backend navdp \
    --hybrid_route phase --revisit_adapter legacy_metric \
    --leg1_mode policy --write_leg1_trace --stop_after_leg1 \
    --native_trace_navdp_checkpoint_sha256 "${navdp_checkpoint_sha}" \
    --leg1_goal_source own --navdp_goal_switch_reset carry \
    --success_dist 1.0 --max_steps 600 --exec_horizon 8 \
    --trajectory_selector server --trajectory_selector_scope all \
    --seed "${base_seed}" --terminal_uturn off --terminal_visual_refine off \
    --deterministic_plan_seeds \
    > "${scene_root}/eval_native_a.log" 2>&1

  # Re-render every eligible successful online trace on the same node/GPU.
  # Renderer differences become fail-closed attrition instead of silently
  # changing historical observations.
  hab_python -u "${ROOT}/MemNavData/materialize_paper_online_a_scene.py" \
    --trace-root "${scene_root}/native_a" --asset-root "${asset_root}" \
    --episode-root "${source_episode_root}" \
    --out "${scene_root}/online_a" \
    > "${scene_root}/materialize_online_a.log" 2>&1
else
  [[ "${FINAL14_POPULATION_MODE}" == 1 ]] || \
    fail "legacy paper collection cannot have an empty scene source"
  # The parent Final14 protocol explicitly retains a scene with zero available
  # source episodes.  Emit the same audited containers without asking the
  # evaluator to interpret an empty --episode_ids argument.
  hab_python - "${scene_root}" <<'PY'
import csv, hashlib, json, sys
from pathlib import Path
root=Path(sys.argv[1]); native=root/"native_a"; online=root/"online_a"
native.mkdir(parents=True,exist_ok=True); online.mkdir(parents=True,exist_ok=True)
with (native/"metric.csv").open("w",newline="") as handle:
    csv.DictWriter(handle,fieldnames=["episode"]).writeheader()
(native/"summary.json").write_text(json.dumps({
    "episodes":0,"leg1_policy_backend":"navdp","stop_after_leg1":True,
},indent=2,sort_keys=True)+"\n")
manifest={
    "schema_version":"shared_online_a_materialized_v1",
    "purpose":"empty Final14 source retained as asset attrition",
    "selection":{
        "all_eligible_traces_attempted":True,"eligible_count":0,
        "requested_count":None,
    },
    "source_trace_count":0,"episodes":[],"attrition":[],
}
path=online/"manifest.json"
path.write_text(json.dumps(manifest,indent=2,sort_keys=True)+"\n")
(online/"manifest.json.sha256").write_text(
    hashlib.sha256(path.read_bytes()).hexdigest()+"  manifest.json\n")
PY
fi
if [[ "${FINAL14_POPULATION_MODE}" == 1 ]]; then
  hab_python -u "${ROOT}/MemNavData/build_final14_role_pair_scene.py" \
    --online-root "${scene_root}/online_a" \
    --out "${scene_root}/role_pairs" --scene-rank "${SCENE_INDEX}" \
    --only-scene "${scene}" \
    --source-episode-order "${episode_csv}" --max-histories-per-scene 3 \
    > "${scene_root}/build_role_pairs.log" 2>&1
else
  hab_python -u "${ROOT}/MemNavData/build_paper_role_pair_scene.py" \
    --online-root "${scene_root}/online_a" \
    --out "${scene_root}/role_pairs" \
    > "${scene_root}/build_role_pairs.log" 2>&1
fi

hab_python - "${scene_root}" "${episode_csv}" \
  "${EXPECTED_MANIFEST_SHA}" "${FINAL14_POPULATION_MODE}" <<'PY'
import csv, hashlib, json, sys
from pathlib import Path

root=Path(sys.argv[1]); episodes=[
    value for value in sys.argv[2].split(",") if value
]
summary=json.loads((root/"native_a"/"summary.json").read_text())
if int(summary["episodes"]) != len(episodes):
    raise SystemExit("native-A summary is incomplete")
if summary["leg1_policy_backend"] != "navdp" or not summary["stop_after_leg1"]:
    raise SystemExit("native-A policy contract changed")
with (root/"native_a"/"metric.csv").open(newline="") as handle:
    rows=list(csv.DictReader(handle))
if [row["episode"] for row in rows] != episodes:
    raise SystemExit("native-A episode identities changed")
traces=[]
for episode in episodes:
    path=root/"native_a"/f"{episode}_leg1_trace.json"
    payload=json.loads(path.read_text())
    if payload["episode"] != episode or payload["source_scene"] != root.name.split("_",1)[1]:
        raise SystemExit("native-A trace identity changed")
    traces.append({
        "episode": episode,
        "reached": bool(payload["reached"]),
        "steps": int(payload["steps"]),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    })
receipt={
    "schema_version":"paper_online_a_scene_receipt_v1_20260814",
    "scene":root.name.split("_",1)[1],
    "manifest_sha256":sys.argv[3],
    "policy":"frozen_native_navdp_imagegoal",
    "query_outcomes_read":False,
    "final14_population_mode":bool(int(sys.argv[4])),
    "traces":traces,
    "materialization":json.loads((root/"online_a"/"manifest.json").read_text())["selection"],
    "materialized_episode_count":len(json.loads((root/"online_a"/"manifest.json").read_text())["episodes"]),
    "materialization_attrition":json.loads((root/"online_a"/"manifest.json").read_text())["attrition"],
    "role_pair_construction":json.loads((root/"role_pairs"/"construction_receipt.json").read_text()),
}
(root/"receipt.json").write_text(json.dumps(receipt,indent=2,sort_keys=True)+"\n")
print(json.dumps(receipt,sort_keys=True))
PY

echo "COMPLETE ${scene_root}"
