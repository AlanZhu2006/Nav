#!/usr/bin/env bash
set -euo pipefail
umask 0022

TASK_ROOT=${TASK_ROOT:?set immutable task bundle}
HM3D_RUNTIME_ROOT=${HM3D_RUNTIME_ROOT:?set exact parent HM3D runtime bundle}
EXEC_SOURCE_ROOT=${EXEC_SOURCE_ROOT:?set frozen Final14 execution bundle}
RUN_ROOT=${RUN_ROOT:?set isolated run root}
PARENT_RUN_ROOT=${PARENT_RUN_ROOT:?set completed HM3D parent run}
PARENT_MANIFEST=${PARENT_MANIFEST:?set copied parent manifest}
PROTOCOL=${PROTOCOL:?set frozen mixed-role protocol}
SCENE_RANK=${SCENE_RANK:?set contiguous scene rank}
HAB_PY=${HAB_PY:?set Habitat Python}

fail() { echo "ABORT: $*" >&2; exit 2; }
[[ "${SCENE_RANK}" =~ ^[0-8]$ ]] || fail "scene rank must be 0..8"

readarray -t identity < <("${HAB_PY}" - "${PROTOCOL}" "${PARENT_MANIFEST}" "${SCENE_RANK}" <<'PY'
import json,pathlib,sys
p=json.load(open(sys.argv[1])); m=json.load(open(sys.argv[2]))
row=p["dataset"]["scenes"][int(sys.argv[3])]
scene=row["scene_id"]; parent_index=int(row["parent_index"])
episodes=[entry["episode"] for entry in m["episodes"][scene]]
asset=pathlib.Path(m["assets"][scene]["glb_path"])
print(scene); print(parent_index); print(asset)
print(m["paths"]["generated_root"]); print(",".join(episodes))
PY
)
[[ "${#identity[@]}" -eq 5 ]] || fail "identity reader failed"
scene=${identity[0]}; parent_index=${identity[1]}; asset=${identity[2]}
episode_root=${identity[3]}; episode_csv=${identity[4]}
trace_root=${PARENT_RUN_ROOT}/scenes/$(printf '%02d' "${parent_index}")_${scene}/trace_source
scene_root=${RUN_ROOT}/construction/traces/$(printf '%02d' "${SCENE_RANK}")_${scene}
[[ -d "${trace_root}" && -f "${asset}" ]] || fail "parent inputs missing"
[[ ! -e "${scene_root}" ]] || fail "scene output already exists"
mkdir -p "${scene_root}" "${RUN_ROOT}/logs"

alias_root=${scene_root}/asset_alias
mkdir -p "${alias_root}/${scene}"
ln -s "${asset}" "${alias_root}/${scene}/${scene}.glb"

parent_pythonpath=${TASK_ROOT}/MemNavData:${HM3D_RUNTIME_ROOT}/MemNavData:${HM3D_RUNTIME_ROOT}
env PYTHONPATH="${parent_pythonpath}" "${HAB_PY}" -u \
  "${TASK_ROOT}/MemNavData/hm3d_materialize_existing_online_a.py" \
  --trace-root "${trace_root}" --asset-root "${alias_root}" \
  --episode-root "${episode_root}" --out "${scene_root}/online_a" \
  >"${scene_root}/materialize_online_a.log" 2>&1

exec_pythonpath=${EXEC_SOURCE_ROOT}:${EXEC_SOURCE_ROOT}/MemNavData
env PYTHONPATH="${exec_pythonpath}" "${HAB_PY}" -u \
  "${EXEC_SOURCE_ROOT}/MemNavData/build_final14_role_pair_scene.py" \
  --online-root "${scene_root}/online_a" --out "${scene_root}/role_pairs" \
  --scene-rank "${SCENE_RANK}" --only-scene "${scene}" \
  --source-episode-order "${episode_csv}" --max-histories-per-scene 3 \
  >"${scene_root}/build_role_pairs.log" 2>&1

"${HAB_PY}" - "${scene_root}" "${scene}" "${SCENE_RANK}" \
  "${parent_index}" "${episode_csv}" <<'PY'
import hashlib,json,pathlib,sys
root=pathlib.Path(sys.argv[1])
online=json.load(open(root/"online_a/manifest.json"))
construction=json.load(open(root/"role_pairs/construction_receipt.json"))
payload={
 "schema_version":"hm3d_mixed_role_scene_completion_v1_20260818",
 "scene":sys.argv[2],"scene_rank":int(sys.argv[3]),
 "parent_scene_index":int(sys.argv[4]),
 "source_episodes":[x for x in sys.argv[5].split(",") if x],
 "source_traces":int(online["source_trace_count"]),
 "materialized_histories":len(online["episodes"]),
 "natural_histories":int(construction["retained_standard_natural_histories"]),
 "query_policy_outcomes_read":False,
}
encoded=(json.dumps(payload,indent=2,sort_keys=True)+"\n").encode()
(root/"completion.json").write_bytes(encoded)
(root/"completion.json.sha256").write_text(
 hashlib.sha256(encoded).hexdigest()+"  completion.json\n")
print(json.dumps(payload,sort_keys=True))
PY

echo "COMPLETE ${scene_root}"
