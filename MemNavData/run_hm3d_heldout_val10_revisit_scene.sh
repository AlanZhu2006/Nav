#!/usr/bin/env bash
# Evaluate one frozen HM3D held-out val10 scene with one shared online Goal-A trace.

set -euo pipefail
umask 0022

ROOT=${ROOT:?set immutable base source ROOT}
TASK_ROOT=${TASK_ROOT:?set immutable task bundle root}
RUN_ROOT=${RUN_ROOT:?set run root}
MANIFEST=${MANIFEST:?set frozen manifest}
EXPECTED_MANIFEST_SHA=${EXPECTED_MANIFEST_SHA:?set manifest SHA}
SCENE_INDEX=${SCENE_INDEX:?set scene index}
HAB_PY=${HAB_PY:?set Habitat Python}
MEMNAV_PORT=${MEMNAV_PORT:?set MemNav port}
NAVDP_PORT=${NAVDP_PORT:?set NavDP port}

fail() { echo "ABORT: $*" >&2; exit 2; }
[[ "${SCENE_INDEX}" =~ ^[0-9]$ ]] || fail "scene index must be 0..9"
[[ "$(sha256sum "${MANIFEST}" | awk '{print $1}')" == \
    "${EXPECTED_MANIFEST_SHA}" ]] || fail "manifest identity changed"

hab_site_packages=$("${HAB_PY}" -c \
  'import sysconfig; print(sysconfig.get_paths()["purelib"])')
hab_pythonpath=${ROOT}:${TASK_ROOT}:${ROOT}/MemNavData:${hab_site_packages}/pip/_vendor${PYTHONPATH:+:${PYTHONPATH}}
hab_python() { env PYTHONPATH="${hab_pythonpath}" "${HAB_PY}" "$@"; }

readarray -t identity < <(hab_python - "${MANIFEST}" "${SCENE_INDEX}" <<'PY'
import hashlib,json,pathlib,sys
def sha(path):
    digest=hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda:handle.read(8<<20),b""):
            digest.update(block)
    return digest.hexdigest()
m=json.load(open(sys.argv[1])); scene=m["scenes"][int(sys.argv[2])]
asset=pathlib.Path(m["assets"][scene]["glb_path"])
if sha(asset) != m["assets"][scene]["glb_sha256"]:
    raise SystemExit("scene asset changed")
episodes=[]
for row in m["episodes"][scene]:
    for record in row["files"].values():
        path=pathlib.Path(record["path"])
        if sha(path) != record["sha256"]:
            raise SystemExit(f"episode file changed: {path}")
    episodes.append(row["episode"])
if len(episodes) != m["episodes_per_scene"]:
    raise SystemExit("episode count changed")
print(scene); print(asset); print(pathlib.Path(m["paths"]["generated_root"])/scene)
print(",".join(episodes)); print(m["evaluation"]["base_seed"])
print(m["evaluation"]["max_steps_per_leg"])
PY
)
[[ "${#identity[@]}" -eq 6 ]] || fail "manifest identity reader failed"
scene=${identity[0]}; scene_file=${identity[1]}; episode_root=${identity[2]}
episode_csv=${identity[3]}; base_seed=${identity[4]}; max_steps=${identity[5]}

scene_root=${RUN_ROOT}/scenes/$(printf '%02d' "${SCENE_INDEX}")_${scene}
[[ ! -e "${scene_root}" ]] || fail "scene output already exists: ${scene_root}"
mkdir -p "${scene_root}/logs"

case $((10#${SCENE_INDEX} % 4)) in
  0) arm_order=(certified_relocalization raw_fixed_oracle_role native geometry_router) ;;
  1) arm_order=(raw_fixed_oracle_role geometry_router certified_relocalization native) ;;
  2) arm_order=(geometry_router native raw_fixed_oracle_role certified_relocalization) ;;
  3) arm_order=(native certified_relocalization geometry_router raw_fixed_oracle_role) ;;
esac

hab_python - "${scene_root}/scene_contract.json" "${scene}" \
  "${SCENE_INDEX}" "${EXPECTED_MANIFEST_SHA}" "${arm_order[@]}" <<'PY'
import json,sys
path,scene,index,manifest_sha,*order=sys.argv[1:]
with open(path,"x",encoding="utf-8") as handle:
    json.dump({
      "schema_version":"hm3d_heldout_val10_revisit_scene_contract_v1_20260816",
      "scene":scene,"scene_index":int(index),
      "manifest_sha256":manifest_sha,"arm_order":order,
      "actual_online_goal_a_trace":True,
      "certified_runtime_role_label_visible":False,
      "raw_fixed_role_oracle":True,
    },handle,indent=2,sort_keys=True); handle.write("\n")
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
  --retrieval_override off
  --certified_cdec_rescue off
  --certified_stagnation_graph off
)

trace_root=${scene_root}/trace_source
mkdir -p "${trace_root}"
hab_python -u "${ROOT}/MemNavData/eval_2leg_habitat.py" \
  "${common[@]}" --port "${MEMNAV_PORT}" --novel_port "${NAVDP_PORT}" \
  --out "${trace_root}" --server_backend hybrid_pose \
  --leg1_mode policy --write_leg1_trace --stop_after_leg1 \
  --hybrid_route phase --revisit_adapter legacy_metric \
  >"${scene_root}/logs/eval_trace_source.log" 2>&1

for arm in "${arm_order[@]}"; do
  arm_root=${scene_root}/${arm}
  mkdir -p "${arm_root}"
  case "${arm}" in
    native)
      extra=(--port "${NAVDP_PORT}" --server_backend navdp \
        --hybrid_route phase)
      ;;
    raw_fixed_oracle_role)
      extra=(--port "${MEMNAV_PORT}" --novel_port "${NAVDP_PORT}" \
        --server_backend hybrid_pose --hybrid_route phase \
        --revisit_controller navdp_mixed \
        --revisit_adapter raw_fixed_bearing_v1)
      ;;
    geometry_router)
      extra=(--port "${MEMNAV_PORT}" --novel_port "${NAVDP_PORT}" \
        --server_backend hybrid_pose --hybrid_route memory_geometry \
        --revisit_controller navdp_mixed --revisit_adapter legacy_metric \
        --router_visual_floor 0.88 --router_min_matches 20 \
        --router_min_inliers 12 --router_min_inlier_ratio 0.50 \
        --router_confirm_plans 2 --router_verify_top_k 8)
      ;;
    certified_relocalization)
      extra=(--port "${MEMNAV_PORT}" --novel_port "${NAVDP_PORT}" \
        --server_backend hybrid_pose \
        --hybrid_route certified_relocalization \
        --revisit_controller navdp_mixed \
        --revisit_adapter verified_bearing_v1)
      ;;
    *) fail "unknown arm ${arm}" ;;
  esac
  hab_python -u "${ROOT}/MemNavData/eval_2leg_habitat.py" \
    "${common[@]}" --leg1_mode shared_trace \
    --shared_leg1_trace_root "${trace_root}" --out "${arm_root}" \
    "${extra[@]}" >"${scene_root}/logs/eval_${arm}.log" 2>&1
done

hab_python - "${scene_root}" "${episode_csv}" <<'PY'
import csv,json,pathlib,sys
root=pathlib.Path(sys.argv[1]); expected=sys.argv[2].split(",")
arms=("native","raw_fixed_oracle_role","geometry_router",
      "certified_relocalization")
rows={}
for arm in ("trace_source",)+arms:
    summary=json.loads((root/arm/"summary.json").read_text())
    if summary.get("episodes") != len(expected):
        raise SystemExit(f"{arm}: incomplete summary")
    with (root/arm/"metric.csv").open(newline="") as handle:
        values=list(csv.DictReader(handle))
    if [row["episode"] for row in values] != expected:
        raise SystemExit(f"{arm}: episode identity/order changed")
    rows[arm]={row["episode"]:row for row in values}
for episode in expected:
    trace=rows["trace_source"][episode]
    reached_a=trace["reached_A"]
    trace_sha=trace.get("leg1_trace_sha256")
    for arm in arms:
        row=rows[arm][episode]
        if row["reached_A"] != reached_a:
            raise SystemExit(f"{arm}/{episode}: Goal-A outcome changed")
        if row.get("leg1_trace_sha256") != trace_sha:
            raise SystemExit(f"{arm}/{episode}: Goal-A trace SHA changed")
        if float(reached_a) <= 0.5 and int(float(row["steps_B"])) != 0:
            raise SystemExit(f"{arm}/{episode}: Goal-B ran after Goal-A failure")
print(json.dumps({"status":"complete","scene":root.name,
                  "episodes":len(expected),"arms":list(arms)}))
PY
