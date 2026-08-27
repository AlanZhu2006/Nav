#!/usr/bin/env bash
# Four-arm interface smoke on one already-consumed HM3D episode.

set -euo pipefail
umask 0022

ROOT=${ROOT:?set verified runtime source root}
TASK_ROOT=${TASK_ROOT:?set immutable interface-repair task root}
SMOKE_DATA_ROOT=${SMOKE_DATA_ROOT:?set immutable consumed-scene data root}
RUN_ROOT=${RUN_ROOT:?set fresh smoke output root}
HAB_PY=${HAB_PY:?set Habitat Python}
MEMNAV_PORT=${MEMNAV_PORT:?set MemNav port}
NAVDP_PORT=${NAVDP_PORT:?set NavDP port}

fail() { echo "ABORT: $*" >&2; exit 2; }
[[ ! -e "${RUN_ROOT}/receipt.json" ]] || fail "smoke receipt already exists"
(cd "${SMOKE_DATA_ROOT}" && sha256sum -c SMOKE_DATA.sha256 >/dev/null) || \
  fail "smoke data bundle changed"

readarray -t smoke < <("${HAB_PY}" - "${SMOKE_DATA_ROOT}" <<'PY'
import json,pathlib,sys
root=pathlib.Path(sys.argv[1]); data=json.loads(
    (root/"smoke_data_manifest.json").read_text())
if data["scope"] != "consumed_engineering_smoke_no_efficacy_claim":
    raise SystemExit("invalid smoke scope")
if data["scene_id"] in set(data["heldout_val10_scene_ids"]):
    raise SystemExit("smoke scene overlaps held-out population")
print(root/data["scene_file"])
print(root/data["episode_root"])
print(data["episode_id"])
print(data["seed"])
print(data["max_steps_per_leg"])
PY
)
[[ "${#smoke[@]}" -eq 5 ]] || fail "smoke manifest reader failed"
scene_file=${smoke[0]}; episode_root=${smoke[1]}; episode_id=${smoke[2]}
seed=${smoke[3]}; max_steps=${smoke[4]}

hab_site_packages=$("${HAB_PY}" -c \
  'import sysconfig; print(sysconfig.get_paths()["purelib"])')
hab_pythonpath=${TASK_ROOT}/MemNavData:${ROOT}/MemNavData:${ROOT}:${hab_site_packages}/pip/_vendor${PYTHONPATH:+:${PYTHONPATH}}
hab_python() { env PYTHONPATH="${hab_pythonpath}" "${HAB_PY}" "$@"; }

hab_python - "${ROOT}/MemNavData/eval_2leg_habitat.py" \
  "${TASK_ROOT}/MemNavData/revisit_bearing_adapter.py" <<'PY'
import hashlib,pathlib,sys
import revisit_bearing_adapter as adapter
runtime_eval=pathlib.Path(sys.argv[1]).resolve()
overlay=pathlib.Path(sys.argv[2]).resolve()
if pathlib.Path(adapter.__file__).resolve() != overlay:
    raise SystemExit(f"wrong adapter imported: {adapter.__file__}")
required={"legacy_metric","raw_fixed_bearing_v1","verified_bearing_v1"}
if not required.issubset(adapter.REVISIT_ADAPTER_MODES):
    raise SystemExit("adapter modes missing")
for path in (runtime_eval,overlay):
    print(hashlib.sha256(path.read_bytes()).hexdigest(),path)
PY
hab_python -m eval_2leg_habitat --help >/dev/null

mkdir -p "${RUN_ROOT}/logs"
common=(
  --episode_root "${episode_root}"
  --episode_ids "${episode_id}"
  --scene "${scene_file}"
  --host 127.0.0.1
  --success_dist 1.0
  --max_steps "${max_steps}"
  --exec_horizon 8
  --trajectory_selector server
  --trajectory_selector_scope all
  --navdp_goal_switch_reset carry
  --leg1_goal_source own
  --seed "${seed}"
  --terminal_uturn off
  --terminal_visual_refine off
  --deterministic_plan_seeds
  --retrieval_override off
  --certified_cdec_rescue off
  --certified_stagnation_graph off
)

trace_root=${RUN_ROOT}/trace_source
mkdir -p "${trace_root}"
hab_python -u -m eval_2leg_habitat \
  "${common[@]}" --port "${MEMNAV_PORT}" --novel_port "${NAVDP_PORT}" \
  --out "${trace_root}" --server_backend hybrid_pose \
  --leg1_mode policy --write_leg1_trace --stop_after_leg1 \
  --hybrid_route phase --revisit_adapter legacy_metric \
  >"${RUN_ROOT}/logs/eval_trace_source.log" 2>&1

arms=(native raw_fixed_oracle_role geometry_router certified_relocalization)
for arm in "${arms[@]}"; do
  arm_root=${RUN_ROOT}/${arm}; mkdir -p "${arm_root}"
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
    *) fail "unknown smoke arm ${arm}" ;;
  esac
  hab_python -u -m eval_2leg_habitat \
    "${common[@]}" --leg1_mode shared_trace \
    --shared_leg1_trace_root "${trace_root}" --out "${arm_root}" \
    "${extra[@]}" >"${RUN_ROOT}/logs/eval_${arm}.log" 2>&1
done

hab_python - "${RUN_ROOT}" "${episode_id}" \
  "${ROOT}/MemNavData/eval_2leg_habitat.py" \
  "${TASK_ROOT}/MemNavData/revisit_bearing_adapter.py" <<'PY'
import csv,hashlib,json,pathlib,sys
root=pathlib.Path(sys.argv[1]); episode=sys.argv[2]
eval_path=pathlib.Path(sys.argv[3]); adapter_path=pathlib.Path(sys.argv[4])
arms=("native","raw_fixed_oracle_role","geometry_router",
      "certified_relocalization")
rows={}
for arm in ("trace_source",)+arms:
    summary_path=root/arm/"summary.json"
    metric_path=root/arm/"metric.csv"
    plans_path=root/arm/f"{episode}_plans.json"
    for path in (summary_path,metric_path,plans_path):
        if not path.is_file():
            raise SystemExit(f"{arm}: missing {path.name}")
    summary=json.loads(summary_path.read_text())
    if summary.get("episodes") != 1:
        raise SystemExit(f"{arm}: incomplete summary")
    with metric_path.open(newline="") as handle:
        values=list(csv.DictReader(handle))
    if len(values) != 1 or values[0].get("episode") != episode:
        raise SystemExit(f"{arm}: bad metric identity")
    rows[arm]=values[0]
trace=rows["trace_source"]
if float(trace["reached_A"]) <= 0.5:
    raise SystemExit("consumed smoke did not exercise Goal B")
trace_sha=trace.get("leg1_trace_sha256")
if not trace_sha:
    raise SystemExit("trace source has no trace SHA")
for arm in arms:
    row=rows[arm]
    if row.get("leg1_trace_sha256") != trace_sha:
        raise SystemExit(f"{arm}: shared Goal-A trace differs")
    if row.get("reached_A") != trace.get("reached_A"):
        raise SystemExit(f"{arm}: shared Goal-A outcome differs")
    if int(float(row.get("steps_B") or 0)) <= 0:
        raise SystemExit(f"{arm}: Goal B was not executed")
raw=rows["raw_fixed_oracle_role"]
if raw.get("revisit_adapter") != "raw_fixed_bearing_v1":
    raise SystemExit("raw-fixed arm imported the wrong adapter")
if int(float(raw.get("revisit_adapter_plan_count") or 0)) <= 0:
    raise SystemExit("raw-fixed arm did not exercise the adapter")
if int(float(raw.get("revisit_adapter_takeover_plan_count") or 0)) <= 0:
    raise SystemExit("raw-fixed arm did not exercise takeover")
cert=rows["certified_relocalization"]
if cert.get("revisit_adapter") != "verified_bearing_v1":
    raise SystemExit("certified arm imported the wrong adapter")
if int(float(cert.get("certified_relocalization_request_count") or 0)) <= 0:
    raise SystemExit("certified request path was not exercised")
if int(float(cert.get("certified_relocalization_runtime_failure_count") or 0)):
    raise SystemExit("certified runtime failure occurred")
receipt={
  "schema_version":"hm3d_runtime_interface_smoke_v1_20260816",
  "scope":"consumed engineering smoke; no efficacy claim",
  "passed":True,
  "episode_id":episode,
  "goal_a_success_required_only_to_exercise_interface":True,
  "shared_goal_a_trace_sha256":trace_sha,
  "runtime_evaluator_sha256":hashlib.sha256(eval_path.read_bytes()).hexdigest(),
  "overlay_adapter_sha256":hashlib.sha256(adapter_path.read_bytes()).hexdigest(),
  "arms":{
    arm:{
      "reached_b_observed_not_used_as_gate":float(rows[arm]["reached_B"])>0.5,
      "steps_b":int(float(rows[arm]["steps_B"])),
      "adapter":rows[arm].get("revisit_adapter"),
      "adapter_plans":int(float(rows[arm].get("revisit_adapter_plan_count") or 0)),
      "adapter_takeovers":int(float(rows[arm].get("revisit_adapter_takeover_plan_count") or 0)),
      "certificate_requests":int(float(rows[arm].get("certified_relocalization_request_count") or 0)),
    } for arm in arms
  },
}
with (root/"receipt.json").open("x",encoding="utf-8") as handle:
    json.dump(receipt,handle,indent=2,sort_keys=True); handle.write("\n")
print(json.dumps(receipt,indent=2,sort_keys=True))
PY
