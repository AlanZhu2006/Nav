#!/usr/bin/env bash
# Run one frozen fresh160 2-leg episode through direct/budget/rescue arms.
# Servers are owned by the caller so all three arms share one loaded process.

set -euo pipefail
umask 0022

ROOT=${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
OUT_ROOT=${OUT_ROOT:?set OUT_ROOT}
ENTRY_INDEX=${ENTRY_INDEX:?set ENTRY_INDEX in 0..9}
MEMNAV_PORT=${MEMNAV_PORT:?set MEMNAV_PORT}
NAVDP_PORT=${NAVDP_PORT:?set NAVDP_PORT}
HAB_PY=${HAB_PY:-/home/asus/miniconda3/envs/habitat/bin/python}
MANIFEST=${MANIFEST:-${ROOT}/MemNavData/certified_2leg_stagnation_graph_manifest_20260813.json}
AUDIT_ONLY=${AUDIT_ONLY:-0}

fail() { echo "ABORT: $*" >&2; exit 2; }
[[ "${ENTRY_INDEX}" =~ ^[0-9]$ ]] || fail "ENTRY_INDEX must be 0..9"
[[ "${AUDIT_ONLY}" =~ ^[01]$ ]] || fail "AUDIT_ONLY must be 0 or 1"
for port in "${MEMNAV_PORT}" "${NAVDP_PORT}"; do
  ss -ltn | awk '{print $4}' | grep -Eq "(^|:)${port}$" || \
    fail "required server port ${port} is not listening"
done

hab_site=$(${HAB_PY} -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
export PYTHONPATH=${ROOT}:${ROOT}/MemNavData:${hab_site}/pip/_vendor${PYTHONPATH:+:${PYTHONPATH}}

readarray -t identity < <("${HAB_PY}" - "${ROOT}" "${MANIFEST}" "${ENTRY_INDEX}" <<'PY'
import csv, hashlib, json, sys
from pathlib import Path

root = Path(sys.argv[1]).resolve()
manifest_path = Path(sys.argv[2]).resolve()
index = int(sys.argv[3])
manifest = json.loads(manifest_path.read_text())
row = manifest["episodes"][index]
if row["index"] != index:
    raise SystemExit("manifest index changed")

def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

mirror = root / manifest["source"]["mirror_root"]
source_manifest_path = mirror / "data_manifest.json"
source_report_path = mirror / "report.json"
if sha(source_manifest_path) != manifest["source"]["data_manifest_sha256"]:
    raise SystemExit("source data manifest changed")
if sha(source_report_path) != manifest["source"]["formal_report_sha256"]:
    raise SystemExit("source formal report changed")
source_manifest = json.loads(source_manifest_path.read_text())

scene = row["scene"]
episode = row["episode"]
scene_dir = mirror / "scenes" / row["source_scene_dir"]
trace = scene_dir / "trace_source" / f"{episode}_leg1_trace.json"
metric = scene_dir / "certified_relocalization" / "metric.csv"
if sha(trace) != row["source_trace_sha256"]:
    raise SystemExit("source online-A trace changed")
if sha(metric) != row["source_metric_sha256"]:
    raise SystemExit("source certified metric changed")

with metric.open(newline="") as handle:
    old = next(item for item in csv.DictReader(handle)
               if item["episode"] == episode)
if (old["reached_B"] == "1.0") is not row["source_B_success"]:
    raise SystemExit("source success label changed")
if int(old["steps_B"]) != row["source_B_steps"]:
    raise SystemExit("source B step changed")
if old["termination_reason_B"] != row["source_B_termination"]:
    raise SystemExit("source B termination changed")
if row["cohort"] == "known_failure" and not (
        old["reached_A"] == "1.0"
        and old["reached_B"] == "0.0"
        and old["termination_reason_B"] == "stuck"
        and int(float(old["certified_relocalization_accept_count_B"])) > 0):
    raise SystemExit("known-failure contract changed")

episode_root = mirror / "data" / "mp3d_2leg" / scene
episode_dir = episode_root / episode
source_episode = next(item for item in source_manifest["episodes"][scene]
                      if item["episode"] == episode)
for key, relative in (
    ("metadata", "meta/gen_meta.json"),
    ("parquet", "data/chunk-000/episode_000000.parquet"),
    ("goal", "goal_image.jpg"),
):
    if sha(episode_dir / relative) != source_episode["files"][key]["sha256"]:
        raise SystemExit(f"source episode {key} changed")
asset = Path(manifest["source"]["asset_root"]) / scene / f"{scene}.glb"
if sha(asset) != source_manifest["assets"][scene]["sha256"]:
    raise SystemExit("scene asset changed")
trace_payload = json.loads(trace.read_text())
if int(trace_payload["episode_seed"]) != row["episode_seed"]:
    raise SystemExit("online-A seed changed")

print(scene)
print(episode)
print(row["cohort"])
print(row["episode_seed"])
print(asset)
print(episode_root)
print(scene_dir / "trace_source")
print(row["source_B_steps"])
print(row["source_B_termination"])
print("1" if row["source_B_success"] else "0")
PY
)
[[ "${#identity[@]}" -eq 10 ]] || fail "manifest identity reader failed"
scene=${identity[0]}; episode=${identity[1]}; cohort=${identity[2]}
episode_seed=${identity[3]}; scene_file=${identity[4]}
episode_root=${identity[5]}; trace_root=${identity[6]}
source_steps=${identity[7]}; source_termination=${identity[8]}
source_success=${identity[9]}

entry_root=${OUT_ROOT}/entries/$(printf '%02d' "${ENTRY_INDEX}")_${scene}_${episode}
if [[ "${AUDIT_ONLY}" == 0 ]]; then
  [[ ! -e "${entry_root}" ]] || fail "entry output already exists: ${entry_root}"
  mkdir -p "${entry_root}/logs"
else
  [[ -d "${entry_root}" ]] || fail "audit-only entry is missing: ${entry_root}"
fi

case $((10#${ENTRY_INDEX} % 3)) in
  0) arm_order=(direct budget_control rescue) ;;
  1) arm_order=(budget_control rescue direct) ;;
  2) arm_order=(rescue direct budget_control) ;;
esac
if [[ "${AUDIT_ONLY}" == 0 ]]; then
  printf '%s\n' "${arm_order[@]}" > "${entry_root}/arm_order.txt"
fi

common=(
  --episode_root "${episode_root}"
  --episode_ids "${episode}"
  --scene "${scene_file}"
  --host 127.0.0.1
  --port "${MEMNAV_PORT}"
  --novel_port "${NAVDP_PORT}"
  --server_backend hybrid_pose
  --leg1_mode shared_trace
  --shared_leg1_trace_root "${trace_root}"
  --leg1_goal_source own
  --seed "${episode_seed}"
  --success_dist 1.0
  --max_steps 500
  --exec_horizon 8
  --stuck_window 150
  --stuck_dist 0.10
  --trajectory_selector server
  --trajectory_selector_scope all
  --navdp_goal_switch_reset carry
  --terminal_uturn off
  --terminal_visual_refine off
  --deterministic_plan_seeds
  --hybrid_route certified_relocalization
  --revisit_controller navdp_mixed
  --revisit_adapter verified_bearing_v1
)

if [[ "${AUDIT_ONLY}" == 0 ]]; then
  for arm in "${arm_order[@]}"; do
    mode=off
    [[ "${arm}" == budget_control ]] && mode=budget_control
    [[ "${arm}" == rescue ]] && mode=rescue
    arm_root=${entry_root}/${arm}
    mkdir "${arm_root}"
    echo "[run] ${ENTRY_INDEX} ${scene}/${episode}/${arm}"
    "${HAB_PY}" -u "${ROOT}/MemNavData/eval_2leg_habitat.py" \
      "${common[@]}" --out "${arm_root}" \
      --certified_stagnation_graph "${mode}" \
      > "${entry_root}/logs/eval_${arm}.log" 2>&1
  done
fi

"${HAB_PY}" - "${entry_root}" "${episode}" "${cohort}" \
  "${source_steps}" "${source_termination}" "${source_success}" <<'PY'
import csv, json, sys
from pathlib import Path

root = Path(sys.argv[1])
episode, cohort = sys.argv[2], sys.argv[3]
source_steps = int(sys.argv[4])
source_termination = sys.argv[5]
source_success = bool(int(sys.argv[6]))

def require(value, message):
    if not value:
        raise SystemExit(message)

records = {}
for arm, mode in (("direct", "off"),
                  ("budget_control", "budget_control"),
                  ("rescue", "rescue")):
    summary = json.loads((root / arm / "summary.json").read_text())
    with (root / arm / "metric.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    payload = json.loads((root / arm / f"{episode}_plans.json").read_text())
    require(summary["episodes"] == 1, f"{arm}: incomplete summary")
    require(summary["certified_stagnation_graph"] == mode,
            f"{arm}: stagnation mode changed")
    require(len(rows) == 1 and rows[0]["episode"] == episode,
            f"{arm}: metric identity changed")
    records[arm] = {"summary": summary, "metric": rows[0], "trace": payload}

direct = records["direct"]
require((direct["metric"]["reached_B"] == "1.0") is source_success,
        "direct did not reproduce source B outcome")
require(direct["metric"]["termination_reason_B"] == source_termination,
        "direct did not reproduce source B termination")
require(int(direct["metric"]["certified_relocalization_accept_count_B"]) > 0,
        "direct no longer has an accepted certificate")

direct_plans = direct["trace"]["legB"]
direct_rollout = direct["trace"]["legB_rollout_trace"]
direct_memory = direct["trace"]["legB_memory_trace"]
direct_steps = int(direct["metric"]["steps_B"])
require(len(direct_rollout) == direct_steps,
        "direct rollout length differs from its metric steps")

def causal_plan(plan):
    """Drop wall-clock diagnostics while retaining every decision field."""
    return {
        key: value for key, value in plan.items()
        if not key.endswith("_ms")
    }

direct_causal_plans = [causal_plan(plan) for plan in direct_plans]
for arm in ("budget_control", "rescue"):
    other = records[arm]
    other_causal_plans = [
        causal_plan(plan)
        for plan in other["trace"]["legB"][:len(direct_plans)]]
    require(other_causal_plans == direct_causal_plans,
            f"{arm}: plan prefix differs before the direct endpoint")
    require(other["trace"]["legB_rollout_trace"][:direct_steps]
            == direct_rollout,
            f"{arm}: physical rollout prefix differs before intervention")
    other_memory_prefix = [
        item for item in other["trace"]["legB_memory_trace"]
        if int(item["step"]) < direct_steps]
    require(other_memory_prefix == direct_memory,
            f"{arm}: memory prefix differs before intervention")

budget_metric = records["budget_control"]["metric"]
rescue_metric = records["rescue"]["metric"]
if cohort == "known_failure":
    require(budget_metric["certified_stagnation_intervention_attempted"]
            == "True", "budget control did not trigger")
    require(rescue_metric["certified_stagnation_intervention_attempted"]
            == "True", "rescue did not trigger")
    require(int(budget_metric["certified_graph_active_plan_count"]) == 0,
            "budget control unexpectedly executed graph plans")
    require(int(rescue_metric["certified_graph_active_plan_count"]) > 0,
            "rescue triggered without an active graph plan")
else:
    for arm in ("direct", "budget_control", "rescue"):
        require(records[arm]["metric"][
            "certified_stagnation_intervention_attempted"] == "False",
            f"{arm}: success control was unexpectedly treated")
    require([causal_plan(plan) for plan in
             records["budget_control"]["trace"]["legB"]]
            == direct_causal_plans,
            "budget control changed a successful plan trace")
    require([causal_plan(plan) for plan in
             records["rescue"]["trace"]["legB"]]
            == direct_causal_plans,
            "rescue changed a successful plan trace")

report = {
    "schema_version": "certified_2leg_stagnation_graph_episode_audit_v1",
    "entry": root.name,
    "cohort": cohort,
    "causal_prefix_exact": True,
    "source_machine_steps_B": source_steps,
    "same_process_direct_steps_B": direct_steps,
    "cross_machine_step_delta": direct_steps - source_steps,
    "outcomes": {
        arm: {
            "B": records[arm]["metric"]["reached_B"] == "1.0",
            "steps_B": int(records[arm]["metric"]["steps_B"]),
            "termination_B": records[arm]["metric"]["termination_reason_B"],
            "final_distance_B": float(records[arm]["metric"]["final_dist_B"]),
            "intervention": records[arm]["metric"][
                "certified_stagnation_intervention_attempted"] == "True",
            "graph_active_plans": int(records[arm]["metric"][
                "certified_graph_active_plan_count"]),
        }
        for arm in ("direct", "budget_control", "rescue")
    },
}
with (root / "episode_audit.json").open("x", encoding="utf-8") as handle:
    json.dump(report, handle, indent=2, sort_keys=True)
    handle.write("\n")
print(json.dumps(report, sort_keys=True))
PY

echo "[complete] ${entry_root}"
