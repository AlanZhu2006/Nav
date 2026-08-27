#!/usr/bin/env bash
# One frozen actual-online 3-leg episode, paired three-arm stagnation audit.

set -euo pipefail
umask 0022

SOURCE_ROOT=${SOURCE_ROOT:?set immutable task source root}
RUN_ROOT=${RUN_ROOT:?set pilot run root}
BENCH_ROOT=${BENCH_ROOT:?set frozen benchmark root}
EXPECTED_MANIFEST_SHA=${EXPECTED_MANIFEST_SHA:?set frozen benchmark SHA}
EPISODE_INDEX=${EPISODE_INDEX:?set frozen manifest selection index}
HAB_PY=${HAB_PY:?set Habitat interpreter}
MEMNAV_PORT=${MEMNAV_PORT:?set MemNav port}
NAVDP_PORT=${NAVDP_PORT:?set NavDP port}
MAX_STEPS=${MAX_STEPS:-600}
EXPERIMENT_KIND=${EXPERIMENT_KIND:-pilot}

fail() { echo "ABORT: $*" >&2; exit 2; }
case "${EXPERIMENT_KIND}" in
  pilot)
    [[ "${EPISODE_INDEX}" =~ ^(0|1|2|3|7|14)$ ]] || \
      fail "EPISODE_INDEX is outside the frozen pilot set"
    experiment_scope="post-hoc failure mechanism pilot; not an SR estimate"
    ;;
  fresh20_expansion)
    [[ "${EPISODE_INDEX}" =~ ^([0-9]|1[0-9])$ ]] || \
      fail "EPISODE_INDEX is outside fresh20"
    experiment_scope="authorized full-fresh20 internal expansion"
    ;;
  *) fail "unknown EXPERIMENT_KIND ${EXPERIMENT_KIND}" ;;
esac
[[ "${MAX_STEPS}" =~ ^[1-9][0-9]*$ ]] || fail "MAX_STEPS must be positive"
[[ "$(sha256sum "${BENCH_ROOT}/manifest.json" | awk '{print $1}')" == \
   "${EXPECTED_MANIFEST_SHA}" ]] || fail "benchmark manifest changed"

HAB_SITE_PACKAGES=$("${HAB_PY}" -c \
  'import sysconfig; print(sysconfig.get_paths()["purelib"])')
HAB_PYTHONPATH=${SOURCE_ROOT}:${SOURCE_ROOT}/MemNavData:${HAB_SITE_PACKAGES}/pip/_vendor${PYTHONPATH:+:${PYTHONPATH}}
hab_python() { env PYTHONPATH="${HAB_PYTHONPATH}" "${HAB_PY}" "$@"; }

readarray -t identity < <(hab_python - \
  "${BENCH_ROOT}/manifest.json" "${EPISODE_INDEX}" <<'PY'
import hashlib,json,sys
from pathlib import Path

def sha(path):
    digest=hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8*1024*1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

manifest=json.load(open(sys.argv[1]))
index=int(sys.argv[2])
row=manifest["episodes"][index]
scene=str(row["scene"]); episode=str(row["episode"])
source=Path(row["source_online_episode"])
receipt=json.load(open(source/"receipt.json"))
trace=json.load(open(source/"online_a_trace.json"))
asset=Path(receipt["source_asset"])
if sha(asset) != receipt["source_asset_sha256"]:
    raise SystemExit("scene asset changed")
if sha(source/"online_a_trace.json") != row["source_online_trace_sha256"]:
    raise SystemExit("online-A trace changed")
if sha(source/"receipt.json") != row["source_online_receipt_sha256"]:
    raise SystemExit("online-A receipt changed")
print(scene); print(episode); print(asset); print(int(trace["episode_seed"]))
PY
)
[[ "${#identity[@]}" -eq 4 ]] || fail "episode identity reader failed"
scene=${identity[0]}; episode=${identity[1]}; scene_file=${identity[2]}
episode_seed=${identity[3]}
label=$(printf '%03d' "${EPISODE_INDEX}")
episode_out=${RUN_ROOT}/scenes/${label}_${scene}_${episode}
[[ ! -e "${episode_out}" ]] || fail "episode output exists: ${episode_out}"
mkdir -p "${episode_out}/logs"

case $((10#${EPISODE_INDEX} % 3)) in
  0) arm_order=(direct budget_control rescue) ;;
  1) arm_order=(budget_control rescue direct) ;;
  2) arm_order=(rescue direct budget_control) ;;
esac
printf '%s\n' "${arm_order[@]}" > "${episode_out}/arm_order.txt"
cohort=control
[[ "${EPISODE_INDEX}" =~ ^(2|7|14)$ ]] && cohort=known_failure
if [[ "${EXPERIMENT_KIND}" == fresh20_expansion \
      && ! "${EPISODE_INDEX}" =~ ^(0|1|2|3|7|14)$ ]]; then
  cohort=unselected_expansion
fi
hab_python - "${episode_out}/episode_contract.json" "${scene}" "${episode}" \
  "${EPISODE_INDEX}" "${episode_seed}" "${EXPECTED_MANIFEST_SHA}" \
  "${cohort}" "${experiment_scope}" "${arm_order[@]}" <<'PY'
import json,sys
path,scene,episode,index,seed,manifest_sha,cohort,scope,*order=sys.argv[1:]
with open(path,"x",encoding="utf-8") as handle:
    json.dump({
        "schema_version":"certified_stagnation_graph_pilot_episode_v2_budget_control",
        "scope":scope,
        "scene":scene,"episode":episode,"selection_index":int(index),
        "episode_seed":int(seed),"benchmark_manifest_sha256":manifest_sha,
        "cohort":cohort,"arm_order":order,
        "primary_contrast":"rescue_B_minus_budget_control_B",
    },handle,indent=2,sort_keys=True); handle.write("\n")
PY

common=(
  --episode_root "${BENCH_ROOT}/${scene}"
  --episode_ids "${episode}"
  --scene "${scene_file}"
  --host 127.0.0.1
  --success_dist 1.0
  --max_steps "${MAX_STEPS}"
  --exec_horizon 8
  --trajectory_selector server
  --trajectory_selector_scope all
  --navdp_goal_switch_reset before_c
  --leg1_mode shared_trace
  --leg1_goal_source own
  --seed "${episode_seed}"
  --terminal_uturn off
  --terminal_visual_refine off
  --deterministic_plan_seeds
  --double_revisit_c_history initial_leg_only
  --shared_online_variant v1_controlled_pose_perturbation
  --shared_online_c_tail_max_covis 0.10
  --port "${MEMNAV_PORT}"
  --novel_port "${NAVDP_PORT}"
  --server_backend hybrid_pose
  --hybrid_route certified_relocalization
  --revisit_controller navdp_mixed
  --revisit_adapter verified_bearing_v1
)

for arm in "${arm_order[@]}"; do
  arm_root=${episode_out}/${arm}
  mkdir "${arm_root}"
  mode=off
  [[ "${arm}" == budget_control ]] && mode=budget_control
  [[ "${arm}" == rescue ]] && mode=rescue
  echo "[run] index=${EPISODE_INDEX} ${scene}/${episode}/${arm}"
  hab_python -u "${SOURCE_ROOT}/MemNavData/eval_shared_online_double_revisit.py" \
    "${common[@]}" --out "${arm_root}" \
    --certified_stagnation_graph "${mode}" \
    > "${episode_out}/logs/eval_${arm}.log" 2>&1
done

hab_python - "${episode_out}" "${episode}" "${EXPECTED_MANIFEST_SHA}" <<'PY'
import csv,json,sys
from pathlib import Path
root=Path(sys.argv[1]); episode=sys.argv[2]; manifest_sha=sys.argv[3]
payload={}
for arm,mode in (("direct","off"),("budget_control","budget_control"),
                 ("rescue","rescue")):
    summary=json.load(open(root/arm/"summary.json"))
    if summary["episodes"] != 1 or summary["benchmark_manifest_sha256"] != manifest_sha:
        raise SystemExit(f"{arm}: incomplete or wrong benchmark")
    if summary["certified_stagnation_graph"] != mode:
        raise SystemExit(f"{arm}: graph mode changed")
    if summary["policy_backends"] != {"B":"navdp_auto","C":"navdp_auto"}:
        raise SystemExit(f"{arm}: controller contract changed")
    with open(root/arm/"metric.csv",newline="") as handle:
        rows=list(csv.DictReader(handle))
    if len(rows) != 1 or rows[0]["episode"] != episode:
        raise SystemExit(f"{arm}: metric identity changed")
    payload[arm]=json.load(open(root/arm/f"{episode}_plans.json"))
for arm in ("budget_control","rescue"):
    if payload["direct"]["replay"] != payload[arm]["replay"]:
        raise SystemExit(f"{arm}: did not restore the same online-A prefix")
print(json.dumps({"episode":root.name,"status":"complete"},sort_keys=True))
PY

echo "[complete] ${episode_out}"
