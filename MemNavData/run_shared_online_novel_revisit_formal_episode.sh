#!/usr/bin/env bash
# One sealed actual-online Novel->Novel->Revisit episode, five paired C arms.

set -euo pipefail
umask 0022

SOURCE_ROOT=${SOURCE_ROOT:?set immutable task source root}
RUN_ROOT=${RUN_ROOT:?set formal run root}
BENCH_ROOT=${BENCH_ROOT:?set sealed benchmark root}
TRACE_ROOT=${TRACE_ROOT:?set extracted native trace root}
EXPECTED_MANIFEST_SHA=${EXPECTED_MANIFEST_SHA:?set sealed manifest SHA}
EXPECTED_TRACE_MANIFEST_SHA=${EXPECTED_TRACE_MANIFEST_SHA:?set trace manifest SHA}
EPISODE_INDEX=${EPISODE_INDEX:?set frozen accepted-population index}
HAB_PY=${HAB_PY:?set Habitat interpreter}
MEMNAV_PORT=${MEMNAV_PORT:?set MemNav port}
NAVDP_PORT=${NAVDP_PORT:?set NavDP port}
MAX_STEPS=${MAX_STEPS:-600}

fail() { echo "ABORT: $*" >&2; exit 2; }
[[ "${EPISODE_INDEX}" =~ ^[0-9]+$ ]] || fail "invalid episode index"
[[ "${MAX_STEPS}" =~ ^[1-9][0-9]*$ ]] || fail "invalid max steps"
[[ "$(sha256sum "${BENCH_ROOT}/manifest.json" | awk '{print $1}')" == \
   "${EXPECTED_MANIFEST_SHA}" ]] || fail "benchmark manifest changed"
[[ "$(sha256sum "${TRACE_ROOT}/manifest.json" | awk '{print $1}')" == \
   "${EXPECTED_TRACE_MANIFEST_SHA}" ]] || fail "trace manifest changed"

HAB_SITE_PACKAGES=$("${HAB_PY}" -c \
  'import sysconfig; print(sysconfig.get_paths()["purelib"])')
HAB_PYTHONPATH=${SOURCE_ROOT}:${SOURCE_ROOT}/MemNavData:${HAB_SITE_PACKAGES}/pip/_vendor${PYTHONPATH:+:${PYTHONPATH}}
hab_python() { env PYTHONPATH="${HAB_PYTHONPATH}" "${HAB_PY}" "$@"; }

readarray -t identity < <(hab_python - \
  "${BENCH_ROOT}/manifest.json" "${EPISODE_INDEX}" <<'PY'
import hashlib,json,sys
from pathlib import Path

def sha(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda:f.read(8<<20),b""): h.update(chunk)
    return h.hexdigest()

m=json.load(open(sys.argv[1])); index=int(sys.argv[2])
rows=m["accepted"]
if index >= len(rows):
    print("SKIP"); print(len(rows)); raise SystemExit(0)
r=rows[index]; scene=str(r["scene"]); episode=str(r["episode"])
asset=Path(r["source_scene_asset"])
if sha(asset) != r["source_scene_asset_sha256"]:
    raise SystemExit("scene asset changed")
print("RUN"); print(scene); print(episode); print(asset); print(index)
PY
)
[[ "${#identity[@]}" -ge 2 ]] || fail "identity reader failed"
if [[ "${identity[0]}" == SKIP ]]; then
  mkdir -p "${RUN_ROOT}/skipped"
  printf 'accepted_population=%s\narray_index=%s\n' \
    "${identity[1]}" "${EPISODE_INDEX}" \
    > "${RUN_ROOT}/skipped/$(printf '%03d' "${EPISODE_INDEX}").txt"
  exit 0
fi
[[ "${#identity[@]}" -eq 5 && "${identity[0]}" == RUN ]] || \
  fail "malformed episode identity"
scene=${identity[1]}; episode=${identity[2]}; scene_file=${identity[3]}
selection_index=${identity[4]}
label=$(printf '%03d' "${selection_index}")
episode_out=${RUN_ROOT}/scenes/${label}_${scene}_${episode}
[[ ! -e "${episode_out}" ]] || fail "episode output exists: ${episode_out}"
mkdir -p "${episode_out}/logs"

arms=(native known_direct certified certified_budget certified_graph)
offset=$((10#${selection_index} % ${#arms[@]}))
arm_order=()
for ((i=0; i<${#arms[@]}; i++)); do
  arm_order+=("${arms[$(((offset+i)%${#arms[@]}))]}")
done
printf '%s\n' "${arm_order[@]}" > "${episode_out}/arm_order.txt"

hab_python - "${episode_out}/episode_contract.json" "${scene}" "${episode}" \
  "${selection_index}" "${EXPECTED_MANIFEST_SHA}" \
  "${EXPECTED_TRACE_MANIFEST_SHA}" "${arm_order[@]}" <<'PY'
import json,sys
path,scene,episode,index,manifest_sha,trace_sha,*order=sys.argv[1:]
with open(path,"x",encoding="utf-8") as f:
    json.dump({
      "schema_version":"shared_online_nnr_paired_episode_v1_20260814",
      "scene":scene,"episode":episode,"selection_index":int(index),
      "benchmark_manifest_sha256":manifest_sha,
      "trace_manifest_sha256":trace_sha,"arm_order":order,
      "arms":["native","known_direct","certified","certified_budget","certified_graph"],
      "primary_method_contrast":"certified_graph_minus_native",
      "diagnostic_contrasts":[
        "known_direct_minus_native","certified_minus_native",
        "certified_graph_minus_certified_budget"
      ],
    },f,indent=2,sort_keys=True); f.write("\n")
PY

common=(
  --episode_root "${BENCH_ROOT}/${scene}"
  --episode_ids "${episode}"
  --scene "${scene_file}"
  --host 127.0.0.1
  --port "${MEMNAV_PORT}"
  --novel_port "${NAVDP_PORT}"
  --server_backend hybrid_pose
  --success_dist 1.0
  --max_steps "${MAX_STEPS}"
  --exec_horizon 8
  --trajectory_selector server
  --trajectory_selector_scope all
  --navdp_goal_switch_reset before_c
  --leg1_mode shared_trace
  --shared_leg1_trace_root "${TRACE_ROOT}/${scene}"
  --leg1_goal_source own
  --seed 0
  --terminal_uturn off
  --terminal_visual_refine off
  --deterministic_plan_seeds
  --retrieval_override off
  --double_revisit_c_history initial_leg_only
  --certified_cdec_rescue off
  --revisit_controller navdp_mixed
)

for arm in "${arm_order[@]}"; do
  arm_root=${episode_out}/${arm}
  mkdir "${arm_root}"
  case "${arm}" in
    native|known_direct)
      route=(--hybrid_route phase --revisit_adapter legacy_metric)
      ;;
    certified|certified_budget|certified_graph)
      route=(--hybrid_route certified_relocalization \
             --revisit_adapter verified_bearing_v1)
      ;;
    *) fail "unknown arm ${arm}" ;;
  esac
  case "${arm}" in
    certified_budget) stagnation=budget_control ;;
    certified_graph) stagnation=rescue ;;
    *) stagnation=off ;;
  esac
  echo "[run] ${selection_index} ${scene}/${episode}/${arm}"
  hab_python -u "${SOURCE_ROOT}/MemNavData/eval_shared_online_novel_revisit.py" \
    "${common[@]}" "${route[@]}" --shared_online_nnr_arm "${arm}" \
    --certified_stagnation_graph "${stagnation}" --out "${arm_root}" \
    > "${episode_out}/logs/eval_${arm}.log" 2>&1
done

hab_python - "${episode_out}" "${episode}" <<'PY'
import csv,hashlib,json,sys
from pathlib import Path
root=Path(sys.argv[1]); episode=sys.argv[2]
arms=("native","known_direct","certified","certified_budget","certified_graph")
payload={}
metrics={}
for arm in arms:
    summary=json.load(open(root/arm/"summary.json"))
    if summary["episodes"] != 1 or summary["arm"] != arm:
        raise SystemExit(f"{arm}: incomplete summary")
    with open(root/arm/"metric.csv",newline="") as f: rows=list(csv.DictReader(f))
    if len(rows)!=1 or rows[0]["episode"]!=episode or rows[0]["arm"]!=arm:
        raise SystemExit(f"{arm}: metric identity changed")
    metrics[arm]=rows[0]
    payload[arm]=json.load(open(root/arm/f"{episode}_plans.json"))

reference=payload["native"]
for arm in arms[1:]:
    other=payload[arm]
    for key in ("frozen_legA","frozen_legB"):
        if reference[key] != other[key]:
            raise SystemExit(f"{arm}: shared {key} plans differ")
    for trace_kind in ("rollout_traces","memory_traces"):
        for leg in ("legA","legB"):
            if reference[trace_kind][leg] != other[trace_kind][leg]:
                raise SystemExit(f"{arm}: shared {trace_kind}/{leg} differs")
    for field in ("online_A_trace_sha256","online_B_trace_sha256"):
        if metrics[arm][field] != metrics["native"][field]:
            raise SystemExit(f"{arm}: prefix hash differs")

receipt={
  "schema_version":"shared_online_nnr_episode_completion_v1_20260814",
  "episode":episode,
  "prefix_equality":True,
  "outcomes":{arm:int(metrics[arm]["reached_C"]) for arm in arms},
  "termination":{arm:metrics[arm]["termination_C"] for arm in arms},
  "final_distance_m":{arm:float(metrics[arm]["final_dist_C"]) for arm in arms},
}
encoded=(json.dumps(receipt,indent=2,sort_keys=True)+"\n").encode()
(root/"completion.json").write_bytes(encoded)
(root/"completion.json.sha256").write_text(
    hashlib.sha256(encoded).hexdigest()+"  completion.json\n")
PY

echo "[complete] ${episode_out}"
