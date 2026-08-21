#!/usr/bin/env bash
# Run all four Novel causal-control arms for one frozen consumed episode.

set -euo pipefail
umask 0022

SOURCE_ROOT=${SOURCE_ROOT:?set immutable task source root}
RUN_ROOT=${RUN_ROOT:?set Novel-control run root}
BENCH_ROOT=${BENCH_ROOT:?set sealed natural-direction benchmark root}
CONTROL_MANIFEST=${CONTROL_MANIFEST:?set frozen control manifest}
EXPECTED_CONTROL_MANIFEST_SHA=${EXPECTED_CONTROL_MANIFEST_SHA:?set control manifest SHA}
EPISODE_INDEX=${EPISODE_INDEX:?set frozen population index}
HAB_PY=${HAB_PY:?set Habitat interpreter}
MEMNAV_PORT=${MEMNAV_PORT:?set MemNav port}
NAVDP_PORT=${NAVDP_PORT:?set NavDP port}
MAX_STEPS=${MAX_STEPS:-600}

fail() { echo "ABORT: $*" >&2; exit 2; }
[[ "${EPISODE_INDEX}" =~ ^[0-9]+$ ]] || fail "invalid episode index"
[[ "${MAX_STEPS}" =~ ^[1-9][0-9]*$ ]] || fail "invalid max steps"
[[ "$(sha256sum "${CONTROL_MANIFEST}" | awk '{print $1}')" == \
   "${EXPECTED_CONTROL_MANIFEST_SHA}" ]] || fail "control manifest changed"

HAB_SITE_PACKAGES=$("${HAB_PY}" -c \
  'import sysconfig; print(sysconfig.get_paths()["purelib"])')
HAB_PYTHONPATH=${SOURCE_ROOT}:${SOURCE_ROOT}/MemNavData:${HAB_SITE_PACKAGES}/pip/_vendor${PYTHONPATH:+:${PYTHONPATH}}
hab_python() { env PYTHONPATH="${HAB_PYTHONPATH}" "${HAB_PY}" "$@"; }

readarray -t identity < <(hab_python - \
  "${CONTROL_MANIFEST}" "${BENCH_ROOT}/manifest.json" \
  "${EPISODE_INDEX}" <<'PY'
import hashlib,json,sys
from pathlib import Path

def sha(path):
    digest=hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda:handle.read(8<<20),b""):
            digest.update(chunk)
    return digest.hexdigest()

control=json.load(open(sys.argv[1])); benchmark=Path(sys.argv[2]); index=int(sys.argv[3])
if sha(benchmark) != control["benchmark_manifest_sha256"]:
    raise SystemExit("benchmark manifest changed")
rows=control["episodes"]
if index >= len(rows):
    print("SKIP"); print(len(rows)); raise SystemExit(0)
row=rows[index]
scene=str(row["scene"]); episode=str(row["episode"])
sidecar=Path(row["role_pairs_path"])
if sha(sidecar) != row["role_pairs_sha256"]:
    raise SystemExit("role-pair sidecar changed")
payload=json.load(open(sidecar))
receipt=json.load(open(Path(payload["online_a_episode"])/"receipt.json"))
asset=Path(receipt["source_asset"])
if not asset.is_file() or sha(asset) != receipt["source_asset_sha256"]:
    raise SystemExit("source scene asset changed")
print("RUN"); print(scene); print(episode); print(asset); print(index)
for arm in row["arm_order"]:
    print(arm)
PY
)
[[ "${#identity[@]}" -ge 2 ]] || fail "identity reader failed"
if [[ "${identity[0]}" == SKIP ]]; then
  mkdir -p "${RUN_ROOT}/evaluation/skipped"
  hab_python - "${RUN_ROOT}/evaluation/skipped/$(printf '%03d' "${EPISODE_INDEX}").json" \
    "${identity[1]}" "${EPISODE_INDEX}" <<'PY'
import json,sys
with open(sys.argv[1],"x",encoding="utf-8") as handle:
    json.dump({"population":int(sys.argv[2]),"array_index":int(sys.argv[3])},
              handle,indent=2,sort_keys=True)
    handle.write("\n")
PY
  exit 0
fi
[[ "${#identity[@]}" -eq 9 && "${identity[0]}" == RUN ]] || \
  fail "malformed episode identity"
scene=${identity[1]}; episode=${identity[2]}; scene_file=${identity[3]}
selection_index=${identity[4]}; label=$(printf '%03d' "${selection_index}")
arm_order=("${identity[@]:5:4}")
episode_out=${RUN_ROOT}/evaluation/${label}_${scene}_${episode}
[[ ! -e "${episode_out}" ]] || fail "episode output exists: ${episode_out}"
mkdir -p "${episode_out}/logs"
printf '%s\n' "${arm_order[@]}" > "${episode_out}/arm_order.txt"

hab_python - "${episode_out}/episode_contract.json" "${scene}" "${episode}" \
  "${selection_index}" "${EXPECTED_CONTROL_MANIFEST_SHA}" \
  "${MAX_STEPS}" "${arm_order[@]}" <<'PY'
import json,sys
(path,scene,episode,index,manifest_sha,max_steps,*order)=sys.argv[1:]
with open(path,"x",encoding="utf-8") as handle:
    json.dump({
      "schema_version":"novel_memory_direction_episode_v1_20260816",
      "scene":scene,"episode":episode,"selection_index":int(index),
      "control_manifest_sha256":manifest_sha,
      "arm_order":order,
      "arms":["native","raw_factual_history","raw_deranged_history",
              "raw_randomized_bearing"],
      "query_role":"novel",
      "evaluation_stage":"consumed_development_mechanism_only",
      "confirmation_claim_allowed":False,
      "method_or_threshold_selection_allowed":False,
      "max_steps":int(max_steps),"success_radius_m":1.0,
      "exec_horizon":8,"deterministic_plan_seeds":True,
      "runtime_role_visibility":"none",
      "primary_contrasts":[
        "raw_factual_history-raw_randomized_bearing",
        "raw_factual_history-raw_deranged_history",
        "raw_factual_history-native",
        "raw_deranged_history-native",
        "raw_randomized_bearing-native"],
    },handle,indent=2,sort_keys=True)
    handle.write("\n")
PY

common=(
  --episode_root "${BENCH_ROOT}/${scene}"
  --episode_ids "${episode}"
  --scene "${scene_file}"
  --scene_identity "${scene}"
  --host 127.0.0.1
  --success_dist 1.0
  --max_steps "${MAX_STEPS}"
  --exec_horizon 8
  --trajectory_selector server
  --trajectory_selector_scope all
  --leg1_mode shared_trace
  --leg1_goal_source own
  --seed 0
  --terminal_uturn off
  --terminal_visual_refine off
  --deterministic_plan_seeds
  --retrieval_override off
  --certified_cdec_rescue off
  --certified_stagnation_graph off
  --revisit_controller navdp_mixed
  --role_pair_scope consumed_integration
  --role_pair_query_role novel
)

timing_args=()
for arm in "${arm_order[@]}"; do
  arm_root=${episode_out}/${arm}
  mkdir "${arm_root}"
  case "${arm}" in
    native)
      route=(--port "${NAVDP_PORT}" --server_backend navdp \
        --hybrid_route phase --revisit_adapter legacy_metric)
      ;;
    raw_factual_history|raw_deranged_history|raw_randomized_bearing)
      route=(--port "${MEMNAV_PORT}" --novel_port "${NAVDP_PORT}" \
        --server_backend hybrid_pose --hybrid_route phase \
        --revisit_adapter raw_fixed_bearing_v1)
      ;;
    *) fail "unknown arm ${arm}" ;;
  esac
  echo "[run] ${selection_index} ${scene}/${episode}/${arm}"
  start_ns=$(date +%s%N)
  env NOVEL_CONTROL_MANIFEST="${CONTROL_MANIFEST}" \
    EXPECTED_NOVEL_CONTROL_MANIFEST_SHA="${EXPECTED_CONTROL_MANIFEST_SHA}" \
    NOVEL_CONTROL_ARM="${arm}" \
    PYTHONPATH="${HAB_PYTHONPATH}" \
    "${HAB_PY}" -u \
      "${SOURCE_ROOT}/MemNavData/eval_novel_memory_direction_control.py" \
      "${common[@]}" "${route[@]}" --out "${arm_root}" \
      > "${episode_out}/logs/eval_${arm}.log" 2>&1
  end_ns=$(date +%s%N)
  timing_args+=("${arm}:$((end_ns-start_ns))")
done

hab_python - "${episode_out}" "${episode}" \
  "${EXPECTED_CONTROL_MANIFEST_SHA}" "${CONTROL_MANIFEST}" \
  "${timing_args[@]}" <<'PY'
import csv,hashlib,json,math,sys
from pathlib import Path

root=Path(sys.argv[1]); episode=sys.argv[2]
manifest_sha=sys.argv[3]; manifest_path=Path(sys.argv[4])
timings={}
for token in sys.argv[5:]:
    name,value=token.split(":",1); timings[name]=int(value)/1e9
arms=("native","raw_factual_history","raw_deranged_history",
      "raw_randomized_bearing")
metrics={}; payloads={}
for arm in arms:
    summary=json.load(open(root/arm/"summary.json"))
    if summary.get("queries") != 1 or summary.get("arm") != arm:
        raise SystemExit(f"{arm}: incomplete or mislabeled summary")
    control=summary.get("novel_causal_control") or {}
    if (summary.get("role_pair_scope") != "consumed_integration"
            or summary.get("role_pair_query_role") != "novel"
            or summary.get("runtime_role_visibility") != "none"
            or control.get("manifest_sha256") != manifest_sha
            or control.get("confirmation_claim_allowed") is not False):
        raise SystemExit(f"{arm}: causal scope audit failed")
    with open(root/arm/"metric.csv",newline="") as handle:
        rows=list(csv.DictReader(handle))
    if len(rows)!=1 or rows[0]["analysis_role"]!="novel":
        raise SystemExit(f"{arm}: Novel query population changed")
    metrics[arm]=rows[0]
    paths=list((root/arm).glob(f"{episode}_*_plans.json"))
    if len(paths)!=1:
        raise SystemExit(f"{arm}: expected one plan ledger")
    payload=json.load(open(paths[0]))
    audit=payload.get("novel_causal_control") or {}
    if (payload.get("arm") != arm
            or payload.get("analysis_role_not_forwarded") is not True
            or audit.get("arm") != arm
            or audit.get("manifest_sha256") != manifest_sha
            or audit.get("confirmation_claim_allowed") is not False):
        raise SystemExit(f"{arm}: plan ledger scope audit failed")
    for plan_index, plan in enumerate(payload["query_leg"]):
        requested=plan.get("requested_diffusion_seed")
        echoed=plan.get("diffusion_seed")
        if requested is None or echoed is None or int(requested)!=int(echoed):
            raise SystemExit(
                f"{arm}: requested/echoed diffusion seed differs at plan {plan_index}")
    payloads[arm]=payload

reference=payloads["native"]
identity_fields=("scene","episode","pair_id","query_id","analysis_role",
                 "seed","shared_A_frames","shared_A_decision_frames",
                 "geodesic_m")
replay_fields=("all_rgb_hashes_verified","decision_frames","decision_steps",
               "diffusion_samples_during_replay","navdp_memory_size",
               "navdp_queue_lengths","online_frames","factual_fifo_scene",
               "factual_fifo_episode","factual_fifo_frames",
               "factual_fifo_decision_sha256")
for arm in arms[1:]:
    right=payloads[arm]
    if reference["legA"] != right["legA"]:
        raise SystemExit(f"{arm}: shared online-A plans differ")
    if reference["rollout_traces"]["legA"] != right["rollout_traces"]["legA"]:
        raise SystemExit(f"{arm}: shared online-A rollout differs")
    for field in replay_fields:
        if reference["replay"].get(field) != right["replay"].get(field):
            raise SystemExit(f"{arm}: factual replay field {field} differs")
    for field in identity_fields:
        if metrics["native"][field] != metrics[arm][field]:
            raise SystemExit(f"{arm}: paired field {field} differs")
    if int(metrics[arm]["adapter_takeover_plans"]) == 0:
        if (reference["rollout_traces"]["query"]
                != right["rollout_traces"]["query"]):
            raise SystemExit(f"{arm}: zero-takeover path is not exact native fallback")

factual=payloads["raw_factual_history"]
deranged=payloads["raw_deranged_history"]
randomized=payloads["raw_randomized_bearing"]
for payload in (factual, randomized):
    replay=payload["replay"]
    if (replay.get("sidecar_is_deranged") is not False
            or replay.get("sidecar_scene") != metrics["native"]["scene"]
            or replay.get("sidecar_episode") != metrics["native"]["episode"]):
        raise SystemExit("factual sidecar identity changed")
if (factual["replay"]["sidecar_memory_sha256"]
        != randomized["replay"]["sidecar_memory_sha256"]
        or factual["memory_traces"]["legA"]
        != randomized["memory_traces"]["legA"]):
    raise SystemExit("factual/randomized long-term history differs")
if (deranged["replay"].get("sidecar_is_deranged") is not True
        or deranged["replay"].get("sidecar_scene")
        == factual["replay"].get("sidecar_scene")
        and deranged["replay"].get("sidecar_episode")
        == factual["replay"].get("sidecar_episode")):
    raise SystemExit("deranged sidecar retained factual identity")
if (deranged["replay"]["sidecar_memory_sha256"]
        == factual["replay"]["sidecar_memory_sha256"]):
    raise SystemExit("deranged sidecar did not change RGB content")

ledger=randomized["novel_causal_control"]["randomized_bearing_ledger"]
if len(ledger) != len(randomized["query_leg"]):
    raise SystemExit("randomized ledger length differs from plan count")
if not ledger:
    raise SystemExit("randomized arm emitted no decisions")
first_factual=factual["query_leg"][0]
first_random=randomized["query_leg"][0]
first_audit=ledger[0]
if bool(first_factual.get("revisit_adapter_takeover")) != bool(
        first_audit["factual_takeover"]):
    raise SystemExit("first randomized proposal availability was not preserved")
if first_audit["factual_takeover"]:
    factual_point=[float(v) for v in first_factual["memory_unbounded_pointgoal"]]
    ledger_point=[float(v) for v in first_audit["factual_raw_pointgoal"]]
    if any(abs(a-b)>1e-6 for a,b in zip(factual_point,ledger_point)):
        raise SystemExit("first factual proposal differs before angle intervention")
    randomized_point=[float(v) for v in first_random["memory_unbounded_pointgoal"]]
    expected=[float(v) for v in first_audit["randomized_controller_pointgoal"]]
    # The plan stores the unbounded transformed vector, while the adapter
    # ledger also stores the fixed-radius controller vector.  Their directions
    # must agree and the executed radius remains 2.5 m.
    cross=randomized_point[0]*expected[1]-randomized_point[1]*expected[0]
    dot=randomized_point[0]*expected[0]+randomized_point[1]*expected[1]
    if abs(cross)>1e-6 or dot<=0:
        raise SystemExit("randomized plan direction differs from frozen ledger")
    if not math.isclose(float(first_random["memory_pointgoal_fixed_radius_m"]),
                        2.5,rel_tol=0.0,abs_tol=1e-12):
        raise SystemExit("randomized execution radius changed")

manifest=json.load(open(manifest_path))
row=manifest["episodes"][int(json.load(open(root/"episode_contract.json"))[
    "selection_index"])]
expected_order=row["arm_order"]
actual_order=(root/"arm_order.txt").read_text().splitlines()
if actual_order != expected_order:
    raise SystemExit("executed arm order differs from frozen manifest")

receipt={
  "schema_version":"novel_memory_direction_completion_v1_20260816",
  "evaluation_stage":"consumed_development_mechanism_only",
  "confirmation_claim_allowed":False,
  "method_or_threshold_selection_allowed":False,
  "control_manifest_sha256":manifest_sha,
  "scene":metrics["native"]["scene"],
  "episode":metrics["native"]["episode"],
  "query_id":metrics["native"]["query_id"],
  "prefix_equality":True,
  "factual_fifo_equality":True,
  "deranged_sidecar_verified":True,
  "randomized_bearing_verified":True,
  "zero_takeover_exact_fallback_verified":True,
  "runtime_role_visibility":"none",
  "wall_time_seconds":timings,
  "outcomes":{arm:int(metrics[arm]["reached"]) for arm in arms},
  "geodesic_m":{arm:float(metrics[arm]["geodesic_m"]) for arm in arms},
  "path_length_m":{arm:float(metrics[arm]["path_len_m"]) for arm in arms},
  "steps":{arm:int(metrics[arm]["steps"]) for arm in arms},
  "spl":{
      arm:(
          int(metrics[arm]["reached"])
          * (float(metrics[arm]["geodesic_m"])
             / max(float(metrics[arm]["geodesic_m"]),
                   float(metrics[arm]["path_len_m"])))
          if max(float(metrics[arm]["geodesic_m"]),
                 float(metrics[arm]["path_len_m"])) > 0.0
          else float(int(metrics[arm]["reached"])))
      for arm in arms},
  "termination":{arm:metrics[arm]["termination_reason"] for arm in arms},
  "final_distance_m":{
    arm:float(metrics[arm]["final_goal_dist_m"]) for arm in arms},
  "final_geodesic_m":{
    arm:float(metrics[arm]["final_goal_geodesic_m"]) for arm in arms},
  "takeover_plans":{
      arm:int(metrics[arm]["adapter_takeover_plans"]) for arm in arms},
  "plan_count":{arm:len(payloads[arm]["query_leg"]) for arm in arms},
  "fallback_plans":{
      arm:(len(payloads[arm]["query_leg"])
           - int(metrics[arm]["adapter_takeover_plans"]))
      for arm in arms},
}
encoded=(json.dumps(receipt,indent=2,sort_keys=True)+"\n").encode()
(root/"completion.json").write_bytes(encoded)
(root/"completion.json.sha256").write_text(
    hashlib.sha256(encoded).hexdigest()+"  completion.json\n")
PY

echo "[complete] ${episode_out}"
