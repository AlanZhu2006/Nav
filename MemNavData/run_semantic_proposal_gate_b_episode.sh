#!/usr/bin/env bash
# One consumed Revisit history, two paired certified-proposal arms.

set -euo pipefail
umask 0022

SOURCE_ROOT=${SOURCE_ROOT:?set immutable task source root}
RUN_ROOT=${RUN_ROOT:?set Gate-B run root}
AUDIT_MANIFEST=${AUDIT_MANIFEST:?set frozen 28-history manifest}
EXPECTED_AUDIT_MANIFEST_SHA=${EXPECTED_AUDIT_MANIFEST_SHA:?set manifest SHA}
POPULATION_INDEX=${POPULATION_INDEX:?set frozen population index}
HAB_PY=${HAB_PY:?set Habitat interpreter}
MEMNAV_PORT=${MEMNAV_PORT:?set MemNav port}
NAVDP_PORT=${NAVDP_PORT:?set NavDP port}
MAX_STEPS=${MAX_STEPS:-600}

fail() { echo "ABORT: $*" >&2; exit 2; }
[[ "${POPULATION_INDEX}" =~ ^[0-9]+$ ]] || fail "invalid population index"
(( POPULATION_INDEX < 28 )) || fail "population index outside frozen Gate-B set"
[[ "${MAX_STEPS}" =~ ^[1-9][0-9]*$ ]] || fail "invalid max steps"
[[ "$(sha256sum "${AUDIT_MANIFEST}" | awk '{print $1}')" == \
   "${EXPECTED_AUDIT_MANIFEST_SHA}" ]] || fail "audit manifest changed"

HAB_SITE_PACKAGES=$("${HAB_PY}" -c \
  'import sysconfig; print(sysconfig.get_paths()["purelib"])')
HAB_PYTHONPATH=${SOURCE_ROOT}:${SOURCE_ROOT}/MemNavData:${HAB_SITE_PACKAGES}/pip/_vendor${PYTHONPATH:+:${PYTHONPATH}}
hab_python() { env PYTHONPATH="${HAB_PYTHONPATH}" "${HAB_PY}" "$@"; }

readarray -t identity < <(hab_python - \
  "${AUDIT_MANIFEST}" "${POPULATION_INDEX}" <<'PY'
import hashlib,json,sys
from pathlib import Path

def sha(path):
    digest=hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda:handle.read(8<<20),b""):
            digest.update(block)
    return digest.hexdigest()

audit=json.load(open(sys.argv[1])); index=int(sys.argv[2])
if audit.get("scope") != "consumed_posthoc_method_development_diagnostic":
    raise SystemExit("audit scope changed")
rows=audit["episodes"]
if len(rows)!=28 or index>=len(rows):
    raise SystemExit("frozen population changed")
entry=rows[index]; root=Path(entry["benchmark_root"])
manifest_path=root/"manifest.json"
if sha(manifest_path)!=entry["benchmark_manifest_sha256"]:
    raise SystemExit("benchmark manifest changed")
manifest=json.load(open(manifest_path)); row=manifest["episodes"][int(entry["population_index"])]
scene=str(row["scene"]); episode=str(row["episode"])
receipt=json.load(open(Path(row["online_a_episode"])/"receipt.json"))
asset=Path(receipt["source_asset"])
if not asset.is_file() or sha(asset)!=receipt["source_asset_sha256"]:
    raise SystemExit("source scene asset changed")
print(entry["cohort"]); print(root); print(entry["benchmark_manifest_sha256"])
print(scene); print(episode); print(asset); print(entry["population_index"])
PY
)
[[ "${#identity[@]}" -eq 7 ]] || fail "malformed frozen identity"
cohort=${identity[0]}; bench_root=${identity[1]}; bench_manifest_sha=${identity[2]}
scene=${identity[3]}; episode=${identity[4]}; scene_file=${identity[5]}
benchmark_index=${identity[6]}; label=$(printf '%03d' "${POPULATION_INDEX}")
episode_out=${RUN_ROOT}/records/${label}_${cohort}_${scene}_${episode}
[[ ! -e "${episode_out}" ]] || fail "episode output exists: ${episode_out}"
mkdir -p "${episode_out}/logs"

arms=(geometry_first semantic_first)
if (( POPULATION_INDEX % 2 == 1 )); then
  arms=(semantic_first geometry_first)
fi
printf '%s\n' "${arms[@]}" > "${episode_out}/arm_order.txt"

hab_python - "${episode_out}/episode_contract.json" \
  "${POPULATION_INDEX}" "${benchmark_index}" "${cohort}" "${scene}" \
  "${episode}" "${bench_root}" "${bench_manifest_sha}" \
  "${EXPECTED_AUDIT_MANIFEST_SHA}" "${MAX_STEPS}" "${arms[@]}" <<'PY'
import json,sys
(path,population_index,benchmark_index,cohort,scene,episode,benchmark_root,
 benchmark_manifest_sha,audit_manifest_sha,max_steps,*arm_order)=sys.argv[1:]
payload={
 "schema_version":"semantic_proposal_gate_b_episode_v1_20260815",
 "scope":"consumed_closed_loop_development_never_confirmation",
 "population_index":int(population_index),
 "benchmark_index":int(benchmark_index),
 "cohort":cohort,
 "scene":scene,
 "episode":episode,
 "benchmark_root":benchmark_root,
 "benchmark_manifest_sha256":benchmark_manifest_sha,
 "audit_manifest_sha256":audit_manifest_sha,
 "arm_order":arm_order,
 "arms":["geometry_first","semantic_first"],
 "query_role":"revisit",
 "runtime_role_visibility":"none",
 "max_steps":int(max_steps),
 "success_radius_m":1.0,
 "exec_horizon":8,
 "deterministic_plan_seeds":True,
}
with open(path,"x") as handle:
    json.dump(payload,handle,indent=2,sort_keys=True); handle.write("\n")
PY

common=(
  --episode_root "${bench_root}/${scene}"
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
  --role_pair_query_role revisit
  --port "${MEMNAV_PORT}"
  --novel_port "${NAVDP_PORT}"
  --server_backend hybrid_pose
  --revisit_adapter verified_bearing_v1
)

timing_args=()
for arm in "${arms[@]}"; do
  arm_root=${episode_out}/${arm}
  mkdir "${arm_root}"
  if [[ "${arm}" == geometry_first ]]; then
    route=(--hybrid_route certified_relocalization)
  else
    route=(--hybrid_route certified_semantic_first)
  fi
  echo "[run] ${POPULATION_INDEX} ${cohort} ${scene}/${episode}/${arm}"
  start_ns=$(date +%s%N)
  hab_python -u "${SOURCE_ROOT}/MemNavData/eval_shared_online_role_pairs.py" \
    "${common[@]}" "${route[@]}" --out "${arm_root}" \
    > "${episode_out}/logs/eval_${arm}.log" 2>&1
  end_ns=$(date +%s%N)
  timing_args+=("${arm}:$((end_ns-start_ns))")
done

hab_python - "${episode_out}" "${POPULATION_INDEX}" \
  "${cohort}" "${scene}" "${episode}" "${timing_args[@]}" <<'PY'
import csv,hashlib,json,sys
from pathlib import Path

root=Path(sys.argv[1]); index=int(sys.argv[2]); cohort=sys.argv[3]
scene=sys.argv[4]; episode=sys.argv[5]
timings={}
for token in sys.argv[6:]:
    name,value=token.split(":",1); timings[name]=int(value)/1e9
arms=("geometry_first","semantic_first")
expected_arm={"geometry_first":"certified",
              "semantic_first":"semantic_first_certified"}
expected_order={"geometry_first":"geometry_first",
                "semantic_first":"dino_first_certified"}
metrics={}; payloads={}; proposal_orders={}; proposal_sources={}
selected_anchors={}
for arm in arms:
    summary=json.load(open(root/arm/"summary.json"))
    if summary.get("queries")!=1 or summary.get("arm")!=expected_arm[arm]:
        raise SystemExit(f"{arm}: incomplete or mislabeled summary")
    if summary.get("role_pair_scope")!="consumed_integration":
        raise SystemExit(f"{arm}: wrong scope")
    if summary.get("role_pair_query_role")!="revisit":
        raise SystemExit(f"{arm}: role filter changed")
    if summary.get("runtime_role_visibility")!="none":
        raise SystemExit(f"{arm}: role leak")
    with open(root/arm/"metric.csv",newline="") as handle:
        rows=list(csv.DictReader(handle))
    if len(rows)!=1 or rows[0]["analysis_role"]!="revisit":
        raise SystemExit(f"{arm}: wrong query population")
    row=rows[0]; metrics[arm]=row
    paths=list((root/arm).glob(f"{episode}_*_plans.json"))
    if len(paths)!=1:
        raise SystemExit(f"{arm}: expected one plan payload")
    payload=json.load(open(paths[0])); payloads[arm]=payload
    if payload.get("analysis_role_not_forwarded") is not True:
        raise SystemExit(f"{arm}: analysis role was forwarded")
    observed=[p.get("certified_relocalization_proposal_order")
              for p in payload["query_leg"]]
    if (not observed or any(value is None for value in observed)
            or set(observed)!={expected_order[arm]}):
        raise SystemExit(f"{arm}: wrong proposal order receipt")
    proposal_orders[arm]=observed
    proposal_sources[arm]=[
        p.get("certified_relocalization_selected_proposal_source")
        for p in payload["query_leg"]
        if p.get("certified_relocalization_selected_proposal_source") is not None]
    selected_anchors[arm]=[
        int(p["router_selected_anchor"])
        for p in payload["query_leg"]
        if p.get("router_selected_anchor") is not None]

left=payloads["geometry_first"]; right=payloads["semantic_first"]
if left["legA"]!=right["legA"]:
    raise SystemExit("shared online-A plans differ")
if left["rollout_traces"]["legA"]!=right["rollout_traces"]["legA"]:
    raise SystemExit("shared online-A rollout differs")
for field in ("all_rgb_hashes_verified","decision_frames","decision_steps",
              "diffusion_samples_during_replay","navdp_memory_size",
              "navdp_queue_lengths","online_frames"):
    if left["replay"][field]!=right["replay"][field]:
        raise SystemExit(f"shared replay field differs: {field}")
for field in ("scene","episode","pair_id","query_id","analysis_role",
              "seed","shared_A_frames","shared_A_decision_frames",
              "geodesic_m"):
    if metrics["geometry_first"][field]!=metrics["semantic_first"][field]:
        raise SystemExit(f"paired metric field differs: {field}")

raw_outcomes={arm:int(metrics[arm]["reached"]) for arm in arms}
runtime_failure_plans={}
endpoint_failure_plans={}
for arm in arms:
    ok_receipts=[
        plan.get("certified_relocalization_ok")
        for plan in payloads[arm]["query_leg"]
    ]
    if not ok_receipts or any(value not in (True, False)
                              for value in ok_receipts):
        raise SystemExit(f"{arm}: missing certified runtime receipt")
    runtime_failure_plans[arm]=sum(value is False for value in ok_receipts)
    endpoint_failure_plans[arm]=int(metrics[arm]["runtime_failure_plans"])
    if endpoint_failure_plans[arm] > runtime_failure_plans[arm]:
        raise SystemExit(f"{arm}: endpoint failure accounting is inconsistent")
# The preregistered Gate-B contract treats an infrastructure/runtime failure as
# an arm failure even when exact native fallback subsequently reaches the goal.
# Keep the raw physical outcome as an audit field; use the conservative value
# for every paired decision.
outcomes={
    arm:int(bool(raw_outcomes[arm]) and runtime_failure_plans[arm] == 0)
    for arm in arms
}
receipt={
 "schema_version":"semantic_proposal_gate_b_completion_v2_20260815",
 "scope":"consumed_closed_loop_development_never_confirmation",
 "population_index":index,"cohort":cohort,"scene":scene,"episode":episode,
 "query_role":"revisit","runtime_role_visibility":"none",
 "prefix_equality":True,
 "arm_order":json.load(open(root/"episode_contract.json"))["arm_order"],
 "wall_time_seconds":timings,
 "raw_outcomes":raw_outcomes,
 "outcomes":outcomes,
 "final_distance_m":{arm:float(metrics[arm]["final_goal_dist_m"])
                     for arm in arms},
 "steps":{arm:int(metrics[arm]["steps"]) for arm in arms},
 "runtime_failure_plans":runtime_failure_plans,
 "certificate_endpoint_failure_plans":endpoint_failure_plans,
 "certificate_accept_plans":{arm:int(metrics[arm]["certificate_accept_plans"])
                             for arm in arms},
 "adapter_takeover_plans":{arm:int(metrics[arm]["adapter_takeover_plans"])
                           for arm in arms},
 "proposal_orders":proposal_orders,"proposal_sources":proposal_sources,
 "selected_anchors":selected_anchors,
}
encoded=(json.dumps(receipt,indent=2,sort_keys=True)+"\n").encode()
(root/"completion.json").write_bytes(encoded)
(root/"completion.json.sha256").write_text(
    hashlib.sha256(encoded).hexdigest()+"  completion.json\n")
PY

echo "[complete] ${episode_out}"
