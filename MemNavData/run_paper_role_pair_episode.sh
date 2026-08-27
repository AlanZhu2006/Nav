#!/usr/bin/env bash
# Evaluate one frozen paper history under one query protocol and all five arms.

set -euo pipefail
umask 0022

SOURCE_ROOT=${SOURCE_ROOT:?set immutable task source root}
RUN_ROOT=${RUN_ROOT:?set paper run root}
BENCH_ROOT=${BENCH_ROOT:?set sealed role-pair benchmark root}
EXPECTED_MANIFEST_SHA=${EXPECTED_MANIFEST_SHA:?set benchmark manifest SHA}
PROTOCOL=${PROTOCOL:?set support_controlled, natural_direction, or hard_support}
EPISODE_INDEX=${EPISODE_INDEX:?set frozen population index}
HAB_PY=${HAB_PY:?set Habitat interpreter}
MEMNAV_PORT=${MEMNAV_PORT:?set MemNav port}
NAVDP_PORT=${NAVDP_PORT:?set NavDP port}
MAX_STEPS=${MAX_STEPS:-600}
INCLUDE_LEARNED_PI3X=${INCLUDE_LEARNED_PI3X:-0}
EXPECTED_PI3X_MODEL_SHA=${EXPECTED_PI3X_MODEL_SHA:-}
EXPECTED_PI3X_PROOF_SHA=${EXPECTED_PI3X_PROOF_SHA:-}

fail() { echo "ABORT: $*" >&2; exit 2; }
[[ "${PROTOCOL}" == support_controlled || "${PROTOCOL}" == natural_direction || \
   "${PROTOCOL}" == hard_support ]] || \
  fail "invalid protocol"
[[ "${EPISODE_INDEX}" =~ ^[0-9]+$ ]] || fail "invalid episode index"
[[ "${MAX_STEPS}" =~ ^[1-9][0-9]*$ ]] || fail "invalid max steps"
[[ "${INCLUDE_LEARNED_PI3X}" == 0 || "${INCLUDE_LEARNED_PI3X}" == 1 ]] || \
  fail "INCLUDE_LEARNED_PI3X must be 0 or 1"
if [[ "${INCLUDE_LEARNED_PI3X}" == 1 ]]; then
  [[ "${EXPECTED_PI3X_MODEL_SHA}" =~ ^[0-9a-f]{64}$ ]] || \
    fail "invalid Pi3X model SHA"
  [[ "${EXPECTED_PI3X_PROOF_SHA}" =~ ^[0-9a-f]{64}$ ]] || \
    fail "invalid Pi3X proof SHA"
fi
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
        for chunk in iter(lambda:handle.read(8<<20),b""):
            digest.update(chunk)
    return digest.hexdigest()

manifest=json.load(open(sys.argv[1])); index=int(sys.argv[2])
rows=manifest["episodes"]
if index >= len(rows):
    print("SKIP"); print(len(rows)); raise SystemExit(0)
row=rows[index]
scene=str(row["scene"]); episode=str(row["episode"])
receipt=json.load(open(Path(row["online_a_episode"])/"receipt.json"))
asset=Path(receipt["source_asset"])
if not asset.is_file() or sha(asset) != receipt["source_asset_sha256"]:
    raise SystemExit("source scene asset changed")
print("RUN"); print(scene); print(episode); print(asset); print(index)
PY
)
[[ "${#identity[@]}" -ge 2 ]] || fail "identity reader failed"
if [[ "${identity[0]}" == SKIP ]]; then
  mkdir -p "${RUN_ROOT}/evaluation/${PROTOCOL}/skipped"
  hab_python - "${RUN_ROOT}/evaluation/${PROTOCOL}/skipped/$(printf '%03d' "${EPISODE_INDEX}").json" \
    "${identity[1]}" "${EPISODE_INDEX}" "${PROTOCOL}" <<'PY'
import json,sys
with open(sys.argv[1],"x",encoding="utf-8") as handle:
    json.dump({"protocol":sys.argv[4],"population":int(sys.argv[2]),
               "array_index":int(sys.argv[3])},handle,indent=2,sort_keys=True)
    handle.write("\n")
PY
  exit 0
fi
[[ "${#identity[@]}" -eq 5 && "${identity[0]}" == RUN ]] || \
  fail "malformed episode identity"
scene=${identity[1]}; episode=${identity[2]}; scene_file=${identity[3]}
selection_index=${identity[4]}; label=$(printf '%03d' "${selection_index}")
episode_out=${RUN_ROOT}/evaluation/${PROTOCOL}/${label}_${scene}_${episode}
[[ ! -e "${episode_out}" ]] || fail "episode output exists: ${episode_out}"
mkdir -p "${episode_out}/logs"

if [[ "${INCLUDE_LEARNED_PI3X}" == 1 ]]; then
  arms=(native raw_fixed_bearing geometry_fixed certified learned_pi3x_spatial)
else
  arms=(native raw_direct raw_fixed_bearing geometry_fixed certified)
fi
protocol_offset=0
[[ "${PROTOCOL}" == natural_direction ]] && protocol_offset=2
[[ "${PROTOCOL}" == hard_support ]] && protocol_offset=4
offset=$(( (10#${selection_index} + protocol_offset) % ${#arms[@]} ))
arm_order=()
for ((i=0; i<${#arms[@]}; i++)); do
  arm_order+=("${arms[$(((offset+i)%${#arms[@]}))]}")
done
printf '%s\n' "${arm_order[@]}" > "${episode_out}/arm_order.txt"

hab_python - "${episode_out}/episode_contract.json" "${scene}" "${episode}" \
  "${selection_index}" "${PROTOCOL}" "${EXPECTED_MANIFEST_SHA}" \
  "${MAX_STEPS}" "${INCLUDE_LEARNED_PI3X}" \
  "${EXPECTED_PI3X_MODEL_SHA}" "${EXPECTED_PI3X_PROOF_SHA}" \
  "${arm_order[@]}" <<'PY'
import json,sys
(path,scene,episode,index,protocol,manifest_sha,max_steps,include_learned,
 model_sha,proof_sha,*order)=sys.argv[1:]
learned=bool(int(include_learned))
arms=(
  ["native","raw_fixed_bearing","geometry_fixed","certified",
   "learned_pi3x_spatial"]
  if learned else
  ["native","raw_direct","raw_fixed_bearing","geometry_fixed","certified"]
)
with open(path,"x",encoding="utf-8") as handle:
    json.dump({
      "schema_version":"paper_role_pair_paired_episode_v2_20260817",
      "scene":scene,"episode":episode,"selection_index":int(index),
      "protocol":protocol,"benchmark_manifest_sha256":manifest_sha,
      "arm_order":order,"arms":arms,
      "paper_arm_names":{
        "raw_direct":"raw_metric","raw_fixed_bearing":"raw_fixed",
        "geometry_fixed":"geometry_fixed",
        "learned_pi3x_spatial":"learned_pi3x_spatial"},
      "max_steps":int(max_steps),"success_radius_m":1.0,
      "exec_horizon":8,"deterministic_plan_seeds":True,
      "runtime_role_visibility":"none",
      "primary_contrasts":["certified-native","certified-raw_fixed_bearing",
                           "certified-geometry_fixed"] + (
          ["learned_pi3x_spatial-native",
           "learned_pi3x_spatial-certified"] if learned else []),
      "learned_pi3x":({
          "model_sha256":model_sha,
          "proof_manifest_sha256":proof_sha,
          "consensus":"2/4","bridge_frames":16,
          "controller_residual_m":2.5,
      } if learned else None),
    },handle,indent=2,sort_keys=True); handle.write("\n")
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
  --role_pair_scope paper_heldout
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
    raw_direct)
      route=(--port "${MEMNAV_PORT}" --novel_port "${NAVDP_PORT}" \
        --server_backend hybrid_pose --hybrid_route phase \
        --revisit_adapter legacy_metric)
      ;;
    raw_fixed_bearing)
      route=(--port "${MEMNAV_PORT}" --novel_port "${NAVDP_PORT}" \
        --server_backend hybrid_pose --hybrid_route phase \
        --revisit_adapter raw_fixed_bearing_v1)
      ;;
    geometry_fixed)
      route=(--port "${MEMNAV_PORT}" --novel_port "${NAVDP_PORT}" \
        --server_backend hybrid_pose --hybrid_route memory_geometry \
        --revisit_adapter verified_bearing_v1 \
        --router_visual_floor 0.88 --router_min_matches 20 \
        --router_min_inliers 12 --router_min_inlier_ratio 0.50 \
        --router_confirm_plans 2 --router_verify_top_k 8)
      ;;
    certified)
      route=(--port "${MEMNAV_PORT}" --novel_port "${NAVDP_PORT}" \
        --server_backend hybrid_pose --hybrid_route certified_relocalization \
        --revisit_adapter verified_bearing_v1)
      ;;
    learned_pi3x_spatial)
      route=(--port "${MEMNAV_PORT}" --novel_port "${NAVDP_PORT}" \
        --server_backend hybrid_pose \
        --hybrid_route learned_pi3x_relocalization \
        --revisit_adapter verified_bearing_v1 \
        --expected_pi3x_model_sha256 "${EXPECTED_PI3X_MODEL_SHA}" \
        --expected_pi3x_proof_manifest_sha256 \
          "${EXPECTED_PI3X_PROOF_SHA}")
      ;;
    *) fail "unknown arm ${arm}" ;;
  esac
  echo "[run] ${PROTOCOL} ${selection_index} ${scene}/${episode}/${arm}"
  start_ns=$(date +%s%N)
  hab_python -u "${SOURCE_ROOT}/MemNavData/eval_shared_online_role_pairs.py" \
    "${common[@]}" "${route[@]}" --out "${arm_root}" \
    > "${episode_out}/logs/eval_${arm}.log" 2>&1
  end_ns=$(date +%s%N)
  timing_args+=("${arm}:$((end_ns-start_ns))")
done

hab_python - "${episode_out}" "${episode}" "${PROTOCOL}" \
  "${INCLUDE_LEARNED_PI3X}" "${EXPECTED_PI3X_MODEL_SHA}" \
  "${EXPECTED_PI3X_PROOF_SHA}" \
  "${timing_args[@]}" <<'PY'
import csv,hashlib,json,math,sys
from pathlib import Path

root=Path(sys.argv[1]); episode=sys.argv[2]; protocol=sys.argv[3]
learned=bool(int(sys.argv[4])); model_sha=sys.argv[5]; proof_sha=sys.argv[6]
timings={}
for token in sys.argv[7:]:
    name,value=token.split(":",1); timings[name]=int(value)/1e9
arms=(
    ("native","raw_fixed_bearing","geometry_fixed","certified",
     "learned_pi3x_spatial")
    if learned else
    ("native","raw_direct","raw_fixed_bearing","geometry_fixed","certified")
)
metrics={}; payloads={}
for arm in arms:
    summary=json.load(open(root/arm/"summary.json"))
    expected_arm=arm
    if summary.get("queries") != 2 or summary.get("arm") != expected_arm:
        raise SystemExit(f"{arm}: incomplete or mislabeled summary")
    if summary.get("scope") != "paper held-out role-pair evaluation":
        raise SystemExit(f"{arm}: wrong evaluation scope")
    if summary.get("runtime_role_visibility") != "none":
        raise SystemExit(f"{arm}: runtime role leak")
    with open(root/arm/"metric.csv",newline="") as handle:
        rows=list(csv.DictReader(handle))
    if len(rows)!=2 or {row["analysis_role"] for row in rows}!={"novel","revisit"}:
        raise SystemExit(f"{arm}: metric role population changed")
    metrics[arm]={row["analysis_role"]:row for row in rows}
    payloads[arm]={}
    for row in rows:
        role=row["analysis_role"]
        path=root/arm/f"{episode}_{row['query_id']}_plans.json"
        payload=json.load(open(path))
        if payload.get("analysis_role_not_forwarded") is not True:
            raise SystemExit(f"{arm}/{role}: role-forwarding audit failed")
        payloads[arm][role]=payload
    if arm == "learned_pi3x_spatial":
        status=summary.get("memnav_server_info",{}).get(
            "learned_pi3x_relocalization")
        if (not isinstance(status,dict) or status.get("enabled") is not True
                or status.get("model_sha256") != model_sha
                or status.get("proof_manifest_sha256") != proof_sha):
            raise SystemExit("learned Pi3X server identity changed")

reference=payloads["native"]
for arm in arms[1:]:
    for role in ("novel","revisit"):
        left=reference[role]; right=payloads[arm][role]
        if left["legA"] != right["legA"]:
            raise SystemExit(f"{arm}/{role}: shared online-A plans differ")
        if left["rollout_traces"]["legA"] != right["rollout_traces"]["legA"]:
            raise SystemExit(f"{arm}/{role}: shared online-A rollout differs")
        for field in ("all_rgb_hashes_verified","decision_frames",
                      "decision_steps","diffusion_samples_during_replay",
                      "navdp_memory_size","navdp_queue_lengths","online_frames"):
            if left["replay"][field] != right["replay"][field]:
                raise SystemExit(f"{arm}/{role}: replay field {field} differs")
        a=metrics["native"][role]; b=metrics[arm][role]
        for field in ("scene","episode","pair_id","query_id","analysis_role",
                      "seed","shared_A_frames","shared_A_decision_frames",
                      "geodesic_m"):
            if a[field] != b[field]:
                raise SystemExit(f"{arm}/{role}: paired field {field} differs")
        if int(b["shared_A_hashes_ok"]) != 1 or int(b["shared_A_diffusion_samples"]) != 0:
            raise SystemExit(f"{arm}/{role}: online-A replay audit failed")

memnav_arms=[arm for arm in arms if arm != "native"]
memory_reference=memnav_arms[0]
for arm in memnav_arms[1:]:
    for role in ("novel","revisit"):
        if (payloads[memory_reference][role]["memory_traces"]["legA"] !=
                payloads[arm][role]["memory_traces"]["legA"]):
            raise SystemExit(f"{arm}/{role}: MemNav online-A memory differs")

if learned:
    for role in ("novel","revisit"):
        row=metrics["learned_pi3x_spatial"][role]
        accepts=int(row["learned_pi3x_accept_plans"])
        takeovers=int(row["adapter_takeover_plans"])
        if int(row["learned_pi3x_initial_inference_plans"]) != 1:
            raise SystemExit(f"learned/{role}: first-query lifecycle changed")
        if accepts != takeovers or int(row["runtime_failure_plans"]) != 0:
            raise SystemExit(f"learned/{role}: takeover/runtime contract failed")

receipt={
  "schema_version":"paper_role_pair_episode_completion_v2_20260817",
  "protocol":protocol,"episode":episode,"prefix_equality":True,
  "runtime_role_visibility":"none",
  "wall_time_seconds":timings,
  "outcomes":{
    arm:{role:int(metrics[arm][role]["reached"])
         for role in ("novel","revisit")} for arm in arms},
  "termination":{
    arm:{role:metrics[arm][role]["termination_reason"]
         for role in ("novel","revisit")} for arm in arms},
  "final_distance_m":{
    arm:{role:float(metrics[arm][role]["final_goal_dist_m"])
         for role in ("novel","revisit")} for arm in arms},
  "learned_pi3x":({
      "model_sha256":model_sha,
      "proof_manifest_sha256":proof_sha,
  } if learned else None),
}
encoded=(json.dumps(receipt,indent=2,sort_keys=True)+"\n").encode()
(root/"completion.json").write_bytes(encoded)
(root/"completion.json.sha256").write_text(
    hashlib.sha256(encoded).hexdigest()+"  completion.json\n")
PY

echo "[complete] ${episode_out}"
