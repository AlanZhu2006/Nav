#!/usr/bin/env bash
# One frozen fresh160 episode, four same-process double-Revisit arms.

set -euo pipefail
umask 0022

SOURCE_ROOT=${SOURCE_ROOT:?set immutable task source root}
RUN_ROOT=${RUN_ROOT:?set run root}
BENCH_ROOT=${BENCH_ROOT:?set frozen benchmark root}
EXPECTED_MANIFEST_SHA=${EXPECTED_MANIFEST_SHA:?set frozen benchmark SHA}
EPISODE_INDEX=${EPISODE_INDEX:?set selection index}
HAB_PY=${HAB_PY:?set Habitat interpreter}
MEMNAV_PORT=${MEMNAV_PORT:?set MemNav port}
NAVDP_PORT=${NAVDP_PORT:?set NavDP port}
MAX_STEPS=${MAX_STEPS:-600}

fail() { echo "ABORT: $*" >&2; exit 2; }
[[ "${EPISODE_INDEX}" =~ ^[0-9]+$ ]] || \
  fail "EPISODE_INDEX must be a non-negative integer"
[[ "${MAX_STEPS}" =~ ^[1-9][0-9]*$ ]] || fail "MAX_STEPS must be positive"
[[ "$(sha256sum "${BENCH_ROOT}/manifest.json" | awk '{print $1}')" == \
   "${EXPECTED_MANIFEST_SHA}" ]] || fail "benchmark manifest changed"

HAB_SITE_PACKAGES=$("${HAB_PY}" -c \
  'import sysconfig; print(sysconfig.get_paths()["purelib"])')
HAB_PYTHONPATH=${SOURCE_ROOT}:${SOURCE_ROOT}/MemNavData:${HAB_SITE_PACKAGES}/pip/_vendor${PYTHONPATH:+:${PYTHONPATH}}
hab_python() { env PYTHONPATH="${HAB_PYTHONPATH}" "${HAB_PY}" "$@"; }

readarray -t identity < <(hab_python - \
  "${BENCH_ROOT}/manifest.json" "${EPISODE_INDEX}" <<'PY'
import hashlib, json, sys
from pathlib import Path

def sha(path):
    digest=hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8*1024*1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

manifest=json.load(open(sys.argv[1]))
index=int(sys.argv[2])
if not 0 <= index < len(manifest["episodes"]):
    raise SystemExit(
        f"episode index {index} outside frozen manifest of "
        f"{len(manifest['episodes'])} episodes"
    )
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
print(scene)
print(episode)
print(asset)
print(int(trace["episode_seed"]))
PY
)
[[ "${#identity[@]}" -eq 4 ]] || fail "episode identity reader failed"
scene=${identity[0]}
episode=${identity[1]}
scene_file=${identity[2]}
episode_seed=${identity[3]}
episode_root=${BENCH_ROOT}/${scene}
episode_label=$(printf '%03d' "${EPISODE_INDEX}")
episode_out=${RUN_ROOT}/scenes/${episode_label}_${scene}_${episode}
[[ ! -e "${episode_out}" ]] || fail "episode output already exists: ${episode_out}"
mkdir -p "${episode_out}/logs"

case $((10#${EPISODE_INDEX} % 4)) in
  0) arm_order=(certified full_memory native memory_b_native_c) ;;
  1) arm_order=(full_memory memory_b_native_c certified native) ;;
  2) arm_order=(memory_b_native_c native full_memory certified) ;;
  3) arm_order=(native certified memory_b_native_c full_memory) ;;
esac
printf '%s\n' "${arm_order[@]}" > "${episode_out}/arm_order.txt"
hab_python - "${episode_out}/episode_contract.json" "${scene}" "${episode}" \
  "${EPISODE_INDEX}" "${episode_seed}" "${EXPECTED_MANIFEST_SHA}" \
  "${arm_order[@]}" <<'PY'
import json,sys
path,scene,episode,index,seed,manifest_sha,*order=sys.argv[1:]
with open(path,"x",encoding="utf-8") as handle:
    json.dump({
        "schema_version":"shared_online_double_revisit_fresh_episode_v1",
        "scene":scene,
        "episode":episode,
        "selection_index":int(index),
        "episode_seed":int(seed),
        "benchmark_manifest_sha256":manifest_sha,
        "arm_order":order,
        "primary_contrast":"full_memory_C_minus_memory_B_native_C",
    },handle,indent=2,sort_keys=True)
    handle.write("\n")
PY

common=(
  --episode_root "${episode_root}"
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
)

for arm in "${arm_order[@]}"; do
  arm_root=${episode_out}/${arm}
  mkdir "${arm_root}"
  echo "[run] index=${EPISODE_INDEX} ${scene}/${episode}/${arm}"
  case "${arm}" in
    full_memory)
      hab_python -u "${SOURCE_ROOT}/MemNavData/eval_shared_online_double_revisit.py" \
        "${common[@]}" --out "${arm_root}" \
        --port "${MEMNAV_PORT}" --novel_port "${NAVDP_PORT}" \
        --server_backend hybrid_pose --hybrid_route phase \
        --revisit_adapter legacy_metric \
        --shared_online_known_revisit_scope both \
        > "${episode_out}/logs/eval_${arm}.log" 2>&1
      ;;
    memory_b_native_c)
      hab_python -u "${SOURCE_ROOT}/MemNavData/eval_shared_online_double_revisit.py" \
        "${common[@]}" --out "${arm_root}" \
        --port "${MEMNAV_PORT}" --novel_port "${NAVDP_PORT}" \
        --server_backend hybrid_pose --hybrid_route phase \
        --revisit_adapter legacy_metric \
        --shared_online_known_revisit_scope b_only \
        > "${episode_out}/logs/eval_${arm}.log" 2>&1
      ;;
    certified)
      hab_python -u "${SOURCE_ROOT}/MemNavData/eval_shared_online_double_revisit.py" \
        "${common[@]}" --out "${arm_root}" \
        --port "${MEMNAV_PORT}" --novel_port "${NAVDP_PORT}" \
        --server_backend hybrid_pose \
        --hybrid_route certified_relocalization \
        --revisit_controller navdp_mixed \
        --revisit_adapter verified_bearing_v1 \
        > "${episode_out}/logs/eval_${arm}.log" 2>&1
      ;;
    native)
      hab_python -u "${SOURCE_ROOT}/MemNavData/eval_shared_online_double_revisit.py" \
        "${common[@]}" --out "${arm_root}" \
        --port "${NAVDP_PORT}" --server_backend navdp --hybrid_route phase \
        > "${episode_out}/logs/eval_${arm}.log" 2>&1
      ;;
    *) fail "unknown arm ${arm}" ;;
  esac
done

hab_python - "${episode_out}" "${episode}" "${EXPECTED_MANIFEST_SHA}" <<'PY'
import csv,json,sys
from pathlib import Path
root=Path(sys.argv[1]); episode=sys.argv[2]; manifest_sha=sys.argv[3]
expected={
 "full_memory":({"B":"navdp_mix","C":"navdp_mix"},True),
 "memory_b_native_c":({"B":"navdp_mix","C":"navdp"},False),
 "certified":({"B":"navdp_auto","C":"navdp_auto"},True),
 "native":({"B":None,"C":None},False),
}
payloads={}
for arm,(backends,long_memory) in expected.items():
    summary=json.load(open(root/arm/"summary.json"))
    if summary["episodes"] != 1 or summary["benchmark_manifest_sha256"] != manifest_sha:
        raise SystemExit(f"{arm}: incomplete or wrong benchmark")
    if summary["shared_A_all_hashes_ok"] is not True or summary["shared_A_total_diffusion_samples"] != 0:
        raise SystemExit(f"{arm}: shared A replay contract failed")
    if summary["policy_backends"] != backends or summary["C_long_memory_enabled"] is not long_memory:
        raise SystemExit(f"{arm}: controller contract changed")
    with open(root/arm/"metric.csv",newline="") as handle:
        rows=list(csv.DictReader(handle))
    if len(rows) != 1 or rows[0]["episode"] != episode:
        raise SystemExit(f"{arm}: metric identity changed")
    payloads[arm]=json.load(open(root/arm/f"{episode}_plans.json"))
full=payloads["full_memory"]
ablation=payloads["memory_b_native_c"]
for key in ("legB",):
    if full[key] != ablation[key]:
        raise SystemExit("B plan prefix differs between causal arms")
for group in ("rollout_traces","memory_traces"):
    if full[group]["legB"] != ablation[group]["legB"]:
        raise SystemExit(f"B {group} differs between causal arms")
print(json.dumps({"episode":root.name,"status":"complete"},sort_keys=True))
PY

echo "[complete] ${episode_out}"
