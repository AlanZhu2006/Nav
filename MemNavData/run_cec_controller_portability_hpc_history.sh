#!/usr/bin/env bash
# Run all four CEC controller projections on one frozen mixed-role history.
set -euo pipefail
umask 0022

SOURCE_ROOT=${SOURCE_ROOT:?set immutable source root}
BASE_SOURCE_ROOT=${BASE_SOURCE_ROOT:?set base dependency source root}
BENCH_ROOT=${BENCH_ROOT:?set sealed natural-direction benchmark root}
EXPECTED_MANIFEST_SHA=${EXPECTED_MANIFEST_SHA:?set benchmark SHA}
HISTORY_INDEX=${HISTORY_INDEX:?set frozen manifest index}
RUN_ROOT=${RUN_ROOT:?set pilot run root}
PORTABILITY_ENV_ROOT=${PORTABILITY_ENV_ROOT:?set immutable env root}
PORTABILITY_CHECKPOINT_ROOT=${PORTABILITY_CHECKPOINT_ROOT:?set checkpoint root}
DEPENDENCY_RECEIPT=${DEPENDENCY_RECEIPT:?set CEC/NavDP dependency receipt}
EXPECTED_DEPENDENCY_RECEIPT_SHA=${EXPECTED_DEPENDENCY_RECEIPT_SHA:?set receipt SHA}

MEMNAV_PY=${MEMNAV_PY:-/scratch/lg154/conda-envs/memnav/bin/python}
HAB_PY=${HAB_PY:-/scratch/lg154/conda-envs/habitat/bin/python}
MAX_STEPS=${MAX_STEPS:-80}
BASE_PORT=${BASE_PORT:-24000}
LINGBOT_REPO=${LINGBOT_REPO:-/scratch/lg154/Research/Nav/NavDP/baselines/memnav/lingbot-map}

fail() { echo "ABORT: $*" >&2; exit 2; }
[[ "${HISTORY_INDEX}" =~ ^[0-9]+$ ]] || fail "bad history index"
[[ "${MAX_STEPS}" =~ ^[1-9][0-9]*$ ]] || fail "bad max steps"
[[ "${BASE_PORT}" =~ ^[0-9]+$ ]] || fail "bad base port"
(( BASE_PORT >= 20000 && BASE_PORT <= 49990 )) || fail "unsafe port range"
[[ "$(sha256sum "${BENCH_ROOT}/manifest.json" | awk '{print $1}')" == \
   "${EXPECTED_MANIFEST_SHA}" ]] || fail "benchmark manifest changed"
[[ "$(sha256sum "${DEPENDENCY_RECEIPT}" | awk '{print $1}')" == \
   "${EXPECTED_DEPENDENCY_RECEIPT_SHA}" ]] || fail "dependency receipt changed"
[[ -r "${PORTABILITY_ENV_ROOT}/environment_receipt.json" ]] || \
  fail "portability environment is not sealed"
(cd "${PORTABILITY_CHECKPOINT_ROOT}" && \
  sha256sum -c --quiet CHECKPOINTS.sha256) || fail "checkpoint receipt failed"

readarray -t identity < <("${MEMNAV_PY}" - \
  "${BENCH_ROOT}/manifest.json" "${HISTORY_INDEX}" <<'PY'
import hashlib,json,sys
from pathlib import Path
manifest=json.load(open(sys.argv[1])); index=int(sys.argv[2])
row=manifest["episodes"][index]
scene=str(row["scene"]); episode=str(row["episode"])
receipt=json.load(open(Path(row["online_a_episode"])/"receipt.json"))
asset=Path(receipt["source_asset"])
h=hashlib.sha256(asset.read_bytes()).hexdigest()
if h != receipt["source_asset_sha256"]: raise SystemExit("scene asset changed")
print(scene); print(episode); print(asset)
PY
)
[[ "${#identity[@]}" -eq 3 ]] || fail "history identity failed"
scene=${identity[0]}; episode=${identity[1]}; scene_file=${identity[2]}
history_root=${RUN_ROOT}/evaluation/$(printf '%03d' "${HISTORY_INDEX}")_${scene}_${episode}
[[ ! -e "${history_root}" ]] || fail "history output exists: ${history_root}"
mkdir -p "${history_root}/logs"

readarray -t dependencies < <("${MEMNAV_PY}" - "${DEPENDENCY_RECEIPT}" <<'PY'
import json,os,sys
p=json.load(open(sys.argv[1]))
for name in ("gatecurr600","navdp_checkpoint","lingbot_map_long"):
 row=p["dependencies"][name]
 if os.stat(row["path"]).st_size != int(row["bytes"]):
  raise SystemExit(f"dependency size changed: {name}")
 print(row["path"])
PY
)
[[ "${#dependencies[@]}" -eq 3 ]] || fail "dependency receipt invalid"
memnav_ckpt=${dependencies[0]}; navdp_ckpt=${dependencies[1]}
lingbot_weights=${dependencies[2]}

controllers=(navdp vint iplanner viplanner)
offset=$(( HISTORY_INDEX % ${#controllers[@]} ))
order=()
for ((i=0; i<${#controllers[@]}; i++)); do
  order+=("${controllers[$(((offset+i)%${#controllers[@]}))]}")
done
printf '%s\n' "${order[@]}" >"${history_root}/controller_order.txt"

timings=()
for controller in "${order[@]}"; do
  controller_root=${history_root}/${controller}
  started=$(date +%s%N)
  env \
    ROOT="${SOURCE_ROOT}" CONTROLLER="${controller}" \
    EVAL_KIND=role_pair_mixed SCENE="${scene}" EPISODE="${episode}" \
    MAX_STEPS="${MAX_STEPS}" EVAL_SEED=0 \
    MEMNAV_PORT="${BASE_PORT}" FALLBACK_PORT="$((BASE_PORT+1))" \
    UPSTREAM_PORT="$((BASE_PORT+2))" PROXY_PORT="$((BASE_PORT+3))" \
    HUB_PORT="$((BASE_PORT+4))" \
    MEMNAV_PY="${MEMNAV_PY}" HAB_PY="${HAB_PY}" \
    VINT_PY="${PORTABILITY_ENV_ROOT}/vint/bin/python" \
    VIPLANNER_PY="${PORTABILITY_ENV_ROOT}/viplanner/bin/python" \
    MEMNAV_CKPT="${memnav_ckpt}" NAVDP_CKPT="${navdp_ckpt}" \
    VINT_CKPT="${PORTABILITY_CHECKPOINT_ROOT}/vint.pth" \
    IPLANNER_CKPT="${PORTABILITY_CHECKPOINT_ROOT}/iplanner.pth" \
    VIPLANNER_CKPT="${PORTABILITY_CHECKPOINT_ROOT}/viplanner.pt" \
    MASK2FORMER_CKPT="${PORTABILITY_CHECKPOINT_ROOT}/mask2former_r50_8xb2-lsj-50e_coco-panoptic_20230118_125535-54df384a.pth" \
    MASK2FORMER_CONFIG="${PORTABILITY_ENV_ROOT}/viplanner/lib/python3.10/site-packages/mmdet/.mim/configs/mask2former/mask2former_r50_8xb2-lsj-50e_coco-panoptic.py" \
    LINGBOT_REPO="${LINGBOT_REPO}" LINGBOT_WEIGHTS="${lingbot_weights}" \
    LIGHTGLUE_REPO="${BASE_SOURCE_ROOT}/third_party/LightGlue" \
    DEPENDENCY_ROOT="${BASE_SOURCE_ROOT}/third_party/python" \
    INTERNNAV_ROOT="${BASE_SOURCE_ROOT}/InternNav" \
    SCENE_FILE="${scene_file}" BENCHMARK_ROOT="${BENCH_ROOT}/${scene}" \
    RUN_ROOT="${controller_root}" \
    bash "${SOURCE_ROOT}/MemNavData/run_cec_controller_portability_smoke_local.sh" \
    >"${history_root}/logs/${controller}.log" 2>&1
  ended=$(date +%s%N)
  timings+=("${controller}:$((ended-started))")
done

audit_args=()
for controller in "${controllers[@]}"; do
  audit_args+=(--run "${controller}=${history_root}/${controller}")
done
"${MEMNAV_PY}" "${SOURCE_ROOT}/MemNavData/audit_cec_controller_portability_pilot.py" \
  "${audit_args[@]}" --out "${history_root}/independent_audit.json"

"${MEMNAV_PY}" - "${history_root}/completion.json" \
  "${history_root}/independent_audit.json" "${scene}" "${episode}" \
  "${HISTORY_INDEX}" "${MAX_STEPS}" \
  "${PORTABILITY_ENV_ROOT}/environment_receipt.json" \
  "${PORTABILITY_CHECKPOINT_ROOT}/CHECKPOINTS.sha256" \
  "${timings[@]}" <<'PY'
import hashlib,json,sys
from pathlib import Path
(out,audit,scene,episode,index,max_steps,env_receipt,checkpoint_receipt,
 *timing_tokens)=sys.argv[1:]
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
timings={}
for token in timing_tokens:
 name,value=token.split(":",1); timings[name]=int(value)/1e9
p={
 "schema":"cec_controller_portability_hpc_history_completion_v1",
 "complete":True,"scene":scene,"episode":episode,
 "history_index":int(index),"max_steps":int(max_steps),
 "controller_wall_seconds":timings,
 "independent_audit_sha256":sha(audit),
 "environment_receipt_sha256":sha(env_receipt),
 "checkpoint_receipt_sha256":sha(checkpoint_receipt),
}
Path(out).write_text(json.dumps(p,indent=2,sort_keys=True)+"\n")
PY
echo "COMPLETE history=${HISTORY_INDEX} scene=${scene} output=${history_root}"
