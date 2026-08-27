#!/usr/bin/env bash
# Same-process paired geometry-certificate vs geometry->CDEC closed loop.
set -euo pipefail
umask 0022

ROOT=${ROOT:?set immutable overlay ROOT}
RUN_ROOT=${RUN_ROOT:?set RUN_ROOT}
MANIFEST=${MANIFEST:?set frozen MANIFEST}
EXPECTED_MANIFEST_SHA=${EXPECTED_MANIFEST_SHA:?set EXPECTED_MANIFEST_SHA}
TRACE_RECEIPT=${TRACE_RECEIPT:?set TRACE_RECEIPT}
EXPECTED_TRACE_RECEIPT_SHA=${EXPECTED_TRACE_RECEIPT_SHA:?set trace receipt SHA}
SOURCE_RECEIPT=${SOURCE_RECEIPT:?set SOURCE_RECEIPT}
EXPECTED_SOURCE_RECEIPT_SHA=${EXPECTED_SOURCE_RECEIPT_SHA:?set source receipt SHA}
GATE_VERIFICATION=${GATE_VERIFICATION:?set independent gate verification}
EXPECTED_GATE_VERIFICATION_SHA=${EXPECTED_GATE_VERIFICATION_SHA:?set gate SHA}
CDEC_ARTIFACT_SHA=${CDEC_ARTIFACT_SHA:?set CDEC artifact SHA}
SCENE_INDEX=${SCENE_INDEX:?set SCENE_INDEX}
HAB_PY=${HAB_PY:?set HAB_PY}
MEMNAV_PORT=${MEMNAV_PORT:?set MEMNAV_PORT}
NAVDP_PORT=${NAVDP_PORT:?set NAVDP_PORT}

fail() { echo "ABORT: $*" >&2; exit 2; }
[[ "${SCENE_INDEX}" =~ ^([0-9]|1[0-9])$ ]] || fail "invalid scene index"
for pair in \
  "${MANIFEST}:${EXPECTED_MANIFEST_SHA}" \
  "${TRACE_RECEIPT}:${EXPECTED_TRACE_RECEIPT_SHA}" \
  "${SOURCE_RECEIPT}:${EXPECTED_SOURCE_RECEIPT_SHA}" \
  "${GATE_VERIFICATION}:${EXPECTED_GATE_VERIFICATION_SHA}"; do
  path=${pair%%:*}; expected=${pair#*:}
  [[ "$(sha256sum "${path}" | awk '{print $1}')" == "${expected}" ]] || \
    fail "input SHA mismatch: ${path}"
done

HAB_SITE_PACKAGES=$("${HAB_PY}" -c \
  'import sysconfig; print(sysconfig.get_paths()["purelib"])')
HAB_PYTHONPATH=${ROOT}:${HAB_SITE_PACKAGES}/pip/_vendor${PYTHONPATH:+:${PYTHONPATH}}
hab_python() { env PYTHONPATH="${HAB_PYTHONPATH}" "${HAB_PY}" "$@"; }

readarray -t identity < <(hab_python - \
  "${MANIFEST}" "${TRACE_RECEIPT}" "${SCENE_INDEX}" <<'PY'
import hashlib,json,sys
from pathlib import Path

def sha(path):
    digest=hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda:handle.read(8<<20),b""):
            digest.update(block)
    return digest.hexdigest()

manifest=json.load(open(sys.argv[1]))
receipt=json.load(open(sys.argv[2]))
index=int(sys.argv[3])
scene=manifest["scenes"][index]
episode_root=Path(manifest["paths"]["episode_root"])/scene
asset=Path(manifest["paths"]["asset_root"])/scene/f"{scene}.glb"
if sha(asset)!=manifest["assets"][scene]["sha256"]:
    raise SystemExit("asset identity changed")
episodes=[]
for row in manifest["episodes"][scene]:
    episode=episode_root/row["episode"]
    paths={
      "metadata":episode/"meta"/"gen_meta.json",
      "parquet":episode/"data"/"chunk-000"/"episode_000000.parquet",
      "goal":episode/"goal_image.jpg",
    }
    for kind,path in paths.items():
        if sha(path)!=row["files"][kind]["sha256"]:
            raise SystemExit(f"episode identity changed: {scene}/{row['episode']}/{kind}")
    episodes.append(row["episode"])
if len(episodes)!=8:
    raise SystemExit("formal comparison requires eight episodes per scene")
scene_receipt=receipt["scenes"].get(scene)
if not isinstance(scene_receipt,dict):
    raise SystemExit("trace receipt lacks scene")
trace_root=Path(scene_receipt["trace_root"])
if sha(trace_root/"summary.json")!=scene_receipt["summary_sha256"]:
    raise SystemExit("trace summary changed")
for episode in episodes:
    path=trace_root/f"{episode}_leg1_trace.json"
    if sha(path)!=scene_receipt["episodes"].get(episode):
        raise SystemExit(f"trace changed: {scene}/{episode}")
print(scene)
print(asset)
print(episode_root)
print(",".join(episodes))
print(manifest["evaluation"]["base_seed"])
print(manifest["evaluation"]["max_steps_per_leg"])
print(trace_root)
PY
)
[[ "${#identity[@]}" -eq 7 ]] || fail "identity reader failed"
scene=${identity[0]}
scene_file=${identity[1]}
episode_root=${identity[2]}
episode_csv=${identity[3]}
base_seed=${identity[4]}
max_steps=${identity[5]}
trace_root=${identity[6]}

scene_root=${RUN_ROOT}/scenes/$(printf '%02d' "${SCENE_INDEX}")_${scene}
[[ ! -e "${scene_root}" ]] || fail "scene output exists: ${scene_root}"
mkdir -p "${scene_root}/logs"
if (( 10#${SCENE_INDEX} % 2 == 0 )); then
  arm_order=(geometry_certificate cdec_cascade)
else
  arm_order=(cdec_cascade geometry_certificate)
fi

hab_python - "${scene_root}/scene_contract.json" "${scene}" \
  "${SCENE_INDEX}" "${EXPECTED_MANIFEST_SHA}" \
  "${EXPECTED_TRACE_RECEIPT_SHA}" "${EXPECTED_GATE_VERIFICATION_SHA}" \
  "${CDEC_ARTIFACT_SHA}" "${arm_order[@]}" <<'PY'
import json,sys
path,scene,index,manifest_sha,trace_sha,gate_sha,artifact_sha,*order=sys.argv[1:]
with open(path,"x",encoding="utf-8") as handle:
    json.dump({
      "schema_version":"cdec_consumed_closed_loop_scene_v1_20260813",
      "scene":scene,"scene_index":int(index),
      "manifest_sha256":manifest_sha,"trace_receipt_sha256":trace_sha,
      "gate_verification_sha256":gate_sha,"cdec_artifact_sha256":artifact_sha,
      "arm_order":order,"primary_contrast":"cdec_cascade_minus_geometry_certificate",
      "stagnation_graph":"off",
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
  --port "${MEMNAV_PORT}" --novel_port "${NAVDP_PORT}"
  --server_backend hybrid_pose
  --leg1_mode shared_trace --shared_leg1_trace_root "${trace_root}"
  --hybrid_route certified_relocalization
  --revisit_controller navdp_mixed
  --revisit_adapter verified_bearing_v1
)

for arm in "${arm_order[@]}"; do
  arm_root=${scene_root}/${arm}
  mkdir -p "${arm_root}"
  if [[ "${arm}" == cdec_cascade ]]; then
    cdec_args=(--certified_cdec_rescue on \
      --expected_cdec_artifact_sha256 "${CDEC_ARTIFACT_SHA}")
  else
    cdec_args=(--certified_cdec_rescue off)
  fi
  hab_python -u "${ROOT}/MemNavData/eval_2leg_habitat.py" \
    "${common[@]}" "${cdec_args[@]}" --out "${arm_root}" \
    > "${scene_root}/logs/eval_${arm}.log" 2>&1
done

hab_python - "${scene_root}" "${episode_csv}" "${CDEC_ARTIFACT_SHA}" <<'PY'
import csv,json,sys
from pathlib import Path
root=Path(sys.argv[1]); expected=sys.argv[2].split(","); artifact=sys.argv[3]
for arm,mode in (("geometry_certificate","off"),("cdec_cascade","on")):
    summary=json.loads((root/arm/"summary.json").read_text())
    if summary.get("episodes")!=len(expected):
        raise SystemExit(f"{arm}: incomplete summary")
    if summary.get("certified_cdec_rescue")!=mode:
        raise SystemExit(f"{arm}: CDEC request mode changed")
    status=summary.get("cdec_server_status") or {}
    if status.get("enabled") is not True or status.get("artifact_sha256")!=artifact:
        raise SystemExit(f"{arm}: CDEC artifact status changed")
    with (root/arm/"metric.csv").open(newline="") as handle:
        rows=list(csv.DictReader(handle))
    if [row["episode"] for row in rows]!=expected:
        raise SystemExit(f"{arm}: metric identity/order mismatch")
    if int(summary.get("certified_cdec_uncached_runtime_failure_count",-1))!=0:
        raise SystemExit(f"{arm}: learned proposal runtime failure")
    if arm=="geometry_certificate" and any(int(summary.get(name,-1))!=0 for name in (
        "certified_cdec_requested_plan_count",
        "certified_cdec_learned_selected_plan_count",
        "certified_cdec_uncached_invocation_count",
    )):
        raise SystemExit("geometry baseline invoked the learned proposal")
print(json.dumps({"scene":root.name,"status":"complete","episodes":len(expected)}))
PY
