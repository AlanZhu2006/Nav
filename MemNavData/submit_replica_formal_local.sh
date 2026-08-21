#!/usr/bin/env bash
# Freeze, bundle, and detach the pre-query-vetoed Replica stress test locally.

set -euo pipefail
umask 0022

LOCAL_ROOT=${LOCAL_ROOT:-$(git rev-parse --show-toplevel)}
COMPATIBILITY=${COMPATIBILITY:-${LOCAL_ROOT}/.diagnostics/replica_full18_compatibility_20260814.json}
CONSTRUCTIBILITY=${CONSTRUCTIBILITY:-${LOCAL_ROOT}/.diagnostics/replica_goal_a_constructibility_10scene_20260814/constructibility_result.json}
PILOT_AUDIT=${PILOT_AUDIT:-${LOCAL_ROOT}/.diagnostics/replica_room0_role_pair_four_arm_full4_600_20260814/outcome_neutral_audit.json}
BUNDLE_BASE=${BUNDLE_BASE:-${LOCAL_ROOT}/.diagnostics/source_bundles}
RESULT_BASE=${RESULT_BASE:-${LOCAL_ROOT}/.diagnostics}
RUN_TAG=${RUN_TAG:-replica_stress_10scene_$(date -u +%Y%m%dT%H%M%SZ)}
RUN_ROOT=${RUN_ROOT:-${RESULT_BASE}/${RUN_TAG}}
MEMNAV_PY=${MEMNAV_PY:-/home/asus/miniconda3/envs/memnav/bin/python}
HAB_PY=${HAB_PY:-/home/asus/miniconda3/envs/habitat/bin/python}

fail() { echo "ABORT: $*" >&2; exit 2; }
[[ "${RUN_TAG}" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]] || fail "unsafe RUN_TAG"
[[ ! -e "${RUN_ROOT}" ]] || fail "run root already exists: ${RUN_ROOT}"
[[ -r "${COMPATIBILITY}" && -r "${CONSTRUCTIBILITY}" \
   && -r "${PILOT_AUDIT}" ]] || fail "gate receipt missing"
[[ "$(sha256sum "${COMPATIBILITY}" | awk '{print $1}')" == \
  f6c14266b5aa5462d8a0d21c1219212e482f95b18fe73808eae204f3fe5752cf ]] || \
  fail "compatibility receipt changed"
[[ "$(sha256sum "${CONSTRUCTIBILITY}" | awk '{print $1}')" == \
  d4a74d41a3d4a874323b3efb28da445165233ef6338822fd1be0b0be5bd6782c ]] || \
  fail "Goal-A constructibility receipt changed"
"${MEMNAV_PY}" - "${CONSTRUCTIBILITY}" <<'PY'
import json,sys
p=json.load(open(sys.argv[1]))
assert p["query_outcomes_read"] is False
assert p["navigation_outcomes_generated"] is False
assert p["fresh_histories_excluding_room_0"] == 20
assert p["fresh_scene_count_with_history_excluding_room_0"] == 5
assert p["target_met"] is False
PY
"${MEMNAV_PY}" - "${PILOT_AUDIT}" <<'PY'
import json,sys
p=json.load(open(sys.argv[1]))
assert p["valid"] is True
assert p["scope"] == "Replica cross-dataset integration only; no SR claim"
assert p["histories"] == 4 and p["max_steps"] == 600
assert p["runtime_role_visibility"] == "none"
assert p["runtime_failure_plans"] == 0
PY

for path in \
  "${LOCAL_ROOT}/MemNavData/freeze_replica_formal_confirmation.py" \
  "${LOCAL_ROOT}/MemNavData/run_replica_formal_local.sh" \
  "${LOCAL_ROOT}/MemNavData/run_paper_role_pair_episode.sh" \
  "${LOCAL_ROOT}/MemNavData/REPLICA_CROSS_DATASET_PROTOCOL_20260814.md"; do
  [[ -f "${path}" && ! -L "${path}" ]] || fail "missing physical source ${path}"
done
"${HAB_PY}" -m py_compile \
  "${LOCAL_ROOT}/MemNavData/freeze_replica_formal_confirmation.py" \
  "${LOCAL_ROOT}/MemNavData/write_replica_formal_scene_receipt.py" \
  "${LOCAL_ROOT}/MemNavData/summarize_replica_online_a.py" \
  "${LOCAL_ROOT}/MemNavData/finalize_paper_role_pairs.py" \
  "${LOCAL_ROOT}/MemNavData/summarize_paper_role_pair_eval.py" \
  "${LOCAL_ROOT}/MemNavData/independent_verify_paper_role_pair_eval.py"
bash -n "${LOCAL_ROOT}/MemNavData/run_replica_formal_local.sh" \
  "${LOCAL_ROOT}/MemNavData/run_paper_role_pair_episode.sh"

staging=$(mktemp -d /tmp/replica_formal_bundle.XXXXXX)
cleanup() {
  if [[ -d "${staging}" ]]; then
    rm -rf -- "${staging}"
  fi
}
trap cleanup EXIT
mkdir -p "${staging}/MemNavData" \
  "${staging}/NavDP/baselines/memnav" \
  "${staging}/NavDP/baselines/navdp" "${staging}/receipts"
while IFS= read -r -d '' path; do
  cp --preserve=mode,timestamps "${path}" \
    "${staging}/MemNavData/$(basename "${path}")"
done < <(find "${LOCAL_ROOT}/MemNavData" -maxdepth 1 -type f -name '*.py' -print0)
for relative in \
  MemNavData/run_replica_formal_local.sh \
  MemNavData/run_paper_role_pair_episode.sh \
  MemNavData/REPLICA_CROSS_DATASET_PROTOCOL_20260814.md \
  MemNavData/REPLICA_ATTEMPT1_PYCACHE_INCIDENT_20260814.json \
  MemNavData/REPLICA_ATTEMPT2_DEPENDENCY_INCIDENT_20260814.json \
  MemNavData/PAPER_EVALUATION_PROTOCOL_20260814.md \
  MemNavData/PAPER_CONSTRUCTION_AMENDMENT_20260814.md; do
  cp --preserve=mode,timestamps "${LOCAL_ROOT}/${relative}" \
    "${staging}/${relative}"
done
for component in memnav navdp; do
  while IFS= read -r -d '' path; do
    cp --preserve=mode,timestamps "${path}" \
      "${staging}/NavDP/baselines/${component}/$(basename "${path}")"
  done < <(find "${LOCAL_ROOT}/NavDP/baselines/${component}" -maxdepth 1 \
    -type f -name '*.py' -print0)
done
# NavDP's top-level policy imports the bundled Depth Anything V2 package.
# Copy that package recursively; copying only navdp/*.py produced an immutable
# but non-runnable attempt-2 bundle.  Bytecode is deliberately excluded and is
# redirected to RUN_ROOT at execution time.
while IFS= read -r -d '' path; do
  relative=${path#"${LOCAL_ROOT}/NavDP/baselines/navdp/"}
  destination=${staging}/NavDP/baselines/navdp/${relative}
  mkdir -p "$(dirname "${destination}")"
  cp --preserve=mode,timestamps "${path}" "${destination}"
done < <(find "${LOCAL_ROOT}/NavDP/baselines/navdp/depth_anything" \
  -type f ! -name '*.pyc' ! -path '*/__pycache__/*' -print0)
cp "${COMPATIBILITY}" "${staging}/receipts/replica_compatibility.json"
cp "${CONSTRUCTIBILITY}" \
  "${staging}/receipts/replica_goal_a_constructibility.json"
cp "${PILOT_AUDIT}" "${staging}/receipts/replica_room0_consumed_pilot.json"
"${HAB_PY}" "${staging}/MemNavData/freeze_replica_formal_confirmation.py" \
  --compatibility "${COMPATIBILITY}" \
  --constructibility "${CONSTRUCTIBILITY}" \
  --out "${staging}/receipts/replica_formal_manifest.json" \
  > "${staging}/receipts/freeze_stdout.json"
FREEZE_SHA=$(sha256sum "${staging}/receipts/replica_formal_manifest.json" | awk '{print $1}')

LOCAL_HEAD=$(git -C "${LOCAL_ROOT}" rev-parse HEAD)
"${MEMNAV_PY}" - "${staging}" "${LOCAL_HEAD}" "${FREEZE_SHA}" <<'PY'
import hashlib,json,sys
from pathlib import Path
root=Path(sys.argv[1]); files={}
for path in sorted(root.rglob("*")):
    if path.is_symlink(): raise SystemExit(f"bundle symlink: {path}")
    if path.is_file() and path.name not in {"source_bundle_manifest.json","SOURCE_BUNDLE.sha256"}:
        files[path.relative_to(root).as_posix()]=hashlib.sha256(path.read_bytes()).hexdigest()
payload={
 "schema_version":"replica_cross_dataset_stress_bundle_v2_20260814",
 "local_git_head_context":sys.argv[2],
 "scope":"ten-scene attempted, five-fresh-scene Replica cross-dataset stress test",
 "freeze_manifest_sha256":sys.argv[3],
 "method_adaptation":"none",
 "query_outcomes_read":False,
 "files":files,
}
(root/"source_bundle_manifest.json").write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n")
PY
(
  cd "${staging}"
  find . -type f ! -name SOURCE_BUNDLE.sha256 -print0 | sort -z | \
    xargs -0 sha256sum > SOURCE_BUNDLE.sha256
  sha256sum -c SOURCE_BUNDLE.sha256 >/dev/null
)
SOURCE_RECEIPT_SHA=$(sha256sum "${staging}/SOURCE_BUNDLE.sha256" | awk '{print $1}')
BUNDLE_MANIFEST_SHA=$(sha256sum "${staging}/source_bundle_manifest.json" | awk '{print $1}')
BUNDLE_ROOT=${BUNDLE_BASE}/replica_stress_${BUNDLE_MANIFEST_SHA:0:16}
mkdir -p "${BUNDLE_BASE}"
if [[ -e "${BUNDLE_ROOT}" ]]; then
  [[ "$(sha256sum "${BUNDLE_ROOT}/SOURCE_BUNDLE.sha256" | awk '{print $1}')" == \
    "${SOURCE_RECEIPT_SHA}" ]] || fail "bundle path collision"
  (cd "${BUNDLE_ROOT}" && sha256sum -c SOURCE_BUNDLE.sha256) >/dev/null
else
  mv "${staging}" "${BUNDLE_ROOT}"
  staging=/nonexistent/replica_formal_bundle_consumed
  chmod -R a-w "${BUNDLE_ROOT}"
fi

mkdir -p "${RUN_ROOT}/logs"
FREEZE_MANIFEST=${BUNDLE_ROOT}/receipts/replica_formal_manifest.json
SOURCE_RECEIPT=${BUNDLE_ROOT}/SOURCE_BUNDLE.sha256
"${MEMNAV_PY}" - "${RUN_ROOT}/submission.json" "${RUN_ROOT}" \
  "${BUNDLE_ROOT}" "${SOURCE_RECEIPT_SHA}" "${FREEZE_SHA}" <<'PY'
import json,os,socket,sys
path,run,bundle,source_sha,freeze_sha=sys.argv[1:]
payload={
 "schema_version":"replica_cross_dataset_stress_submission_v2_20260814",
 "run_root":run,"source_bundle":bundle,
 "source_receipt_sha256":source_sha,"freeze_manifest_sha256":freeze_sha,
 "host":socket.gethostname(),"submitter_uid":os.getuid(),
 "gpu_index":0,"execution":"detached_single_gpu_sequential",
 "pilot_scene_excluded_from_primary":"room_0",
 "formal_confirmation_authorized":False,
 "reporting_scope":"underpowered cross-dataset stress only",
 "query_outcomes_read":False,
}
open(path,"x").write(json.dumps(payload,indent=2,sort_keys=True)+"\n")
PY

nohup setsid env CUDA_VISIBLE_DEVICES=0 \
  SOURCE_ROOT="${BUNDLE_ROOT}" SOURCE_RECEIPT="${SOURCE_RECEIPT}" \
  EXPECTED_SOURCE_RECEIPT_SHA="${SOURCE_RECEIPT_SHA}" \
  RUN_ROOT="${RUN_ROOT}" FREEZE_MANIFEST="${FREEZE_MANIFEST}" \
  EXPECTED_FREEZE_SHA="${FREEZE_SHA}" \
  bash "${BUNDLE_ROOT}/MemNavData/run_replica_formal_local.sh" \
  > "${RUN_ROOT}/pipeline.log" 2>&1 < /dev/null &
pipeline_pid=$!
"${MEMNAV_PY}" - "${RUN_ROOT}/launch_receipt.json" "${pipeline_pid}" <<'PY'
import json,sys,time
open(sys.argv[1],"x").write(json.dumps({
 "schema_version":"replica_formal_local_launch_v1_20260814",
 "pipeline_pid":int(sys.argv[2]),"launch_unix_time":time.time(),
},indent=2,sort_keys=True)+"\n")
PY
sleep 2
kill -0 "${pipeline_pid}" 2>/dev/null || {
  tail -n 160 "${RUN_ROOT}/pipeline.log" >&2
  fail "detached Replica pipeline exited during launch"
}

echo "RUN_ROOT=${RUN_ROOT}"
echo "SOURCE_BUNDLE=${BUNDLE_ROOT}"
echo "SOURCE_RECEIPT_SHA=${SOURCE_RECEIPT_SHA}"
echo "FREEZE_MANIFEST_SHA=${FREEZE_SHA}"
echo "PIPELINE_PID=${pipeline_pid}"
