#!/usr/bin/env bash
# Run the frozen Replica cross-dataset stress construction and five-arm eval.

set -euo pipefail
umask 0022

SOURCE_ROOT=${SOURCE_ROOT:?set immutable source bundle root}
SOURCE_RECEIPT=${SOURCE_RECEIPT:?set SOURCE_BUNDLE.sha256}
EXPECTED_SOURCE_RECEIPT_SHA=${EXPECTED_SOURCE_RECEIPT_SHA:?set source receipt SHA}
RUN_ROOT=${RUN_ROOT:?set fresh result root}
FREEZE_MANIFEST=${FREEZE_MANIFEST:?set frozen Replica manifest}
EXPECTED_FREEZE_SHA=${EXPECTED_FREEZE_SHA:?set frozen manifest SHA}

MEMNAV_PY=${MEMNAV_PY:-/home/asus/miniconda3/envs/memnav/bin/python}
HAB_PY=${HAB_PY:-/home/asus/miniconda3/envs/habitat/bin/python}
MEMNAV_CKPT=${MEMNAV_CKPT:-/home/asus/Research/Nav-axis-uturn/.diagnostics/unseen_scene_eval_20260803/checkpoints/gatecurr600.memnav.ckpt}
NAVDP_CKPT=${NAVDP_CKPT:-/home/asus/Research/Nav/NavDP/baselines/navdp/checkpoints/navdp_checkpoint.ckpt}
LINGBOT_REPO=${LINGBOT_REPO:-/home/asus/Research/Nav/NavDP/baselines/memnav/lingbot-map}
LINGBOT_WEIGHTS=${LINGBOT_WEIGHTS:-${LINGBOT_REPO}/weights/lingbot-map-long.pt}
LIGHTGLUE_REPO=${LIGHTGLUE_REPO:-/home/asus/Research/Nav-graph-blind/.diagnostics/dependencies/LightGlue}
DEPENDENCY_ROOT=${DEPENDENCY_ROOT:-/home/asus/Research/Nav-graph-blind/.diagnostics/dependencies/python}
INTERNNAV_ROOT=${INTERNNAV_ROOT:-/home/asus/Research/Nav-graph-blind/InternNav}

EXPECTED_MEMNAV_SHA=9b7a5811ff0aea212503f58b45258ba4f66b06420f87c350946aead39db6fdb7
EXPECTED_NAVDP_SHA=3bb3ad4ab241e857bb57a4021cc6aab76d5263e81fbf80298d579053ef011947
EXPECTED_LINGBOT_SHA=832bc82cbae0bc9bbe946ef5ee1f7226abd8c0e183ccf8beddbb3d133576f409

fail() { echo "ABORT: $*" >&2; exit 2; }
[[ -d "${RUN_ROOT}" ]] || fail "submission did not create RUN_ROOT"
[[ ! -e "${RUN_ROOT}/PIPELINE_STARTED" ]] || fail "pipeline already started"
[[ "$(sha256sum "${SOURCE_RECEIPT}" | awk '{print $1}')" == \
  "${EXPECTED_SOURCE_RECEIPT_SHA}" ]] || fail "source receipt changed"
(cd "${SOURCE_ROOT}" && sha256sum -c "${SOURCE_RECEIPT}") >/dev/null || \
  fail "immutable source bundle changed"
[[ "$(sha256sum "${FREEZE_MANIFEST}" | awk '{print $1}')" == \
  "${EXPECTED_FREEZE_SHA}" ]] || fail "Replica freeze manifest changed"
[[ "$(sha256sum "${MEMNAV_CKPT}" | awk '{print $1}')" == \
  "${EXPECTED_MEMNAV_SHA}" ]] || fail "MemNav checkpoint changed"
[[ "$(sha256sum "${NAVDP_CKPT}" | awk '{print $1}')" == \
  "${EXPECTED_NAVDP_SHA}" ]] || fail "NavDP checkpoint changed"
[[ "$(sha256sum "${LINGBOT_WEIGHTS}" | awk '{print $1}')" == \
  "${EXPECTED_LINGBOT_SHA}" ]] || fail "LingBot weights changed"
for required in "${MEMNAV_PY}" "${HAB_PY}" "${LIGHTGLUE_REPO}" \
  "${DEPENDENCY_ROOT}" "${INTERNNAV_ROOT}"; do
  [[ -e "${required}" ]] || fail "missing dependency ${required}"
done

# The source bundle is intentionally read-only.  Route bytecode outside it so
# preflight compilation cannot attempt to create bundle-local __pycache__.
export PYTHONPYCACHEPREFIX=${RUN_ROOT}/pycache
mkdir -p "${PYTHONPYCACHEPREFIX}"

HAB_SITE_PACKAGES=$("${HAB_PY}" -c \
  'import sysconfig; print(sysconfig.get_paths()["purelib"])')
HAB_PYTHONPATH=${SOURCE_ROOT}:${SOURCE_ROOT}/MemNavData:${HAB_SITE_PACKAGES}/pip/_vendor${PYTHONPATH:+:${PYTHONPATH}}
hab_python() { env PYTHONPATH="${HAB_PYTHONPATH}" "${HAB_PY}" "$@"; }

hab_python -m py_compile \
  "${SOURCE_ROOT}/MemNavData/generate_twoleg.py" \
  "${SOURCE_ROOT}/MemNavData/eval_2leg_habitat.py" \
  "${SOURCE_ROOT}/MemNavData/materialize_paper_online_a_scene.py" \
  "${SOURCE_ROOT}/MemNavData/build_paper_role_pair_scene.py" \
  "${SOURCE_ROOT}/MemNavData/finalize_paper_role_pairs.py" \
  "${SOURCE_ROOT}/MemNavData/eval_shared_online_role_pairs.py" \
  "${SOURCE_ROOT}/MemNavData/summarize_paper_role_pair_eval.py" \
  "${SOURCE_ROOT}/MemNavData/independent_verify_paper_role_pair_eval.py"
bash -n "${SOURCE_ROOT}/MemNavData/run_paper_role_pair_episode.sh"

# Verify every compatibility-gated asset without reading any navigation result.
hab_python - "${FREEZE_MANIFEST}" <<'PY'
import hashlib,json,sys
from pathlib import Path
manifest=json.load(open(sys.argv[1]))
assert manifest["query_outcomes_read"] is False
assert manifest["formal_confirmation_authorized"] is False
assert manifest["cross_dataset_stress_evaluation_authorized"] is True
assert len(manifest["scenes"]) == 10
for scene in manifest["scenes"]:
    assert scene["generator_contract"]["goal_a_source_only"] is True
    for row in scene["files"].values():
        path=Path(row["path"])
        assert path.is_file() and path.stat().st_size == int(row["bytes"]), path
        digest=hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda:handle.read(32<<20),b""):
                digest.update(chunk)
        assert digest.hexdigest() == row["sha256"], path
print("Replica asset hashes verified")
PY

mkdir -p "${RUN_ROOT}/logs" "${RUN_ROOT}/source_episodes" \
  "${RUN_ROOT}/traces" "${RUN_ROOT}/buffer"
: > "${RUN_ROOT}/PIPELINE_STARTED"
hab_python - "${FREEZE_MANIFEST}" "${RUN_ROOT}/asset_map.json" <<'PY'
import json,sys
p=json.load(open(sys.argv[1]))
mapping={row["scene"]:row["stage"] for row in p["scenes"]}
open(sys.argv[2],"x").write(json.dumps(mapping,indent=2,sort_keys=True)+"\n")
PY

# Generate exactly the four pre-registered source attempts per scene.  An
# exhausted generator is retained as source attrition and is never replaced.
while IFS=$'\t' read -r index scene stage navmesh seed dmin dmax bmin multiplier max_attempts; do
  scene_source=${RUN_ROOT}/source_episodes/${scene}
  [[ ! -e "${scene_source}" ]] || fail "source output exists: ${scene_source}"
  set +e
  hab_python -u "${SOURCE_ROOT}/MemNavData/generate_twoleg.py" \
    --scene "${stage}" --navmesh "${navmesh}" --out "${scene_source}" \
    --n 4 --n_legs 2 --goal_a_source_only \
    --seed "${seed}" --dA_min "${dmin}" \
    --dA_max "${dmax}" --b_min "${bmin}" \
    --episode_attempt_multiplier "${multiplier}" \
    --max_attempts "${max_attempts}" \
    > "${RUN_ROOT}/logs/generate_$(printf '%02d' "${index}")_${scene}.log" 2>&1
  generator_rc=$?
  set -e
  hab_python - "${scene_source}/generation_summary.json" \
    "${generator_rc}" "${scene_source}/generator_exit.json" <<'PY'
import json,sys
summary=json.load(open(sys.argv[1])); rc=int(sys.argv[2])
if summary["complete"] and rc != 0:
    raise SystemExit("complete generation returned nonzero")
if not summary["complete"] and rc == 0:
    raise SystemExit("incomplete formal generation returned success")
open(sys.argv[3],"x").write(json.dumps({
  "exit_code":rc,"complete":bool(summary["complete"]),
  "generated_episodes":int(summary["generated_episodes"]),
  "requested_episodes":int(summary["requested_episodes"]),
  "navigation_outcome_read":False,
},indent=2,sort_keys=True)+"\n")
PY
done < <(hab_python - "${FREEZE_MANIFEST}" <<'PY'
import json,sys
for row in json.load(open(sys.argv[1]))["scenes"]:
 c=row["generator_contract"]
 print("\t".join(map(str,(
   row["index"],row["scene"],row["stage"],row["navmesh"],
   row["generator_seed"],c["dA_min_m"],c["dA_max_m"],c["b_min_m"],
   c["episode_attempt_multiplier"],c["max_attempts"],
 ))))
PY
)
(
  cd "${RUN_ROOT}/source_episodes"
  find . -type f ! -name SOURCE_EPISODES_FILES.sha256 -print0 | sort -z | \
    xargs -0 sha256sum > SOURCE_EPISODES_FILES.sha256
  sha256sum -c SOURCE_EPISODES_FILES.sha256 >/dev/null
)

runtime_root=$(mktemp -d /tmp/replica_formal.XXXXXX)
NAVDP_PID=; MEMNAV_PID=
cleanup() {
  for pid in "${MEMNAV_PID}" "${NAVDP_PID}"; do
    if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
      kill "${pid}" 2>/dev/null || true
      wait "${pid}" 2>/dev/null || true
    fi
  done
  rm -rf -- "${runtime_root}"
}
trap cleanup EXIT INT TERM
SERVER_PYTHONPATH=${SOURCE_ROOT}:${DEPENDENCY_ROOT}:${LIGHTGLUE_REPO}:${INTERNNAV_ROOT}/src/diffusion-policy${PYTHONPATH:+:${PYTHONPATH}}
mkdir -p "${runtime_root}/navdp" "${runtime_root}/memnav"

(
  exec env NAVDP_DISABLE_VIDEO=1 PYTHONUNBUFFERED=1 \
    PYTHONPATH="${SERVER_PYTHONPATH}" \
    "${MEMNAV_PY}" -u "${SOURCE_ROOT}/MemNavData/retrying_server_launcher.py" \
      --base-port 24041 --range-start 20000 --range-end 49999 \
      --stride 1994 --max-attempts 12 --ready-timeout 720 \
      --port-file "${runtime_root}/navdp.port" \
      --receipt-file "${RUN_ROOT}/logs/navdp_launcher.json" \
      --log-prefix "${RUN_ROOT}/logs/navdp_server.child" \
      --cwd "${runtime_root}/navdp" -- \
      "${MEMNAV_PY}" -u "${SOURCE_ROOT}/NavDP/baselines/navdp/navdp_server.py" \
        --port '{port}' --checkpoint "${NAVDP_CKPT}"
) > "${RUN_ROOT}/logs/navdp_launcher.log" 2>&1 &
NAVDP_PID=$!
for _ in $(seq 1 420); do
  kill -0 "${NAVDP_PID}" 2>/dev/null || fail "NavDP launcher exited"
  [[ -s "${runtime_root}/navdp.port" ]] && break
  sleep 2
done
[[ -s "${runtime_root}/navdp.port" ]] || fail "NavDP startup timed out"
NAVDP_PORT=$(<"${runtime_root}/navdp.port")

# Collect native online-A, materialize every eligible trace, and construct both
# query protocols.  No query arm is executed in this phase.
while IFS=$'\t' read -r index scene stage online_seed; do
  scene_source=${RUN_ROOT}/source_episodes/${scene}
  scene_root=${RUN_ROOT}/traces/$(printf '%02d' "${index}")_${scene}
  mkdir "${scene_root}"
  episode_csv=$(hab_python - "${scene_source}" <<'PY'
import sys
from pathlib import Path
print(",".join(sorted(p.name for p in Path(sys.argv[1]).glob("episode_*"))))
PY
  )
  if [[ -n "${episode_csv}" ]]; then
    hab_python -u "${SOURCE_ROOT}/MemNavData/eval_2leg_habitat.py" \
      --episode_root "${scene_source}" --episode_ids "${episode_csv}" \
      --scene "${stage}" --scene_identity "${scene}" --host 127.0.0.1 \
      --port "${NAVDP_PORT}" --out "${scene_root}/native_a" \
      --server_backend navdp --hybrid_route phase \
      --revisit_adapter legacy_metric --leg1_mode policy \
      --write_leg1_trace --stop_after_leg1 --leg1_goal_source own \
      --native_trace_navdp_checkpoint_sha256 "${EXPECTED_NAVDP_SHA}" \
      --navdp_goal_switch_reset carry --success_dist 1.0 --max_steps 600 \
      --exec_horizon 8 --trajectory_selector server \
      --trajectory_selector_scope all --seed "${online_seed}" \
      --terminal_uturn off --terminal_visual_refine off \
      --deterministic_plan_seeds \
      > "${scene_root}/eval_native_a.log" 2>&1
    hab_python -u "${SOURCE_ROOT}/MemNavData/materialize_paper_online_a_scene.py" \
      --trace-root "${scene_root}/native_a" \
      --asset-root "$(dirname "$(dirname "${stage}")")" \
      --episode-root "${RUN_ROOT}/source_episodes" \
      --asset-map-json "${RUN_ROOT}/asset_map.json" \
      --out "${scene_root}/online_a" \
      > "${scene_root}/materialize_online_a.log" 2>&1
  else
    mkdir "${scene_root}/native_a" "${scene_root}/online_a"
    hab_python - "${scene_root}/online_a/manifest.json" \
      "${scene_root}/native_a" <<'PY'
import json,sys
p={
 "schema_version":"shared_online_a_materialized_v1",
 "purpose":"empty formal fragment after frozen source-generation attrition",
 "trace_root":sys.argv[2],
 "selection":{
  "requested_count":None,"eligible_count":0,"distinct_scene_first":False,
  "anchor_margin":39,"anchor_end_margin":16,
  "anchor_requirement":"one_runtime_eligible_frame",
  "minimum_anchor_gap_frames":None,"minimum_preselection_score_m":None,
  "goals_frozen":False,"all_eligible_traces_attempted":True,
 },
 "source_trace_count":0,"attrition":[],"episodes":[],
}
open(sys.argv[1],"x").write(json.dumps(p,indent=2,sort_keys=True)+"\n")
PY
    sha256sum "${scene_root}/online_a/manifest.json" \
      > "${scene_root}/online_a/manifest.json.sha256"
  fi
  hab_python -u "${SOURCE_ROOT}/MemNavData/build_paper_role_pair_scene.py" \
    --online-root "${scene_root}/online_a" \
    --out "${scene_root}/role_pairs" \
    > "${scene_root}/build_role_pairs.log" 2>&1
  hab_python -u "${SOURCE_ROOT}/MemNavData/write_replica_formal_scene_receipt.py" \
    --run-root "${RUN_ROOT}" --freeze-manifest "${FREEZE_MANIFEST}" \
    --scene-index "${index}" --out "${scene_root}/receipt.json" \
    > "${scene_root}/receipt.stdout.json"
done < <(hab_python - "${FREEZE_MANIFEST}" <<'PY'
import json,sys
for row in json.load(open(sys.argv[1]))["scenes"]:
 print("\t".join(map(str,(row["index"],row["scene"],row["stage"],row["online_a_seed"]))))
PY
)

hab_python -u "${SOURCE_ROOT}/MemNavData/summarize_replica_online_a.py" \
  --root "${RUN_ROOT}" --freeze-manifest "${FREEZE_MANIFEST}" \
  --out "${RUN_ROOT}/online_a_inventory.json" \
  > "${RUN_ROOT}/logs/online_a_inventory.stdout.json"
sha256sum "${RUN_ROOT}/online_a_inventory.json" \
  > "${RUN_ROOT}/online_a_inventory.json.sha256"
hab_python -u "${SOURCE_ROOT}/MemNavData/finalize_paper_role_pairs.py" \
  --run-root "${RUN_ROOT}" --out "${RUN_ROOT}/benchmarks" \
  --expected-scene-count 10 --target-histories 20 --target-scenes 8 \
  --benchmark-scope replica_cross_dataset_stress_underpowered \
  > "${RUN_ROOT}/logs/finalize_benchmarks.stdout.json"

# The fresh-primary population is frozen before the first query arm starts.
hab_python - "${RUN_ROOT}/benchmarks/population_receipt.json" \
  "${RUN_ROOT}/benchmarks/primary_population_receipt.json" <<'PY'
import json,sys
p=json.load(open(sys.argv[1])); pilot="room_0"
population=[row for row in p["population"] if row["scene"] != pilot]
scenes=len({row["scene"] for row in population})
out={
 "schema_version":"replica_cross_dataset_stress_population_v2_20260814",
 "pilot_scene_excluded":pilot,"population":population,
 "histories":len(population),"scene_clusters":scenes,
 "target_histories":20,"target_scene_clusters":8,
 "target_met":len(population)>=20 and scenes>=8,
 "underpowered_if_target_not_met":not(len(population)>=20 and scenes>=8),
 "formal_confirmation_authorized":False,
 "reporting_scope":"cross-dataset stress only",
 "policy_query_outcomes_read":False,
}
open(sys.argv[2],"x").write(json.dumps(out,indent=2,sort_keys=True)+"\n")
PY
(
  cd "${RUN_ROOT}/benchmarks"
  find . -type f ! -name BENCHMARK_FILES.sha256 \
    ! -name BENCHMARK_FILES.sha256.sha256 ! -name SEALED \
    -print0 | sort -z | xargs -0 sha256sum > BENCHMARK_FILES.sha256
  sha256sum -c BENCHMARK_FILES.sha256 >/dev/null
  sha256sum BENCHMARK_FILES.sha256 > BENCHMARK_FILES.sha256.sha256
  : > SEALED
  chmod -R a-w .
)
chmod -R a-w "${RUN_ROOT}/traces" "${RUN_ROOT}/source_episodes"
: > "${RUN_ROOT}/CONSTRUCTION_SEALED"

# A cross-dataset stress run with no fresh constructible history has no policy
# estimand.  Stop from the sealed population alone, before starting MemNav or a
# query arm; consumed-pilot-only queries cannot repair an empty fresh sample.
fresh_population=$(hab_python - \
  "${RUN_ROOT}/benchmarks/primary_population_receipt.json" <<'PY'
import json,sys
p=json.load(open(sys.argv[1]))
assert p["policy_query_outcomes_read"] is False
print(int(p["histories"]))
PY
)
if (( fresh_population == 0 )); then
  : > "${RUN_ROOT}/POPULATION_GATE_STOPPED"
  echo "Replica stress stopped: sealed fresh population is empty"
  exit 0
fi

# Start the frozen memory server only after construction and primary admission
# are sealed.  This prevents any query outcome from feeding back into sampling.
(
  exec env PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    PYTHONPATH="${SERVER_PYTHONPATH}" \
    LINGBOT_REPO="${LINGBOT_REPO}" LINGBOT_WEIGHTS="${LINGBOT_WEIGHTS}" \
    MEMNAV_WINDOW=32 MEMNAV_NUM_SCALE=8 MEMNAV_MAX_FRAME_NUM=2048 \
    MEMNAV_GROUND_SCALE_MAX=6.0 MEMNAV_GATE_FUSION=complementary \
    MEMNAV_AUX_POSE_CALIBRATION=empirical MEMNAV_COLLISION_SELECT=1 \
    MEMNAV_REPORT_TO=none \
    "${MEMNAV_PY}" -u "${SOURCE_ROOT}/MemNavData/retrying_server_launcher.py" \
      --base-port 24040 --range-start 20000 --range-end 49999 \
      --stride 1994 --max-attempts 12 --ready-timeout 720 \
      --port-file "${runtime_root}/memnav.port" \
      --receipt-file "${RUN_ROOT}/logs/memnav_launcher.json" \
      --log-prefix "${RUN_ROOT}/logs/memnav_server.child" \
      --cwd "${runtime_root}/memnav" -- \
      "${MEMNAV_PY}" -u "${SOURCE_ROOT}/NavDP/baselines/memnav/memnav_server.py" \
        --port '{port}' --checkpoint "${MEMNAV_CKPT}" \
        --internnav_root "${INTERNNAV_ROOT}" --num_samples 16 \
        --exclude_recent 32 --retrieval raw \
        --retrieval_candidate_top_k 32 --retrieval_candidate_min_gap 16 \
        --graph_subgoal_spacing_m 0.0 --graph_subgoal_arrival_m 0.60 \
        --flow_gate auto --buffer_root "${RUN_ROOT}/buffer" \
        --certified_relocalization --lightglue_repo "${LIGHTGLUE_REPO}" \
        --lightglue_dependency_root "${DEPENDENCY_ROOT}" \
        --lightglue_max_keypoints 2048
) > "${RUN_ROOT}/logs/memnav_launcher.log" 2>&1 &
MEMNAV_PID=$!
for _ in $(seq 1 420); do
  kill -0 "${MEMNAV_PID}" 2>/dev/null || fail "MemNav launcher exited"
  [[ -s "${runtime_root}/memnav.port" ]] && break
  sleep 2
done
[[ -s "${runtime_root}/memnav.port" ]] || fail "MemNav startup timed out"
MEMNAV_PORT=$(<"${runtime_root}/memnav.port")

for protocol in support_controlled natural_direction; do
  bench_root=${RUN_ROOT}/benchmarks/${protocol}
  expected_manifest_sha=$(awk '{print $1}' "${bench_root}/manifest.json.sha256")
  count=$(hab_python - "${bench_root}/manifest.json" <<'PY'
import json,sys
print(len(json.load(open(sys.argv[1]))["episodes"]))
PY
  )
  for ((index=0; index<count; index++)); do
    env SOURCE_ROOT="${SOURCE_ROOT}" RUN_ROOT="${RUN_ROOT}" \
      BENCH_ROOT="${bench_root}" \
      EXPECTED_MANIFEST_SHA="${expected_manifest_sha}" \
      PROTOCOL="${protocol}" EPISODE_INDEX="${index}" \
      HAB_PY="${HAB_PY}" MEMNAV_PORT="${MEMNAV_PORT}" \
      NAVDP_PORT="${NAVDP_PORT}" MAX_STEPS=600 \
      bash "${SOURCE_ROOT}/MemNavData/run_paper_role_pair_episode.sh" \
      >> "${RUN_ROOT}/logs/query_progress.log" 2>&1
  done
done

hab_python -u "${SOURCE_ROOT}/MemNavData/summarize_paper_role_pair_eval.py" \
  --root "${RUN_ROOT}" --out "${RUN_ROOT}/evaluation_summary_all10_descriptive.json" \
  > "${RUN_ROOT}/logs/summary_all10.stdout.json"
hab_python -u "${SOURCE_ROOT}/MemNavData/independent_verify_paper_role_pair_eval.py" \
  --root "${RUN_ROOT}" \
  --summary "${RUN_ROOT}/evaluation_summary_all10_descriptive.json" \
  --out "${RUN_ROOT}/independent_verification_all10.json" \
  > "${RUN_ROOT}/logs/verify_all10.stdout.json"
hab_python -u "${SOURCE_ROOT}/MemNavData/summarize_paper_role_pair_eval.py" \
  --root "${RUN_ROOT}" --exclude-scene room_0 \
  --out "${RUN_ROOT}/evaluation_summary_fresh_stress.json" \
  > "${RUN_ROOT}/logs/summary_fresh_stress.stdout.json"
hab_python -u "${SOURCE_ROOT}/MemNavData/independent_verify_paper_role_pair_eval.py" \
  --root "${RUN_ROOT}" \
  --summary "${RUN_ROOT}/evaluation_summary_fresh_stress.json" \
  --exclude-scene room_0 \
  --out "${RUN_ROOT}/independent_verification_fresh_stress.json" \
  > "${RUN_ROOT}/logs/verify_fresh_stress.stdout.json"
(
  cd "${RUN_ROOT}"
  sha256sum evaluation_summary_all10_descriptive.json \
    evaluation_summary_fresh_stress.json \
    independent_verification_all10.json \
    independent_verification_fresh_stress.json \
    > FINAL_RESULTS.sha256
  sha256sum -c FINAL_RESULTS.sha256 >/dev/null
)
: > "${RUN_ROOT}/PIPELINE_COMPLETE"
echo "Replica cross-dataset stress evaluation complete: ${RUN_ROOT}"
