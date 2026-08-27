#!/usr/bin/env bash
# Runtime smoke for the paper-only geometry_fixed role-pair arm.

set -euo pipefail
umask 0022

ROOT=${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
BENCH_ROOT=${BENCH_ROOT:-${ROOT}/.diagnostics/shared_online_role_pair_natural_heading_v1_smoke_20260814}
OUT_ROOT=${OUT_ROOT:-${ROOT}/.diagnostics/paper_geometry_fixed_arm_smoke_20260814}
MEMNAV_PORT=${MEMNAV_PORT:-21840}
NAVDP_PORT=${NAVDP_PORT:-21841}
MEMNAV_PY=${MEMNAV_PY:-/home/asus/miniconda3/envs/memnav/bin/python}
HAB_PY=${HAB_PY:-/home/asus/miniconda3/envs/habitat/bin/python}
MEMNAV_CKPT=${MEMNAV_CKPT:-/home/asus/Research/Nav-axis-uturn/.diagnostics/unseen_scene_eval_20260803/checkpoints/gatecurr600.memnav.ckpt}
NAVDP_CKPT=${NAVDP_CKPT:-/home/asus/Research/Nav/NavDP/baselines/navdp/checkpoints/navdp_checkpoint.ckpt}
LINGBOT_REPO=${LINGBOT_REPO:-/home/asus/Research/Nav/NavDP/baselines/memnav/lingbot-map}
LINGBOT_WEIGHTS=${LINGBOT_WEIGHTS:-${LINGBOT_REPO}/weights/lingbot-map-long.pt}
LIGHTGLUE_REPO=${LIGHTGLUE_REPO:-${ROOT}/.diagnostics/dependencies/LightGlue}
DEPENDENCY_ROOT=${DEPENDENCY_ROOT:-${ROOT}/.diagnostics/dependencies/python}
INTERNNAV_ROOT=${INTERNNAV_ROOT:-${ROOT}/InternNav}

fail() { echo "ABORT: $*" >&2; exit 2; }
[[ ! -e "${OUT_ROOT}" ]] || fail "output exists: ${OUT_ROOT}"
for port in "${MEMNAV_PORT}" "${NAVDP_PORT}"; do
  ! ss -ltn | awk '{print $4}' | grep -Eq "(^|:)${port}$" || \
    fail "port ${port} is already in use"
done

manifest_sha=$(sha256sum "${BENCH_ROOT}/manifest.json" | awk '{print $1}')
readarray -t identity < <("${HAB_PY}" - "${BENCH_ROOT}/manifest.json" <<'PY'
import json,sys
row=json.load(open(sys.argv[1]))["episodes"][0]
receipt=json.load(open(row["online_a_episode"]+"/receipt.json"))
print(row["scene"]); print(row["episode"]); print(receipt["source_asset"])
PY
)
[[ "${#identity[@]}" -eq 3 ]] || fail "benchmark identity read failed"
scene=${identity[0]}; episode=${identity[1]}; scene_file=${identity[2]}

mkdir -p "${OUT_ROOT}/logs" "${OUT_ROOT}/buffer" "${OUT_ROOT}/result"
runtime_root=$(mktemp -d /tmp/paper_geometry_fixed_smoke.XXXXXX)
MEMNAV_PID=; NAVDP_PID=
cleanup() {
  for pid in "${NAVDP_PID}" "${MEMNAV_PID}"; do
    if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
      kill "${pid}" 2>/dev/null || true
      wait "${pid}" 2>/dev/null || true
    fi
  done
  rm -rf -- "${runtime_root}"
}
trap cleanup EXIT INT TERM

hab_site_packages=$("${HAB_PY}" -c \
  'import sysconfig; print(sysconfig.get_paths()["purelib"])')
hab_pythonpath=${ROOT}:${ROOT}/MemNavData:${hab_site_packages}/pip/_vendor${PYTHONPATH:+:${PYTHONPATH}}
server_pythonpath=${ROOT}:${DEPENDENCY_ROOT}:${LIGHTGLUE_REPO}:${INTERNNAV_ROOT}/src/diffusion-policy${PYTHONPATH:+:${PYTHONPATH}}
mkdir -p "${runtime_root}/memnav" "${runtime_root}/navdp"
(
  cd "${runtime_root}/memnav"
  exec env PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    PYTHONPATH="${server_pythonpath}" LINGBOT_REPO="${LINGBOT_REPO}" \
    LINGBOT_WEIGHTS="${LINGBOT_WEIGHTS}" MEMNAV_WINDOW=32 \
    MEMNAV_NUM_SCALE=8 MEMNAV_MAX_FRAME_NUM=2048 \
    MEMNAV_GROUND_SCALE_MAX=6.0 MEMNAV_GATE_FUSION=complementary \
    MEMNAV_AUX_POSE_CALIBRATION=empirical MEMNAV_COLLISION_SELECT=1 \
    MEMNAV_REPORT_TO=none "${MEMNAV_PY}" -u \
    "${ROOT}/NavDP/baselines/memnav/memnav_server.py" \
    --port "${MEMNAV_PORT}" --checkpoint "${MEMNAV_CKPT}" \
    --internnav_root "${INTERNNAV_ROOT}" --num_samples 16 \
    --exclude_recent 32 --retrieval raw --retrieval_candidate_top_k 32 \
    --retrieval_candidate_min_gap 16 --graph_subgoal_spacing_m 0.0 \
    --graph_subgoal_arrival_m 0.60 --flow_gate auto \
    --buffer_root "${OUT_ROOT}/buffer" --certified_relocalization \
    --lightglue_repo "${LIGHTGLUE_REPO}" \
    --lightglue_dependency_root "${DEPENDENCY_ROOT}" \
    --lightglue_max_keypoints 2048
) >"${OUT_ROOT}/logs/server_memnav.log" 2>&1 &
MEMNAV_PID=$!
(
  cd "${runtime_root}/navdp"
  exec env NAVDP_DISABLE_VIDEO=1 PYTHONUNBUFFERED=1 \
    PYTHONPATH="${server_pythonpath}" "${MEMNAV_PY}" -u \
    "${ROOT}/NavDP/baselines/navdp/navdp_server.py" \
    --port "${NAVDP_PORT}" --checkpoint "${NAVDP_CKPT}"
) >"${OUT_ROOT}/logs/server_navdp.log" 2>&1 &
NAVDP_PID=$!

for spec in \
  "memnav:${MEMNAV_PID}:${MEMNAV_PORT}:${OUT_ROOT}/logs/server_memnav.log" \
  "navdp:${NAVDP_PID}:${NAVDP_PORT}:${OUT_ROOT}/logs/server_navdp.log"; do
  IFS=: read -r label pid port log <<<"${spec}"
  ready=0
  for _ in $(seq 1 240); do
    kill -0 "${pid}" 2>/dev/null || { tail -n 120 "${log}" >&2; fail "${label} exited"; }
    if ss -ltn | awk '{print $4}' | grep -Eq "(^|:)${port}$"; then
      ready=1; break
    fi
    sleep 2
  done
  [[ "${ready}" -eq 1 ]] || fail "${label} did not bind"
done

env PYTHONPATH="${hab_pythonpath}" "${HAB_PY}" -u \
  "${ROOT}/MemNavData/eval_shared_online_role_pairs.py" \
  --episode_root "${BENCH_ROOT}/${scene}" --episode_ids "${episode}" \
  --scene "${scene_file}" --host 127.0.0.1 --success_dist 1.0 \
  --max_steps 120 --exec_horizon 8 --trajectory_selector server \
  --trajectory_selector_scope all --leg1_mode shared_trace \
  --leg1_goal_source own --seed 0 --terminal_uturn off \
  --terminal_visual_refine off --deterministic_plan_seeds \
  --retrieval_override off --certified_cdec_rescue off \
  --certified_stagnation_graph off --revisit_controller navdp_mixed \
  --role_pair_scope paper_heldout --port "${MEMNAV_PORT}" \
  --novel_port "${NAVDP_PORT}" --server_backend hybrid_pose \
  --hybrid_route memory_geometry --revisit_adapter verified_bearing_v1 \
  --router_visual_floor 0.88 --router_min_matches 20 \
  --router_min_inliers 12 --router_min_inlier_ratio 0.50 \
  --router_confirm_plans 2 --router_verify_top_k 8 \
  --out "${OUT_ROOT}/result" >"${OUT_ROOT}/logs/eval.log" 2>&1

env PYTHONPATH="${hab_pythonpath}" "${HAB_PY}" - \
  "${OUT_ROOT}/result/summary.json" "${OUT_ROOT}/receipt.json" \
  "${manifest_sha}" <<'PY'
import json,sys
summary=json.load(open(sys.argv[1]))
assert summary["arm"]=="geometry_fixed"
assert summary["role_pair_scope"]=="paper_heldout"
assert summary["scope"]=="paper held-out role-pair evaluation"
assert summary["queries"]==2 and summary["runtime_role_visibility"]=="none"
assert summary["shared_A_all_hashes_ok"] is True
assert summary["shared_A_total_diffusion_samples"]==0
receipt={
 "schema_version":"paper_geometry_fixed_arm_smoke_v1_20260814",
 "scope":"consumed implementation smoke; no efficacy claim",
 "passed":True,"benchmark_manifest_sha256":sys.argv[3],
 "arm":summary["arm"],"role_pair_scope":summary["role_pair_scope"],
 "queries":summary["queries"],"runtime_role_visibility":"none",
 "shared_A_all_hashes_ok":True,"shared_A_total_diffusion_samples":0,
}
with open(sys.argv[2],"x") as handle:
 json.dump(receipt,handle,indent=2,sort_keys=True); handle.write("\n")
PY
sha256sum "${OUT_ROOT}/receipt.json"
