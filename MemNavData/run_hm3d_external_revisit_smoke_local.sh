#!/usr/bin/env bash
# Consumed-scene implementation smoke for the HM3D external Revisit transfer.
#
# This is deliberately not an efficacy estimate.  It collects Goal-A once with
# frozen native NavDP, replays that byte-identical causal trace, and exercises
# native, raw-DINO fixed-bearing, and role-free certified control on Goal-B.

set -euo pipefail
umask 0022

ROOT=${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
EPISODE_ROOT=${EPISODE_ROOT:-${ROOT}/.diagnostics/hm3d_external_revisit_smoke_20260816/scene_5cd}
EPISODE_ID=${EPISODE_ID:-episode_0000}
SCENE_FILE=${SCENE_FILE:-${ROOT}/.diagnostics/datasets/goat-smoke-hm3d-20260814/data/scene_datasets/hm3d/val/00853-5cdEh9F2hJL/5cdEh9F2hJL.basis.glb}
OUT_ROOT=${OUT_ROOT:-${ROOT}/.diagnostics/hm3d_external_revisit_smoke_20260816/closed_loop}
MEMNAV_PORT=${MEMNAV_PORT:-22140}
NAVDP_PORT=${NAVDP_PORT:-22141}
MAX_STEPS=${MAX_STEPS:-500}

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
[[ "${MAX_STEPS}" =~ ^[1-9][0-9]*$ ]] || fail "MAX_STEPS must be positive"
[[ ! -e "${OUT_ROOT}" ]] || fail "output already exists: ${OUT_ROOT}"
for path in \
  "${EPISODE_ROOT}/${EPISODE_ID}/meta/gen_meta.json" \
  "${EPISODE_ROOT}/${EPISODE_ID}/data/chunk-000/episode_000000.parquet" \
  "${EPISODE_ROOT}/${EPISODE_ID}/goal_image.jpg" \
  "${SCENE_FILE}" "${MEMNAV_PY}" "${HAB_PY}" "${MEMNAV_CKPT}" \
  "${NAVDP_CKPT}" "${LINGBOT_WEIGHTS}" "${LIGHTGLUE_REPO}/lightglue" \
  "${DEPENDENCY_ROOT}/kornia"; do
  test -r "${path}" || fail "missing input: ${path}"
done
for port in "${MEMNAV_PORT}" "${NAVDP_PORT}"; do
  ! ss -ltn | awk '{print $4}' | grep -Eq "(^|:)${port}$" || \
    fail "port ${port} is already in use"
done

mkdir -p "${OUT_ROOT}/logs" "${OUT_ROOT}/buffer"
runtime_root=$(mktemp -d /tmp/hm3d_external_revisit_smoke.XXXXXX)
MEMNAV_PID=
NAVDP_PID=
cleanup() {
  for process_id in "${NAVDP_PID}" "${MEMNAV_PID}"; do
    if [[ -n "${process_id}" ]] && kill -0 "${process_id}" 2>/dev/null; then
      kill "${process_id}" 2>/dev/null || true
      wait "${process_id}" 2>/dev/null || true
    fi
  done
  rm -rf -- "${runtime_root}"
}
trap cleanup EXIT INT TERM

hab_site_packages=$("${HAB_PY}" -c \
  'import sysconfig; print(sysconfig.get_paths()["purelib"])')
hab_pythonpath=${ROOT}:${ROOT}/MemNavData:${hab_site_packages}/pip/_vendor${PYTHONPATH:+:${PYTHONPATH}}
server_pythonpath=${ROOT}:${DEPENDENCY_ROOT}:${LIGHTGLUE_REPO}:${INTERNNAV_ROOT}/src/diffusion-policy${PYTHONPATH:+:${PYTHONPATH}}

"${HAB_PY}" -m py_compile \
  "${ROOT}/MemNavData/eval_2leg_habitat.py" \
  "${ROOT}/MemNavData/certified_relocalization_runtime.py" \
  "${ROOT}/MemNavData/revisit_bearing_adapter.py"

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
  IFS=: read -r label process_id port log_path <<<"${spec}"
  ready=0
  for _ in $(seq 1 240); do
    kill -0 "${process_id}" 2>/dev/null || {
      tail -n 160 "${log_path}" >&2
      fail "${label} server exited during startup"
    }
    if ss -ltn | awk '{print $4}' | grep -Eq "(^|:)${port}$"; then
      ready=1
      break
    fi
    sleep 2
  done
  [[ "${ready}" -eq 1 ]] || fail "${label} server did not bind port ${port}"
done

common=(
  --episode_root "${EPISODE_ROOT}"
  --episode_ids "${EPISODE_ID}"
  --scene "${SCENE_FILE}"
  --host 127.0.0.1
  --success_dist 1.0
  --max_steps "${MAX_STEPS}"
  --exec_horizon 8
  --trajectory_selector server
  --trajectory_selector_scope all
  --navdp_goal_switch_reset carry
  --leg1_goal_source own
  --seed 2026081602
  --terminal_uturn off
  --terminal_visual_refine off
  --deterministic_plan_seeds
)

trace_root=${OUT_ROOT}/trace_source
mkdir -p "${trace_root}"
env PYTHONPATH="${hab_pythonpath}" "${HAB_PY}" -u \
  "${ROOT}/MemNavData/eval_2leg_habitat.py" "${common[@]}" \
  --port "${MEMNAV_PORT}" --novel_port "${NAVDP_PORT}" \
  --out "${trace_root}" --server_backend hybrid_pose \
  --leg1_mode policy --write_leg1_trace --stop_after_leg1 \
  --hybrid_route phase --revisit_adapter legacy_metric \
  >"${OUT_ROOT}/logs/eval_trace_source.log" 2>&1

for arm in native raw_fixed certified; do
  arm_root=${OUT_ROOT}/${arm}
  mkdir -p "${arm_root}"
  case "${arm}" in
    native)
      extra=(--port "${NAVDP_PORT}" --server_backend navdp \
        --hybrid_route phase)
      ;;
    raw_fixed)
      extra=(--port "${MEMNAV_PORT}" --novel_port "${NAVDP_PORT}" \
        --server_backend hybrid_pose --hybrid_route phase \
        --revisit_controller navdp_mixed \
        --revisit_adapter raw_fixed_bearing_v1)
      ;;
    certified)
      extra=(--port "${MEMNAV_PORT}" --novel_port "${NAVDP_PORT}" \
        --server_backend hybrid_pose \
        --hybrid_route certified_relocalization \
        --revisit_controller navdp_mixed \
        --revisit_adapter verified_bearing_v1)
      ;;
    *) fail "unknown arm ${arm}" ;;
  esac
  env PYTHONPATH="${hab_pythonpath}" "${HAB_PY}" -u \
    "${ROOT}/MemNavData/eval_2leg_habitat.py" "${common[@]}" \
    --leg1_mode shared_trace --shared_leg1_trace_root "${trace_root}" \
    --out "${arm_root}" "${extra[@]}" \
    >"${OUT_ROOT}/logs/eval_${arm}.log" 2>&1
done

env PYTHONPATH="${hab_pythonpath}" "${HAB_PY}" - \
  "${OUT_ROOT}" "${EPISODE_ROOT}/${EPISODE_ID}/meta/gen_meta.json" \
  "${EPISODE_ID}" <<'PY'
import csv
import hashlib
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
metadata = pathlib.Path(sys.argv[2])
episode_id = sys.argv[3]
rows = {}
for arm in ("native", "raw_fixed", "certified"):
    with (root / arm / "metric.csv").open(newline="") as handle:
        values = list(csv.DictReader(handle))
    if len(values) != 1 or values[0]["episode"] != episode_id:
        raise SystemExit(f"{arm}: incomplete metric identity")
    rows[arm] = values[0]
trace_rows = list(csv.DictReader((root / "trace_source" / "metric.csv").open()))
if len(trace_rows) != 1:
    raise SystemExit("trace source is incomplete")
receipt = {
    "schema_version": "hm3d_external_revisit_smoke_v1_20260816",
    "scope": "consumed HM3D engineering smoke; no efficacy claim",
    "actual_online_goal_a_trace": True,
    "goal_a_success": float(trace_rows[0]["reached_A"]) > 0.5,
    "episode_id": episode_id,
    "metadata_sha256": hashlib.sha256(metadata.read_bytes()).hexdigest(),
    "arms": {
        arm: {
            "reached_a": float(row["reached_A"]) > 0.5,
            "reached_b": float(row["reached_B"]) > 0.5,
            "steps_b": int(float(row["steps_B"])),
            "certificate_accept_count": int(float(
                row.get("certified_relocalization_accept_count") or 0)),
            "certificate_request_count": int(float(
                row.get("certified_relocalization_request_count") or 0)),
            "takeover_plan_count": int(float(
                row.get("revisit_adapter_takeover_plan_count") or 0)),
        }
        for arm, row in rows.items()
    },
}
receipt["full_goal_b_execution"] = (
    receipt["goal_a_success"]
    and all(item["reached_a"] for item in receipt["arms"].values())
    and all(item["steps_b"] > 0 for item in receipt["arms"].values())
    and receipt["arms"]["certified"]["certificate_request_count"] > 0
)
receipt["passed"] = receipt["full_goal_b_execution"]
with (root / "receipt.json").open("x", encoding="utf-8") as handle:
    json.dump(receipt, handle, indent=2, sort_keys=True)
    handle.write("\n")
print(json.dumps(receipt, indent=2, sort_keys=True))
PY
