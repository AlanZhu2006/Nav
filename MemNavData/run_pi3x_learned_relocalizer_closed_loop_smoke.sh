#!/usr/bin/env bash
# Consumed-scene transport smoke for the frozen learned Pi3X relocalizer.
#
# This is an implementation/schema gate, not an efficacy estimate.  Goal A is
# collected once with native NavDP and then replayed byte-identically through
# a native Goal-B arm and the learned relocalization arm.

set -euo pipefail
umask 0022

ROOT=${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
BASE_ROOT=${BASE_ROOT:-${ROOT}}
EPISODE_ROOT=${EPISODE_ROOT:?set consumed two-leg episode root}
EPISODE_ID=${EPISODE_ID:-episode_0000}
SCENE_FILE=${SCENE_FILE:?set consumed scene GLB}
OUT_ROOT=${OUT_ROOT:?set a fresh output root}
MEMNAV_PORT=${MEMNAV_PORT:-22940}
NAVDP_PORT=${NAVDP_PORT:-22941}
MAX_STEPS=${MAX_STEPS:-500}
SMOKE_SCOPE=${SMOKE_SCOPE:-consumed MP3D transport/schema smoke; no efficacy claim}

MEMNAV_PY=${MEMNAV_PY:-/home/asus/miniconda3/envs/memnav/bin/python}
HAB_PY=${HAB_PY:-/home/asus/miniconda3/envs/habitat/bin/python}
MEMNAV_CKPT=${MEMNAV_CKPT:?set frozen MemNav checkpoint}
NAVDP_CKPT=${NAVDP_CKPT:?set frozen NavDP checkpoint}
INTERNNAV_ROOT=${INTERNNAV_ROOT:-${BASE_ROOT}/InternNav}
LINGBOT_REPO=${LINGBOT_REPO:?set LingBot repository}
LINGBOT_WEIGHTS=${LINGBOT_WEIGHTS:?set frozen LingBot weights}
PI3X_ROOT=${PI3X_ROOT:?set official Pi3 source root}
PI3X_SNAPSHOT=${PI3X_SNAPSHOT:?set frozen Pi3X snapshot directory}
PI3X_PROOF_MANIFEST=${PI3X_PROOF_MANIFEST:?set frozen proof manifest}
PI3X_MODEL_SHA256=${PI3X_MODEL_SHA256:-69972d6e1c4492cb4d737a84fe940e357087d81c52f5c9b7c160b49c1f41669a}
PI3X_PROOF_MANIFEST_SHA256=${PI3X_PROOF_MANIFEST_SHA256:-1a05aaa7cf75296cb68e32f9ea57fba6bcce2b9f57313a8cede05b7c7b0cffdd}
MEMNAV_CKPT_SHA256=${MEMNAV_CKPT_SHA256:-9b7a5811ff0aea212503f58b45258ba4f66b06420f87c350946aead39db6fdb7}
NAVDP_CKPT_SHA256=${NAVDP_CKPT_SHA256:-3bb3ad4ab241e857bb57a4021cc6aab76d5263e81fbf80298d579053ef011947}
SOURCE_BUNDLE_MANIFEST=${SOURCE_BUNDLE_MANIFEST:-${ROOT}/SOURCE_BUNDLE.sha256}
EXPECTED_SOURCE_RECEIPT_SHA256=${EXPECTED_SOURCE_RECEIPT_SHA256:-}

fail() { echo "ABORT: $*" >&2; exit 2; }
[[ "${MAX_STEPS}" =~ ^[1-9][0-9]*$ ]] || fail "MAX_STEPS must be positive"
[[ ! -e "${OUT_ROOT}" ]] || fail "output already exists: ${OUT_ROOT}"
for path in \
  "${ROOT}/MemNavData/eval_2leg_habitat.py" \
  "${ROOT}/MemNavData/generate_twoleg.py" \
  "${ROOT}/MemNavData/terminal_uturn.py" \
  "${ROOT}/MemNavData/visual_yaw_refinement.py" \
  "${ROOT}/MemNavData/deterministic_eval_protocol.py" \
  "${ROOT}/MemNavData/arrival_shadow.py" \
  "${ROOT}/MemNavData/navdp_goal_switch.py" \
  "${ROOT}/MemNavData/revisit_bearing_adapter.py" \
  "${ROOT}/MemNavData/revisit_action_shadow.py" \
  "${ROOT}/MemNavData/shared_online_double_revisit_runtime.py" \
  "${ROOT}/MemNavData/xnavdp_revisit_contract.py" \
  "${ROOT}/MemNavData/pi3x_online_relocalizer.py" \
  "${ROOT}/MemNavData/pi3x_spatial_proof_runtime.py" \
  "${ROOT}/MemNavData/pi3x_spatial_reliability_model.py" \
  "${ROOT}/NavDP/baselines/memnav/memnav_server.py" \
  "${ROOT}/NavDP/baselines/memnav/policy_agent.py" \
  "${ROOT}/NavDP/baselines/navdp/navdp_server.py" \
  "${ROOT}/NavDP/baselines/navdp/policy_agent.py" \
  "${ROOT}/NavDP/baselines/navdp/deterministic_seed.py" \
  "${SOURCE_BUNDLE_MANIFEST}" \
  "${EPISODE_ROOT}/${EPISODE_ID}/meta/gen_meta.json" \
  "${EPISODE_ROOT}/${EPISODE_ID}/data/chunk-000/episode_000000.parquet" \
  "${EPISODE_ROOT}/${EPISODE_ID}/goal_1.jpg" \
  "${EPISODE_ROOT}/${EPISODE_ID}/goal_image.jpg" \
  "${SCENE_FILE}" "${MEMNAV_PY}" "${HAB_PY}" "${MEMNAV_CKPT}" \
  "${NAVDP_CKPT}" "${LINGBOT_WEIGHTS}" \
  "${PI3X_ROOT}/pi3/models/pi3x.py" \
  "${PI3X_SNAPSHOT}/model.safetensors" "${PI3X_PROOF_MANIFEST}"; do
  test -r "${path}" || fail "missing input: ${path}"
done

(
  cd "${ROOT}"
  sha256sum -c "${SOURCE_BUNDLE_MANIFEST}" >/dev/null
) || fail "source bundle manifest validation failed"
SOURCE_RECEIPT_SHA256=$(sha256sum "${SOURCE_BUNDLE_MANIFEST}" | awk '{print $1}')
if [[ -n "${EXPECTED_SOURCE_RECEIPT_SHA256}" ]]; then
  [[ "${SOURCE_RECEIPT_SHA256}" == "${EXPECTED_SOURCE_RECEIPT_SHA256}" ]] || \
    fail "source bundle receipt changed"
fi

[[ "$(sha256sum "${PI3X_SNAPSHOT}/model.safetensors" | awk '{print $1}')" == \
    "${PI3X_MODEL_SHA256}" ]] || fail "Pi3X model identity changed"
[[ "$(sha256sum "${PI3X_PROOF_MANIFEST}" | awk '{print $1}')" == \
    "${PI3X_PROOF_MANIFEST_SHA256}" ]] || fail "proof manifest changed"
[[ "$(sha256sum "${MEMNAV_CKPT}" | awk '{print $1}')" == \
    "${MEMNAV_CKPT_SHA256}" ]] || fail "MemNav checkpoint changed"
[[ "$(sha256sum "${NAVDP_CKPT}" | awk '{print $1}')" == \
    "${NAVDP_CKPT_SHA256}" ]] || fail "NavDP checkpoint changed"
(
  cd "$(dirname "${PI3X_PROOF_MANIFEST}")"
  sha256sum -c OUTPUTS.sha256 >/dev/null
) || fail "proof deployment files changed"
for port in "${MEMNAV_PORT}" "${NAVDP_PORT}"; do
  ! ss -ltn | awk '{print $4}' | grep -Eq "(^|:)${port}$" || \
    fail "port ${port} is already in use"
done

mkdir -p "${OUT_ROOT}/logs" "${OUT_ROOT}/buffer"
runtime_root=$(mktemp -d /tmp/pi3x_learned_closed_loop.XXXXXX)
MEMNAV_PID=; NAVDP_PID=
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
hab_pythonpath=${ROOT}:${ROOT}/MemNavData:${BASE_ROOT}:${BASE_ROOT}/MemNavData:${hab_site_packages}/pip/_vendor${PYTHONPATH:+:${PYTHONPATH}}
server_pythonpath=${ROOT}:${ROOT}/MemNavData:${BASE_ROOT}:${BASE_ROOT}/MemNavData:${BASE_ROOT}/NavDP/baselines/navdp:${BASE_ROOT}/NavDP/baselines/memnav:${INTERNNAV_ROOT}/src/diffusion-policy${PYTHONPATH:+:${PYTHONPATH}}
export PYTHONPYCACHEPREFIX=${OUT_ROOT}/pycache

env PYTHONPATH="${hab_pythonpath}" "${HAB_PY}" \
  "${ROOT}/MemNavData/eval_2leg_habitat.py" --help >/dev/null
"${HAB_PY}" -m py_compile \
  "${ROOT}/MemNavData/eval_2leg_habitat.py" \
  "${ROOT}/MemNavData/revisit_bearing_adapter.py"
"${MEMNAV_PY}" -m py_compile \
  "${ROOT}/MemNavData/pi3x_online_relocalizer.py" \
  "${ROOT}/MemNavData/pi3x_spatial_proof_runtime.py" \
  "${ROOT}/MemNavData/pi3x_spatial_reliability_model.py" \
  "${ROOT}/NavDP/baselines/memnav/memnav_server.py" \
  "${ROOT}/NavDP/baselines/memnav/policy_agent.py" \
  "${ROOT}/NavDP/baselines/navdp/navdp_server.py" \
  "${ROOT}/NavDP/baselines/navdp/policy_agent.py" \
  "${ROOT}/NavDP/baselines/navdp/deterministic_seed.py"

mkdir -p "${runtime_root}/memnav" "${runtime_root}/navdp"
(
  cd "${runtime_root}/memnav"
  exec env PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    PYTHONPATH="${server_pythonpath}" \
    PYTHONPYCACHEPREFIX="${PYTHONPYCACHEPREFIX}" \
    LINGBOT_REPO="${LINGBOT_REPO}" LINGBOT_WEIGHTS="${LINGBOT_WEIGHTS}" \
    MEMNAV_WINDOW=32 MEMNAV_NUM_SCALE=8 MEMNAV_MAX_FRAME_NUM=2048 \
    MEMNAV_GROUND_SCALE_MAX=6.0 MEMNAV_GATE_FUSION=complementary \
    MEMNAV_AUX_POSE_CALIBRATION=empirical MEMNAV_COLLISION_SELECT=1 \
    MEMNAV_REPORT_TO=none "${MEMNAV_PY}" -u \
    "${ROOT}/NavDP/baselines/memnav/memnav_server.py" \
      --port "${MEMNAV_PORT}" --checkpoint "${MEMNAV_CKPT}" \
      --internnav_root "${INTERNNAV_ROOT}" --num_samples 16 \
      --exclude_recent 32 --retrieval raw --retrieval_candidate_top_k 32 \
      --retrieval_candidate_min_gap 16 --graph_subgoal_spacing_m 0.0 \
      --graph_subgoal_arrival_m 0.60 --flow_gate auto \
      --buffer_root "${OUT_ROOT}/buffer" \
      --pi3x_learned_relocalizer \
      --pi3x_root "${PI3X_ROOT}" \
      --pi3x_snapshot "${PI3X_SNAPSHOT}" \
      --pi3x_model_sha256 "${PI3X_MODEL_SHA256}" \
      --pi3x_spatial_proof_manifest "${PI3X_PROOF_MANIFEST}" \
      --pi3x_inference_dtype auto
) >"${OUT_ROOT}/logs/server_memnav.log" 2>&1 &
MEMNAV_PID=$!
(
  cd "${runtime_root}/navdp"
  exec env NAVDP_DISABLE_VIDEO=1 PYTHONUNBUFFERED=1 \
    PYTHONPYCACHEPREFIX="${PYTHONPYCACHEPREFIX}" \
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
  for _ in $(seq 1 300); do
    kill -0 "${process_id}" 2>/dev/null || {
      tail -n 200 "${log_path}" >&2
      fail "${label} server exited during startup"
    }
    if ss -ltn | awk '{print $4}' | grep -Eq "(^|:)${port}$"; then
      ready=1
      break
    fi
    sleep 2
  done
  [[ "${ready}" -eq 1 ]] || fail "${label} server did not bind"
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
  --seed 2026081701
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

for arm in native learned_pi3x; do
  arm_root=${OUT_ROOT}/${arm}
  mkdir -p "${arm_root}"
  case "${arm}" in
    native)
      extra=(--port "${NAVDP_PORT}" --server_backend navdp \
        --hybrid_route phase)
      ;;
    learned_pi3x)
      extra=(--port "${MEMNAV_PORT}" --novel_port "${NAVDP_PORT}" \
        --server_backend hybrid_pose \
        --hybrid_route learned_pi3x_relocalization \
        --revisit_controller navdp_mixed \
        --revisit_adapter verified_bearing_v1 \
        --expected_pi3x_model_sha256 "${PI3X_MODEL_SHA256}" \
        --expected_pi3x_proof_manifest_sha256 \
          "${PI3X_PROOF_MANIFEST_SHA256}")
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
  "${EPISODE_ROOT}/${EPISODE_ID}/goal_1.jpg" \
  "${EPISODE_ID}" "${PI3X_MODEL_SHA256}" \
  "${PI3X_PROOF_MANIFEST_SHA256}" "${SOURCE_BUNDLE_MANIFEST}" \
  "${SOURCE_RECEIPT_SHA256}" "${SMOKE_SCOPE}" <<'PY'
import csv
import hashlib
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
metadata = pathlib.Path(sys.argv[2])
goal_b_image = pathlib.Path(sys.argv[3])
episode_id, model_sha, proof_sha, source_manifest, source_receipt_sha, scope = (
    sys.argv[4:]
)

def one_row(arm):
    with (root / arm / "metric.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 1 or rows[0]["episode"] != episode_id:
        raise SystemExit(f"{arm}: incomplete metric identity")
    return rows[0]

trace = one_row("trace_source")
native = one_row("native")
learned = one_row("learned_pi3x")
trace_sha = trace.get("leg1_trace_sha256")
if not trace_sha or any(
    row.get("leg1_trace_sha256") != trace_sha for row in (native, learned)
):
    raise SystemExit("Goal-A trace identity changed across arms")
if any(row["reached_A"] != trace["reached_A"] for row in (native, learned)):
    raise SystemExit("Goal-A outcome changed across arms")

summary = json.loads((root / "learned_pi3x" / "summary.json").read_text())
server = summary.get("learned_pi3x_relocalization_server") or {}
requests = int(float(
    learned.get("learned_pi3x_relocalization_request_count") or 0))
failures = int(float(
    learned.get("learned_pi3x_relocalization_runtime_failure_count") or 0))
accepts = int(float(
    learned.get("learned_pi3x_relocalization_accept_count") or 0))
takeovers = int(float(
    learned.get("revisit_adapter_takeover_plan_count") or 0))
goal_a_success = float(trace["reached_A"]) > 0.5
steps_b = int(float(learned["steps_B"]))
passed = bool(
    server.get("enabled") is True
    and server.get("model_sha256") == model_sha
    and server.get("proof_manifest_sha256") == proof_sha
    and server.get("certificate_components_consumed") is False
    and server.get("simulator_pose_or_depth_consumed") is False
    and goal_a_success and steps_b > 0 and requests > 0 and failures == 0
    and (accepts == 0 or takeovers > 0)
)
receipt = {
    "schema_version": "pi3x_learned_closed_loop_smoke_v2_20260817",
    "scope": scope,
    "passed": passed,
    "episode_id": episode_id,
    "metadata_sha256": hashlib.sha256(metadata.read_bytes()).hexdigest(),
    "goal_b_image_sha256": hashlib.sha256(goal_b_image.read_bytes()).hexdigest(),
    "goal_a_success": goal_a_success,
    "shared_goal_a_trace_sha256": trace_sha,
    "source_bundle": {
        "manifest": source_manifest,
        "manifest_sha256": source_receipt_sha,
    },
    "native": {
        "reached_b": float(native["reached_B"]) > 0.5,
        "steps_b": int(float(native["steps_B"])),
    },
    "learned_pi3x": {
        "reached_b": float(learned["reached_B"]) > 0.5,
        "steps_b": steps_b,
        "request_count": requests,
        "accept_count": accepts,
        "runtime_failure_count": failures,
        "takeover_plan_count": takeovers,
    },
    "server_contract": server,
}
with (root / "receipt.json").open("x", encoding="utf-8") as handle:
    json.dump(receipt, handle, indent=2, sort_keys=True)
    handle.write("\n")
print(json.dumps(receipt, indent=2, sort_keys=True))
if not passed:
    raise SystemExit("learned closed-loop transport smoke failed")
PY

sha256sum "${OUT_ROOT}/receipt.json" >"${OUT_ROOT}/receipt.json.sha256"
echo "[complete] ${OUT_ROOT}"
