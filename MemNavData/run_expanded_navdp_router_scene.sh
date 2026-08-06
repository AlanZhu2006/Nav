#!/usr/bin/env bash
# Run three paired arms for one scene from the frozen expanded benchmark:
# native NavDP, the original top-1 geometry router, and temporal top-K geometry.

set -euo pipefail
umask 0022
export GIT_OPTIONAL_LOCKS=0

ROOT=${ROOT:?set ROOT}
RUN_ROOT=${RUN_ROOT:?set RUN_ROOT}
MANIFEST=${MANIFEST:?set MANIFEST}
EXPECTED_MANIFEST_SHA=${EXPECTED_MANIFEST_SHA:?set EXPECTED_MANIFEST_SHA}
EXPECTED_COMMIT=${EXPECTED_COMMIT:?set EXPECTED_COMMIT}
SCENE_INDEX=${SCENE_INDEX:?set SCENE_INDEX}
HAB_PY=${HAB_PY:?set HAB_PY}
MEMNAV_PY=${MEMNAV_PY:?set MEMNAV_PY}
MAX_STEPS=${MAX_STEPS:-500}
EPISODE_LIMIT=${EPISODE_LIMIT:-0}
LEG1_MODE=${LEG1_MODE:-policy}
STOP_AFTER_LEG1=${STOP_AFTER_LEG1:-0}
WRITE_LEG1_TRACE=${WRITE_LEG1_TRACE:-0}
DETERMINISTIC_PLAN_SEEDS=${DETERMINISTIC_PLAN_SEEDS:-0}
NAVDP_GOAL_SWITCH_RESET=${NAVDP_GOAL_SWITCH_RESET:-carry}
SHARED_LEG1_ROOT=${SHARED_LEG1_ROOT:-}
RUN_NAVDP_NATIVE=${RUN_NAVDP_NATIVE:-1}
RUN_GEOMETRY_TOP1=${RUN_GEOMETRY_TOP1:-1}
RUN_GEOMETRY_ROUTER=${RUN_GEOMETRY_ROUTER:-1}
RETRIEVAL_CANDIDATE_MIN_GAP=${RETRIEVAL_CANDIDATE_MIN_GAP:-16}
GRAPH_SUBGOAL_SPACING_M=${GRAPH_SUBGOAL_SPACING_M:-0.0}
GRAPH_SUBGOAL_ARRIVAL_M=${GRAPH_SUBGOAL_ARRIVAL_M:-0.60}
TERMINAL_UTURN=${TERMINAL_UTURN:-off}
TERMINAL_VISUAL_REFINE=${TERMINAL_VISUAL_REFINE:-off}
ARRIVAL_SHADOW=${ARRIVAL_SHADOW:-off}
UNIT_TEST_MODULE=${UNIT_TEST_MODULE:-MemNavData.test_expanded_navdp_router_eval}
EXPECTED_HAB_REQUESTS_VERSION=${EXPECTED_HAB_REQUESTS_VERSION:-2.32.4}
EXPECTED_HAB_REQUESTS_INIT_BYTES=5057
EXPECTED_HAB_REQUESTS_INIT_SHA=1e507f1f386bcc6b5f0ff69a614c14875cd65cb67be7f6022f28adef9774573f
EXPECTED_HAB_REQUESTS_VERSION_BYTES=435
EXPECTED_HAB_REQUESTS_VERSION_SHA=${EXPECTED_HAB_REQUESTS_VERSION_SHA:-143abaf3563712f063743a7952aa65319dbcb934d894cfc989bd2c015f8da577}

# The frozen Habitat Python has requests only in pip's pure-Python vendor
# directory.  Scope that path to Habitat subprocesses so it cannot pollute the
# Python 3.10 policy environment.
HAB_SITE_PACKAGES=$("${HAB_PY}" -c \
  'import sysconfig; print(sysconfig.get_paths()["purelib"])')
HAB_REQUESTS_VENDOR=${HAB_REQUESTS_VENDOR:-${HAB_SITE_PACKAGES}/pip/_vendor}
HAB_PYTHONPATH=${HAB_REQUESTS_VENDOR}${PYTHONPATH:+:${PYTHONPATH}}
hab_python() {
  env PYTHONPATH="${HAB_PYTHONPATH}" "${HAB_PY}" "$@"
}

EVALUATOR=${EVALUATOR:-${ROOT}/MemNavData/eval_2leg_habitat.py}
VALIDATOR=${VALIDATOR:-${ROOT}/MemNavData/validate_expanded_navdp_router_eval.py}
MEMNAV_SERVER=${ROOT}/NavDP/baselines/memnav/memnav_server.py
NAVDP_SERVER=${ROOT}/NavDP/baselines/navdp/navdp_server.py
INTERNNAV_ROOT=${ROOT}/InternNav
LONGCLIP_ENTRY=${INTERNNAV_ROOT}/internnav/model/basemodel/LongCLIP/model/longclip.py
LINGBOT_REPO=${LINGBOT_REPO:-/scratch/lg154/Research/Nav/NavDP/baselines/memnav/lingbot-map}
LINGBOT_WEIGHTS=${LINGBOT_WEIGHTS:-${LINGBOT_REPO}/weights/lingbot-map-long.pt}
EXPECTED_LINGBOT_COMMIT=${EXPECTED_LINGBOT_COMMIT:-7ff6f3ed0913d4d326f8f13bbb429c4ffc0195c2}
MEMNAV_CKPT=${MEMNAV_CKPT:-/scratch/yz11502/Research/Nav-axis-uturn/.diagnostics/unseen_scene_eval_20260803/checkpoints/gatecurr600.memnav.ckpt}
NAVDP_CKPT=${NAVDP_CKPT:-/scratch/yz11502/Research/Nav-axis-uturn/.diagnostics/unseen_scene_eval_20260803/checkpoints/navdp_checkpoint.ckpt}
ASSET_ROOT_OVERRIDE=${ASSET_ROOT_OVERRIDE:-}
EPISODE_ROOT_OVERRIDE=${EPISODE_ROOT_OVERRIDE:-}
RUN_CONDITIONAL_ORACLES=${RUN_CONDITIONAL_ORACLES:-0}
RUN_CONDITIONAL_ORACLE_ANCHOR=${RUN_CONDITIONAL_ORACLE_ANCHOR:-${RUN_CONDITIONAL_ORACLES}}
RUN_CONDITIONAL_ORACLE_POINT=${RUN_CONDITIONAL_ORACLE_POINT:-${RUN_CONDITIONAL_ORACLES}}

TASK_FILES=(
  MemNavData/eval_2leg_habitat.py
  MemNavData/validate_expanded_navdp_router_eval.py
  MemNavData/summarize_expanded_navdp_router_eval.py
  MemNavData/test_expanded_navdp_router_eval.py
  MemNavData/test_router_candidates.py
  MemNavData/test_reverse_memory_graph.py
  MemNavData/test_policy_agent_graph.py
  MemNavData/terminal_uturn.py
  MemNavData/visual_yaw_refinement.py
  MemNavData/summarize_terminal_uturn.py
  MemNavData/test_terminal_uturn.py
  MemNavData/arrival_shadow.py
  MemNavData/test_arrival_shadow.py
  MemNavData/summarize_arrival_shadow.py
  MemNavData/test_summarize_arrival_shadow.py
  MemNavData/deterministic_eval_protocol.py
  MemNavData/navdp_goal_switch.py
  MemNavData/test_navdp_goal_switch.py
  MemNavData/test_deterministic_eval_protocol.py
  MemNavData/test_navdp_memory_replay.py
  MemNavData/conditional_c_protocol.py
  MemNavData/test_conditional_c_protocol.py
  MemNavData/summarize_conditional_c_eval.py
  MemNavData/test_summarize_conditional_c_eval.py
  MemNavData/run_expanded_navdp_router_scene.sh
  MemNavData/slurm_expanded_navdp_router_eval.sbatch
  MemNavData/expanded_navdp_router_eval_20260805.json
  NavDP/baselines/memnav/memnav_server.py
  NavDP/baselines/memnav/policy_agent.py
  NavDP/baselines/memnav/pose_alignment.py
  NavDP/baselines/memnav/router_candidates.py
  NavDP/baselines/memnav/reverse_memory_graph.py
  InternNav/internnav/model/basemodel/memnav/memnav_policy.py
  NavDP/baselines/navdp/navdp_server.py
  NavDP/baselines/navdp/deterministic_seed.py
  NavDP/baselines/navdp/policy_agent.py
  NavDP/baselines/navdp/policy_network.py
)
[[ "${MAX_STEPS}" =~ ^[1-9][0-9]*$ ]] || {
  echo "ABORT: MAX_STEPS must be a positive integer" >&2
  exit 1
}
[[ "${EPISODE_LIMIT}" =~ ^[0-9]+$ ]] || {
  echo "ABORT: EPISODE_LIMIT must be a non-negative integer" >&2
  exit 1
}
[[ "${UNIT_TEST_MODULE}" =~ ^[A-Za-z_][A-Za-z0-9_.]*$ ]] || {
  echo "ABORT: invalid UNIT_TEST_MODULE=${UNIT_TEST_MODULE}" >&2
  exit 1
}
for flag in RUN_CONDITIONAL_ORACLES RUN_CONDITIONAL_ORACLE_ANCHOR \
            RUN_CONDITIONAL_ORACLE_POINT; do
  [[ "${!flag}" =~ ^[01]$ ]] || {
    echo "ABORT: ${flag} must be 0 or 1" >&2; exit 1; }
done
for flag in RUN_NAVDP_NATIVE RUN_GEOMETRY_TOP1 RUN_GEOMETRY_ROUTER; do
  [[ "${!flag}" =~ ^[01]$ ]] || {
    echo "ABORT: ${flag} must be 0 or 1" >&2; exit 1; }
done
for flag in STOP_AFTER_LEG1 WRITE_LEG1_TRACE DETERMINISTIC_PLAN_SEEDS; do
  [[ "${!flag}" =~ ^[01]$ ]] || {
    echo "ABORT: ${flag} must be 0 or 1" >&2; exit 1; }
done
[[ "${NAVDP_GOAL_SWITCH_RESET}" =~ ^(carry|before_b|every_goal)$ ]] || {
  echo "ABORT: invalid NAVDP_GOAL_SWITCH_RESET=${NAVDP_GOAL_SWITCH_RESET}" >&2
  exit 1
}
if [[ "${NAVDP_GOAL_SWITCH_RESET}" != carry \
      && "${DETERMINISTIC_PLAN_SEEDS}" -ne 1 ]]; then
  echo "ABORT: goal-switch reset ablation requires deterministic plan seeds" >&2
  exit 1
fi
[[ "${LEG1_MODE}" =~ ^(policy|replay|shared_trace)$ ]] || {
  echo "ABORT: invalid LEG1_MODE=${LEG1_MODE}" >&2; exit 1; }
[[ "${TERMINAL_UTURN}" =~ ^(off|oracle|lingbot_yaw|lingbot_local|lingbot)$ ]] || {
  echo "ABORT: invalid TERMINAL_UTURN=${TERMINAL_UTURN}" >&2; exit 1; }
[[ "${TERMINAL_VISUAL_REFINE}" =~ ^(off|verify|refine)$ ]] || {
  echo "ABORT: invalid TERMINAL_VISUAL_REFINE=${TERMINAL_VISUAL_REFINE}" >&2
  exit 1
}
[[ "${ARRIVAL_SHADOW}" =~ ^(off|diagnostic)$ ]] || {
  echo "ABORT: invalid ARRIVAL_SHADOW=${ARRIVAL_SHADOW}" >&2; exit 1; }
if [[ "${TERMINAL_UTURN}" == off && "${TERMINAL_VISUAL_REFINE}" != off ]]; then
  echo "ABORT: terminal visual refinement requires a terminal U-turn" >&2
  exit 1
fi
if [[ "${LEG1_MODE}" == shared_trace ]]; then
  [[ -n "${SHARED_LEG1_ROOT}" ]] || {
    echo "ABORT: shared_trace requires SHARED_LEG1_ROOT" >&2; exit 1; }
else
  [[ -z "${SHARED_LEG1_ROOT}" ]] || {
    echo "ABORT: SHARED_LEG1_ROOT requires shared_trace" >&2; exit 1; }
fi
if [[ "${STOP_AFTER_LEG1}" -eq 1 || "${WRITE_LEG1_TRACE}" -eq 1 ]]; then
  [[ "${LEG1_MODE}" == policy ]] || {
    echo "ABORT: trace writing/Goal-A-only requires policy LEG1_MODE" >&2
    exit 1
  }
fi
(( RUN_NAVDP_NATIVE + RUN_GEOMETRY_TOP1 + RUN_GEOMETRY_ROUTER > 0 )) || {
  echo "ABORT: at least one evaluation arm must be enabled" >&2; exit 1; }
[[ "${RETRIEVAL_CANDIDATE_MIN_GAP}" =~ ^[1-9][0-9]*$ ]] || {
  echo "ABORT: RETRIEVAL_CANDIDATE_MIN_GAP must be positive" >&2; exit 1; }
"${HAB_PY}" - "${GRAPH_SUBGOAL_SPACING_M}" \
  "${GRAPH_SUBGOAL_ARRIVAL_M}" <<'PY'
import math, sys
spacing, arrival = map(float, sys.argv[1:])
if not math.isfinite(spacing) or spacing < 0:
    raise SystemExit("GRAPH_SUBGOAL_SPACING_M must be finite and non-negative")
if not math.isfinite(arrival) or arrival <= 0:
    raise SystemExit("GRAPH_SUBGOAL_ARRIVAL_M must be finite and positive")
PY
if (( RUN_CONDITIONAL_ORACLE_ANCHOR + RUN_CONDITIONAL_ORACLE_POINT > 0 )); then
  [[ "$(basename "${EVALUATOR}")" == "eval_conditional_c_habitat.py" ]] || {
    echo "ABORT: conditional oracle arms require eval_conditional_c_habitat.py" >&2
    exit 1
  }
fi
UNIT_TEST_PATH=${ROOT}/${UNIT_TEST_MODULE//./\/}.py
for dynamic_path in "${EVALUATOR}" "${VALIDATOR}" "${MANIFEST}" \
                    "${UNIT_TEST_PATH}"; do
  dynamic_relative=$(realpath --relative-to="${ROOT}" "${dynamic_path}")
  [[ "${dynamic_relative}" != ../* && "${dynamic_relative}" != ".." ]] || {
    echo "ABORT: benchmark source must live inside ${ROOT}: ${dynamic_path}" >&2
    exit 1
  }
  TASK_FILES+=("${dynamic_relative}")
done

actual_commit=$(git -C "${ROOT}" rev-parse HEAD)
[[ "${actual_commit}" == "${EXPECTED_COMMIT}" ]] || {
  echo "ABORT: code commit ${actual_commit} != ${EXPECTED_COMMIT}" >&2
  exit 1
}
git -C "${ROOT}" ls-files --error-unmatch -- "${TASK_FILES[@]}" \
  >/dev/null || {
    echo "ABORT: a benchmark task input is not tracked by the commit" >&2
    exit 1
  }
[[ -z "$(git -C "${ROOT}" status --porcelain --untracked-files=all)" ]] || {
  echo "ABORT: benchmark worktree is not completely clean" >&2
  git -C "${ROOT}" status --short >&2
  exit 1
}
git -C "${ROOT}" diff --quiet -- "${TASK_FILES[@]}" || {
  echo "ABORT: benchmark task files differ from the checked-out commit" >&2
  exit 1
}
git -C "${ROOT}" diff --cached --quiet -- "${TASK_FILES[@]}" || {
  echo "ABORT: staged benchmark task files differ from the checked-out commit" >&2
  exit 1
}

for required in "${HAB_PY}" "${MEMNAV_PY}" "${EVALUATOR}" "${VALIDATOR}" \
                "${MEMNAV_SERVER}" "${NAVDP_SERVER}" "${LINGBOT_WEIGHTS}" \
                "${MEMNAV_CKPT}" "${NAVDP_CKPT}" "${MANIFEST}" \
                "${UNIT_TEST_PATH}" "${LONGCLIP_ENTRY}" \
                "${HAB_REQUESTS_VENDOR}/requests/__init__.py" \
                "${HAB_REQUESTS_VENDOR}/requests/__version__.py"; do
  test -r "${required}" || { echo "ABORT: missing dependency ${required}" >&2; exit 1; }
done
actual_lingbot_commit=$(git -c safe.directory="${LINGBOT_REPO}" \
  -C "${LINGBOT_REPO}" rev-parse HEAD)
[[ "${actual_lingbot_commit}" == "${EXPECTED_LINGBOT_COMMIT}" ]] || {
  echo "ABORT: LingBot commit ${actual_lingbot_commit} != ${EXPECTED_LINGBOT_COMMIT}" >&2
  exit 1
}
git -c safe.directory="${LINGBOT_REPO}" -C "${LINGBOT_REPO}" \
  diff --quiet || {
    echo "ABORT: LingBot tracked files differ from its pinned commit" >&2
    exit 1
  }
git -c safe.directory="${LINGBOT_REPO}" -C "${LINGBOT_REPO}" \
  diff --cached --quiet || {
    echo "ABORT: LingBot has staged changes" >&2
    exit 1
  }
REQUESTS_INIT=${HAB_REQUESTS_VENDOR}/requests/__init__.py
REQUESTS_VERSION=${HAB_REQUESTS_VENDOR}/requests/__version__.py
[[ "$(stat -c '%s' "${REQUESTS_INIT}")" == "${EXPECTED_HAB_REQUESTS_INIT_BYTES}" ]] || {
  echo "ABORT: vendored requests size mismatch" >&2
  exit 1
}
[[ "$(sha256sum "${REQUESTS_INIT}" | awk '{print $1}')" == \
    "${EXPECTED_HAB_REQUESTS_INIT_SHA}" ]] || {
  echo "ABORT: vendored requests SHA256 mismatch" >&2
  exit 1
}
[[ "$(stat -c '%s' "${REQUESTS_VERSION}")" == \
    "${EXPECTED_HAB_REQUESTS_VERSION_BYTES}" ]] || {
  echo "ABORT: vendored requests version-file size mismatch" >&2
  exit 1
}
[[ "$(sha256sum "${REQUESTS_VERSION}" | awk '{print $1}')" == \
    "${EXPECTED_HAB_REQUESTS_VERSION_SHA}" ]] || {
  echo "ABORT: vendored requests version-file SHA256 mismatch" >&2
  exit 1
}

mkdir -p "${RUN_ROOT}/preflight" "${RUN_ROOT}/scenes"
VALIDATOR_ARGS=(
  --manifest "${MANIFEST}"
  --expected-manifest-sha "${EXPECTED_MANIFEST_SHA}"
  --scene-index "${SCENE_INDEX}"
  --gatecurr-checkpoint "${MEMNAV_CKPT}"
  --navdp-checkpoint "${NAVDP_CKPT}"
  --lingbot-weights "${LINGBOT_WEIGHTS}"
)
[[ -z "${ASSET_ROOT_OVERRIDE}" ]] || \
  VALIDATOR_ARGS+=(--asset-root "${ASSET_ROOT_OVERRIDE}")
[[ -z "${EPISODE_ROOT_OVERRIDE}" ]] || \
  VALIDATOR_ARGS+=(--episode-root "${EPISODE_ROOT_OVERRIDE}")
hab_python "${VALIDATOR}" "${VALIDATOR_ARGS[@]}" \
  > "${RUN_ROOT}/preflight/scene_$(printf '%02d' "${SCENE_INDEX}").json"

scene=$(hab_python - "${MANIFEST}" "${SCENE_INDEX}" <<'PY'
import json, sys
print(json.load(open(sys.argv[1]))["selection"]["selected_scenes"][int(sys.argv[2])])
PY
)
mapfile -t EPISODE_IDS < <(hab_python - "${MANIFEST}" "${scene}" <<'PY'
import json, sys
manifest = json.load(open(sys.argv[1]))
print(*(row["episode"] for row in manifest["episodes"][sys.argv[2]]), sep="\n")
PY
)
if (( EPISODE_LIMIT > 0 && EPISODE_LIMIT < ${#EPISODE_IDS[@]} )); then
  EPISODE_IDS=("${EPISODE_IDS[@]:0:EPISODE_LIMIT}")
fi
(( ${#EPISODE_IDS[@]} > 0 )) || {
  echo "ABORT: no episodes remain after EPISODE_LIMIT" >&2; exit 1; }
episode_csv=$(IFS=,; echo "${EPISODE_IDS[*]}")

SCENE_ROOT=${RUN_ROOT}/scenes/$(printf '%02d' "${SCENE_INDEX}")_${scene}
if [[ -e "${SCENE_ROOT}" ]]; then
  echo "ABORT: scene output already exists: ${SCENE_ROOT}" >&2
  exit 1
fi
mkdir -p "${SCENE_ROOT}/logs" "${SCENE_ROOT}/buffer"
exec > >(tee "${SCENE_ROOT}/run.log") 2>&1

hab_python -m py_compile \
  "${EVALUATOR}" \
  "${VALIDATOR}" \
  "${ROOT}/MemNavData/terminal_uturn.py" \
  "${ROOT}/MemNavData/arrival_shadow.py" \
  "${ROOT}/MemNavData/summarize_arrival_shadow.py" \
  "${ROOT}/MemNavData/visual_yaw_refinement.py" \
  "${ROOT}/MemNavData/summarize_terminal_uturn.py"
"${MEMNAV_PY}" -m py_compile \
  "${ROOT}/MemNavData/deterministic_eval_protocol.py" \
  "${ROOT}/MemNavData/navdp_goal_switch.py" \
  "${MEMNAV_SERVER}" \
  "${ROOT}/NavDP/baselines/memnav/policy_agent.py" \
  "${ROOT}/NavDP/baselines/memnav/router_candidates.py" \
  "${ROOT}/NavDP/baselines/memnav/reverse_memory_graph.py" \
  "${NAVDP_SERVER}" \
  "${ROOT}/NavDP/baselines/navdp/deterministic_seed.py"
(
  cd "${ROOT}"
  hab_python -m unittest \
    "${UNIT_TEST_MODULE}" \
    MemNavData.test_router_candidates \
    MemNavData.test_reverse_memory_graph \
    MemNavData.test_terminal_uturn \
    MemNavData.test_arrival_shadow \
    MemNavData.test_summarize_arrival_shadow \
    MemNavData.test_conditional_c_protocol \
    MemNavData.test_navdp_goal_switch \
    MemNavData.test_summarize_conditional_c_eval -v
)
(
  cd "${ROOT}"
  "${MEMNAV_PY}" -m unittest \
    MemNavData.test_policy_agent_graph \
    MemNavData.test_deterministic_eval_protocol \
    MemNavData.test_navdp_memory_replay -v
)
hab_python -c \
  'import habitat_sim,numpy,pandas,pyarrow,PIL,requests,scipy,quaternion,sys; assert requests.__version__ == sys.argv[1]; print("Habitat dependencies OK", habitat_sim.__version__, "requests", requests.__version__)' \
  "${EXPECTED_HAB_REQUESTS_VERSION}"
"${MEMNAV_PY}" -c \
  'import torch,torchvision,transformers,diffusers,cv2,flask,imageio; assert torch.cuda.is_available(); print("Policy dependencies OK", torch.__version__)'

port_key=$(( (${SLURM_JOB_ID:-1000} + SCENE_INDEX * 37) % 15000 ))
MEMNAV_PORT=${MEMNAV_PORT:-$((20000 + port_key * 2))}
NAVDP_PORT=${NAVDP_PORT:-$((MEMNAV_PORT + 1))}
for port in "${MEMNAV_PORT}" "${NAVDP_PORT}"; do
  if ss -ltn | awk '{print $4}' | grep -Eq "(^|:)${port}$"; then
    echo "ABORT: port ${port} is already in use" >&2
    exit 1
  fi
done

RUNTIME_ROOT=${SLURM_TMPDIR:-/tmp}/memnav_expanded_${SLURM_JOB_ID:-local}_${SCENE_INDEX}
mkdir -p "${RUNTIME_ROOT}/memnav" "${RUNTIME_ROOT}/navdp"
MEMNAV_PID=
NAVDP_PID=
cleanup() {
  for pid in "${NAVDP_PID}" "${MEMNAV_PID}"; do
    if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
      kill "${pid}" 2>/dev/null || true
      wait "${pid}" 2>/dev/null || true
    fi
  done
}
trap cleanup EXIT INT TERM

(
  cd "${RUNTIME_ROOT}/memnav"
  exec env \
    PYTHONUNBUFFERED=1 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    PYTHONPATH="${INTERNNAV_ROOT}/src/diffusion-policy:${PYTHONPATH:-}" \
    LINGBOT_REPO="${LINGBOT_REPO}" \
    LINGBOT_WEIGHTS="${LINGBOT_WEIGHTS}" \
    MEMNAV_WINDOW=32 \
    MEMNAV_NUM_SCALE=8 \
    MEMNAV_MAX_FRAME_NUM=2048 \
    MEMNAV_GROUND_SCALE_MAX=6.0 \
    MEMNAV_GATE_FUSION=complementary \
    MEMNAV_AUX_POSE_CALIBRATION=empirical \
    MEMNAV_COLLISION_SELECT=1 \
    MEMNAV_REPORT_TO=none \
    "${MEMNAV_PY}" -u "${MEMNAV_SERVER}" \
      --port "${MEMNAV_PORT}" \
      --checkpoint "${MEMNAV_CKPT}" \
      --internnav_root "${INTERNNAV_ROOT}" \
      --num_samples 16 \
      --exclude_recent 32 \
      --retrieval raw \
      --retrieval_candidate_top_k 32 \
      --retrieval_candidate_min_gap "${RETRIEVAL_CANDIDATE_MIN_GAP}" \
      --graph_subgoal_spacing_m "${GRAPH_SUBGOAL_SPACING_M}" \
      --graph_subgoal_arrival_m "${GRAPH_SUBGOAL_ARRIVAL_M}" \
      --flow_gate auto \
      --buffer_root "${SCENE_ROOT}/buffer"
) > "${SCENE_ROOT}/logs/server_memnav.log" 2>&1 &
MEMNAV_PID=$!

(
  cd "${RUNTIME_ROOT}/navdp"
  exec env NAVDP_DISABLE_VIDEO=1 PYTHONUNBUFFERED=1 \
    "${MEMNAV_PY}" -u "${NAVDP_SERVER}" \
      --port "${NAVDP_PORT}" --checkpoint "${NAVDP_CKPT}"
) > "${SCENE_ROOT}/logs/server_navdp.log" 2>&1 &
NAVDP_PID=$!

for spec in \
    "memnav:${MEMNAV_PID}:${MEMNAV_PORT}:${SCENE_ROOT}/logs/server_memnav.log" \
    "navdp:${NAVDP_PID}:${NAVDP_PORT}:${SCENE_ROOT}/logs/server_navdp.log"; do
  IFS=: read -r label pid port log <<<"${spec}"
  ready=0
  for _ in $(seq 1 240); do
    if ! kill -0 "${pid}" 2>/dev/null; then
      echo "ABORT: ${label} server exited during startup" >&2
      tail -n 120 "${log}" >&2
      exit 1
    fi
    if ss -ltn | awk '{print $4}' | grep -Eq "(^|:)${port}$"; then
      ready=1
      break
    fi
    sleep 2
  done
  [[ "${ready}" -eq 1 ]] || {
    echo "ABORT: ${label} server did not bind port ${port}" >&2
    tail -n 120 "${log}" >&2
    exit 1
  }
done

MANIFEST_ASSET_ROOT=$(hab_python - "${MANIFEST}" <<'PY'
import json, sys
print(json.load(open(sys.argv[1]))["paths"]["asset_root"])
PY
)
MANIFEST_EPISODE_ROOT=$(hab_python - "${MANIFEST}" "${scene}" <<'PY'
import json, sys
manifest = json.load(open(sys.argv[1]))
if "episode_root" in manifest["paths"]:
    key = "episode_root"
else:
    key = ("legacy_anchor_episode_root"
           if sys.argv[2] in manifest["selection"]["anchor_scenes"]
           else "expanded_episode_root")
print(manifest["paths"][key])
PY
)
ASSET_ROOT=${ASSET_ROOT_OVERRIDE:-${MANIFEST_ASSET_ROOT}}
EPISODE_ROOT=${EPISODE_ROOT_OVERRIDE:-${MANIFEST_EPISODE_ROOT}}
SCENE_FILE=${ASSET_ROOT}/${scene}/${scene}.glb
EPISODE_SCENE_ROOT=${EPISODE_ROOT}/${scene}
test -r "${SCENE_FILE}" || {
  echo "ABORT: missing scene asset ${SCENE_FILE}" >&2; exit 1; }
test -d "${EPISODE_SCENE_ROOT}" || {
  echo "ABORT: missing episode scene root ${EPISODE_SCENE_ROOT}" >&2; exit 1; }
COMMON_ARGS=(
  --episode_root "${EPISODE_SCENE_ROOT}"
  --scene "${SCENE_FILE}"
  --host 127.0.0.1
  --leg1_mode "${LEG1_MODE}"
  --success_dist 1.0
  --max_steps "${MAX_STEPS}"
  --exec_horizon 8
  --trajectory_selector server
  --navdp_goal_switch_reset "${NAVDP_GOAL_SWITCH_RESET}"
  --leg1_goal_source own
  --seed 20260803
  --terminal_uturn "${TERMINAL_UTURN}"
  --terminal_visual_refine "${TERMINAL_VISUAL_REFINE}"
  --arrival_shadow "${ARRIVAL_SHADOW}"
  --episode_ids "${episode_csv}"
)
if [[ "${STOP_AFTER_LEG1}" -eq 1 ]]; then
  COMMON_ARGS+=(--stop_after_leg1)
fi
if [[ "${WRITE_LEG1_TRACE}" -eq 1 ]]; then
  COMMON_ARGS+=(--write_leg1_trace)
fi
if [[ "${DETERMINISTIC_PLAN_SEEDS}" -eq 1 ]]; then
  COMMON_ARGS+=(--deterministic_plan_seeds)
fi
if [[ "${LEG1_MODE}" == shared_trace ]]; then
  SHARED_SCENE_ROOT=${SHARED_LEG1_ROOT}/scenes/$(printf '%02d' "${SCENE_INDEX}")_${scene}/geometry_router
  test -d "${SHARED_SCENE_ROOT}" || {
    echo "ABORT: shared Goal-A trace root missing: ${SHARED_SCENE_ROOT}" >&2
    exit 1
  }
  COMMON_ARGS+=(--shared_leg1_trace_root "${SHARED_SCENE_ROOT}")
fi

ARMS=()
if [[ "${RUN_NAVDP_NATIVE}" -eq 1 ]]; then
  echo "[eval] scene=${scene} arm=navdp_native episodes=${episode_csv}"
  mkdir -p "${SCENE_ROOT}/navdp_native"
  (
    cd "${RUNTIME_ROOT}"
    hab_python -u "${EVALUATOR}" \
      "${COMMON_ARGS[@]}" \
      --port "${NAVDP_PORT}" \
      --out "${SCENE_ROOT}/navdp_native" \
      --server_backend navdp
  ) > "${SCENE_ROOT}/logs/eval_navdp_native.log" 2>&1
  ARMS+=(navdp_native)
fi

run_geometry_arm() {
  local arm=$1
  local verify_top_k=$2
  echo "[eval] scene=${scene} arm=${arm} verify_top_k=${verify_top_k} episodes=${episode_csv}"
  mkdir -p "${SCENE_ROOT}/${arm}"
  (
    cd "${RUNTIME_ROOT}"
    hab_python -u "${EVALUATOR}" \
      "${COMMON_ARGS[@]}" \
      --port "${MEMNAV_PORT}" \
      --novel_port "${NAVDP_PORT}" \
      --out "${SCENE_ROOT}/${arm}" \
      --server_backend hybrid_pose \
      --hybrid_route memory_geometry \
      --router_visual_floor 0.88 \
      --router_min_matches 20 \
      --router_min_inliers 12 \
      --router_min_inlier_ratio 0.50 \
      --router_confirm_plans 2 \
      --router_verify_top_k "${verify_top_k}"
  ) > "${SCENE_ROOT}/logs/eval_${arm}.log" 2>&1
}

# Run top-1 first so its state cannot inherit the top-K accepted anchor.  Each
# evaluator calls the audited reset endpoint for every episode, resetting the
# policy RNG, streaming memory, and router latch to the same episode seed.
if [[ "${RUN_GEOMETRY_TOP1}" -eq 1 ]]; then
  run_geometry_arm geometry_top1 1
  ARMS+=(geometry_top1)
fi
if [[ "${RUN_GEOMETRY_ROUTER}" -eq 1 ]]; then
  run_geometry_arm geometry_router 8
  ARMS+=(geometry_router)
fi
run_conditional_oracle_arm() {
  local arm=$1
  local mode=$2
  echo "[eval] scene=${scene} arm=${arm} conditional_mode=${mode} episodes=${episode_csv}"
  mkdir -p "${SCENE_ROOT}/${arm}"
  (
    cd "${RUNTIME_ROOT}"
    hab_python -u "${EVALUATOR}" \
      --conditional_c_mode "${mode}" \
      "${COMMON_ARGS[@]}" \
      --port "${MEMNAV_PORT}" \
      --novel_port "${NAVDP_PORT}" \
      --out "${SCENE_ROOT}/${arm}" \
      --server_backend hybrid_pose \
      --hybrid_route memory_geometry \
      --router_visual_floor 0.88 \
      --router_min_matches 20 \
      --router_min_inliers 12 \
      --router_min_inlier_ratio 0.50 \
      --router_confirm_plans 2 \
      --router_verify_top_k 8
  ) > "${SCENE_ROOT}/logs/eval_${arm}.log" 2>&1
}

if [[ "${RUN_CONDITIONAL_ORACLE_ANCHOR}" -eq 1 ]]; then
  run_conditional_oracle_arm oracle_anchor oracle_anchor
  ARMS+=(oracle_anchor)
fi
if [[ "${RUN_CONDITIONAL_ORACLE_POINT}" -eq 1 ]]; then
  run_conditional_oracle_arm oracle_point oracle_point
  ARMS+=(oracle_point)
fi

for arm in "${ARMS[@]}"; do
  test -s "${SCENE_ROOT}/${arm}/metric.csv" || {
    echo "ABORT: ${arm} did not produce metric.csv" >&2; exit 1; }
  test -s "${SCENE_ROOT}/${arm}/summary.json" || {
    echo "ABORT: ${arm} did not produce summary.json" >&2; exit 1; }
  count=$(($(wc -l < "${SCENE_ROOT}/${arm}/metric.csv") - 1))
  [[ "${count}" -eq "${#EPISODE_IDS[@]}" ]] || {
    echo "ABORT: ${arm} produced ${count} rows, expected ${#EPISODE_IDS[@]}" >&2
    exit 1
  }
done

echo "[complete] scene=${scene} output=${SCENE_ROOT}"
