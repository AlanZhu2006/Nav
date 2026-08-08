#!/usr/bin/env bash
# One frozen 20-scene task: all three Novel-A arms share one NavDP process.

set -euo pipefail
umask 0022
export GIT_OPTIONAL_LOCKS=0

ROOT=${ROOT:?set ROOT}
RUN_ROOT=${RUN_ROOT:?set RUN_ROOT}
EXPECTED_COMMIT=${EXPECTED_COMMIT:?set EXPECTED_COMMIT}
SCENE_INDEX=${SCENE_INDEX:?set SCENE_INDEX}
HAB_PY=${HAB_PY:?set HAB_PY}
POLICY_PY=${POLICY_PY:?set POLICY_PY}
MANIFEST=${MANIFEST:-${ROOT}/MemNavData/expanded_navdp_router_eval_20260805.json}
PROTOCOL=${PROTOCOL:-${ROOT}/MemNavData/novel_a_bearing_gate_protocol_20260808.json}
INPUT_OVERLAY=${INPUT_OVERLAY:-${ROOT}/MemNavData/novel_a_bearing_inputs_20260808.json}
EXPECTED_MANIFEST_SHA=${EXPECTED_MANIFEST_SHA:-ba8f72cb504768c801e6c9c386436ccdc66dea07a5e5fac2d7b4248738946a61}
EXPECTED_PROTOCOL_SHA=${EXPECTED_PROTOCOL_SHA:-4006f9a62b8376c6a55a6394f0bce026739d2c7c968712b542b15b7f1158b6c8}
EXPECTED_INPUT_OVERLAY_SHA=${EXPECTED_INPUT_OVERLAY_SHA:-401d43723a37465fa00778fd21b27eecbe46cf114abb074a3582b524451ce901}
NAVDP_CKPT=${NAVDP_CKPT:-${ROOT}/.diagnostics/unseen_scene_eval_20260803/checkpoints/navdp_checkpoint.ckpt}
ASSET_ROOT_OVERRIDE=${ASSET_ROOT_OVERRIDE:-}
EPISODE_ROOT_OVERRIDE=${EPISODE_ROOT_OVERRIDE:-}
SMOKE=${SMOKE:-0}
MAX_STEPS=${MAX_STEPS:-500}

[[ "${SCENE_INDEX}" =~ ^([0-9]|1[0-9])$ ]] || {
  echo "ABORT: scene index must be in [0,19]" >&2; exit 1; }
[[ "${SMOKE}" =~ ^[01]$ ]] || {
  echo "ABORT: SMOKE must be 0 or 1" >&2; exit 1; }
if [[ "${SMOKE}" -eq 0 ]]; then
  [[ "${MAX_STEPS}" -eq 500 ]] || {
    echo "ABORT: formal run requires MAX_STEPS=500" >&2; exit 1; }
fi

EVALUATOR=${ROOT}/MemNavData/eval_novel_a_bearing_gate_habitat.py
VALIDATOR=${ROOT}/MemNavData/validate_novel_a_bearing_gate.py
NAVDP_SERVER=${ROOT}/NavDP/baselines/navdp/navdp_server.py
TASK_FILES=(
  MemNavData/NOVEL_A_BEARING_GATE_PROTOCOL_20260808.md
  MemNavData/novel_a_bearing_gate_protocol_20260808.json
  MemNavData/novel_a_bearing_inputs_20260808.json
  MemNavData/novel_a_bearing_gate.py
  MemNavData/eval_novel_a_bearing_gate_habitat.py
  MemNavData/validate_novel_a_bearing_gate.py
  MemNavData/summarize_novel_a_bearing_gate.py
  MemNavData/test_novel_a_bearing_gate.py
  MemNavData/eval_2leg_habitat.py
  MemNavData/generate_twoleg.py
  MemNavData/deterministic_eval_protocol.py
  MemNavData/validate_expanded_navdp_router_eval.py
  MemNavData/terminal_uturn.py
  MemNavData/visual_yaw_refinement.py
  MemNavData/arrival_shadow.py
  MemNavData/navdp_goal_switch.py
  MemNavData/run_novel_a_bearing_scene.sh
  MemNavData/slurm_novel_a_bearing_gate.sbatch
  MemNavData/expanded_navdp_router_eval_20260805.json
  NavDP/baselines/navdp/navdp_server.py
  NavDP/baselines/navdp/deterministic_seed.py
  NavDP/baselines/navdp/policy_agent.py
  NavDP/baselines/navdp/policy_network.py
)

actual_commit=$(git -C "${ROOT}" rev-parse HEAD)
[[ "${actual_commit}" == "${EXPECTED_COMMIT}" ]] || {
  echo "ABORT: commit ${actual_commit} != ${EXPECTED_COMMIT}" >&2; exit 1; }
git -C "${ROOT}" ls-files --error-unmatch -- "${TASK_FILES[@]}" >/dev/null || {
  echo "ABORT: a task input is not tracked" >&2; exit 1; }
[[ -z "$(git -C "${ROOT}" status --porcelain --untracked-files=all)" ]] || {
  echo "ABORT: benchmark worktree is not clean" >&2
  git -C "${ROOT}" status --short >&2
  exit 1
}
for required in "${HAB_PY}" "${POLICY_PY}" "${MANIFEST}" "${PROTOCOL}" \
                "${INPUT_OVERLAY}" "${NAVDP_CKPT}" "${EVALUATOR}" \
                "${VALIDATOR}" "${NAVDP_SERVER}"; do
  test -r "${required}" || {
    echo "ABORT: missing dependency ${required}" >&2; exit 1; }
done
[[ "$(sha256sum "${MANIFEST}" | awk '{print $1}')" == \
    "${EXPECTED_MANIFEST_SHA}" ]] || {
  echo "ABORT: manifest SHA mismatch" >&2; exit 1; }
[[ "$(sha256sum "${PROTOCOL}" | awk '{print $1}')" == \
    "${EXPECTED_PROTOCOL_SHA}" ]] || {
  echo "ABORT: protocol SHA mismatch" >&2; exit 1; }
[[ "$(sha256sum "${INPUT_OVERLAY}" | awk '{print $1}')" == \
    "${EXPECTED_INPUT_OVERLAY_SHA}" ]] || {
  echo "ABORT: input-overlay SHA mismatch" >&2; exit 1; }

# Habitat's frozen environment obtains requests from pip's vendored copy.
HAB_SITE_PACKAGES=$("${HAB_PY}" -c \
  'import sysconfig; print(sysconfig.get_paths()["purelib"])')
HAB_REQUESTS_VENDOR=${HAB_REQUESTS_VENDOR:-${HAB_SITE_PACKAGES}/pip/_vendor}
HAB_PYTHONPATH=${HAB_REQUESTS_VENDOR}${PYTHONPATH:+:${PYTHONPATH}}
hab_python() {
  env PYTHONPATH="${HAB_PYTHONPATH}" "${HAB_PY}" "$@"
}

mkdir -p "${RUN_ROOT}/preflight" "${RUN_ROOT}/scenes"
scene=$(hab_python - "${MANIFEST}" "${SCENE_INDEX}" <<'PY'
import json, sys
print(json.load(open(sys.argv[1]))["selection"]["selected_scenes"][int(sys.argv[2])])
PY
)
if [[ -n "${EPISODE_ROOT_OVERRIDE}" ]]; then
  episode_root=${EPISODE_ROOT_OVERRIDE}
else
  episode_root=$(hab_python - "${MANIFEST}" "${scene}" <<'PY'
import json, sys
m=json.load(open(sys.argv[1])); s=sys.argv[2]
key="legacy_anchor_episode_root" if s in m["selection"]["anchor_scenes"] else "expanded_episode_root"
print(m["paths"][key])
PY
)
fi
if [[ -n "${ASSET_ROOT_OVERRIDE}" ]]; then
  asset_root=${ASSET_ROOT_OVERRIDE}
else
  asset_root=$(hab_python - "${MANIFEST}" <<'PY'
import json, sys
print(json.load(open(sys.argv[1]))["paths"]["asset_root"])
PY
)
fi
scene_file=${asset_root}/${scene}/${scene}.glb
episode_scene_root=${episode_root}/${scene}
mapfile -t episode_ids < <(hab_python - "${MANIFEST}" "${scene}" <<'PY'
import json, sys
m=json.load(open(sys.argv[1])); print(*(x["episode"] for x in m["episodes"][sys.argv[2]]), sep="\n")
PY
)
episode_csv=$(IFS=,; echo "${episode_ids[*]}")

validator_args=(
  --manifest "${MANIFEST}"
  --expected-manifest-sha "${EXPECTED_MANIFEST_SHA}"
  --protocol "${PROTOCOL}"
  --expected-protocol-sha "${EXPECTED_PROTOCOL_SHA}"
  --input-overlay "${INPUT_OVERLAY}"
  --expected-input-overlay-sha "${EXPECTED_INPUT_OVERLAY_SHA}"
  --scene-index "${SCENE_INDEX}"
  --navdp-checkpoint "${NAVDP_CKPT}"
  --asset-root "${asset_root}"
  --episode-root "${episode_root}"
)
hab_python "${VALIDATOR}" "${validator_args[@]}" \
  > "${RUN_ROOT}/preflight/scene_$(printf '%02d' "${SCENE_INDEX}").json"

scene_root=${RUN_ROOT}/scenes/$(printf '%02d' "${SCENE_INDEX}")_${scene}
[[ ! -e "${scene_root}" ]] || {
  echo "ABORT: scene output already exists: ${scene_root}" >&2; exit 1; }
mkdir -p "${scene_root}/logs"
exec > >(tee "${scene_root}/run.log") 2>&1

(
  cd "${ROOT}"
  hab_python -m unittest MemNavData.test_novel_a_bearing_gate -v
)
hab_python -m py_compile "${EVALUATOR}" "${VALIDATOR}" \
  "${ROOT}/MemNavData/summarize_novel_a_bearing_gate.py"
"${POLICY_PY}" -m py_compile "${NAVDP_SERVER}" \
  "${ROOT}/NavDP/baselines/navdp/policy_agent.py" \
  "${ROOT}/NavDP/baselines/navdp/policy_network.py"
"${POLICY_PY}" -c \
  'import torch,diffusers,cv2,flask; assert torch.cuda.is_available(); print(torch.__version__)'

port_key=$(( (${SLURM_JOB_ID:-1000} + SCENE_INDEX * 37) % 15000 ))
NAVDP_PORT=${NAVDP_PORT:-$((20000 + port_key))}
if ss -ltn | awk '{print $4}' | grep -Eq "(^|:)${NAVDP_PORT}$"; then
  echo "ABORT: port ${NAVDP_PORT} is already in use" >&2; exit 1
fi
runtime_root=${SLURM_TMPDIR:-/tmp}/bearing_gate_${SLURM_JOB_ID:-local}_${SCENE_INDEX}
mkdir -p "${runtime_root}/navdp"
NAVDP_PID=
cleanup() {
  if [[ -n "${NAVDP_PID}" ]] && kill -0 "${NAVDP_PID}" 2>/dev/null; then
    kill "${NAVDP_PID}" 2>/dev/null || true
    wait "${NAVDP_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM
(
  cd "${runtime_root}/navdp"
  exec env NAVDP_DISABLE_VIDEO=1 PYTHONUNBUFFERED=1 \
    "${POLICY_PY}" -u "${NAVDP_SERVER}" \
      --port "${NAVDP_PORT}" --checkpoint "${NAVDP_CKPT}"
) > "${scene_root}/logs/server_navdp.log" 2>&1 &
NAVDP_PID=$!
ready=0
for _ in $(seq 1 240); do
  if ! kill -0 "${NAVDP_PID}" 2>/dev/null; then
    echo "ABORT: NavDP server exited during startup" >&2
    tail -n 120 "${scene_root}/logs/server_navdp.log" >&2
    exit 1
  fi
  if ss -ltn | awk '{print $4}' | grep -Eq "(^|:)${NAVDP_PORT}$"; then
    ready=1; break
  fi
  sleep 2
done
[[ "${ready}" -eq 1 ]] || {
  echo "ABORT: NavDP server did not bind" >&2; exit 1; }

smoke_env=()
if [[ "${SMOKE}" -eq 1 ]]; then
  smoke_env+=(NOVEL_A_BEARING_SMOKE=1)
fi
env \
  NOVEL_A_BEARING_PROTOCOL="${PROTOCOL}" \
  NOVEL_A_BEARING_MANIFEST="${MANIFEST}" \
  NOVEL_A_BEARING_INPUTS="${INPUT_OVERLAY}" \
  NOVEL_A_BEARING_SCENE_INDEX="${SCENE_INDEX}" \
  "${smoke_env[@]}" \
  PYTHONPATH="${HAB_PYTHONPATH}" \
  "${HAB_PY}" -u "${EVALUATOR}" \
    --episode_root "${episode_scene_root}" \
    --scene "${scene_file}" \
    --host 127.0.0.1 --port "${NAVDP_PORT}" \
    --out "${scene_root}" \
    --server_backend navdp \
    --leg1_mode policy --stop_after_leg1 \
    --success_dist 1.0 --max_steps "${MAX_STEPS}" --exec_horizon 8 \
    --trajectory_selector server --seed 20260803 \
    --terminal_uturn off --terminal_visual_refine off \
    --episode_ids "${episode_csv}" --deterministic_plan_seeds \
  > "${scene_root}/logs/evaluator.log" 2>&1

test -s "${scene_root}/run_meta.json" -a -s "${scene_root}/bearing_arms.csv" || {
  echo "ABORT: evaluator did not produce complete outputs" >&2; exit 1; }
expected_rows=$(( ${#episode_ids[@]} * 3 ))
actual_rows=$(( $(wc -l < "${scene_root}/bearing_arms.csv") - 1 ))
[[ "${actual_rows}" -eq "${expected_rows}" ]] || {
  echo "ABORT: expected ${expected_rows} arm rows, got ${actual_rows}" >&2; exit 1; }
sha256sum "${TASK_FILES[@]/#/${ROOT}/}" \
  > "${scene_root}/source_inputs.sha256"
echo "[complete] scene=${scene} rows=${actual_rows} output=${scene_root}"
