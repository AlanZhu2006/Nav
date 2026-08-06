#!/usr/bin/env bash
# Execute 2-leg and 3-leg closed-loop benchmarks plus the learned-router blind
# audit sequentially inside one H200 allocation.

set -euo pipefail
umask 0022

ROOT=${ROOT:?set ROOT}
RUN_ROOT=${RUN_ROOT:?set RUN_ROOT}
EXPECTED_COMMIT=${EXPECTED_COMMIT:?set EXPECTED_COMMIT}
HAB_PY=${HAB_PY:?set HAB_PY}
MEMNAV_PY=${MEMNAV_PY:?set MEMNAV_PY}
LINGBOT_REPO=${LINGBOT_REPO:?set LINGBOT_REPO}
EPISODE_OVERLAY_ROOT=${EPISODE_OVERLAY_ROOT:?set EPISODE_OVERLAY_ROOT}

EXPANDED_MANIFEST=${ROOT}/MemNavData/expanded_navdp_router_eval_20260805.json
EXPANDED_MANIFEST_SHA=ba8f72cb504768c801e6c9c386436ccdc66dea07a5e5fac2d7b4248738946a61
EXPANDED_RUNNER=${ROOT}/MemNavData/run_expanded_navdp_router_scene.sh
EXPANDED_SUMMARIZER=${ROOT}/MemNavData/summarize_expanded_navdp_router_eval.py
THREE_LEG_MANIFEST=${ROOT}/MemNavData/expanded_3leg_router_eval_20260805.json
THREE_LEG_MANIFEST_SHA=55096038d0493723d2280fe9abbcb2ea9ea7f2059c50b6eb08d02f560cdb281b
THREE_LEG_EVALUATOR=${ROOT}/MemNavData/eval_3leg_habitat.py
THREE_LEG_VALIDATOR=${ROOT}/MemNavData/validate_expanded_3leg_router_eval.py
THREE_LEG_SUMMARIZER=${ROOT}/MemNavData/summarize_expanded_3leg_router_eval.py
FINAL_SPLIT=${ROOT}/MemNavData/router_multiscene_final_blind_20260805.json
FINAL_SPLIT_SHA=8683e37787afdfa0399351cafcc699a9ab57f0565c921465902049134780f419
ROUTER_RUNNER=${ROOT}/MemNavData/run_patch_temporal_router_multiscene.sh
REFERENCE_ROUTER_MODEL=/scratch/yz11502/Research/Nav-axis-uturn-results/patch_router_multiscene_20260805/full_job_15315411/patch_temporal/diagnostic_patch_temporal_router_not_for_deployment.json
REFERENCE_ROUTER_SHA=7a3605d9bb8891a280286408d6c54c161ae1604246b95dac4e64a0671f318723

TASK_FILES=(
  MemNavData/run_all_in_one_router_eval.sh
  MemNavData/slurm_all_in_one_router_eval.sbatch
  MemNavData/run_expanded_navdp_router_scene.sh
  MemNavData/slurm_expanded_navdp_router_eval.sbatch
  MemNavData/summarize_expanded_navdp_router_eval.py
  MemNavData/validate_expanded_navdp_router_eval.py
  MemNavData/expanded_navdp_router_eval_20260805.json
  MemNavData/eval_3leg_habitat.py
  MemNavData/navdp_goal_switch.py
  MemNavData/test_navdp_goal_switch.py
  MemNavData/validate_expanded_3leg_router_eval.py
  MemNavData/summarize_expanded_3leg_router_eval.py
  MemNavData/test_expanded_3leg_router_eval.py
  MemNavData/test_router_candidates.py
  MemNavData/expanded_3leg_router_eval_20260805.json
  MemNavData/run_patch_temporal_router_multiscene.sh
  MemNavData/slurm_patch_temporal_router_multiscene.sbatch
  MemNavData/validate_frozen_router_blind.py
  MemNavData/router_multiscene_final_blind_20260805.json
  NavDP/baselines/memnav/memnav_server.py
  NavDP/baselines/memnav/policy_agent.py
  NavDP/baselines/memnav/router_candidates.py
)

actual_commit=$(git -C "${ROOT}" rev-parse HEAD)
[[ "${actual_commit}" == "${EXPECTED_COMMIT}" ]] || {
  echo "ABORT: code commit ${actual_commit} != ${EXPECTED_COMMIT}" >&2
  exit 1
}
git -C "${ROOT}" diff --quiet -- "${TASK_FILES[@]}" || {
  echo "ABORT: all-in-one task files differ from the checked-out commit" >&2
  exit 1
}
git -C "${ROOT}" diff --cached --quiet -- "${TASK_FILES[@]}" || {
  echo "ABORT: staged all-in-one task files differ from the checked-out commit" >&2
  exit 1
}
for required in "${EXPANDED_MANIFEST}" "${EXPANDED_RUNNER}" \
                "${EXPANDED_SUMMARIZER}" "${THREE_LEG_MANIFEST}" \
                "${THREE_LEG_EVALUATOR}" "${THREE_LEG_VALIDATOR}" \
                "${THREE_LEG_SUMMARIZER}" "${FINAL_SPLIT}" \
                "${ROUTER_RUNNER}" "${REFERENCE_ROUTER_MODEL}"; do
  test -r "${required}" || { echo "ABORT: missing ${required}" >&2; exit 1; }
done

# The final-blind source scenes are intentionally absent from training, but
# they still need two raw episodes each.  Check this before spending hours on
# the preceding Habitat stages.
"${MEMNAV_PY}" - "${FINAL_SPLIT}" "${EPISODE_OVERLAY_ROOT}" <<'PY'
import json
from pathlib import Path
import sys

manifest = json.load(open(sys.argv[1], encoding="utf-8"))
episode_root = Path(sys.argv[2])
missing = {}
for scene in manifest["development"]:
    scene_root = episode_root / scene
    episodes = ([path for path in scene_root.iterdir() if path.is_dir()]
                if scene_root.is_dir() else [])
    if len(episodes) < 2:
        missing[scene] = len(episodes)
if missing:
    raise RuntimeError(
        f"final-blind scenes need two episodes before evaluation: {missing}")
print("final-blind episode preflight OK")
PY

if [[ -e "${RUN_ROOT}" ]]; then
  echo "ABORT: all-in-one output already exists: ${RUN_ROOT}" >&2
  exit 1
fi
mkdir -p "${RUN_ROOT}"
exec > >(tee "${RUN_ROOT}/run.log") 2>&1

echo "[all-in-one] commit=${actual_commit} job=${SLURM_JOB_ID:-local}"
echo "[stage 1/5] 20-scene, 40-episode paired 2-leg closed-loop evaluation"
EXPANDED_ROOT=${RUN_ROOT}/expanded_navdp_router
for scene_index in $(seq 0 19); do
  echo "[scene $((scene_index + 1))/20] index=${scene_index}"
  env \
    ROOT="${ROOT}" \
    RUN_ROOT="${EXPANDED_ROOT}" \
    MANIFEST="${EXPANDED_MANIFEST}" \
    EXPECTED_MANIFEST_SHA="${EXPANDED_MANIFEST_SHA}" \
    EXPECTED_COMMIT="${EXPECTED_COMMIT}" \
    SCENE_INDEX="${scene_index}" \
    HAB_PY="${HAB_PY}" \
    MEMNAV_PY="${MEMNAV_PY}" \
    "${EXPANDED_RUNNER}"
done

echo "[stage 2/5] aggregate 2-leg Novel, Revisit, joint SR and router activation"
"${HAB_PY}" "${EXPANDED_SUMMARIZER}" \
  --manifest "${EXPANDED_MANIFEST}" \
  --run-root "${EXPANDED_ROOT}" \
  > "${EXPANDED_ROOT}/summary.json"
cat "${EXPANDED_ROOT}/summary.json"

echo "[stage 3/5] 10-scene paired true 3-leg A/B/C closed-loop evaluation"
THREE_LEG_ROOT=${RUN_ROOT}/expanded_3leg_router
for scene_index in $(seq 0 9); do
  echo "[3-leg scene $((scene_index + 1))/10] index=${scene_index}"
  env \
    ROOT="${ROOT}" \
    RUN_ROOT="${THREE_LEG_ROOT}" \
    MANIFEST="${THREE_LEG_MANIFEST}" \
    EXPECTED_MANIFEST_SHA="${THREE_LEG_MANIFEST_SHA}" \
    EXPECTED_COMMIT="${EXPECTED_COMMIT}" \
    SCENE_INDEX="${scene_index}" \
    HAB_PY="${HAB_PY}" \
    MEMNAV_PY="${MEMNAV_PY}" \
    EVALUATOR="${THREE_LEG_EVALUATOR}" \
    VALIDATOR="${THREE_LEG_VALIDATOR}" \
    UNIT_TEST_MODULE=MemNavData.test_expanded_3leg_router_eval \
    MAX_STEPS=1200 \
    "${EXPANDED_RUNNER}"
done

echo "[stage 4/5] aggregate Novel-A, Novel-B, Revisit-C and joint 3-leg SR"
"${HAB_PY}" "${THREE_LEG_SUMMARIZER}" \
  --manifest "${THREE_LEG_MANIFEST}" \
  --run-root "${THREE_LEG_ROOT}" \
  > "${THREE_LEG_ROOT}/summary.json"
cat "${THREE_LEG_ROOT}/summary.json"

echo "[stage 5/5] frozen learned-router audit on four final-reserved scenes"
env \
  MODE=full \
  ROOT="${ROOT}" \
  RUN_ROOT="${RUN_ROOT}/learned_router_final_blind" \
  EPISODE_ROOT="${EPISODE_OVERLAY_ROOT}" \
  MEMNAV_PY="${MEMNAV_PY}" \
  LINGBOT_REPO="${LINGBOT_REPO}" \
  SPLIT_MANIFEST="${FINAL_SPLIT}" \
  EXPECTED_SPLIT_SHA="${FINAL_SPLIT_SHA}" \
  REFERENCE_ROUTER_MODEL="${REFERENCE_ROUTER_MODEL}" \
  EXPECTED_REFERENCE_ROUTER_SHA="${REFERENCE_ROUTER_SHA}" \
  EXPECTED_COMMIT="${EXPECTED_COMMIT}" \
  "${ROUTER_RUNNER}" full

echo "[complete] one allocation produced 2-leg, 3-leg, and blind results"
echo "two_leg_summary=${EXPANDED_ROOT}/summary.json"
echo "three_leg_summary=${THREE_LEG_ROOT}/summary.json"
echo "blind_report=${RUN_ROOT}/learned_router_final_blind/patch_temporal/report.json"
