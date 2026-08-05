#!/usr/bin/env bash
# Execute the expanded closed-loop benchmark, summary, and learned-router blind
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
  MemNavData/run_patch_temporal_router_multiscene.sh
  MemNavData/slurm_patch_temporal_router_multiscene.sbatch
  MemNavData/validate_frozen_router_blind.py
  MemNavData/router_multiscene_final_blind_20260805.json
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
                "${EXPANDED_SUMMARIZER}" "${FINAL_SPLIT}" \
                "${ROUTER_RUNNER}" "${REFERENCE_ROUTER_MODEL}"; do
  test -r "${required}" || { echo "ABORT: missing ${required}" >&2; exit 1; }
done

if [[ -e "${RUN_ROOT}" ]]; then
  echo "ABORT: all-in-one output already exists: ${RUN_ROOT}" >&2
  exit 1
fi
mkdir -p "${RUN_ROOT}"
exec > >(tee "${RUN_ROOT}/run.log") 2>&1

echo "[all-in-one] commit=${actual_commit} job=${SLURM_JOB_ID:-local}"
echo "[stage 1/3] 20 scene-disjoint paired closed-loop evaluations"
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

echo "[stage 2/3] aggregate Novel, Revisit, joint SR and router activation"
"${HAB_PY}" "${EXPANDED_SUMMARIZER}" \
  --manifest "${EXPANDED_MANIFEST}" \
  --run-root "${EXPANDED_ROOT}" \
  > "${EXPANDED_ROOT}/summary.json"
cat "${EXPANDED_ROOT}/summary.json"

echo "[stage 3/3] frozen learned-router audit on four final-reserved scenes"
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

echo "[complete] one allocation produced both closed-loop and blind results"
echo "expanded_summary=${EXPANDED_ROOT}/summary.json"
echo "blind_report=${RUN_ROOT}/learned_router_final_blind/patch_temporal/report.json"
