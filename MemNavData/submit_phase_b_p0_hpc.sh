#!/usr/bin/env bash
# Submit strict Phase-B P0 pairing: one formal smoke scene, then 19 scenes.

set -euo pipefail
umask 0022

ROOT=$(git rev-parse --show-toplevel)
RUN_TAG=${RUN_TAG:?export a unique RUN_TAG}
LAUNCHER=${ROOT}/MemNavData/slurm_expanded_navdp_router_eval.sbatch
MANIFEST=${ROOT}/MemNavData/expanded_navdp_router_eval_20260805.json
PHASE_B_CKPT=${PHASE_B_CKPT:-${ROOT}/.diagnostics/phase_b_model_20260808/lingbot_native_phase_b.pt}
EXPECTED_PHASE_B_CKPT_SHA=${EXPECTED_PHASE_B_CKPT_SHA:-1232a426458cedf36869304116a2dd5c779bbcdaca587f76abd5ed3572164f2c}

[[ "${RUN_TAG}" =~ ^[A-Za-z0-9._-]+$ ]] || {
  echo "ABORT: RUN_TAG contains unsafe characters" >&2
  exit 1
}
for command_name in git sbatch sha256sum; do
  command -v "${command_name}" >/dev/null || {
    echo "ABORT: ${command_name} is unavailable" >&2
    exit 1
  }
done
for required in "${LAUNCHER}" "${MANIFEST}" "${PHASE_B_CKPT}"; do
  test -r "${required}" || {
    echo "ABORT: missing input ${required}" >&2
    exit 1
  }
done
[[ -z "$(git -C "${ROOT}" status --porcelain --untracked-files=all)" ]] || {
  echo "ABORT: HPC evaluation requires a completely clean committed worktree" >&2
  git -C "${ROOT}" status --short >&2
  exit 1
}
actual_checkpoint_sha=$(sha256sum "${PHASE_B_CKPT}" | awk '{print $1}')
[[ "${actual_checkpoint_sha}" == "${EXPECTED_PHASE_B_CKPT_SHA}" ]] || {
  echo "ABORT: Phase-B checkpoint SHA256 mismatch" >&2
  exit 1
}

expected_commit=$(git -C "${ROOT}" rev-parse HEAD)
expected_manifest_sha=$(sha256sum "${MANIFEST}" | awk '{print $1}')
exports="ALL,RUN_TAG=${RUN_TAG},EXPECTED_COMMIT=${expected_commit},EXPECTED_MANIFEST_SHA=${expected_manifest_sha},P0_SHARED_PREFIX=1,RUN_NAVDP_NATIVE=1,RUN_GEOMETRY_TOP1=0,RUN_GEOMETRY_ROUTER=1,RUN_LEARNED_RANK_GEOMETRY=1,PHASE_B_CKPT=${PHASE_B_CKPT},EXPECTED_PHASE_B_CKPT_SHA=${EXPECTED_PHASE_B_CKPT_SHA},MAX_STEPS=500,EPISODE_LIMIT=0,LEG1_MODE=policy,STOP_AFTER_LEG1=0,WRITE_LEG1_TRACE=0,DETERMINISTIC_PLAN_SEEDS=1,NAVDP_GOAL_SWITCH_RESET=carry,TRAJECTORY_SELECTOR=server,TRAJECTORY_SELECTOR_SCOPE=all,RUN_CONDITIONAL_ORACLES=0,RUN_CONDITIONAL_ORACLE_ANCHOR=0,RUN_CONDITIONAL_ORACLE_POINT=0"

sbatch --test-only --array=0 --export="${exports}" "${LAUNCHER}" >/dev/null
smoke_result=$(sbatch --parsable --array=0 --export="${exports}" "${LAUNCHER}")
smoke_job=${smoke_result%%;*}
[[ "${smoke_job}" =~ ^[0-9]+$ ]] || {
  echo "ABORT: unexpected smoke submission result: ${smoke_result}" >&2
  exit 1
}

sbatch --test-only --array=1-19%2 \
  --dependency="afterok:${smoke_job}" --kill-on-invalid-dep=yes \
  --export="${exports}" "${LAUNCHER}" >/dev/null
full_result=$(sbatch --parsable --array=1-19%2 \
  --dependency="afterok:${smoke_job}" --kill-on-invalid-dep=yes \
  --export="${exports}" "${LAUNCHER}")
full_job=${full_result%%;*}
[[ "${full_job}" =~ ^[0-9]+$ ]] || {
  echo "ABORT: unexpected full submission result: ${full_result}" >&2
  exit 1
}

echo "Phase-B P0 submitted: smoke_job=${smoke_job} full_job=${full_job} run_tag=${RUN_TAG}"
