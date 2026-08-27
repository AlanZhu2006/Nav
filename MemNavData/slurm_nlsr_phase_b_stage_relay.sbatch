#!/usr/bin/env bash
# Serialize formal Phase-B GPU stages under the one-GPU-per-user QoS.  This
# relay runs only after the preceding GPU job is terminal, then either resumes
# a wall-time-interrupted role, advances train -> development, or launches the
# two independent audits and the final pinned training relay.
#SBATCH --job-name=nlsr_pB_stage
#SBATCH --account=torch_pr_769_tandon_advanced
#SBATCH --partition=cpu_short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/%x_%j.out
#SBATCH --error=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/%x_%j.err

set -euo pipefail
umask 0022

: "${REPO_ROOT:?export the exact clean Phase-B checkout}"
: "${EXPECTED_COMMIT:?export its full commit SHA}"
: "${EXPECTED_LAUNCHER_SHA:?export the committed stage-relay SHA256}"
: "${RUN_TAG:?export the audited NLSR run tag}"
: "${PHASE_B_ROLE:?export train or development}"
: "${PHASE_B_ATTEMPT:?export the completed GPU attempt number}"
: "${PHASE_B_GPU_JOB_ID:?export the terminal GPU job ID}"

TRAIN_PARTITION="${TRAIN_PARTITION:-a100_tandon}"
DEVELOPMENT_PARTITION="${DEVELOPMENT_PARTITION:-h100_tandon}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-3}"
RESULT_BASE="${RESULT_BASE:-/scratch/yz11502/Research/Nav-axis-uturn-results/nlsr_gapfill_20260807}"
RUN_ROOT="${RESULT_BASE}/${RUN_TAG}"
FEATURE_ROOT="${RUN_ROOT}/phase_b_lingbot_features_${EXPECTED_COMMIT:0:12}"

COLLECT_REL="MemNavData/slurm_nlsr_phase_b_collect.sbatch"
AUDIT_REL="MemNavData/slurm_nlsr_phase_b_audit.sbatch"
STAGE_RELAY_REL="MemNavData/slurm_nlsr_phase_b_stage_relay.sbatch"
TRAIN_RELAY_REL="MemNavData/slurm_nlsr_phase_b_train_relay.sbatch"
COLLECT="${REPO_ROOT}/${COLLECT_REL}"
AUDIT="${REPO_ROOT}/${AUDIT_REL}"
STAGE_RELAY="${REPO_ROOT}/${STAGE_RELAY_REL}"
TRAIN_RELAY="${REPO_ROOT}/${TRAIN_RELAY_REL}"

fail() { echo "ABORT: $*" >&2; exit 2; }
committed_sha() {
  git -C "${REPO_ROOT}" show "${EXPECTED_COMMIT}:$1" | sha256sum | awk '{print $1}'
}
verify_committed() {
  local relative="$1" actual expected
  actual="$(sha256sum "${REPO_ROOT}/${relative}" | awk '{print $1}')"
  expected="$(committed_sha "${relative}")"
  [[ "${actual}" == "${expected}" ]] || fail "${relative} differs from commit"
}

for command_name in git sbatch sacct sha256sum; do
  command -v "${command_name}" >/dev/null || fail "${command_name} is unavailable"
done
[[ "${PHASE_B_ROLE}" =~ ^(train|development)$ ]] || fail "invalid role"
[[ "${PHASE_B_ATTEMPT}" =~ ^[1-9][0-9]*$ ]] || fail "invalid attempt"
[[ "${PHASE_B_GPU_JOB_ID}" =~ ^[0-9]+$ ]] || fail "invalid GPU job ID"
[[ "${MAX_ATTEMPTS}" =~ ^[1-9][0-9]*$ ]] || fail "invalid MAX_ATTEMPTS"
[[ "${TRAIN_PARTITION}" =~ ^(a100_tandon|h100_tandon)$ ]] || fail "invalid train partition"
[[ "${DEVELOPMENT_PARTITION}" =~ ^(a100_tandon|h100_tandon)$ ]] || \
  fail "invalid development partition"
[[ "$(git -C "${REPO_ROOT}" rev-parse HEAD)" == "${EXPECTED_COMMIT}" ]] || \
  fail "stage-relay checkout commit changed"
[[ -z "$(git -C "${REPO_ROOT}" status --porcelain)" ]] || fail "checkout is not clean"
[[ "$(sha256sum "$0" | awk '{print $1}')" == "${EXPECTED_LAUNCHER_SHA}" ]] || \
  fail "spooled stage relay changed"
for relative in "${COLLECT_REL}" "${AUDIT_REL}" "${STAGE_RELAY_REL}" \
                "${TRAIN_RELAY_REL}" "MemNavData/slurm_nlsr_phase_b_train.sbatch"; do
  verify_committed "${relative}"
done

job_state="$(sacct -n -X -j "${PHASE_B_GPU_JOB_ID}" -o State -P | \
  awk -F'|' 'NF && $1 != "" {sub(/[+ ].*$/, "", $1); print $1; exit}')"
[[ -n "${job_state}" ]] || fail "cannot resolve preceding GPU job state"

if [[ "${PHASE_B_ROLE}" == "train" ]]; then
  TOP_K=2
  EXPECTED_SESSIONS=480
  EXPECTED_SEEDS=1098
  PARTITION="${TRAIN_PARTITION}"
else
  TOP_K=8
  EXPECTED_SESSIONS=120
  EXPECTED_SEEDS=933
  PARTITION="${DEVELOPMENT_PARTITION}"
fi
ROLE_ROOT="${FEATURE_ROOT}/${PHASE_B_ROLE}_top${TOP_K}_gap16"

role_complete=0
if [[ -f "${ROLE_ROOT}/lingbot_goal_loop_closure_progress.json" ]]; then
  if python3 - "${ROLE_ROOT}" "${EXPECTED_SESSIONS}" "${EXPECTED_SEEDS}" <<'PY'
import json
from pathlib import Path
import sys
root = Path(sys.argv[1])
progress = json.loads((root / "lingbot_goal_loop_closure_progress.json").read_text())
complete = (
    progress.get("status") == "complete"
    and progress.get("completed_sessions") == int(sys.argv[2])
    and progress.get("completed_seeds") == int(sys.argv[3])
    and (root / "lingbot_goal_loop_closure_rows.csv").is_file()
)
raise SystemExit(0 if complete else 1)
PY
  then
    role_complete=1
  fi
fi

COLLECT_SHA="$(sha256sum "${COLLECT}" | awk '{print $1}')"
AUDIT_SHA="$(sha256sum "${AUDIT}" | awk '{print $1}')"
STAGE_RELAY_SHA="$(sha256sum "${STAGE_RELAY}" | awk '{print $1}')"
TRAIN_RELAY_SHA="$(sha256sum "${TRAIN_RELAY}" | awk '{print $1}')"

submit_gpu() {
  local role="$1" partition="$2" exports job
  exports="ALL,REPO_ROOT=${REPO_ROOT},EXPECTED_COMMIT=${EXPECTED_COMMIT}"
  exports+=",EXPECTED_LAUNCHER_SHA=${COLLECT_SHA},RUN_TAG=${RUN_TAG}"
  exports+=",PHASE_B_ROLE=${role}"
  sbatch --test-only --partition="${partition}" --export="${exports}" \
    "${COLLECT}" >/dev/null
  job="$(sbatch --parsable --partition="${partition}" --export="${exports}" \
    "${COLLECT}")"
  job="${job%%;*}"
  [[ "${job}" =~ ^[0-9]+$ ]] || fail "unexpected GPU submission result"
  printf '%s' "${job}"
}

submit_stage_relay() {
  local role="$1" attempt="$2" gpu_job="$3" exports job
  exports="ALL,REPO_ROOT=${REPO_ROOT},EXPECTED_COMMIT=${EXPECTED_COMMIT}"
  exports+=",EXPECTED_LAUNCHER_SHA=${STAGE_RELAY_SHA},RUN_TAG=${RUN_TAG}"
  exports+=",PHASE_B_ROLE=${role},PHASE_B_ATTEMPT=${attempt}"
  exports+=",PHASE_B_GPU_JOB_ID=${gpu_job},MAX_ATTEMPTS=${MAX_ATTEMPTS}"
  exports+=",TRAIN_PARTITION=${TRAIN_PARTITION}"
  exports+=",DEVELOPMENT_PARTITION=${DEVELOPMENT_PARTITION}"
  job="$(sbatch --parsable --dependency="afterany:${gpu_job}" \
    --kill-on-invalid-dep=yes --export="${exports}" "${STAGE_RELAY}")"
  job="${job%%;*}"
  [[ "${job}" =~ ^[0-9]+$ ]] || fail "unexpected stage-relay submission"
  printf '%s' "${job}"
}

if [[ "${role_complete}" -eq 0 ]]; then
  case "${job_state}" in
    TIMEOUT|PREEMPTED|NODE_FAIL)
      ;;
    *)
      fail "${PHASE_B_ROLE} is incomplete after non-resumable state ${job_state}"
      ;;
  esac
  [[ -f "${ROLE_ROOT}/lingbot_goal_loop_closure_checkpoint.sqlite3" ]] || \
    fail "resumable state lacks a session-atomic checkpoint"
  (( PHASE_B_ATTEMPT < MAX_ATTEMPTS )) || fail "maximum GPU attempts exhausted"
  next_attempt=$((PHASE_B_ATTEMPT + 1))
  next_gpu="$(submit_gpu "${PHASE_B_ROLE}" "${PARTITION}")"
  next_relay="$(submit_stage_relay \
    "${PHASE_B_ROLE}" "${next_attempt}" "${next_gpu}")"
  echo "resumed role=${PHASE_B_ROLE} attempt=${next_attempt} gpu=${next_gpu} relay=${next_relay}"
  exit 0
fi

if [[ "${PHASE_B_ROLE}" == "train" ]]; then
  next_gpu="$(submit_gpu development "${DEVELOPMENT_PARTITION}")"
  next_relay="$(submit_stage_relay development 1 "${next_gpu}")"
  echo "advanced train->development gpu=${next_gpu} relay=${next_relay}"
  exit 0
fi

# Both roles are complete.  Audit them independently, then allow the existing
# training relay to freeze the two audit hashes and submit the three-seed fit.
submit_audit() {
  local role="$1" exports job
  exports="ALL,REPO_ROOT=${REPO_ROOT},EXPECTED_COMMIT=${EXPECTED_COMMIT}"
  exports+=",EXPECTED_LAUNCHER_SHA=${AUDIT_SHA},RUN_TAG=${RUN_TAG}"
  exports+=",PHASE_B_ROLE=${role}"
  sbatch --test-only --export="${exports}" "${AUDIT}" >/dev/null
  job="$(sbatch --parsable --export="${exports}" "${AUDIT}")"
  job="${job%%;*}"
  [[ "${job}" =~ ^[0-9]+$ ]] || fail "unexpected audit submission"
  printf '%s' "${job}"
}

train_audit="$(submit_audit train)"
development_audit="$(submit_audit development)"
train_exports="ALL,REPO_ROOT=${REPO_ROOT},EXPECTED_COMMIT=${EXPECTED_COMMIT}"
train_exports+=",EXPECTED_LAUNCHER_SHA=${TRAIN_RELAY_SHA},RUN_TAG=${RUN_TAG}"
train_exports+=",TRAIN_AUDIT_JOB_ID=${train_audit}"
train_exports+=",DEVELOPMENT_AUDIT_JOB_ID=${development_audit}"
dependency="afterok:${train_audit}:${development_audit}"
sbatch --test-only --dependency="${dependency}" --kill-on-invalid-dep=yes \
  --export="${train_exports}" "${TRAIN_RELAY}" >/dev/null
training_relay="$(sbatch --parsable --dependency="${dependency}" \
  --kill-on-invalid-dep=yes --export="${train_exports}" "${TRAIN_RELAY}")"
training_relay="${training_relay%%;*}"
[[ "${training_relay}" =~ ^[0-9]+$ ]] || fail "unexpected training relay submission"
echo "submitted audits train=${train_audit} development=${development_audit} training_relay=${training_relay}"
