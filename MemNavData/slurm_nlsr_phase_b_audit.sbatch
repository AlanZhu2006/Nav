#!/usr/bin/env bash
# Independently audit one completed formal Phase-B LingBot feature artifact.
#SBATCH --job-name=nlsr_pB_audit
#SBATCH --account=torch_pr_769_tandon_advanced
#SBATCH --partition=cpu_short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/%x_%j.out
#SBATCH --error=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/%x_%j.err

set -euo pipefail
umask 0022

: "${REPO_ROOT:?export the exact clean Phase-B checkout}"
: "${EXPECTED_COMMIT:?export its full commit SHA}"
: "${EXPECTED_LAUNCHER_SHA:?export the committed launcher SHA256}"
: "${RUN_TAG:?export the audited NLSR run tag}"
: "${PHASE_B_ROLE:?export train or development}"

RESULT_BASE="${RESULT_BASE:-/scratch/yz11502/Research/Nav-axis-uturn-results/nlsr_gapfill_20260807}"
RUN_ROOT="${RESULT_BASE}/${RUN_TAG}"
UPSTREAM="${RUN_ROOT}/causal_covisibility_teacher_a55ae5d40ba3_bc6bf58536f6/teacher_bundle"
TEACHER_CSV="${UPSTREAM}/teacher.csv"
TEACHER_AUDIT="${UPSTREAM}/audit.json"
MANIFEST="${RUN_ROOT}/manifest_multistage/candidate_manifest_multistage.json"
SCALE_ARTIFACT="${RUN_ROOT}/causal_ground_scale/causal_ground_scale.json"
SCALE_ACCEPTANCE="${RUN_ROOT}/causal_ground_scale_acceptance_c8f499dc9c4f_f6f28e20b9f2/causal_ground_scale_acceptance.json"
SPLIT_MANIFEST="${REPO_ROOT}/MemNavData/router_multiscene_split_20260805.json"
OUTPUT_ROOT="${RUN_ROOT}/phase_b_lingbot_features_${EXPECTED_COMMIT:0:12}"
AUDITOR_REL="MemNavData/audit_lingbot_native_localizer_artifact.py"
LAUNCHER_REL="MemNavData/slurm_nlsr_phase_b_audit.sbatch"
MEMNAV_PY="/scratch/lg154/conda-envs/memnav/bin/python"
BASE_SIF="/share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif"

EXPECTED_TEACHER_SHA="fd52dcfcb7e8d1bdac0703f7ba0b6d10f04067d4b8793386adf65ede6bc8e313"
EXPECTED_TEACHER_AUDIT_SHA="cd7091bb8c398ac5a7f558003a3fb467cc450a9b65e4416e29b7d27e1fad14c6"
EXPECTED_MANIFEST_SHA="bc6bf58536f6c159d1898ac03abe365eadba65c22cd246e39401821962abb34c"
EXPECTED_SCALE_SHA="f6f28e20b9f21a764132af04b425f23b65e1dc779798d831a571a7df26bd78db"
EXPECTED_SCALE_ACCEPTANCE_SHA="0ced1b3a56fe0e089bad35c5807c219da27daad48e6bbd4bb29436ef2b8eb3ac"
EXPECTED_SCALE_ACCEPTANCE_COMMIT="c8f499dc9c4f9112087d26a0daba46f3c3c82d59"
EXPECTED_SPLIT_SHA="97309c183e25cb3dd65472908748d55a94798a636db6157ab6fe120fca05cf7a"

fail() { echo "ABORT: $*" >&2; exit 2; }
full_sha_is() {
  local path="$1" expected="$2" label="$3"
  [[ -f "${path}" && ! -L "${path}" ]] || fail "missing ${label}: ${path}"
  [[ "$(sha256sum "${path}" | awk '{print $1}')" == "${expected}" ]] || \
    fail "${label} SHA mismatch"
}
committed_file_is() {
  local relative="$1" actual expected
  actual="$(sha256sum "${REPO_ROOT}/${relative}" | awk '{print $1}')"
  expected="$(git -C "${REPO_ROOT}" show "${EXPECTED_COMMIT}:${relative}" | \
    sha256sum | awk '{print $1}')"
  [[ "${actual}" == "${expected}" ]] || fail "${relative} differs from commit"
}

[[ "${PHASE_B_ROLE}" =~ ^(train|development)$ ]] || fail "invalid PHASE_B_ROLE"
[[ "$(git -C "${REPO_ROOT}" rev-parse HEAD)" == "${EXPECTED_COMMIT}" ]] || \
  fail "audit checkout commit changed"
[[ -z "$(git -C "${REPO_ROOT}" status --porcelain)" ]] || fail "checkout is not clean"
full_sha_is "$0" "${EXPECTED_LAUNCHER_SHA}" "spooled audit launcher"
for relative in "${AUDITOR_REL}" "${LAUNCHER_REL}" \
                "MemNavData/phase_b_upstream_receipts.py" \
                "MemNavData/external_causal_scale_contract.py" \
                "MemNavData/test_lingbot_native_training.py" \
                "MemNavData/test_phase_b_upstream_receipts.py" \
                "MemNavData/router_multiscene_split_20260805.json"; do
  committed_file_is "${relative}"
done
full_sha_is "${TEACHER_CSV}" "${EXPECTED_TEACHER_SHA}" "teacher CSV"
full_sha_is "${TEACHER_AUDIT}" "${EXPECTED_TEACHER_AUDIT_SHA}" "teacher audit"
full_sha_is "${MANIFEST}" "${EXPECTED_MANIFEST_SHA}" "causal manifest"
full_sha_is "${SCALE_ARTIFACT}" "${EXPECTED_SCALE_SHA}" "scale artifact"
full_sha_is "${SCALE_ACCEPTANCE}" "${EXPECTED_SCALE_ACCEPTANCE_SHA}" "scale acceptance"
full_sha_is "${SPLIT_MANIFEST}" "${EXPECTED_SPLIT_SHA}" "scene split"

if [[ "${PHASE_B_ROLE}" == "train" ]]; then
  TOP_K=2
  EXPECTED_SCENES=40
  EXPECTED_SESSIONS=480
  EXPECTED_SEEDS=1098
else
  TOP_K=8
  EXPECTED_SCENES=10
  EXPECTED_SESSIONS=120
  EXPECTED_SEEDS=933
fi
ROLE_ROOT="${OUTPUT_ROOT}/${PHASE_B_ROLE}_top${TOP_K}_gap16"
for required in \
  "${ROLE_ROOT}/lingbot_goal_loop_closure_checkpoint.sqlite3" \
  "${ROLE_ROOT}/lingbot_goal_loop_closure_progress.json" \
  "${ROLE_ROOT}/lingbot_goal_loop_closure_rows.csv" \
  "${ROLE_ROOT}/diagnostic_lingbot_goal_loop_closure.json"; do
  [[ -f "${required}" && ! -L "${required}" ]] || fail "missing collector artifact ${required}"
done

export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export APPTAINERENV_PYTHONPATH="${PYTHONPATH}"
export SINGULARITYENV_PYTHONPATH="${PYTHONPATH}"
CONTAINER=(singularity exec -B /scratch/lg154 -B /scratch/yz11502 "${BASE_SIF}")
"${CONTAINER[@]}" "${MEMNAV_PY}" -m py_compile \
  "${REPO_ROOT}/${AUDITOR_REL}" \
  "${REPO_ROOT}/MemNavData/phase_b_upstream_receipts.py"
"${CONTAINER[@]}" "${MEMNAV_PY}" -m unittest -v \
  MemNavData.test_phase_b_upstream_receipts \
  MemNavData.test_lingbot_native_training

"${CONTAINER[@]}" "${MEMNAV_PY}" -u "${REPO_ROOT}/${AUDITOR_REL}" \
  --run-dir "${ROLE_ROOT}" \
  --teacher-csv "${TEACHER_CSV}" \
  --split-manifest "${SPLIT_MANIFEST}" \
  --expected-role "${PHASE_B_ROLE}" \
  --expected-scenes "${EXPECTED_SCENES}" \
  --expected-sessions "${EXPECTED_SESSIONS}" \
  --expected-seeds "${EXPECTED_SEEDS}" \
  --expected-rows "${EXPECTED_SEEDS}" \
  --expected-source-commit "${EXPECTED_COMMIT}" \
  --expected-teacher-sha256 "${EXPECTED_TEACHER_SHA}" \
  --expected-split-sha256 "${EXPECTED_SPLIT_SHA}" \
  --causal-teacher-audit "${TEACHER_AUDIT}" \
  --expected-causal-teacher-audit-sha256 "${EXPECTED_TEACHER_AUDIT_SHA}" \
  --causal-scale-acceptance "${SCALE_ACCEPTANCE}" \
  --expected-causal-scale-acceptance-sha256 "${EXPECTED_SCALE_ACCEPTANCE_SHA}" \
  --expected-causal-scale-acceptance-commit "${EXPECTED_SCALE_ACCEPTANCE_COMMIT}"

echo "[complete] ${PHASE_B_ROLE} artifact audit: ${ROLE_ROOT}/lingbot_native_training_artifact_audit.json"
