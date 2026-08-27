#!/usr/bin/env bash
# Formal three-seed Phase-B localizer training from two independently pinned
# feature-artifact audits.  The frozen NavDP/LingBot weights are not updated.
#SBATCH --job-name=nlsr_pB_train
#SBATCH --account=torch_pr_769_tandon_advanced
#SBATCH --partition=cpu_short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/%x_%j.out
#SBATCH --error=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/%x_%j.err

set -euo pipefail
umask 0022

: "${REPO_ROOT:?export the exact clean Phase-B checkout}"
: "${EXPECTED_COMMIT:?export its full commit SHA}"
: "${EXPECTED_LAUNCHER_SHA:?export the committed launcher SHA256}"
: "${RUN_TAG:?export the audited NLSR run tag}"
: "${EXPECTED_TRAIN_AUDIT_SHA:?relay must pin the train audit SHA}"
: "${EXPECTED_DEVELOPMENT_AUDIT_SHA:?relay must pin the development audit SHA}"

RESULT_BASE="${RESULT_BASE:-/scratch/yz11502/Research/Nav-axis-uturn-results/nlsr_gapfill_20260807}"
RUN_ROOT="${RESULT_BASE}/${RUN_TAG}"
FEATURE_ROOT="${RUN_ROOT}/phase_b_lingbot_features_${EXPECTED_COMMIT:0:12}"
TRAIN_ROOT="${FEATURE_ROOT}/train_top2_gap16"
DEVELOPMENT_ROOT="${FEATURE_ROOT}/development_top8_gap16"
TRAIN_AUDIT="${TRAIN_ROOT}/lingbot_native_training_artifact_audit.json"
DEVELOPMENT_AUDIT="${DEVELOPMENT_ROOT}/lingbot_native_training_artifact_audit.json"
TEACHER_CSV="${RUN_ROOT}/causal_covisibility_teacher_a55ae5d40ba3_bc6bf58536f6/teacher_bundle/teacher.csv"
SPLIT_MANIFEST="${REPO_ROOT}/MemNavData/router_multiscene_split_20260805.json"
OUTPUT_BASE="${RUN_ROOT}/phase_b_models_${EXPECTED_COMMIT:0:12}"
OUTPUT_ROOT="${OUTPUT_BASE}/three_seed_job_${SLURM_JOB_ID:-manual}"

TRAINER_REL="MemNavData/train_lingbot_native_localizer.py"
LAUNCHER_REL="MemNavData/slurm_nlsr_phase_b_train.sbatch"
MEMNAV_PY="/scratch/lg154/conda-envs/memnav/bin/python"
BASE_SIF="/share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif"
EXPECTED_TEACHER_SHA="fd52dcfcb7e8d1bdac0703f7ba0b6d10f04067d4b8793386adf65ede6bc8e313"
EXPECTED_SPLIT_SHA="97309c183e25cb3dd65472908748d55a94798a636db6157ab6fe120fca05cf7a"

WANDB_MODE="${WANDB_MODE:-offline}"
WANDB_PROJECT="${WANDB_PROJECT:-memnav}"
WANDB_NAME="${WANDB_NAME:-nlsr_phase_b_${EXPECTED_COMMIT:0:8}_${SLURM_JOB_ID:-manual}}"
[[ "${WANDB_MODE}" =~ ^(disabled|offline|online)$ ]] || {
  echo "ABORT: invalid WANDB_MODE=${WANDB_MODE}" >&2; exit 2; }
if [[ "${WANDB_MODE}" == "online" && -z "${WANDB_API_KEY:-}" ]]; then
  echo "ABORT: online W&B requested without WANDB_API_KEY" >&2
  exit 2
fi

fail() { echo "ABORT: $*" >&2; exit 2; }
full_sha_is() {
  local path="$1" expected="$2" label="$3"
  [[ -f "${path}" && ! -L "${path}" ]] || fail "missing ${label}: ${path}"
  [[ "$(sha256sum "${path}" | awk '{print $1}')" == "${expected}" ]] || \
    fail "${label} SHA mismatch"
}
sidecar_is() {
  local artifact="$1" expected="$2"
  [[ "$(<"${artifact}.sha256")" == "${expected}  $(basename "${artifact}")" ]] || \
    fail "audit sidecar mismatch: ${artifact}"
}
committed_file_is() {
  local relative="$1" actual expected
  actual="$(sha256sum "${REPO_ROOT}/${relative}" | awk '{print $1}')"
  expected="$(git -C "${REPO_ROOT}" show "${EXPECTED_COMMIT}:${relative}" | \
    sha256sum | awk '{print $1}')"
  [[ "${actual}" == "${expected}" ]] || fail "${relative} differs from commit"
}

[[ "$(git -C "${REPO_ROOT}" rev-parse HEAD)" == "${EXPECTED_COMMIT}" ]] || \
  fail "training checkout commit changed"
[[ -z "$(git -C "${REPO_ROOT}" status --porcelain)" ]] || fail "checkout is not clean"
full_sha_is "$0" "${EXPECTED_LAUNCHER_SHA}" "spooled trainer launcher"
for relative in "${TRAINER_REL}" "${LAUNCHER_REL}" \
                "MemNavData/phase_b_feature_schema.py" \
                "MemNavData/phase_b_deployment_inference_contract.py" \
                "MemNavData/external_causal_scale_contract.py" \
                "MemNavData/test_lingbot_native_training.py" \
                "MemNavData/test_phase_b_training_preflight.py" \
                "MemNavData/router_multiscene_split_20260805.json"; do
  committed_file_is "${relative}"
done
full_sha_is "${TRAIN_AUDIT}" "${EXPECTED_TRAIN_AUDIT_SHA}" "train artifact audit"
sidecar_is "${TRAIN_AUDIT}" "${EXPECTED_TRAIN_AUDIT_SHA}"
full_sha_is "${DEVELOPMENT_AUDIT}" "${EXPECTED_DEVELOPMENT_AUDIT_SHA}" \
  "development artifact audit"
sidecar_is "${DEVELOPMENT_AUDIT}" "${EXPECTED_DEVELOPMENT_AUDIT_SHA}"
full_sha_is "${TEACHER_CSV}" "${EXPECTED_TEACHER_SHA}" "teacher CSV"
full_sha_is "${SPLIT_MANIFEST}" "${EXPECTED_SPLIT_SHA}" "scene split"
[[ ! -e "${OUTPUT_ROOT}" ]] || fail "training output already exists: ${OUTPUT_ROOT}"

mkdir -p "${OUTPUT_BASE}" "${OUTPUT_ROOT}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-12}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-12}"
export PYTHONHASHSEED=0
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export APPTAINERENV_PYTHONPATH="${PYTHONPATH}"
export SINGULARITYENV_PYTHONPATH="${PYTHONPATH}"
CONTAINER=(singularity exec -B /scratch/lg154 -B /scratch/yz11502 "${BASE_SIF}")

"${CONTAINER[@]}" "${MEMNAV_PY}" -m py_compile \
  "${REPO_ROOT}/${TRAINER_REL}" \
  "${REPO_ROOT}/MemNavData/phase_b_feature_schema.py"
"${CONTAINER[@]}" "${MEMNAV_PY}" - <<'PY'
import importlib
import json
versions = {}
for name in ("numpy", "pandas", "sklearn", "torch", "wandb"):
    module = importlib.import_module(name)
    versions[name] = getattr(module, "__version__", "unknown")
print(json.dumps({"dependency_preflight": "passed", "versions": versions},
                 sort_keys=True))
PY
"${CONTAINER[@]}" "${MEMNAV_PY}" -m unittest -v \
  MemNavData.test_lingbot_native_training \
  MemNavData.test_phase_b_training_preflight \
  MemNavData.test_phase_b_deployment_inference_contract

TRAIN_ARGS=(
  --train-run-dir "${TRAIN_ROOT}"
  --train-audit "${TRAIN_AUDIT}"
  --expected-train-audit-sha256 "${EXPECTED_TRAIN_AUDIT_SHA}"
  --development-csv "${DEVELOPMENT_ROOT}/lingbot_goal_loop_closure_rows.csv"
  --development-audit "${DEVELOPMENT_AUDIT}"
  --expected-development-audit-sha256 "${EXPECTED_DEVELOPMENT_AUDIT_SHA}"
  --teacher-csv "${TEACHER_CSV}"
  --split-manifest "${SPLIT_MANIFEST}"
  --epochs 200
  --patience 25
  --batch-size 32
  --learning-rate 3e-4
  --weight-decay 1e-4
  --pose-weight 1.0
  --pose-tail-weight 0.5
  --pose-tail-fraction 0.2
  --hidden-dim 64
  --dropout 0.10
  --device cpu
  --require-external-causal-scale
)

# The first invocation reads the exact production artifacts and requires a
# finite, non-zero backward signal in every trainable head.  It writes no model.
"${CONTAINER[@]}" "${MEMNAV_PY}" -u "${REPO_ROOT}/${TRAINER_REL}" \
  "${TRAIN_ARGS[@]}" \
  --out-dir "${OUTPUT_ROOT}/unused_gradient_preflight" \
  --wandb-mode disabled \
  --preflight-only | tee "${OUTPUT_ROOT}/gradient_preflight.json"

# This train-only gate is deliberately allowed to memorize 16 sessions.  Its
# purpose is mechanical: prove that real joined features/labels can drive the
# rank and no-match heads before spending a development evaluation.
"${CONTAINER[@]}" "${MEMNAV_PY}" -u "${REPO_ROOT}/${TRAINER_REL}" \
  "${TRAIN_ARGS[@]}" \
  --out-dir "${OUTPUT_ROOT}/unused_overfit_preflight" \
  --wandb-mode disabled \
  --overfit-preflight-only | tee "${OUTPUT_ROOT}/overfit_preflight.json"

"${CONTAINER[@]}" "${MEMNAV_PY}" -u "${REPO_ROOT}/${TRAINER_REL}" \
  "${TRAIN_ARGS[@]}" \
  --out-dir "${OUTPUT_ROOT}/model" \
  --wandb-mode "${WANDB_MODE}" \
  --wandb-project "${WANDB_PROJECT}" \
  --wandb-name "${WANDB_NAME}" | tee "${OUTPUT_ROOT}/training_stdout.json"

echo "[complete] formal Phase-B model: ${OUTPUT_ROOT}/model/report.json"
