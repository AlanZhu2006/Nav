#!/usr/bin/env bash
#SBATCH --job-name=h3T2L3Nav
#SBATCH --output=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/%x_%A_%a.out
#SBATCH --error=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/%x_%A_%a.err
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=72G
#SBATCH --partition=h100_tandon,a100_tandon
#SBATCH --qos=gpu48
#SBATCH --gres=gpu:1
#SBATCH --account=torch_pr_769_tandon_advanced

# One parent-scene rank per array cell. Empty ranks exit before model loading.
set -euo pipefail
umask 0022
export PYTHONDONTWRITEBYTECODE=1

: "${SLURM_ARRAY_TASK_ID:?submit as a scene array}"
: "${PHASE:?smoke or formal}"
: "${TASK_ROOT:?}" "${TASK_RECEIPT:?}" "${EXPECTED_TASK_RECEIPT_SHA:?}"
: "${BASE_SOURCE_ROOT:?}" "${BASE_RECEIPT:?}" "${EXPECTED_BASE_RECEIPT_SHA:?}"
: "${SERVER_SOURCE_ROOT:?}" "${SERVER_SOURCE_RECEIPT:?}"
: "${EXPECTED_SERVER_SOURCE_RECEIPT_SHA:?}"
: "${SOURCE_RUN_ROOT:?}" "${BENCH_ROOT:?}" "${PARENT_MANIFEST:?}"
: "${PROTOCOL:?}" "${CONSTRUCTION_VERIFICATION:?}"
: "${EXPECTED_CONSTRUCTION_VERIFICATION_SHA:?}" "${FORMAL_RUN_ROOT:?}"
: "${POLICY_LAUNCHER_ROOT:?}" "${POLICY_LAUNCHER_RECEIPT:?}"
: "${EXPECTED_POLICY_LAUNCHER_RECEIPT_SHA:?}"
: "${RUNTIME_CLOSURE_ROOT:?}" "${RUNTIME_CLOSURE_RECEIPT:?}"
: "${EXPECTED_RUNTIME_CLOSURE_RECEIPT_SHA:?}"
ROLE_PAIR_SCOPE=${ROLE_PAIR_SCOPE:-paper_heldout}

SIF=/share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif
RUNNER=${POLICY_LAUNCHER_ROOT}/MemNavData/run_hm3d_fullmono_server_scene.sh
fail() { echo "ABORT: $*" >&2; exit 2; }

[[ "${PHASE}" == smoke || "${PHASE}" == formal ]] || fail "invalid phase"
[[ "${ROLE_PAIR_SCOPE}" == paper_heldout ]] || \
  fail "Table-2 requires a complete hidden-role population"
[[ "$(sha256sum "${TASK_RECEIPT}" | awk '{print $1}')" == \
   "${EXPECTED_TASK_RECEIPT_SHA}" ]] || fail "task receipt changed"
(cd "${TASK_ROOT}" && sha256sum -c --quiet "${TASK_RECEIPT}") || \
  fail "task bundle changed"
[[ "$(sha256sum "${POLICY_LAUNCHER_RECEIPT}" | awk '{print $1}')" == \
   "${EXPECTED_POLICY_LAUNCHER_RECEIPT_SHA}" ]] || \
  fail "policy wrapper receipt changed"
(cd "${POLICY_LAUNCHER_ROOT}" && sha256sum -c --quiet \
  "${POLICY_LAUNCHER_RECEIPT}") || fail "policy wrapper changed"
[[ "$(sha256sum "${RUNTIME_CLOSURE_RECEIPT}" | awk '{print $1}')" == \
   "${EXPECTED_RUNTIME_CLOSURE_RECEIPT_SHA}" ]] || \
  fail "runtime closure receipt changed"
(cd "${RUNTIME_CLOSURE_ROOT}" && sha256sum -c --quiet \
  "${RUNTIME_CLOSURE_RECEIPT}") || fail "runtime closure changed"
[[ "$(sha256sum "${BASE_RECEIPT}" | awk '{print $1}')" == \
   "${EXPECTED_BASE_RECEIPT_SHA}" ]] || fail "base receipt changed"
(cd "${BASE_SOURCE_ROOT}" && sha256sum -c --quiet "${BASE_RECEIPT}") || \
  fail "base bundle changed"
[[ "$(sha256sum "${SERVER_SOURCE_RECEIPT}" | awk '{print $1}')" == \
   "${EXPECTED_SERVER_SOURCE_RECEIPT_SHA}" ]] || \
  fail "server overlay receipt changed"
(cd "${SERVER_SOURCE_ROOT}" && \
  sha256sum -c --quiet "${SERVER_SOURCE_RECEIPT}") || \
  fail "server overlay changed"
[[ "$(sha256sum "${CONSTRUCTION_VERIFICATION}" | awk '{print $1}')" == \
   "${EXPECTED_CONSTRUCTION_VERIFICATION_SHA}" ]] || \
  fail "construction verification changed"

readarray -t gate < <(/scratch/lg154/conda-envs/memnav/bin/python - \
  "${CONSTRUCTION_VERIFICATION}" "${BENCH_ROOT}/manifest.json" <<'PY'
import hashlib,json,sys
verification=json.load(open(sys.argv[1])); manifest=sys.argv[2]
digest=hashlib.sha256(open(manifest,"rb").read()).hexdigest()
print(str(verification.get("verified") is True).lower())
print(str(verification.get("construction_only") is True).lower())
print(str(verification.get("formal_policy_evaluation_authorized") is True).lower())
print(str(verification.get("policy_outcomes_read") is False).lower())
print(verification.get("benchmark_manifest_sha256"))
print(digest)
PY
)
[[ "${#gate[@]}" -eq 6 && "${gate[0]}" == true \
   && "${gate[1]}" == true && "${gate[2]}" == true \
   && "${gate[3]}" == true && "${gate[4]}" == "${gate[5]}" ]] || \
  fail "Table-2 construction power gate or manifest binding failed"

if [[ "${PHASE}" == smoke ]]; then
  RUN_ROOT=${NAVDP_SMOKE_RUN_ROOT:-${FORMAL_RUN_ROOT}/smoke/navdp}
  case "${RUN_ROOT}" in
    "${FORMAL_RUN_ROOT}"/smoke/navdp*) ;;
    *) fail "NavDP smoke root escaped the formal run" ;;
  esac
  MODE=smoke
  SCENE_INDEX=0
  MAX_STEPS=80
else
  RUN_ROOT=${FORMAL_RUN_ROOT}/formal/navdp
  MODE=eval
  SCENE_INDEX=${SLURM_ARRAY_TASK_ID}
  MAX_STEPS=600
fi

exec singularity exec --nv -B /scratch/lg154 -B /scratch/yz11502 \
  "${SIF}" env MODE="${MODE}" TASK_ROOT="${TASK_ROOT}" \
  WRAPPER_ROOT="${POLICY_LAUNCHER_ROOT}" QUERY_SOURCE_ROOT="${TASK_ROOT}" \
  RUNTIME_CLOSURE_ROOT="${RUNTIME_CLOSURE_ROOT}" \
  SERVER_SOURCE_ROOT="${SERVER_SOURCE_ROOT}" \
  SERVER_SOURCE_RECEIPT="${SERVER_SOURCE_RECEIPT}" \
  EXPECTED_SERVER_SOURCE_RECEIPT_SHA="${EXPECTED_SERVER_SOURCE_RECEIPT_SHA}" \
  BASE_SOURCE_ROOT="${BASE_SOURCE_ROOT}" RUN_ROOT="${RUN_ROOT}" \
  PARENT_MANIFEST="${PARENT_MANIFEST}" PROTOCOL="${PROTOCOL}" \
  BENCH_ROOT="${BENCH_ROOT}" SCENE_INDEX="${SCENE_INDEX}" \
  MAX_STEPS="${MAX_STEPS}" QUERY_ARMS=mono_native,mono_cec \
  HISTORY_CONTRACT=actual_ab ROLE_PAIR_SCOPE="${ROLE_PAIR_SCOPE}" \
  HAB_PY=/scratch/lg154/conda-envs/habitat/bin/python \
  MEMNAV_PY=/scratch/lg154/conda-envs/memnav/bin/python \
  TASK_RECEIPT="${TASK_RECEIPT}" \
  EXPECTED_TASK_RECEIPT_SHA="${EXPECTED_TASK_RECEIPT_SHA}" \
  BASE_RECEIPT="${BASE_RECEIPT}" \
  EXPECTED_BASE_RECEIPT_SHA="${EXPECTED_BASE_RECEIPT_SHA}" \
  bash "${RUNNER}"
