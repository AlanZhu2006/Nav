#!/usr/bin/env bash
#SBATCH --job-name=h3AuthSpec
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

set -euo pipefail
umask 0022
export PYTHONDONTWRITEBYTECODE=1

: "${SLURM_ARRAY_TASK_ID:?submit as a history array}"
: "${PHASE:?smoke or formal}" "${TASK_ROOT:?}" "${TASK_RECEIPT:?}"
: "${EXPECTED_TASK_RECEIPT_SHA:?}" "${BASE_SOURCE_ROOT:?}"
: "${BASE_RECEIPT:?}" "${EXPECTED_BASE_RECEIPT_SHA:?}"
: "${SERVER_SOURCE_ROOT:?}" "${SERVER_SOURCE_RECEIPT:?}"
: "${EXPECTED_SERVER_SOURCE_RECEIPT_SHA:?}" "${RUNTIME_CLOSURE_ROOT:?}"
: "${RUNTIME_CLOSURE_RECEIPT:?}" "${EXPECTED_RUNTIME_CLOSURE_RECEIPT_SHA:?}"
: "${BENCH_ROOT:?}" "${PARENT_MANIFEST:?}" "${FORMAL_RUN_ROOT:?}"
: "${CONSTRUCTION_VERIFICATION:?}" "${EXPECTED_CONSTRUCTION_VERIFICATION_SHA:?}"

fail() { echo "ABORT: $*" >&2; exit 2; }
[[ "${PHASE}" == smoke || "${PHASE}" == formal ]] || fail "bad phase"
[[ "${SLURM_ARRAY_TASK_ID}" =~ ^([0-9]|1[0-9]|2[0-7])$ ]] || \
  fail "history index outside frozen 0..27"

for spec in \
  "${TASK_ROOT}:${TASK_RECEIPT}:${EXPECTED_TASK_RECEIPT_SHA}" \
  "${BASE_SOURCE_ROOT}:${BASE_RECEIPT}:${EXPECTED_BASE_RECEIPT_SHA}" \
  "${SERVER_SOURCE_ROOT}:${SERVER_SOURCE_RECEIPT}:${EXPECTED_SERVER_SOURCE_RECEIPT_SHA}" \
  "${RUNTIME_CLOSURE_ROOT}:${RUNTIME_CLOSURE_RECEIPT}:${EXPECTED_RUNTIME_CLOSURE_RECEIPT_SHA}"; do
  IFS=: read -r root receipt expected <<<"${spec}"
  [[ "$(sha256sum "${receipt}" | awk '{print $1}')" == "${expected}" ]] || \
    fail "bundle receipt changed: ${root}"
  (cd "${root}" && sha256sum -c --quiet "${receipt}") || \
    fail "bundle contents changed: ${root}"
done
[[ "$(sha256sum "${CONSTRUCTION_VERIFICATION}" | awk '{print $1}')" == \
   "${EXPECTED_CONSTRUCTION_VERIFICATION_SHA}" ]] || \
  fail "construction verification changed"

HAB_PY=/scratch/lg154/conda-envs/habitat/bin/python
scene_rank=$(${HAB_PY} - "${BENCH_ROOT}/manifest.json" \
  "${SLURM_ARRAY_TASK_ID}" <<'PY'
import json,sys
row=json.load(open(sys.argv[1]))["episodes"][int(sys.argv[2])]
print(int(row["final14_scene_rank"]))
PY
)
[[ "${scene_rank}" =~ ^[0-9]+$ ]] || fail "invalid scene rank"

if [[ "${PHASE}" == smoke ]]; then
  mode=smoke
  run_root=${FORMAL_RUN_ROOT}/smoke
  max_steps=80
else
  mode=eval
  run_root=${FORMAL_RUN_ROOT}/formal
  max_steps=600
fi

SIF=/share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif
exec singularity exec --nv -B /scratch/lg154 -B /scratch/yz11502 \
  "${SIF}" env \
  MODE="${mode}" TASK_ROOT="${TASK_ROOT}" WRAPPER_ROOT="${TASK_ROOT}" \
  QUERY_SOURCE_ROOT="${RUNTIME_CLOSURE_ROOT}" \
  QUERY_RUNNER="${TASK_ROOT}/MemNavData/run_hm3d_table1_authority_spectrum_history.py" \
  RUNTIME_CLOSURE_ROOT="${RUNTIME_CLOSURE_ROOT}" \
  SERVER_SOURCE_ROOT="${SERVER_SOURCE_ROOT}" \
  SERVER_SOURCE_RECEIPT="${SERVER_SOURCE_RECEIPT}" \
  EXPECTED_SERVER_SOURCE_RECEIPT_SHA="${EXPECTED_SERVER_SOURCE_RECEIPT_SHA}" \
  BASE_SOURCE_ROOT="${BASE_SOURCE_ROOT}" RUN_ROOT="${run_root}" \
  RUNTIME_ATTEMPT="history_${SLURM_ARRAY_TASK_ID}" \
  PARENT_MANIFEST="${PARENT_MANIFEST}" \
  PROTOCOL="${TASK_ROOT}/MemNavData/hm3d_table1_authority_spectrum_protocol_20260904.json" \
  BENCH_ROOT="${BENCH_ROOT}" SCENE_INDEX="${scene_rank}" \
  SCENE_RANK_FIELD=final14_scene_rank \
  FORMAL_INDICES_OVERRIDE="${SLURM_ARRAY_TASK_ID}" \
  HISTORY_CONTRACT=goal_a ROLE_PAIR_SCOPE=paper_heldout \
  QUERY_ARMS=mono_native,mono_raw_fixed,mono_unthresholded_witness,mono_cec \
  MAX_STEPS="${max_steps}" \
  HAB_PY="${HAB_PY}" MEMNAV_PY=/scratch/lg154/conda-envs/memnav/bin/python \
  TASK_RECEIPT="${TASK_RECEIPT}" \
  EXPECTED_TASK_RECEIPT_SHA="${EXPECTED_TASK_RECEIPT_SHA}" \
  BASE_RECEIPT="${BASE_RECEIPT}" \
  EXPECTED_BASE_RECEIPT_SHA="${EXPECTED_BASE_RECEIPT_SHA}" \
  bash "${TASK_ROOT}/MemNavData/run_hm3d_fullmono_server_scene.sh"
