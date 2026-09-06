#!/usr/bin/env bash
#SBATCH --job-name=h3T1MetFix
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

: "${SLURM_ARRAY_TASK_ID:?submit exact scene-rank index 35}"
: "${QUERY_REPAIR_ROOT:?}" "${QUERY_REPAIR_RECEIPT:?}"
: "${EXPECTED_QUERY_REPAIR_RECEIPT_SHA:?}"
: "${TASK_ROOT:?}" "${TASK_RECEIPT:?}" "${EXPECTED_TASK_RECEIPT_SHA:?}"
: "${BASE_SOURCE_ROOT:?}" "${BASE_RECEIPT:?}"
: "${EXPECTED_BASE_RECEIPT_SHA:?}" "${PARENT_MANIFEST:?}"
: "${BENCH_ROOT:?}" "${PROTOCOL:?}" "${CONSTRUCTION_VERIFICATION:?}"
: "${EXPECTED_CONSTRUCTION_VERIFICATION_SHA:?}" "${FORMAL_RUN_ROOT:?}"

[[ "${SLURM_ARRAY_TASK_ID}" == 35 ]] || {
  echo "exact retry is sealed to scene-rank 35" >&2
  exit 2
}
[[ "$(sha256sum "${QUERY_REPAIR_RECEIPT}" | awk '{print $1}')" == \
   "${EXPECTED_QUERY_REPAIR_RECEIPT_SHA}" ]]
(cd "${QUERY_REPAIR_ROOT}" && \
  sha256sum -c --quiet "${QUERY_REPAIR_RECEIPT}")
[[ "$(sha256sum "${TASK_RECEIPT}" | awk '{print $1}')" == \
   "${EXPECTED_TASK_RECEIPT_SHA}" ]]
(cd "${TASK_ROOT}" && sha256sum -c --quiet "${TASK_RECEIPT}")
[[ "$(sha256sum "${BASE_RECEIPT}" | awk '{print $1}')" == \
   "${EXPECTED_BASE_RECEIPT_SHA}" ]]
(cd "${BASE_SOURCE_ROOT}" && sha256sum -c --quiet "${BASE_RECEIPT}")
[[ "$(sha256sum "${CONSTRUCTION_VERIFICATION}" | awk '{print $1}')" == \
   "${EXPECTED_CONSTRUCTION_VERIFICATION_SHA}" ]]

readarray -t gate < <(/scratch/lg154/conda-envs/memnav/bin/python - \
  "${CONSTRUCTION_VERIFICATION}" "${BENCH_ROOT}/manifest.json" <<'PY'
import hashlib,json,sys
verification=json.load(open(sys.argv[1])); manifest=sys.argv[2]
digest=hashlib.sha256(open(manifest,"rb").read()).hexdigest()
print(str(verification.get("verified") is True).lower())
print(str(verification.get("formal_policy_evaluation_authorized") is True).lower())
print(verification.get("benchmark_manifest_sha256"))
print(digest)
print(len(json.load(open(manifest))["episodes"]))
PY
)
[[ "${#gate[@]}" -eq 5 && "${gate[0]}" == true \
   && "${gate[1]}" == true && "${gate[2]}" == "${gate[3]}" \
   && "${gate[4]}" == 28 ]]

SIF=/share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif
RUNNER=${TASK_ROOT}/MemNavData/run_hm3d_fullmono_server_scene.sh
QUERY_RUNNER=${QUERY_REPAIR_ROOT}/MemNavData/run_hm3d_table1_metric_distance_revisit_pair.py

exec singularity exec --nv -B /scratch/lg154 -B /scratch/yz11502 \
  "${SIF}" env MODE=eval TASK_ROOT="${TASK_ROOT}" \
  WRAPPER_ROOT="${QUERY_REPAIR_ROOT}" QUERY_SOURCE_ROOT="${TASK_ROOT}" \
  QUERY_RUNNER="${QUERY_RUNNER}" SERVER_SOURCE_ROOT="${TASK_ROOT}" \
  BASE_SOURCE_ROOT="${BASE_SOURCE_ROOT}" \
  RUN_ROOT="${FORMAL_RUN_ROOT}/formal" PARENT_MANIFEST="${PARENT_MANIFEST}" \
  PROTOCOL="${PROTOCOL}" BENCH_ROOT="${BENCH_ROOT}" \
  SCENE_INDEX=35 SCENE_RANK_FIELD=final14_scene_rank MAX_STEPS=600 \
  QUERY_ARMS=fixed_2p5m,full_metric HISTORY_CONTRACT=goal_a \
  ROLE_PAIR_SCOPE=consumed_integration RUNTIME_ATTEMPT=metricAuditRepair001 \
  HAB_PY=/scratch/lg154/conda-envs/habitat/bin/python \
  MEMNAV_PY=/scratch/lg154/conda-envs/memnav/bin/python \
  TASK_RECEIPT="${TASK_RECEIPT}" \
  EXPECTED_TASK_RECEIPT_SHA="${EXPECTED_TASK_RECEIPT_SHA}" \
  BASE_RECEIPT="${BASE_RECEIPT}" \
  EXPECTED_BASE_RECEIPT_SHA="${EXPECTED_BASE_RECEIPT_SHA}" \
  bash "${RUNNER}"
