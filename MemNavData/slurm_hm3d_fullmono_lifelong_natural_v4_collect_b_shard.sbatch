#!/usr/bin/env bash
#SBATCH --job-name=h3lifeV4B
#SBATCH --output=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/%x_%A_%a.out
#SBATCH --error=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/%x_%A_%a.err
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=72G
#SBATCH --partition=h100_tandon,a100_tandon
#SBATCH --gres=gpu:1
#SBATCH --account=torch_pr_769_tandon_advanced

set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
: "${SLURM_ARRAY_TASK_ID:?factual-B shard index required}"
: "${TASK_ROOT:?}" "${SERVER_SOURCE_ROOT:?}" "${BASE_SOURCE_ROOT:?}"
: "${RUN_ROOT:?}" "${PARENT_ROOT:?}" "${PROTOCOL:?}"
: "${SHARD_MANIFEST:?}" "${EXPECTED_SHARD_MANIFEST_SHA:?}"
: "${TASK_RECEIPT:?}" "${EXPECTED_TASK_RECEIPT_SHA:?}"
: "${SERVER_SOURCE_RECEIPT:?}" "${EXPECTED_SERVER_SOURCE_RECEIPT_SHA:?}"
: "${BASE_RECEIPT:?}" "${EXPECTED_BASE_RECEIPT_SHA:?}"
RUNTIME_ATTEMPT_PREFIX=${RUNTIME_ATTEMPT_PREFIX:-shard}
[[ "${RUNTIME_ATTEMPT_PREFIX}" =~ ^[A-Za-z][A-Za-z0-9_.-]*$ ]]
[[ "$(sha256sum "${SHARD_MANIFEST}" | awk '{print $1}')" == \
   "${EXPECTED_SHARD_MANIFEST_SHA}" ]]
(cd "$(dirname "${SHARD_MANIFEST}")" && \
  sha256sum -c --quiet "$(basename "${SHARD_MANIFEST}").sha256")
readarray -t shard < <(/scratch/lg154/conda-envs/memnav/bin/python - \
  "${SHARD_MANIFEST}" "${SLURM_ARRAY_TASK_ID}" <<'PY'
import json, sys
p=json.load(open(sys.argv[1])); index=int(sys.argv[2])
assert p['all_candidates_partitioned_once'] is True
assert p['query_policy_outcomes_read'] is False
assert p['navigation_outcomes_read'] is False
assert 0 <= index < p['shard_count'] == len(p['shards'])
row=p['shards'][index]
assert row['shard_index'] == index
assert row['history_count'] == len(row['history_indices'])
assert 0 < row['history_count'] <= p['maximum_histories_per_shard']
assert row['navigation_outcomes_read'] is False
print(row['scene_index'])
print(' '.join(str(x) for x in row['history_indices']))
PY
)
[[ "${#shard[@]}" -eq 2 && -n "${shard[1]}" ]]
SCENE_INDEX=${shard[0]}
FORMAL_INDICES_OVERRIDE=${shard[1]}
BASE_SIF=/share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif
exec singularity exec --nv -B /scratch/lg154 -B /scratch/yz11502 \
  "${BASE_SIF}" env MODE=lifelong_b TASK_ROOT="${TASK_ROOT}" \
  SERVER_SOURCE_ROOT="${SERVER_SOURCE_ROOT}" \
  BASE_SOURCE_ROOT="${BASE_SOURCE_ROOT}" RUN_ROOT="${RUN_ROOT}" \
  PARENT_MANIFEST="${PARENT_ROOT}/sealed_inputs/parent_manifest.json" \
  PROTOCOL="${PROTOCOL}" SCENE_INDEX="${SCENE_INDEX}" \
  FORMAL_INDICES_OVERRIDE="${FORMAL_INDICES_OVERRIDE}" \
  RUNTIME_ATTEMPT="$(printf '%s%03d' "${RUNTIME_ATTEMPT_PREFIX}" \
    "${SLURM_ARRAY_TASK_ID}")" \
  BENCH_ROOT="${RUN_ROOT}/ab_population/role_pairs" MAX_STEPS=600 \
  HAB_PY=/scratch/lg154/conda-envs/habitat/bin/python \
  MEMNAV_PY=/scratch/lg154/conda-envs/memnav/bin/python \
  TASK_RECEIPT="${TASK_RECEIPT}" \
  EXPECTED_TASK_RECEIPT_SHA="${EXPECTED_TASK_RECEIPT_SHA}" \
  SERVER_SOURCE_RECEIPT="${SERVER_SOURCE_RECEIPT}" \
  EXPECTED_SERVER_SOURCE_RECEIPT_SHA="${EXPECTED_SERVER_SOURCE_RECEIPT_SHA}" \
  BASE_RECEIPT="${BASE_RECEIPT}" \
  EXPECTED_BASE_RECEIPT_SHA="${EXPECTED_BASE_RECEIPT_SHA}" \
  EXPECTED_HAB_REQUESTS_VERSION=2.32.4 \
  EXPECTED_HAB_REQUESTS_INIT_BYTES=5057 \
  EXPECTED_HAB_REQUESTS_INIT_SHA=1e507f1f386bcc6b5f0ff69a614c14875cd65cb67be7f6022f28adef9774573f \
  EXPECTED_HAB_REQUESTS_VERSION_BYTES=435 \
  EXPECTED_HAB_REQUESTS_VERSION_SHA=143abaf3563712f063743a7952aa65319dbcb934d894cfc989bd2c015f8da577 \
  bash "${TASK_ROOT}/MemNavData/run_hm3d_fullmono_server_scene.sh"
