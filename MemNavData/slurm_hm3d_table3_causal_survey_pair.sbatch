#!/usr/bin/env bash
#SBATCH --job-name=h3T3SurveyPair
#SBATCH --output=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/%x_%A_%a.out
#SBATCH --error=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/%x_%A_%a.err
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=72G
#SBATCH --partition=h100_tandon,a100_tandon
#SBATCH --gres=gpu:1
#SBATCH --account=torch_pr_769_tandon_advanced

set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
: "${SLURM_ARRAY_TASK_ID:?}" "${SURVEY_SOURCE_ROOT:?}" "${RUN_ROOT:?}"
: "${SURVEY_SOURCE_RECEIPT:?}" "${EXPECTED_SURVEY_SOURCE_RECEIPT_SHA:?}"
: "${CANDIDATE_PLAN:?}" "${EXPECTED_CANDIDATE_PLAN_SHA:?}"
: "${EXPECTED_POPULATION_VERIFICATION_SHA:?}"
: "${EXPECTED_BENCHMARK_MANIFEST_SHA:?}"
: "${RUNTIME_CLOSURE_ROOT:?}" "${RUNTIME_CLOSURE_RECEIPT:?}"
: "${EXPECTED_RUNTIME_CLOSURE_RECEIPT_SHA:?}"
POPULATION_RELATIVE_ROOT=${POPULATION_RELATIVE_ROOT:-query_population}
[[ "${POPULATION_RELATIVE_ROOT}" == query_population \
   || "${POPULATION_RELATIVE_ROOT}" == merged_query_population ]] || {
  echo "invalid Table-III population root" >&2; exit 2; }
[[ "$(sha256sum "${SURVEY_SOURCE_RECEIPT}" | awk '{print $1}')" == \
   "${EXPECTED_SURVEY_SOURCE_RECEIPT_SHA}" ]]
(cd "${SURVEY_SOURCE_ROOT}" && \
  sha256sum -c --quiet "${SURVEY_SOURCE_RECEIPT}")
[[ "$(sha256sum "${RUNTIME_CLOSURE_RECEIPT}" | awk '{print $1}')" == \
   "${EXPECTED_RUNTIME_CLOSURE_RECEIPT_SHA}" ]]
(cd "${RUNTIME_CLOSURE_ROOT}" && \
  sha256sum -c --quiet "${RUNTIME_CLOSURE_RECEIPT}")
[[ "$(sha256sum "${CANDIDATE_PLAN}" | awk '{print $1}')" == \
   "${EXPECTED_CANDIDATE_PLAN_SHA}" ]]
population=${RUN_ROOT}/${POPULATION_RELATIVE_ROOT}
verification=${population}/independent_verification.json
(cd "${population}" && sha256sum -c --quiet independent_verification.json.sha256)
[[ "$(sha256sum "${verification}" | awk '{print $1}')" == \
   "${EXPECTED_POPULATION_VERIFICATION_SHA}" ]]
[[ "$(sha256sum "${population}/role_pairs/manifest.json" | awk '{print $1}')" == \
   "${EXPECTED_BENCHMARK_MANIFEST_SHA}" ]]
readarray -t row < <(/scratch/lg154/conda-envs/memnav/bin/python - \
  "${population}/role_pairs/manifest.json" "${verification}" \
  "${SLURM_ARRAY_TASK_ID}" <<'PY'
import json,math,sys
m=json.load(open(sys.argv[1])); v=json.load(open(sys.argv[2])); i=int(sys.argv[3])
assert v['verified'] is True and v['formal_policy_evaluation_authorized'] is True
assert v['history_source']=='controlled_causal_rgb_geodesic_survey'
assert 0 <= i < len(m['episodes']) == 48
r=m['episodes'][i]
distances=[float(q['geodesic_from_a_end_m']) for q in r['pairs'][0]['queries']]
budget=max(600,math.ceil(2.5*max(distances)/0.0376)); assert budget <= 3400
print(r['scene_index']); print(budget)
PY
)
[[ "${#row[@]}" -eq 2 ]]
BASE_SIF=/share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif
exec singularity exec --nv -B /scratch/lg154 -B /scratch/yz11502 \
  "${BASE_SIF}" env MODE=eval TASK_ROOT="${TASK_ROOT}" \
  WRAPPER_ROOT="${SURVEY_SOURCE_ROOT}" QUERY_SOURCE_ROOT="${SURVEY_SOURCE_ROOT}" \
  RUNTIME_CLOSURE_ROOT="${RUNTIME_CLOSURE_ROOT}" \
  SERVER_SOURCE_ROOT="${SERVER_SOURCE_ROOT}" BASE_SOURCE_ROOT="${BASE_SOURCE_ROOT}" \
  RUN_ROOT="${RUN_ROOT}" PARENT_MANIFEST="${CANDIDATE_PLAN}" \
  PROTOCOL="${SURVEY_PROTOCOL}" SCENE_INDEX="${row[0]}" \
  SCENE_RANK_FIELD=scene_index FORMAL_INDICES_OVERRIDE="${SLURM_ARRAY_TASK_ID}" \
  RUNTIME_ATTEMPT="table3Survey$(printf '%03d' "${SLURM_ARRAY_TASK_ID}")" \
  BENCH_ROOT="${population}/role_pairs" MAX_STEPS="${row[1]}" \
  HISTORY_CONTRACT=causal_survey ROLE_PAIR_SCOPE=table3_length \
  QUERY_ARMS=mono_native,mono_cec \
  HAB_PY=/scratch/lg154/conda-envs/habitat/bin/python \
  MEMNAV_PY=/scratch/lg154/conda-envs/memnav/bin/python \
  TASK_RECEIPT="${TASK_RECEIPT}" EXPECTED_TASK_RECEIPT_SHA="${EXPECTED_TASK_RECEIPT_SHA}" \
  SERVER_SOURCE_RECEIPT="${SERVER_SOURCE_RECEIPT}" \
  EXPECTED_SERVER_SOURCE_RECEIPT_SHA="${EXPECTED_SERVER_SOURCE_RECEIPT_SHA}" \
  BASE_RECEIPT="${BASE_RECEIPT}" EXPECTED_BASE_RECEIPT_SHA="${EXPECTED_BASE_RECEIPT_SHA}" \
  EXPECTED_HAB_REQUESTS_VERSION=2.32.4 EXPECTED_HAB_REQUESTS_INIT_BYTES=5057 \
  EXPECTED_HAB_REQUESTS_INIT_SHA=1e507f1f386bcc6b5f0ff69a614c14875cd65cb67be7f6022f28adef9774573f \
  EXPECTED_HAB_REQUESTS_VERSION_BYTES=435 \
  EXPECTED_HAB_REQUESTS_VERSION_SHA=143abaf3563712f063743a7952aa65319dbcb934d894cfc989bd2c015f8da577 \
  bash "${SURVEY_SOURCE_ROOT}/MemNavData/run_hm3d_fullmono_server_scene.sh"
