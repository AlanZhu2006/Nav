#!/usr/bin/env bash
#SBATCH --job-name=h3VisRoute
#SBATCH --output=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/%x_%A_%a.out
#SBATCH --error=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/%x_%A_%a.err
#SBATCH --time=02:00:00
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

: "${SLURM_ARRAY_TASK_ID:?submit indices 0-47}"
: "${SOURCE_ROOT:?}" "${SOURCE_RECEIPT:?}" "${EXPECTED_SOURCE_RECEIPT_SHA:?}"
: "${RUN_ROOT:?}"

SURVEY=/scratch/yz11502/Research/Nav-axis-uturn-source-bundles/hm3d_table3_query_api_closure_e226a4b456846283
SURVEY_SHA=e226a4b4568462831e726a4f486f485233dc4c90bfef72012898cc34ba4744e6
TASK=/scratch/yz11502/Research/Nav-axis-uturn-source-bundles/hm3d_lifelong_natural_b_expansion_execution_1f4979a7fd37d467
TASK_SHA=1f4979a7fd37d46700011558063be34a8fba0a0b8746668469dba7e7955f4282
BASE=/scratch/yz11502/Research/Nav-axis-uturn-source-bundles/final14_mono_factorial_5690569a4373f2d2
BASE_RECEIPT=${BASE}/source_inputs.sha256
BASE_SHA=5690569a4373f2d2768671418f0c604c4a03aa4b0ffe01baf70b288af03ba216
RUNTIME=/scratch/yz11502/Research/Nav-axis-uturn-source-bundles/hm3d_table1_navdp_authority_transaction_repair_82e71f19ee7f4e52
RUNTIME_SHA=82e71f19ee7f4e5233fae499633ce5a233c9c036bb41b9e2bf7d4f0f18effd7d
PARENT=/scratch/yz11502/Research/Nav-axis-uturn-results/hm3d_table3_causal_survey_expansion_20260831/expansion_20260830T165729Z_a8dd232e
POP=${PARENT}/merged_query_population
MANIFEST=${POP}/role_pairs/manifest.json
MANIFEST_SHA=cbc518cea991fd252893f97fd5e730c277e4d899369932536a745351d47e7451
POP_VERIFY=${POP}/independent_verification.json
POP_VERIFY_SHA=8398456ec71503698b3cef60f3b49be6917111d7c76b5e038279dff44b0b4ace
PROTOCOL=${SURVEY}/MemNavData/hm3d_table3_causal_survey_protocol_20260830.json
SIF=/share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif
HAB_PY=/scratch/lg154/conda-envs/habitat/bin/python
MEMNAV_PY=/scratch/lg154/conda-envs/memnav/bin/python

fail() { echo "ABORT: $*" >&2; exit 2; }
verify_bundle() {
  local root=$1 receipt=$2 expected=$3 label=$4
  [[ -d "${root}" && -f "${receipt}" ]] || fail "missing ${label} bundle"
  [[ "$(sha256sum "${receipt}" | awk '{print $1}')" == "${expected}" ]] || \
    fail "${label} receipt changed"
  (cd "${root}" && sha256sum -c --quiet "${receipt}") || \
    fail "${label} bundle content changed"
}

[[ "$(id -un)" == yz11502 ]] || fail "wrong HPC identity"
[[ "${SLURM_ARRAY_TASK_ID}" =~ ^([0-9]|[1-3][0-9]|4[0-7])$ ]] || \
  fail "array index escaped frozen 0-47 population"
verify_bundle "${SOURCE_ROOT}" "${SOURCE_RECEIPT}" \
  "${EXPECTED_SOURCE_RECEIPT_SHA}" visual_route
verify_bundle "${SURVEY}" "${SURVEY}/SOURCE_BUNDLE.sha256" \
  "${SURVEY_SHA}" survey
verify_bundle "${TASK}" "${TASK}/SOURCE_BUNDLE.sha256" "${TASK_SHA}" task
verify_bundle "${BASE}" "${BASE_RECEIPT}" "${BASE_SHA}" base
verify_bundle "${RUNTIME}" "${RUNTIME}/SOURCE_BUNDLE.sha256" \
  "${RUNTIME_SHA}" runtime
[[ "$(sha256sum "${MANIFEST}" | awk '{print $1}')" == "${MANIFEST_SHA}" ]] || \
  fail "benchmark manifest changed"
[[ "$(sha256sum "${POP_VERIFY}" | awk '{print $1}')" == "${POP_VERIFY_SHA}" ]] || \
  fail "population verification changed"
(cd "${POP}" && sha256sum -c --quiet independent_verification.json.sha256) || \
  fail "population verification sidecar changed"

readarray -t row < <("${MEMNAV_PY}" - "${MANIFEST}" \
  "${SLURM_ARRAY_TASK_ID}" <<'PY'
import json,math,sys
m=json.load(open(sys.argv[1])); i=int(sys.argv[2])
assert len(m['episodes']) == 48 and 0 <= i < 48
r=m['episodes'][i]
distances=[float(q['geodesic_from_a_end_m']) for q in r['pairs'][0]['queries']]
budget=max(600, math.ceil(2.5*max(distances)/0.0376))
assert 600 <= budget <= 3400
print(r['scene_index']); print(budget)
PY
)
[[ "${#row[@]}" -eq 2 ]] || fail "manifest row resolution failed"
label=$(printf '%03d' "${SLURM_ARRAY_TASK_ID}")
if compgen -G "${RUN_ROOT}/development/monotone_visual_route/${label}_*" \
    >/dev/null; then
  fail "development output already exists for index ${SLURM_ARRAY_TASK_ID}"
fi

exec singularity exec --nv -B /scratch/lg154 -B /scratch/yz11502 \
  "${SIF}" env MODE=eval TASK_ROOT="${TASK}" \
  WRAPPER_ROOT="${SURVEY}" QUERY_SOURCE_ROOT="${SOURCE_ROOT}" \
  QUERY_RUNNER="${SOURCE_ROOT}/MemNavData/run_hm3d_monotone_visual_route_dev.py" \
  RUNTIME_CLOSURE_ROOT="${RUNTIME}" SERVER_SOURCE_ROOT="${SOURCE_ROOT}" \
  BASE_SOURCE_ROOT="${BASE}" RUN_ROOT="${RUN_ROOT}" \
  PARENT_MANIFEST="${MANIFEST}" PROTOCOL="${PROTOCOL}" \
  SCENE_INDEX="${row[0]}" SCENE_RANK_FIELD=scene_index \
  FORMAL_INDICES_OVERRIDE="${SLURM_ARRAY_TASK_ID}" \
  RUNTIME_ATTEMPT="visRoute${label}" BENCH_ROOT="${POP}/role_pairs" \
  MAX_STEPS="${row[1]}" HISTORY_CONTRACT=causal_survey \
  ROLE_PAIR_SCOPE=table3_length CERTIFIED_ROUTE_DEPTH_CACHE_STRIDE=8 \
  HAB_PY="${HAB_PY}" MEMNAV_PY="${MEMNAV_PY}" \
  TASK_RECEIPT="${TASK}/SOURCE_BUNDLE.sha256" \
  EXPECTED_TASK_RECEIPT_SHA="${TASK_SHA}" \
  SERVER_SOURCE_RECEIPT="${SOURCE_RECEIPT}" \
  EXPECTED_SERVER_SOURCE_RECEIPT_SHA="${EXPECTED_SOURCE_RECEIPT_SHA}" \
  BASE_RECEIPT="${BASE_RECEIPT}" EXPECTED_BASE_RECEIPT_SHA="${BASE_SHA}" \
  EXPECTED_HAB_REQUESTS_VERSION=2.32.4 EXPECTED_HAB_REQUESTS_INIT_BYTES=5057 \
  EXPECTED_HAB_REQUESTS_INIT_SHA=1e507f1f386bcc6b5f0ff69a614c14875cd65cb67be7f6022f28adef9774573f \
  EXPECTED_HAB_REQUESTS_VERSION_BYTES=435 \
  EXPECTED_HAB_REQUESTS_VERSION_SHA=143abaf3563712f063743a7952aa65319dbcb934d894cfc989bd2c015f8da577 \
  bash "${SOURCE_ROOT}/MemNavData/run_hm3d_fullmono_server_scene.sh"
