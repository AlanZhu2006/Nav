#!/usr/bin/env bash
#SBATCH --job-name=h3RearAlign
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

: "${SLURM_ARRAY_TASK_ID:?}" "${PHASE:?gate or formal}"
: "${SOURCE_ROOT:?}" "${SOURCE_RECEIPT:?}"
: "${EXPECTED_SOURCE_RECEIPT_SHA:?}" "${RUN_ROOT:?}"

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
[[ "${PHASE}" == gate || "${PHASE}" == formal ]] || fail "bad phase"
case ",3,4,5,8,10,11,13,15,21," in
  *,"${SLURM_ARRAY_TASK_ID}",*) ;;
  *) fail "history index escaped the frozen consumed subset" ;;
esac
verify_bundle "${SOURCE_ROOT}" "${SOURCE_RECEIPT}" \
  "${EXPECTED_SOURCE_RECEIPT_SHA}" rear_alignment

TASK=/scratch/yz11502/Research/Nav-axis-uturn-source-bundles/hm3d_lifelong_natural_b_expansion_execution_1f4979a7fd37d467
TASK_SHA=1f4979a7fd37d46700011558063be34a8fba0a0b8746668469dba7e7955f4282
BASE=/scratch/yz11502/Research/Nav-axis-uturn-source-bundles/final14_mono_factorial_5690569a4373f2d2
BASE_RECEIPT=${BASE}/source_inputs.sha256
BASE_SHA=5690569a4373f2d2768671418f0c604c4a03aa4b0ffe01baf70b288af03ba216
RUNTIME=/scratch/yz11502/Research/Nav-axis-uturn-source-bundles/hm3d_table1_navdp_authority_transaction_repair_82e71f19ee7f4e52
RUNTIME_SHA=82e71f19ee7f4e5233fae499633ce5a233c9c036bb41b9e2bf7d4f0f18effd7d
verify_bundle "${TASK}" "${TASK}/SOURCE_BUNDLE.sha256" "${TASK_SHA}" task
verify_bundle "${BASE}" "${BASE_RECEIPT}" "${BASE_SHA}" base
verify_bundle "${RUNTIME}" "${RUNTIME}/SOURCE_BUNDLE.sha256" \
  "${RUNTIME_SHA}" runtime

FORMAL_PARENT=/scratch/yz11502/Research/Nav-axis-uturn-results/hm3d_longrange_route_tangent_20260903/fresh_751efdcbb436c99b
POPULATION=${FORMAL_PARENT}/fresh_population
MANIFEST=${POPULATION}/role_pairs/manifest.json
PROTOCOL=${SOURCE_ROOT}/MemNavData/hm3d_longrange_rear_alignment_protocol_20260904.json
PROTOCOL_SHA=f61dbd3021df5ff8603b68666ead03863d77a0f91d5f760301f28b2a596dbe4f
SUMMARY_SHA=b3f55c8bf90f1897556931f05795dd94787864a0682ec4b6ba59995b7bea421a
VERIFY_SHA=3cb3d3c77d9de2a013fe223183750ca3588c595b83dbf33f3b67f18c26e2c2d5
MANIFEST_SHA=0491632af1925fb14fe0bf69d9aaf8d769590552f343df38e2e30a0380655a47
[[ "$(sha256sum "${PROTOCOL}" | awk '{print $1}')" == "${PROTOCOL_SHA}" ]] || \
  fail "rear-alignment protocol changed"
[[ "$(sha256sum "${MANIFEST}" | awk '{print $1}')" == "${MANIFEST_SHA}" ]] || \
  fail "formal benchmark manifest changed"
[[ "$(sha256sum "${FORMAL_PARENT}/result/summary.json" | awk '{print $1}')" == \
   "${SUMMARY_SHA}" ]] || fail "formal summary changed"
[[ "$(sha256sum "${FORMAL_PARENT}/result/independent_verification.json" | awk '{print $1}')" == \
   "${VERIFY_SHA}" ]] || fail "formal verifier changed"

MEMNAV_PY=/scratch/lg154/conda-envs/memnav/bin/python
readarray -t row < <("${MEMNAV_PY}" - "${MANIFEST}" "${PROTOCOL}" \
  "${SLURM_ARRAY_TASK_ID}" <<'PY'
import json, math, sys
m=json.load(open(sys.argv[1])); p=json.load(open(sys.argv[2])); i=int(sys.argv[3])
assert i in p['selected_history_indices']
assert p['outcomes_read_before_design'] is True
assert 0 <= i < len(m['episodes']) == 23
r=m['episodes'][i]
assert r['longrange_route_tangent_fresh'] is True
assert float(r['revisit_vertical_error_m']) <= 0.5 + 1e-12
q=[q for pair in r['pairs'] for q in pair['queries']
   if q['analysis_role']=='revisit']
assert len(q)==1 and 20.0 <= float(q[0]['geodesic_from_a_end_m']) < 30.0
budget=max(600, math.ceil(2.5*float(q[0]['geodesic_from_a_end_m'])/0.0376))
assert 600 <= budget <= 2000
print(r['scene_index']); print(budget)
PY
)
[[ "${#row[@]}" -eq 2 ]] || fail "manifest row resolution failed"

SIF=/share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif
HAB_PY=/scratch/lg154/conda-envs/habitat/bin/python
BUFFER=${SLURM_TMPDIR:?}/cec_rear_alignment_buffer
mkdir -p "${BUFFER}"
exec singularity exec --nv -B /scratch/lg154 -B /scratch/yz11502 \
  "${SIF}" env MODE=eval TASK_ROOT="${TASK}" \
  WRAPPER_ROOT="${SOURCE_ROOT}" QUERY_SOURCE_ROOT="${SOURCE_ROOT}" \
  QUERY_RUNNER="${SOURCE_ROOT}/MemNavData/run_hm3d_longrange_rear_alignment_history.py" \
  RUNTIME_CLOSURE_ROOT="${RUNTIME}" SERVER_SOURCE_ROOT="${SOURCE_ROOT}" \
  BASE_SOURCE_ROOT="${BASE}" RUN_ROOT="${RUN_ROOT}" \
  PARENT_MANIFEST="${MANIFEST}" PROTOCOL="${PROTOCOL}" \
  EXPECTED_PROTOCOL_SHA256="${PROTOCOL_SHA}" \
  SCENE_INDEX="${row[0]}" SCENE_RANK_FIELD=scene_index \
  FORMAL_INDICES_OVERRIDE="${SLURM_ARRAY_TASK_ID}" \
  RUNTIME_ATTEMPT="rearAlign${PHASE}$(printf '%03d' "${SLURM_ARRAY_TASK_ID}")" \
  BENCH_ROOT="${POPULATION}/role_pairs" MAX_STEPS="${row[1]}" \
  HISTORY_CONTRACT=causal_survey \
  ROLE_PAIR_SCOPE=table3_longrange_route_tangent \
  QUERY_ARMS=route_tangent_unaligned,route_tangent_rear_aligned \
  CERTIFIED_ROUTE_DEPTH_CACHE_STRIDE=8 MEMNAV_BUFFER_ROOT="${BUFFER}" \
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
