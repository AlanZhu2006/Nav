#!/usr/bin/env bash
# Run only inside the authenticated yz11502 Torch login shell.
set -euo pipefail
umask 0022
export PYTHONDONTWRITEBYTECODE=1

SOURCE_ROOT=${SOURCE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
SOURCE_RECEIPT=${SOURCE_ROOT}/SOURCE_BUNDLE.sha256
EXPECTED_SOURCE_RECEIPT_SHA=${EXPECTED_SOURCE_RECEIPT_SHA:?}
SOURCE_SHA=$(sha256sum "${SOURCE_RECEIPT}" | awk '{print $1}')
RUN_ROOT=${RUN_ROOT:-/scratch/yz11502/Research/Nav-axis-uturn-results/hm3d_se2_route_compass_20260902/gate_${SOURCE_SHA:0:16}}
SIF=/share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif
HAB_PY=/scratch/lg154/conda-envs/habitat/bin/python
RUNTIME=/scratch/yz11502/Research/Nav-axis-uturn-source-bundles/hm3d_table1_navdp_authority_transaction_repair_82e71f19ee7f4e52
TASK=/scratch/yz11502/Research/Nav-axis-uturn-source-bundles/hm3d_lifelong_natural_b_expansion_execution_1f4979a7fd37d467
BASE=/scratch/yz11502/Research/Nav-axis-uturn-source-bundles/final14_mono_factorial_5690569a4373f2d2
PROTOCOL_SHA=9c8d5b2da5300d34f2b042a53cc6968ec3e990dddc3cea10c2614d4d83e0d8f7
COUNTERFACTUAL_SHA=9a8f55728f4a10f5bc091005ed183ceb20dc3042bbe713b6f6a0c9034a216615
COUNTERFACTUAL=${SOURCE_ROOT}/.diagnostics/se2_projection_replay_20260902/se2_local_odometry_counterfactual_audit.json
DRY_RUN=${DRY_RUN:-0}

fail() { echo "ABORT: $*" >&2; exit 2; }
job_id() { awk -F';' 'NR==1{print $1}'; }
[[ "$(id -un)" == yz11502 ]] || fail "wrong HPC identity"
[[ "${DRY_RUN}" =~ ^[01]$ ]] || fail "DRY_RUN must be 0 or 1"
[[ "${SOURCE_SHA}" == "${EXPECTED_SOURCE_RECEIPT_SHA}" ]] || \
  fail "source receipt changed"
(cd "${SOURCE_ROOT}" && sha256sum -c --quiet "${SOURCE_RECEIPT}") || \
  fail "source bundle content changed"
[[ "$(sha256sum "${SOURCE_ROOT}/MemNavData/LONG_RANGE_SE2_PROGRESS_PROJECTION_DEV_20260902.md" | awk '{print $1}')" == "${PROTOCOL_SHA}" ]] || \
  fail "SE(2) development protocol changed"
[[ -r "${COUNTERFACTUAL}" ]] || fail "counterfactual audit is absent"
[[ "$(sha256sum "${COUNTERFACTUAL}" | awk '{print $1}')" == "${COUNTERFACTUAL_SHA}" ]] || \
  fail "counterfactual audit changed"
[[ ! -e "${RUN_ROOT}" ]] || fail "run root already exists: ${RUN_ROOT}"
grep -q 'causal_survey' \
  /scratch/yz11502/Research/Nav-axis-uturn-source-bundles/hm3d_table3_query_api_closure_e226a4b456846283/MemNavData/run_hm3d_fullmono_server_scene.sh || \
  fail "verified survey runner lacks causal_survey contract"
[[ -r /scratch/yz11502/Research/Nav-axis-uturn-source-bundles/hm3d_table3_query_api_closure_e226a4b456846283/MemNavData/slurm_port_pair.sh ]] || \
  fail "verified survey runner lacks lifetime-held port allocator"

HAB_SITE=$("${HAB_PY}" -c \
  'import sysconfig; print(sysconfig.get_paths()["purelib"])')
HAB_REQUESTS=${HAB_SITE}/pip/_vendor
[[ -r "${HAB_REQUESTS}/requests/__init__.py" ]] || \
  fail "Habitat vendored requests is unavailable"
PYTHONPATH_VALUE=${SOURCE_ROOT}:${SOURCE_ROOT}/MemNavData:${RUNTIME}:${RUNTIME}/MemNavData:${TASK}:${TASK}/MemNavData:${BASE}:${BASE}/MemNavData:${HAB_REQUESTS}
singularity exec -B /scratch/lg154 -B /scratch/yz11502 "${SIF}" \
  env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="${PYTHONPATH_VALUE}" \
  "${HAB_PY}" "${SOURCE_ROOT}/MemNavData/eval_2leg_habitat.py" \
    --episode_root /tmp/contract_only --scene /tmp/contract_only.glb \
    --out /tmp/contract_only --server_backend hybrid_pose \
    --port 18888 --novel_port 18889 \
    --hybrid_route certified_relocalization \
    --revisit_adapter verified_bearing_v1 \
    --navdp_depth_source monocular_sidecar \
    --certified_guidance_mode se2_route_compass \
    --contract_dry_run >/dev/null

source "${SOURCE_ROOT}/MemNavData/slurm_safe_submit.sh"
SBATCH=${SOURCE_ROOT}/MemNavData/slurm_hm3d_se2_route_compass_gate.sbatch
lint_sbatch_template "${SBATCH}" || fail "sbatch lint failed"
exports="ALL,SOURCE_ROOT=${SOURCE_ROOT},SOURCE_RECEIPT=${SOURCE_RECEIPT},EXPECTED_SOURCE_RECEIPT_SHA=${SOURCE_SHA},RUN_ROOT=${RUN_ROOT}"
safe_sbatch --lint-fatal --test-only \
  --partition=h100_tandon,a100_tandon \
  --account=torch_pr_769_tandon_advanced --qos=gpu48 --gres=gpu:1 \
  --cpus-per-task=10 --mem=72G --time=01:00:00 \
  --export="${exports}" "${SBATCH}" >/dev/null
if [[ "${DRY_RUN}" == 1 ]]; then
  echo "DRY_RUN_OK index=40 time=01:00:00 source=${SOURCE_SHA}"
  exit 0
fi

mkdir -p "${RUN_ROOT}"
job=$(safe_sbatch --lint-fatal --parsable --job-name=h3SE2RouteGate \
  --partition=h100_tandon,a100_tandon \
  --account=torch_pr_769_tandon_advanced --qos=gpu48 --gres=gpu:1 \
  --cpus-per-task=10 --mem=72G --time=01:00:00 \
  --export="${exports}" "${SBATCH}" | job_id)
[[ "${job}" =~ ^[0-9]+$ ]] || fail "invalid submitted job id: ${job}"

"/scratch/lg154/conda-envs/memnav/bin/python" - \
  "${RUN_ROOT}/submission.json" "${job}" "${SOURCE_ROOT}" "${SOURCE_SHA}" \
  "${PROTOCOL_SHA}" "${COUNTERFACTUAL_SHA}" <<'PY'
import datetime,json,os,sys
path,job,source,source_sha,protocol_sha,counterfactual_sha=sys.argv[1:]
payload={
 'schema_version':'hm3d_se2_route_compass_gate_submission_v1_20260902',
 'submitted_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),
 'job_id':int(job), 'history_index':40,
 'partition':['h100_tandon','a100_tandon'], 'qos':'gpu48',
 'requested_time':'01:00:00', 'query_step_guard':80,
 'source_root':source, 'source_receipt_sha256':source_sha,
 'protocol_sha256':protocol_sha,
 'counterfactual_audit_sha256':counterfactual_sha,
 'benchmark_manifest_sha256':'cbc518cea991fd252893f97fd5e730c277e4d899369932536a745351d47e7451',
 'stage':'SR-hidden local-SE2 route-compass runtime gate',
 'executor_odometry_source':'habitat_pose_difference_odometry_proxy_v1',
 'world_pose_consumed_by_policy':False,
 'pure_rgb_claim_authorized':False,
 'navigation_success_read_by_gate':False,
 'navigation_final_distance_read_by_gate':False,
 'canonical_cec_default_changed':False,
 'scalar_action_coordinate_mode_changed':False,
}
fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o444)
with os.fdopen(fd,'w') as handle:
    json.dump(payload,handle,indent=2,sort_keys=True); handle.write('\n')
PY
sha256sum "${RUN_ROOT}/submission.json" \
  >"${RUN_ROOT}/submission.json.sha256"
chmod a-w "${RUN_ROOT}/submission.json" "${RUN_ROOT}/submission.json.sha256"
printf 'JOB_ID=%s\nRUN_ROOT=%s\nSOURCE_SHA256=%s\n' \
  "${job}" "${RUN_ROOT}" "${SOURCE_SHA}"
