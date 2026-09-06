#!/usr/bin/env bash
# Submit only after the sealed SR-hidden controller gate has passed.
set -euo pipefail
umask 0022
export PYTHONDONTWRITEBYTECODE=1

SOURCE_ROOT=${SOURCE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
SOURCE_RECEIPT=${SOURCE_ROOT}/SOURCE_BUNDLE.sha256
EXPECTED_SOURCE_RECEIPT_SHA=${EXPECTED_SOURCE_RECEIPT_SHA:?}
SOURCE_SHA=$(sha256sum "${SOURCE_RECEIPT}" | awk '{print $1}')
GATE_ROOT=${GATE_ROOT:?set the completed action-coordinate gate root}
RUN_ROOT=${RUN_ROOT:-/scratch/yz11502/Research/Nav-axis-uturn-results/hm3d_action_coordinate_compass_20260902/pair_${SOURCE_SHA:0:16}}
SIF=/share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif
HAB_PY=/scratch/lg154/conda-envs/habitat/bin/python
RUNTIME=/scratch/yz11502/Research/Nav-axis-uturn-source-bundles/hm3d_table1_navdp_authority_transaction_repair_82e71f19ee7f4e52
TASK=/scratch/yz11502/Research/Nav-axis-uturn-source-bundles/hm3d_lifelong_natural_b_expansion_execution_1f4979a7fd37d467
BASE=/scratch/yz11502/Research/Nav-axis-uturn-source-bundles/final14_mono_factorial_5690569a4373f2d2
PROTOCOL_SHA=d5b004825f5af2c6e3faa8485fdc605ceda939229c0a2e27db27a994fd4d1305
DRY_RUN=${DRY_RUN:-0}

fail() { echo "ABORT: $*" >&2; exit 2; }
job_id() { awk -F';' 'NR==1{print $1}'; }
[[ "$(id -un)" == yz11502 ]] || fail "wrong HPC identity"
[[ "${DRY_RUN}" =~ ^[01]$ ]] || fail "DRY_RUN must be 0 or 1"
[[ "${SOURCE_SHA}" == "${EXPECTED_SOURCE_RECEIPT_SHA}" ]] || \
  fail "source receipt changed"
(cd "${SOURCE_ROOT}" && sha256sum -c --quiet "${SOURCE_RECEIPT}") || \
  fail "source bundle content changed"
[[ "$(sha256sum "${SOURCE_ROOT}/MemNavData/LONG_RANGE_ACTION_COORDINATE_COMPASS_PROTOCOL_20260902.md" | awk '{print $1}')" == "${PROTOCOL_SHA}" ]] || \
  fail "frozen scientific protocol changed"
[[ ! -e "${RUN_ROOT}" ]] || fail "run root already exists: ${RUN_ROOT}"

GATE_SOURCE_SHA=$("/scratch/lg154/conda-envs/memnav/bin/python" - "${GATE_ROOT}" <<'PY'
import hashlib,json,sys
from pathlib import Path
root=Path(sys.argv[1])
submission=root/'submission.json'
submission_digest=hashlib.sha256(submission.read_bytes()).hexdigest()
submission_sidecar=(root/'submission.json.sha256').read_text().split()
assert submission_sidecar and submission_sidecar[0] == submission_digest
submission_row=json.loads(submission.read_text())
paths=sorted(root.glob('deployment_gate/040_*/gate_completion.json'))
assert len(paths) == 1, f'expected one index-40 gate, found {len(paths)}'
path=paths[0]
digest=hashlib.sha256(path.read_bytes()).hexdigest()
sidecar=path.with_name(path.name+'.sha256').read_text().split()
assert sidecar and sidecar[0] == digest
row=json.loads(path.read_text())
assert row['schema_version'] == 'hm3d_action_coordinate_compass_deployment_gate_v1_20260902'
assert row['status'] == 'deployment_gate_passed'
assert row['history_index'] == 40
assert row['navigation_success_read'] is False
assert row['navigation_final_distance_read'] is False
assert row['navigation_sr_computed'] is False
assert row['closed_loop_comparison_authorized'] is True
assert row['runtime_role_filter_used'] is False
audit=row['action_coordinate_audit']
assert audit['progress_finite_monotone'] is True
assert audit['positive_translation_update_count'] > 0
assert audit['unit_bearing_norm_verified'] is True
assert audit['fixed_controller_radius_m'] == 2.5
assert audit['visual_gate_after_authorization'] is False
assert audit['distance_regime_present'] is False
assert audit['endpoint_fallback_available'] is False
assert audit['native_fallback_after_authorization_available'] is False
assert audit['evaluator_pose_consumed'] is False
assert audit['habitat_path_consumed'] is False
assert audit['metric_scale_consumed'] is False
gate_source_sha=submission_row['source_receipt_sha256']
assert len(gate_source_sha) == 64
print(gate_source_sha)
PY
)
[[ "${GATE_SOURCE_SHA}" =~ ^[0-9a-f]{64}$ ]] || \
  fail "invalid gate source receipt: ${GATE_SOURCE_SHA}"
echo "ACTION_COORDINATE_GATE_VERIFIED source=${GATE_SOURCE_SHA}"

HAB_SITE=$("${HAB_PY}" -c \
  'import sysconfig; print(sysconfig.get_paths()["purelib"])')
HAB_REQUESTS=${HAB_SITE}/pip/_vendor
[[ -r "${HAB_REQUESTS}/requests/__init__.py" ]] || \
  fail "Habitat vendored requests is unavailable"
PYTHONPATH_VALUE=${SOURCE_ROOT}:${SOURCE_ROOT}/MemNavData:${RUNTIME}:${RUNTIME}/MemNavData:${TASK}:${TASK}/MemNavData:${BASE}:${BASE}/MemNavData:${HAB_REQUESTS}
for mode in endpoint_bearing action_coordinate_compass; do
  singularity exec -B /scratch/lg154 -B /scratch/yz11502 "${SIF}" \
    env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="${PYTHONPATH_VALUE}" \
    "${HAB_PY}" "${SOURCE_ROOT}/MemNavData/eval_2leg_habitat.py" \
      --episode_root /tmp/contract_only --scene /tmp/contract_only.glb \
      --out /tmp/contract_only --server_backend hybrid_pose \
      --port 18888 --novel_port 18889 \
      --hybrid_route certified_relocalization \
      --revisit_adapter verified_bearing_v1 \
      --navdp_depth_source monocular_sidecar \
      --certified_guidance_mode "${mode}" --contract_dry_run >/dev/null
done

source "${SOURCE_ROOT}/MemNavData/slurm_safe_submit.sh"
SBATCH=${SOURCE_ROOT}/MemNavData/slurm_hm3d_action_coordinate_compass_pair.sbatch
ANALYSIS_SBATCH=${SOURCE_ROOT}/MemNavData/slurm_hm3d_action_coordinate_compass_analysis.sbatch
lint_sbatch_template "${SBATCH}" || fail "sbatch lint failed"
lint_sbatch_template "${ANALYSIS_SBATCH}" || fail "analysis sbatch lint failed"
exports="ALL,SOURCE_ROOT=${SOURCE_ROOT},SOURCE_RECEIPT=${SOURCE_RECEIPT},EXPECTED_SOURCE_RECEIPT_SHA=${SOURCE_SHA},RUN_ROOT=${RUN_ROOT}"
safe_sbatch --lint-fatal --test-only \
  --partition=h100_tandon,a100_tandon \
  --account=torch_pr_769_tandon_advanced --qos=gpu48 --gres=gpu:1 \
  --cpus-per-task=10 --mem=72G --time=02:00:00 --array=0-47%3 \
  --export="${exports}" "${SBATCH}" >/dev/null
for mode in aggregate verify; do
  safe_sbatch --lint-fatal --test-only --partition=cpu_short \
    --account=torch_pr_769_tandon_advanced \
    --export="ALL,MODE=${mode},SOURCE_ROOT=${SOURCE_ROOT},SOURCE_RECEIPT=${SOURCE_RECEIPT},EXPECTED_SOURCE_RECEIPT_SHA=${SOURCE_SHA},RUN_ROOT=${RUN_ROOT}" \
    "${ANALYSIS_SBATCH}" >/dev/null
done
if [[ "${DRY_RUN}" == 1 ]]; then
  echo "DRY_RUN_OK indices=0-47%3 time=02:00:00 source=${SOURCE_SHA}"
  exit 0
fi

mkdir -p "${RUN_ROOT}"
job=$(safe_sbatch --lint-fatal --parsable --job-name=h3ActCoordPair \
  --partition=h100_tandon,a100_tandon \
  --account=torch_pr_769_tandon_advanced --qos=gpu48 --gres=gpu:1 \
  --cpus-per-task=10 --mem=72G --time=02:00:00 --array=0-47%3 \
  --export="${exports}" "${SBATCH}" | job_id)
[[ "${job}" =~ ^[0-9]+$ ]] || fail "invalid submitted job id: ${job}"
summary_exports="ALL,MODE=aggregate,SOURCE_ROOT=${SOURCE_ROOT},SOURCE_RECEIPT=${SOURCE_RECEIPT},EXPECTED_SOURCE_RECEIPT_SHA=${SOURCE_SHA},RUN_ROOT=${RUN_ROOT}"
summary_job=$(safe_sbatch --lint-fatal --parsable \
  --partition=cpu_short --account=torch_pr_769_tandon_advanced \
  --dependency=afterany:"${job}" --kill-on-invalid-dep=yes \
  --export="${summary_exports}" "${ANALYSIS_SBATCH}" | job_id)
[[ "${summary_job}" =~ ^[0-9]+$ ]] || \
  fail "invalid summary job id: ${summary_job}"
verify_exports="ALL,MODE=verify,SOURCE_ROOT=${SOURCE_ROOT},SOURCE_RECEIPT=${SOURCE_RECEIPT},EXPECTED_SOURCE_RECEIPT_SHA=${SOURCE_SHA},RUN_ROOT=${RUN_ROOT}"
verify_job=$(safe_sbatch --lint-fatal --parsable \
  --partition=cpu_short --account=torch_pr_769_tandon_advanced \
  --dependency=afterok:"${summary_job}" --kill-on-invalid-dep=yes \
  --export="${verify_exports}" "${ANALYSIS_SBATCH}" | job_id)
[[ "${verify_job}" =~ ^[0-9]+$ ]] || \
  fail "invalid verification job id: ${verify_job}"

"/scratch/lg154/conda-envs/memnav/bin/python" - \
  "${RUN_ROOT}/submission.json" "${job}" "${summary_job}" "${verify_job}" \
  "${SOURCE_ROOT}" "${SOURCE_SHA}" "${GATE_ROOT}" "${GATE_SOURCE_SHA}" \
  "${PROTOCOL_SHA}" <<'PY'
import datetime,json,os,sys
path,job,summary_job,verify_job,source,source_sha,gate_root,gate_source_sha,protocol_sha=sys.argv[1:]
payload={
 'schema_version':'hm3d_action_coordinate_compass_pair_submission_v1_20260902',
 'submitted_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),
 'job_id':int(job), 'array_indices':'0-47', 'array_concurrency':3,
 'summary_job_id':int(summary_job), 'verification_job_id':int(verify_job),
 'summary_dependency':'afterany evaluation array; fail closed on missing output',
 'verification_dependency':'afterok summary',
 'partition':['h100_tandon','a100_tandon'], 'qos':'gpu48',
 'requested_time_per_element':'02:00:00',
 'source_root':source, 'source_receipt_sha256':source_sha,
 'gate_root':gate_root, 'gate_history_index':40,
 'gate_source_receipt_sha256':gate_source_sha,
 'protocol_sha256':protocol_sha,
 'arms':['mono_cec_endpoint','mono_cec_action_coordinate'],
 'benchmark_manifest_sha256':'cbc518cea991fd252893f97fd5e730c277e4d899369932536a745351d47e7451',
 'population_scope':'consumed development; not paper confirmation',
 'runtime_role_visibility':'none',
 'navigation_outcomes_read_before_submission':False,
 'canonical_cec_default_changed':False,
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
printf 'SUMMARY_JOB_ID=%s\nVERIFY_JOB_ID=%s\n' \
  "${summary_job}" "${verify_job}"
