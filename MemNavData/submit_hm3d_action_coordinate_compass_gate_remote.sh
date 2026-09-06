#!/usr/bin/env bash
# Run only inside the authenticated yz11502 Torch login shell.
set -euo pipefail
umask 0022
export PYTHONDONTWRITEBYTECODE=1

SOURCE_ROOT=${SOURCE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
SOURCE_RECEIPT=${SOURCE_ROOT}/SOURCE_BUNDLE.sha256
EXPECTED_SOURCE_RECEIPT_SHA=${EXPECTED_SOURCE_RECEIPT_SHA:?}
SOURCE_SHA=$(sha256sum "${SOURCE_RECEIPT}" | awk '{print $1}')
RUN_ROOT=${RUN_ROOT:-/scratch/yz11502/Research/Nav-axis-uturn-results/hm3d_action_coordinate_compass_20260902/gate_${SOURCE_SHA:0:16}}
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
    --certified_guidance_mode action_coordinate_compass \
    --contract_dry_run >/dev/null

source "${SOURCE_ROOT}/MemNavData/slurm_safe_submit.sh"
SBATCH=${SOURCE_ROOT}/MemNavData/slurm_hm3d_action_coordinate_compass_gate.sbatch
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
job=$(safe_sbatch --lint-fatal --parsable --job-name=h3ActCoordGate \
  --partition=h100_tandon,a100_tandon \
  --account=torch_pr_769_tandon_advanced --qos=gpu48 --gres=gpu:1 \
  --cpus-per-task=10 --mem=72G --time=01:00:00 \
  --export="${exports}" "${SBATCH}" | job_id)
[[ "${job}" =~ ^[0-9]+$ ]] || fail "invalid submitted job id: ${job}"

"/scratch/lg154/conda-envs/memnav/bin/python" - \
  "${RUN_ROOT}/submission.json" "${job}" "${SOURCE_ROOT}" "${SOURCE_SHA}" \
  "${PROTOCOL_SHA}" <<'PY'
import datetime,json,os,sys
path,job,source,source_sha,protocol_sha=sys.argv[1:]
payload={
 'schema_version':'hm3d_action_coordinate_compass_gate_submission_v1_20260902',
 'submitted_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),
 'job_id':int(job), 'history_index':40,
 'partition':['h100_tandon','a100_tandon'], 'qos':'gpu48',
 'requested_time':'01:00:00', 'query_step_guard':80,
 'source_root':source, 'source_receipt_sha256':source_sha,
 'protocol_sha256':protocol_sha,
 'benchmark_manifest_sha256':'cbc518cea991fd252893f97fd5e730c277e4d899369932536a745351d47e7451',
 'prospective_offline_result_sha256':'09934b9075b8d9a52ee9fe5c8b2e0b1b2199dc19921f4370858b88d38e134376',
 'stage':'SR-hidden action-coordinate controller gate',
 'navigation_success_read_before_submission':False,
 'navigation_final_distance_read_before_submission':False,
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
