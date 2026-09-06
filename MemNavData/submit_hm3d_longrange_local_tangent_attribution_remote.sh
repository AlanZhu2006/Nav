#!/usr/bin/env bash
set -euo pipefail
umask 0022
export PYTHONDONTWRITEBYTECODE=1

SOURCE_ROOT=${SOURCE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
SOURCE_RECEIPT=${SOURCE_ROOT}/SOURCE_BUNDLE.sha256
EXPECTED_SOURCE_RECEIPT_SHA=${EXPECTED_SOURCE_RECEIPT_SHA:?}
SOURCE_SHA=$(sha256sum "${SOURCE_RECEIPT}" | awk '{print $1}')
REQUESTED_TIME=${REQUESTED_TIME:-02:00:00}
RUN_ROOT=${RUN_ROOT:-/scratch/yz11502/Research/Nav-axis-uturn-results/hm3d_longrange_local_tangent_20260903/consumed_${SOURCE_SHA:0:16}}
PROTOCOL_SHA=3e48de6deeff53da7ed4e7732efbee4ecf3e02691480e35021ea2e29abc4a3ba
SIF=/share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif
HAB_PY=/scratch/lg154/conda-envs/habitat/bin/python
RUNTIME=/scratch/yz11502/Research/Nav-axis-uturn-source-bundles/hm3d_table1_navdp_authority_transaction_repair_82e71f19ee7f4e52
TASK=/scratch/yz11502/Research/Nav-axis-uturn-source-bundles/hm3d_lifelong_natural_b_expansion_execution_1f4979a7fd37d467
BASE=/scratch/yz11502/Research/Nav-axis-uturn-source-bundles/final14_mono_factorial_5690569a4373f2d2
DRY_RUN=${DRY_RUN:-0}

fail() { echo "ABORT: $*" >&2; exit 2; }
job_id() { awk -F';' 'NR==1{print $1}'; }
[[ "$(id -un)" == yz11502 ]] || fail "wrong HPC identity"
[[ "${DRY_RUN}" =~ ^[01]$ ]] || fail "DRY_RUN must be 0 or 1"
[[ "${SOURCE_SHA}" == "${EXPECTED_SOURCE_RECEIPT_SHA}" ]] || \
  fail "source receipt changed"
(cd "${SOURCE_ROOT}" && sha256sum -c --quiet "${SOURCE_RECEIPT}") || \
  fail "source bundle content changed"
[[ "$(sha256sum "${SOURCE_ROOT}/MemNavData/LONG_RANGE_LOCAL_TANGENT_ATTRIBUTION_PROTOCOL_20260903.md" | awk '{print $1}')" == "${PROTOCOL_SHA}" ]] || \
  fail "frozen v2 protocol changed"
[[ ! -e "${RUN_ROOT}" ]] || fail "run root already exists: ${RUN_ROOT}"

HAB_SITE=$("${HAB_PY}" -c \
  'import sysconfig; print(sysconfig.get_paths()["purelib"])')
HAB_REQUESTS=${HAB_SITE}/pip/_vendor
PYTHONPATH_VALUE=${SOURCE_ROOT}:${SOURCE_ROOT}/MemNavData:${RUNTIME}:${RUNTIME}/MemNavData:${TASK}:${TASK}/MemNavData:${BASE}:${BASE}/MemNavData:${HAB_REQUESTS}
for arm in oracle_chord_mixed_realized oracle_tangent_mixed_realized oracle_tangent_then_native_realized; do
  singularity exec -B /scratch/lg154 -B /scratch/yz11502 "${SIF}" \
    env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="${PYTHONPATH_VALUE}" \
    HM3D_LONGRANGE_TANGENT_ARM="${arm}" \
    "${HAB_PY}" \
    "${SOURCE_ROOT}/MemNavData/eval_hm3d_longrange_local_tangent_attribution.py" \
      --contract_dry_run --episode_root /contract/dry/scene0 \
      --scene /contract/dry/scene.glb --scene_identity scene0 \
      --out /contract/dry/out --host 127.0.0.1 --port 18888 \
      --novel_port 18889 --server_backend hybrid_pose --success_dist 1.0 \
      --max_steps 3400 --stuck_window 3401 --exec_horizon 8 \
      --trajectory_selector server --trajectory_selector_scope all \
      --leg1_mode shared_trace --leg1_goal_source own --seed 0 \
      --terminal_uturn off --terminal_visual_refine off \
      --deterministic_plan_seeds --retrieval_override off \
      --certified_cdec_rescue off --certified_stagnation_graph off \
      --revisit_controller navdp_mixed \
      --role_pair_scope table3_longrange_oracle \
      --role_pair_query_role revisit \
      --revisit_adapter verified_bearing_v1 \
      --navdp_depth_source monocular_sidecar \
      --hybrid_route certified_relocalization \
      --certified_guidance_mode endpoint_bearing >/dev/null
done

source "${SOURCE_ROOT}/MemNavData/slurm_safe_submit.sh"
SBATCH=${SOURCE_ROOT}/MemNavData/slurm_hm3d_longrange_local_tangent_attribution.sbatch
lint_sbatch_template "${SBATCH}" || fail "v2 sbatch lint failed"
exports="ALL,SOURCE_ROOT=${SOURCE_ROOT},SOURCE_RECEIPT=${SOURCE_RECEIPT},EXPECTED_SOURCE_RECEIPT_SHA=${SOURCE_SHA},RUN_ROOT=${RUN_ROOT}"
safe_sbatch --lint-fatal --test-only \
  --partition=h100_tandon,a100_tandon \
  --account=torch_pr_769_tandon_advanced --qos=gpu48 --gres=gpu:1 \
  --cpus-per-task=10 --mem=72G --time="${REQUESTED_TIME}" \
  --array=32 --export="${exports}" "${SBATCH}" >/dev/null
if [[ "${DRY_RUN}" == 1 ]]; then
  echo "DRY_RUN_OK index=32 time=${REQUESTED_TIME}"
  exit 0
fi

mkdir -p "${RUN_ROOT}"
job=$(safe_sbatch --lint-fatal --parsable --job-name=h3TangentV2 \
  --partition=h100_tandon,a100_tandon \
  --account=torch_pr_769_tandon_advanced --qos=gpu48 --gres=gpu:1 \
  --cpus-per-task=10 --mem=72G --time="${REQUESTED_TIME}" \
  --array=32 --export="${exports}" "${SBATCH}" | job_id)
[[ "${job}" =~ ^[0-9]+$ ]] || fail "invalid evaluation job id"

"${HAB_PY}" - "${RUN_ROOT}/submission.json" "${job}" \
  "${SOURCE_ROOT}" "${SOURCE_SHA}" "${PROTOCOL_SHA}" \
  "${REQUESTED_TIME}" <<'PY'
import datetime,json,os,sys
path,job,source,source_sha,protocol_sha,requested_time=sys.argv[1:]
payload={
 'schema_version':'hm3d_longrange_local_tangent_submission_v2_20260903',
 'submitted_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),
 'job_id':int(job), 'array_index':32,
 'partition':['h100_tandon','a100_tandon'], 'qos':'gpu48',
 'requested_time':requested_time,
 'source_root':source, 'source_receipt_sha256':source_sha,
 'protocol_sha256':protocol_sha,
 'population_scope':'one consumed development failure; privileged diagnosis',
 'paper_result_eligible':False,
 'navigation_outcomes_read_before_submission':False,
 'canonical_cec_default_changed':False,
 'runtime_rgb_buffer_storage':'SLURM_TMPDIR',
}
fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o444)
with os.fdopen(fd,'w') as handle:
    json.dump(payload,handle,indent=2,sort_keys=True); handle.write('\n')
PY
sha256sum "${RUN_ROOT}/submission.json" >"${RUN_ROOT}/submission.json.sha256"
chmod a-w "${RUN_ROOT}/submission.json" "${RUN_ROOT}/submission.json.sha256"
printf 'JOB_ID=%s\nRUN_ROOT=%s\nSOURCE_SHA256=%s\n' \
  "${job}" "${RUN_ROOT}" "${SOURCE_SHA}"
