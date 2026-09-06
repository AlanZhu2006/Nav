#!/usr/bin/env bash
set -euo pipefail
umask 0022
export PYTHONDONTWRITEBYTECODE=1

SOURCE_ROOT=${SOURCE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
SOURCE_RECEIPT=${SOURCE_ROOT}/SOURCE_BUNDLE.sha256
EXPECTED_SOURCE_RECEIPT_SHA=${EXPECTED_SOURCE_RECEIPT_SHA:?}
SOURCE_SHA=$(sha256sum "${SOURCE_RECEIPT}" | awk '{print $1}')
PHASE=${PHASE:-gate}
REQUESTED_TIME=${REQUESTED_TIME:-04:00:00}
RUN_ROOT=${RUN_ROOT:-/scratch/yz11502/Research/Nav-axis-uturn-results/hm3d_longrange_oracle_attribution_20260903/${PHASE}_${SOURCE_SHA:0:16}}
GATE_ROOT=${GATE_ROOT:-}
SIF=/share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif
HAB_PY=/scratch/lg154/conda-envs/habitat/bin/python
RUNTIME=/scratch/yz11502/Research/Nav-axis-uturn-source-bundles/hm3d_table1_navdp_authority_transaction_repair_82e71f19ee7f4e52
TASK=/scratch/yz11502/Research/Nav-axis-uturn-source-bundles/hm3d_lifelong_natural_b_expansion_execution_1f4979a7fd37d467
BASE=/scratch/yz11502/Research/Nav-axis-uturn-source-bundles/final14_mono_factorial_5690569a4373f2d2
PROTOCOL_SHA=9606f129c1bfc80d57d070151dcb97b960a28f72d9875d0a39779052395e2640
MANIFEST_SHA=cbc518cea991fd252893f97fd5e730c277e4d899369932536a745351d47e7451
DRY_RUN=${DRY_RUN:-0}

fail() { echo "ABORT: $*" >&2; exit 2; }
job_id() { awk -F';' 'NR==1{print $1}'; }
[[ "$(id -un)" == yz11502 ]] || fail "wrong HPC identity"
[[ "${PHASE}" == gate || "${PHASE}" == formal ]] || fail "invalid PHASE"
[[ "${DRY_RUN}" =~ ^[01]$ ]] || fail "DRY_RUN must be 0 or 1"
[[ "${REQUESTED_TIME}" =~ ^[0-9]{2}:[0-9]{2}:[0-9]{2}$ ]] || \
  fail "REQUESTED_TIME must be HH:MM:SS"
[[ "${SOURCE_SHA}" == "${EXPECTED_SOURCE_RECEIPT_SHA}" ]] || \
  fail "source receipt changed"
(cd "${SOURCE_ROOT}" && sha256sum -c --quiet "${SOURCE_RECEIPT}") || \
  fail "source bundle content changed"
[[ "$(sha256sum "${SOURCE_ROOT}/MemNavData/LONG_RANGE_ORACLE_ATTRIBUTION_PROTOCOL_20260903.md" | awk '{print $1}')" == "${PROTOCOL_SHA}" ]] || \
  fail "frozen oracle protocol changed"
[[ ! -e "${RUN_ROOT}" ]] || fail "run root already exists: ${RUN_ROOT}"

if [[ "${PHASE}" == formal ]]; then
  [[ -n "${GATE_ROOT}" ]] || fail "formal phase requires GATE_ROOT"
  "${HAB_PY}" - "${GATE_ROOT}" "${SOURCE_SHA}" <<'PY'
import hashlib,json,sys
from pathlib import Path
root=Path(sys.argv[1]); expected_source=sys.argv[2]
sub=root/'submission.json'
assert hashlib.sha256(sub.read_bytes()).hexdigest() == (root/'submission.json.sha256').read_text().split()[0]
s=json.loads(sub.read_text())
assert s['phase'] == 'gate' and s['source_receipt_sha256'] == expected_source
paths=sorted(root.glob('development/longrange_oracle_attribution/032_*/completion.json'))
assert len(paths) == 1
p=paths[0]
assert hashlib.sha256(p.read_bytes()).hexdigest() == p.with_name(p.name+'.sha256').read_text().split()[0]
r=json.loads(p.read_text())
assert r['schema_version'] == 'hm3d_longrange_oracle_attribution_episode_v1_20260903'
assert r['history_index'] == 32 and r['paper_result_eligible'] is False
assert r['prefix_equality'] is True and r['target_proof_equality'] is True
assert set(r['wall_time_seconds']) == {
 'action_coordinate_mixed','oracle_route_mixed',
 'oracle_geodesic_mixed','oracle_geodesic_point'}
print(sum(float(v) for v in r['wall_time_seconds'].values()))
PY
fi

HAB_SITE=$("${HAB_PY}" -c \
  'import sysconfig; print(sysconfig.get_paths()["purelib"])')
HAB_REQUESTS=${HAB_SITE}/pip/_vendor
[[ -r "${HAB_REQUESTS}/requests/__init__.py" ]] || \
  fail "Habitat vendored requests unavailable"
PYTHONPATH_VALUE=${SOURCE_ROOT}:${SOURCE_ROOT}/MemNavData:${RUNTIME}:${RUNTIME}/MemNavData:${TASK}:${TASK}/MemNavData:${BASE}:${BASE}/MemNavData:${HAB_REQUESTS}
for arm in action_coordinate_mixed oracle_route_mixed oracle_geodesic_mixed oracle_geodesic_point; do
  mode=endpoint_bearing
  if [[ "${arm}" == action_coordinate_mixed ]]; then
    mode=action_coordinate_compass
  fi
  singularity exec -B /scratch/lg154 -B /scratch/yz11502 "${SIF}" \
    env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="${PYTHONPATH_VALUE}" \
    HM3D_LONGRANGE_ATTRIBUTION_ARM="${arm}" \
    "${HAB_PY}" \
    "${SOURCE_ROOT}/MemNavData/eval_hm3d_longrange_oracle_attribution.py" \
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
      --certified_guidance_mode "${mode}" >/dev/null
done

source "${SOURCE_ROOT}/MemNavData/slurm_safe_submit.sh"
SBATCH=${SOURCE_ROOT}/MemNavData/slurm_hm3d_longrange_oracle_attribution.sbatch
ANALYSIS=${SOURCE_ROOT}/MemNavData/slurm_hm3d_longrange_oracle_attribution_analysis.sbatch
lint_sbatch_template "${SBATCH}" || fail "evaluation sbatch lint failed"
lint_sbatch_template "${ANALYSIS}" || fail "analysis sbatch lint failed"
indices=32
concurrency=1
if [[ "${PHASE}" == formal ]]; then
  indices=32-47
  concurrency=3
fi
exports="ALL,SOURCE_ROOT=${SOURCE_ROOT},SOURCE_RECEIPT=${SOURCE_RECEIPT},EXPECTED_SOURCE_RECEIPT_SHA=${SOURCE_SHA},RUN_ROOT=${RUN_ROOT}"
safe_sbatch --lint-fatal --test-only \
  --partition=h100_tandon,a100_tandon \
  --account=torch_pr_769_tandon_advanced --qos=gpu48 --gres=gpu:1 \
  --cpus-per-task=10 --mem=72G --time="${REQUESTED_TIME}" \
  --array="${indices}%${concurrency}" --export="${exports}" \
  "${SBATCH}" >/dev/null
if [[ "${PHASE}" == formal ]]; then
  for mode in aggregate verify; do
    safe_sbatch --lint-fatal --test-only --partition=cpu_short \
      --account=torch_pr_769_tandon_advanced \
      --export="ALL,MODE=${mode},SOURCE_ROOT=${SOURCE_ROOT},SOURCE_RECEIPT=${SOURCE_RECEIPT},EXPECTED_SOURCE_RECEIPT_SHA=${SOURCE_SHA},RUN_ROOT=${RUN_ROOT}" \
      "${ANALYSIS}" >/dev/null
  done
fi
if [[ "${DRY_RUN}" == 1 ]]; then
  echo "DRY_RUN_OK phase=${PHASE} indices=${indices}%${concurrency} time=${REQUESTED_TIME}"
  exit 0
fi

mkdir -p "${RUN_ROOT}"
job=$(safe_sbatch --lint-fatal --parsable --job-name=h3OracleAttr \
  --partition=h100_tandon,a100_tandon \
  --account=torch_pr_769_tandon_advanced --qos=gpu48 --gres=gpu:1 \
  --cpus-per-task=10 --mem=72G --time="${REQUESTED_TIME}" \
  --array="${indices}%${concurrency}" --export="${exports}" \
  "${SBATCH}" | job_id)
[[ "${job}" =~ ^[0-9]+$ ]] || fail "invalid evaluation job id"
summary_job=
verify_job=
if [[ "${PHASE}" == formal ]]; then
  summary_job=$(safe_sbatch --lint-fatal --parsable \
    --partition=cpu_short --account=torch_pr_769_tandon_advanced \
    --dependency=afterany:"${job}" --kill-on-invalid-dep=yes \
    --export="ALL,MODE=aggregate,SOURCE_ROOT=${SOURCE_ROOT},SOURCE_RECEIPT=${SOURCE_RECEIPT},EXPECTED_SOURCE_RECEIPT_SHA=${SOURCE_SHA},RUN_ROOT=${RUN_ROOT}" \
    "${ANALYSIS}" | job_id)
  verify_job=$(safe_sbatch --lint-fatal --parsable \
    --partition=cpu_short --account=torch_pr_769_tandon_advanced \
    --dependency=afterok:"${summary_job}" --kill-on-invalid-dep=yes \
    --export="ALL,MODE=verify,SOURCE_ROOT=${SOURCE_ROOT},SOURCE_RECEIPT=${SOURCE_RECEIPT},EXPECTED_SOURCE_RECEIPT_SHA=${SOURCE_SHA},RUN_ROOT=${RUN_ROOT}" \
    "${ANALYSIS}" | job_id)
  [[ "${summary_job}" =~ ^[0-9]+$ && "${verify_job}" =~ ^[0-9]+$ ]] || \
    fail "invalid downstream job id"
fi

"${HAB_PY}" - "${RUN_ROOT}/submission.json" "${PHASE}" "${job}" \
  "${summary_job}" "${verify_job}" "${SOURCE_ROOT}" "${SOURCE_SHA}" \
  "${PROTOCOL_SHA}" "${REQUESTED_TIME}" "${indices}" "${concurrency}" <<'PY'
import datetime,json,os,sys
(path,phase,job,summary,verify,source,source_sha,protocol_sha,
 requested_time,indices,concurrency)=sys.argv[1:]
payload={
 'schema_version':'hm3d_longrange_oracle_attribution_submission_v1_20260903',
 'submitted_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),
 'phase':phase, 'job_id':int(job), 'array_indices':indices,
 'array_concurrency':int(concurrency),
 'summary_job_id':int(summary) if summary else None,
 'verification_job_id':int(verify) if verify else None,
 'partition':['h100_tandon','a100_tandon'], 'qos':'gpu48',
 'requested_time_per_element':requested_time,
 'source_root':source, 'source_receipt_sha256':source_sha,
 'protocol_sha256':protocol_sha,
 'benchmark_manifest_sha256':'cbc518cea991fd252893f97fd5e730c277e4d899369932536a745351d47e7451',
 'population_scope':'consumed development; privileged attribution only',
 'paper_result_eligible':False,
 'navigation_outcomes_read_before_submission':False,
 'canonical_cec_default_changed':False,
}
fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o444)
with os.fdopen(fd,'w') as handle:
    json.dump(payload,handle,indent=2,sort_keys=True); handle.write('\n')
PY
sha256sum "${RUN_ROOT}/submission.json" >"${RUN_ROOT}/submission.json.sha256"
chmod a-w "${RUN_ROOT}/submission.json" "${RUN_ROOT}/submission.json.sha256"
printf 'JOB_ID=%s\nRUN_ROOT=%s\nSOURCE_SHA256=%s\n' \
  "${job}" "${RUN_ROOT}" "${SOURCE_SHA}"
if [[ -n "${summary_job}" ]]; then
  printf 'SUMMARY_JOB_ID=%s\nVERIFY_JOB_ID=%s\n' \
    "${summary_job}" "${verify_job}"
fi
