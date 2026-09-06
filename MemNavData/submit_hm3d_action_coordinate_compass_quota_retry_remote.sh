#!/usr/bin/env bash
# Run only on the authenticated yz11502 Torch login node.
set -euo pipefail
umask 0022
export PYTHONDONTWRITEBYTECODE=1

REPAIR_SOURCE_ROOT=${REPAIR_SOURCE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
REPAIR_SOURCE_RECEIPT=${REPAIR_SOURCE_ROOT}/SOURCE_BUNDLE.sha256
REPAIR_SOURCE_SHA=$(sha256sum "${REPAIR_SOURCE_RECEIPT}" | awk '{print $1}')
SOURCE_ROOT=/scratch/yz11502/Research/Nav-axis-uturn-source-bundles/hm3d_action_coordinate_compass_b7de41263049e415
SOURCE_RECEIPT=${SOURCE_ROOT}/SOURCE_BUNDLE.sha256
SOURCE_SHA=b7de41263049e415755f3595338f9e064373966e44992d234ebe453e55bb802a
RUN_ROOT=/scratch/yz11502/Research/Nav-axis-uturn-results/hm3d_action_coordinate_compass_20260902/pair_b7de41263049e415
MANIFEST=/scratch/yz11502/Research/Nav-axis-uturn-results/hm3d_table3_causal_survey_expansion_20260831/expansion_20260830T165729Z_a8dd232e/merged_query_population/role_pairs/manifest.json
MANIFEST_SHA=cbc518cea991fd252893f97fd5e730c277e4d899369932536a745351d47e7451
REPAIR_TAG=16781348_disk_quota_exact_retry_20260902
REPAIR_STATE_ROOT=${RUN_ROOT}/repairs/action_coordinate_quota_exact_retry_20260902
ARCHIVE=${RUN_ROOT}/failed_attempts/${REPAIR_TAG}
PY=/scratch/lg154/conda-envs/memnav/bin/python
DRY_RUN=${DRY_RUN:-0}

fail() { echo "ABORT: $*" >&2; exit 2; }
job_id() { awk -F';' 'NR==1{print $1}'; }
verify_bundle() {
  local root=$1 receipt=$2 expected=$3 label=$4
  [[ -d "${root}" && -f "${receipt}" ]] || fail "missing ${label} bundle"
  [[ "$(sha256sum "${receipt}" | awk '{print $1}')" == "${expected}" ]] ||
    fail "${label} receipt changed"
  (cd "${root}" && sha256sum -c --quiet "${receipt}") ||
    fail "${label} bundle content changed"
}

[[ "$(id -un)" == yz11502 ]] || fail "wrong HPC identity"
[[ "${DRY_RUN}" =~ ^[01]$ ]] || fail "DRY_RUN must be 0 or 1"
verify_bundle "${REPAIR_SOURCE_ROOT}" "${REPAIR_SOURCE_RECEIPT}" \
  "${REPAIR_SOURCE_SHA}" repair
verify_bundle "${SOURCE_ROOT}" "${SOURCE_RECEIPT}" "${SOURCE_SHA}" scientific
[[ "$(sha256sum "${MANIFEST}" | awk '{print $1}')" == "${MANIFEST_SHA}" ]] ||
  fail "benchmark manifest changed"
(cd "${RUN_ROOT}" && sha256sum -c --quiet submission.json.sha256) ||
  fail "original submission receipt changed"

"${PY}" - "${RUN_ROOT}" "${MANIFEST}" "${REPAIR_STATE_ROOT}" <<'PY'
import hashlib, json, pathlib, sys
run=pathlib.Path(sys.argv[1]); manifest=pathlib.Path(sys.argv[2]); repair=pathlib.Path(sys.argv[3])
m=json.loads(manifest.read_text()); assert len(m['episodes']) == 48
sub=json.loads((run/'submission.json').read_text())
assert sub['job_id'] == 16781348 and sub['summary_job_id'] == 16781349
assert sub['verification_job_id'] == 16781350
assert sub['source_receipt_sha256'] == 'b7de41263049e415755f3595338f9e064373966e44992d234ebe453e55bb802a'
sealed=[]; partial=[]; absent=[]
for i,row in enumerate(m['episodes']):
    root=run/'development/action_coordinate_compass'/f"{i:03d}_{row['scene']}_{row['episode']}"
    completion=root/'completion.json'; sidecar=root/'completion.json.sha256'
    if completion.is_file():
        if not sidecar.is_file(): raise SystemExit(f'missing sidecar at index {i}')
        expected=sidecar.read_text().split()[0]
        if hashlib.sha256(completion.read_bytes()).hexdigest() != expected:
            raise SystemExit(f'completion hash mismatch at index {i}')
        sealed.append(i)
    elif root.exists(): partial.append(i)
    else: absent.append(i)
if sealed != list(range(18)): raise SystemExit(f'retained set changed: {sealed}')
if partial != [18,19,20]: raise SystemExit(f'partial set changed: {partial}')
if absent != list(range(21,48)): raise SystemExit(f'absent set changed: {absent}')
for path in (run/'summary/action_coordinate_compass_result.json',
             run/'summary/action_coordinate_compass_independent_verification.json',
             repair/'exact_retry_submission.json'):
    if path.exists(): raise SystemExit('downstream/retry output already exists: '+str(path))
bootstrap=repair/'bootstrap_cleanup.json'; side=repair/'bootstrap_cleanup.json.sha256'
if not bootstrap.is_file() or not side.is_file(): raise SystemExit('missing bootstrap cleanup receipt')
if hashlib.sha256(bootstrap.read_bytes()).hexdigest() != side.read_text().split()[0]:
    raise SystemExit('bootstrap cleanup receipt changed')
row=json.loads(bootstrap.read_text())
assert row['index'] == 0 and row['completion_verified_before_cleanup'] is True
assert row['navigation_outcomes_read'] is False
PY

for index in 18 19 20; do
  label=$(printf '%03d' "${index}")
  readarray -t roots < <(compgen -G \
    "${RUN_ROOT}/development/action_coordinate_compass/${label}_*" || true)
  readarray -t runtimes < <(compgen -G \
    "${RUN_ROOT}/runtime/eval_*_actCoordPair${label}" || true)
  [[ "${#roots[@]}" -eq 1 && -d "${roots[0]}" && \
     ! -e "${roots[0]}/completion.json" ]] ||
    fail "partial evaluation state changed for index ${index}"
  [[ "${#runtimes[@]}" -eq 1 && -d "${runtimes[0]}" ]] ||
    fail "partial runtime state changed for index ${index}"
  grep -R -q --include='*.log' --include='*.out' --include='*.err' \
    -- 'Disk quota exceeded' "${roots[0]}" "${runtimes[0]}" ||
    fail "quota failure signature missing for index ${index}"
done
[[ ! -e "${ARCHIVE}" ]] || fail "failure archive already exists"

source "${REPAIR_SOURCE_ROOT}/MemNavData/slurm_safe_submit.sh"
RETRY_SBATCH=${REPAIR_SOURCE_ROOT}/MemNavData/slurm_hm3d_action_coordinate_compass_quota_retry.sbatch
ANALYSIS_SBATCH=${SOURCE_ROOT}/MemNavData/slurm_hm3d_action_coordinate_compass_analysis.sbatch
lint_sbatch_template "${RETRY_SBATCH}" || fail "retry sbatch lint failed"
lint_sbatch_template "${ANALYSIS_SBATCH}" || fail "analysis sbatch lint failed"
exports="ALL,REPAIR_SOURCE_ROOT=${REPAIR_SOURCE_ROOT},REPAIR_SOURCE_RECEIPT=${REPAIR_SOURCE_RECEIPT},EXPECTED_REPAIR_SOURCE_RECEIPT_SHA=${REPAIR_SOURCE_SHA},SOURCE_ROOT=${SOURCE_ROOT},SOURCE_RECEIPT=${SOURCE_RECEIPT},EXPECTED_SOURCE_RECEIPT_SHA=${SOURCE_SHA},RUN_ROOT=${RUN_ROOT},REPAIR_STATE_ROOT=${REPAIR_STATE_ROOT}"
safe_sbatch --lint-fatal --test-only --partition=h100_tandon,a100_tandon \
  --account=torch_pr_769_tandon_advanced --qos=gpu48 --gres=gpu:1 \
  --cpus-per-task=10 --mem=72G --time=02:00:00 --array=18-47%3 \
  --export="${exports}" "${RETRY_SBATCH}" >/dev/null
for mode in aggregate verify; do
  safe_sbatch --lint-fatal --test-only --partition=cpu_short \
    --account=torch_pr_769_tandon_advanced --cpus-per-task=4 --mem=24G \
    --time=00:30:00 \
    --export="ALL,MODE=${mode},SOURCE_ROOT=${SOURCE_ROOT},SOURCE_RECEIPT=${SOURCE_RECEIPT},EXPECTED_SOURCE_RECEIPT_SHA=${SOURCE_SHA},RUN_ROOT=${RUN_ROOT}" \
    "${ANALYSIS_SBATCH}" >/dev/null
done
if [[ "${DRY_RUN}" == 1 ]]; then
  echo "DRY_RUN_OK retained=0-17 exact_retry=18-47%3"
  exit 0
fi

# The bootstrap deletion freed enough entries to create this receipt. Build the
# full retained-buffer audit in node-local /tmp, delete only verified completed
# buffers 1-17, then persist the receipt back under the run root.
cleanup_tmp=$(mktemp /tmp/action_coordinate_cleanup.XXXXXX.json)
"${PY}" - "${RUN_ROOT}" "${MANIFEST}" "${cleanup_tmp}" <<'PY'
import datetime, hashlib, json, os, pathlib, shutil, sys
run=pathlib.Path(sys.argv[1]); m=json.load(open(sys.argv[2])); out=pathlib.Path(sys.argv[3])
targets=[]
for i in range(1,18):
    row=m['episodes'][i]
    completion=run/'development/action_coordinate_compass'/f"{i:03d}_{row['scene']}_{row['episode']}"/'completion.json'
    side=completion.with_name('completion.json.sha256')
    digest=hashlib.sha256(completion.read_bytes()).hexdigest()
    if digest != side.read_text().split()[0]: raise SystemExit(f'completion hash mismatch {i}')
    candidates=list((run/'runtime').glob(f'eval_*_actCoordPair{i:03d}/buffer'))
    if len(candidates) != 1: raise SystemExit(f'buffer set changed {i}: {candidates}')
    target=candidates[0].resolve(); runtime=(run/'runtime').resolve()
    if target.parent.parent != runtime or target.name != 'buffer':
        raise SystemExit(f'unsafe target {target}')
    files=0; bytes_=0
    for root,dirs,names in os.walk(target):
        files += len(names)
        for name in names:
            bytes_ += (pathlib.Path(root)/name).stat().st_size
    targets.append({'index':i,'target':str(target),'files':files,'bytes':bytes_,
                    'completion_sha256':digest})
payload={'schema_version':'hm3d_action_coordinate_retained_buffer_cleanup_v1_20260902',
 'created_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),
 'indices':list(range(1,18)), 'targets':targets,
 'total_files_removed':sum(x['files'] for x in targets),
 'total_bytes_removed':sum(x['bytes'] for x in targets),
 'completion_sidecars_verified':True, 'navigation_outcomes_read':False,
 'data_class':'regenerable_runtime_rgb_buffers',
 'scientific_execution_changed':False}
out.write_text(json.dumps(payload,indent=2,sort_keys=True)+'\n')
for item in targets:
    shutil.rmtree(item['target'])
    if pathlib.Path(item['target']).exists(): raise SystemExit('cleanup failed: '+item['target'])
PY
cleanup=${REPAIR_STATE_ROOT}/retained_buffer_cleanup.json
[[ ! -e "${cleanup}" ]] || fail "retained cleanup receipt already exists"
mv -- "${cleanup_tmp}" "${cleanup}"
sha256sum "${cleanup}" >"${cleanup}.sha256"
chmod a-w "${cleanup}" "${cleanup}.sha256"

mkdir -p "${ARCHIVE}/development/action_coordinate_compass" "${ARCHIVE}/runtime"
for index in 18 19 20; do
  label=$(printf '%03d' "${index}")
  readarray -t roots < <(compgen -G \
    "${RUN_ROOT}/development/action_coordinate_compass/${label}_*" || true)
  readarray -t runtimes < <(compgen -G \
    "${RUN_ROOT}/runtime/eval_*_actCoordPair${label}" || true)
  mv -- "${roots[0]}" "${ARCHIVE}/development/action_coordinate_compass/"
  mv -- "${runtimes[0]}" "${ARCHIVE}/runtime/"
done
"${PY}" - "${ARCHIVE}" <<'PY'
import datetime, hashlib, json, os, pathlib, sys
root=pathlib.Path(sys.argv[1]); log_files=[]; files=0; bytes_=0
for path in root.rglob('*'):
    if not path.is_file(): continue
    files += 1; bytes_ += path.stat().st_size
    if path.suffix in {'.log','.out','.err'}:
        log_files.append({'path':path.relative_to(root).as_posix(),
                          'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
payload={'schema_version':'hm3d_action_coordinate_quota_failure_archive_v1_20260902',
 'created_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),
 'failed_indices':[18,19,20], 'failure_class':'scratch_file_count_quota_exhaustion',
 'tree_file_count':files, 'tree_bytes':bytes_, 'log_files':sorted(log_files,key=lambda x:x['path']),
 'partial_navigation_outcomes_read':False, 'partial_sr_computed':False,
 'deleted':False}
path=root/'archive_manifest.json'
fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o444)
with os.fdopen(fd,'w') as f: json.dump(payload,f,indent=2,sort_keys=True); f.write('\n')
PY
sha256sum "${ARCHIVE}/archive_manifest.json" >"${ARCHIVE}/archive_manifest.json.sha256"
chmod a-w "${ARCHIVE}/archive_manifest.json" "${ARCHIVE}/archive_manifest.json.sha256"

retry=$(safe_sbatch --lint-fatal --parsable --job-name=h3ActCoordQR \
  --partition=h100_tandon,a100_tandon \
  --account=torch_pr_769_tandon_advanced --qos=gpu48 --gres=gpu:1 \
  --cpus-per-task=10 --mem=72G --time=02:00:00 --array=18-47%3 \
  --export="${exports}" "${RETRY_SBATCH}" | job_id)
summary_exports="ALL,MODE=aggregate,SOURCE_ROOT=${SOURCE_ROOT},SOURCE_RECEIPT=${SOURCE_RECEIPT},EXPECTED_SOURCE_RECEIPT_SHA=${SOURCE_SHA},RUN_ROOT=${RUN_ROOT}"
summary=$(safe_sbatch --lint-fatal --parsable --partition=cpu_short \
  --account=torch_pr_769_tandon_advanced --cpus-per-task=4 --mem=24G \
  --time=00:30:00 --dependency=afterok:${retry} --kill-on-invalid-dep=yes \
  --export="${summary_exports}" "${ANALYSIS_SBATCH}" | job_id)
verify_exports="ALL,MODE=verify,SOURCE_ROOT=${SOURCE_ROOT},SOURCE_RECEIPT=${SOURCE_RECEIPT},EXPECTED_SOURCE_RECEIPT_SHA=${SOURCE_SHA},RUN_ROOT=${RUN_ROOT}"
verify=$(safe_sbatch --lint-fatal --parsable --partition=cpu_short \
  --account=torch_pr_769_tandon_advanced --cpus-per-task=4 --mem=24G \
  --time=00:30:00 --dependency=afterok:${summary} --kill-on-invalid-dep=yes \
  --export="${verify_exports}" "${ANALYSIS_SBATCH}" | job_id)
for id in "${retry}" "${summary}" "${verify}"; do
  [[ "${id}" =~ ^[0-9]+$ ]] || fail "invalid job id: ${id}"
done

cleanup_sha=$(sha256sum "${cleanup}" | awk '{print $1}')
archive_sha=$(sha256sum "${ARCHIVE}/archive_manifest.json" | awk '{print $1}')
submission=${REPAIR_STATE_ROOT}/exact_retry_submission.json
"${PY}" - "${submission}" "${retry}" "${summary}" "${verify}" \
  "${REPAIR_SOURCE_ROOT}" "${REPAIR_SOURCE_SHA}" "${cleanup_sha}" \
  "${ARCHIVE}" "${archive_sha}" <<'PY'
import datetime,json,os,sys
path,retry,summary,verify,bundle,bundle_sha,cleanup_sha,archive,archive_sha=sys.argv[1:]
payload={'schema_version':'hm3d_action_coordinate_quota_retry_submission_v1_20260902',
 'submitted_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),
 'failure_class':'scratch_file_count_quota_exhaustion',
 'retained_completion_count':18, 'retained_indices':'0-17',
 'exact_retry_indices':'18-47', 'array_concurrency':3,
 'jobs':{'exact_retry':int(retry),'summary':int(summary),'independent_verifier':int(verify)},
 'dependency_contract':'exact retry -> aggregate -> independent verifier; all afterok',
 'scientific_source_receipt_sha256':'b7de41263049e415755f3595338f9e064373966e44992d234ebe453e55bb802a',
 'repair_bundle':bundle, 'repair_bundle_receipt_sha256':bundle_sha,
 'retained_buffer_cleanup_sha256':cleanup_sha,
 'failure_archive':archive, 'failure_archive_manifest_sha256':archive_sha,
 'method_or_population_changed':False, 'arms_or_seed_changed':False,
 'episode_budget_changed':False, 'completed_indices_rerun':False,
 'navigation_outcomes_read_for_repair':False, 'partial_sr_computed':False,
 'fallback_completion_allowed':False,
 'successful_retry_buffer_policy':'delete only after sealed completion sidecar verification'}
fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o444)
with os.fdopen(fd,'w') as f: json.dump(payload,f,indent=2,sort_keys=True); f.write('\n')
PY
sha256sum "${submission}" >"${submission}.sha256"
chmod a-w "${submission}" "${submission}.sha256"
printf 'EXACT_RETRY=%s SUMMARY=%s VERIFY=%s\n' "${retry}" "${summary}" "${verify}"
