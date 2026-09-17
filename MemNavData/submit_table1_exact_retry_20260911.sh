#!/usr/bin/env bash
# Rerun only the two failed cells, using the original immutable runtime/plan.
# Original successes, failures, and active jobs are never moved or overwritten.
set -euo pipefail
: "${TABLE1_RETRY_RUN:?Set a new additive retry directory}"
[[ "$(id -un)" == yz11502 ]]

export TABLE1_ORIGINAL_RUN=/scratch/yz11502/Research/Nav-axis-uturn-results/table1_repaired_20260910/run_7b6bffa77891ef4f
export REPAIRED_BUNDLE=/scratch/yz11502/Research/Nav-axis-uturn-source-bundles/table1_repaired_7b6bffa77891ef4f
export EXPECTED_RUNTIME_SHA=7b6bffa77891ef4f041508dc7dc31b40fb56b7e1885334a9e180a1f06d76e099
export TABLE1_PLAN=${TABLE1_ORIGINAL_RUN}/plan.json
export EXPECTED_PLAN_SHA=0964f52738e22f2d16b099196605ef2e9681f57afce20fbdb99b417d8b535622
export TABLE1_RUN=${TABLE1_RETRY_RUN}
export TABLE1_EVAL_ROOT=${TABLE1_RETRY_RUN}/evaluation
TABLE1_PY=/scratch/lg154/conda-envs/memnav/bin/python

[[ "$(sha256sum "${REPAIRED_BUNDLE}/SOURCE_BUNDLE.sha256" | awk '{print $1}')" == "${EXPECTED_RUNTIME_SHA}" ]]
(cd "${REPAIRED_BUNDLE}" && sha256sum -c --quiet SOURCE_BUNDLE.sha256)
[[ "$(sha256sum "${TABLE1_PLAN}" | awk '{print $1}')" == "${EXPECTED_PLAN_SHA}" ]]

"${TABLE1_PY}" - <<'PY'
import hashlib, json, os, subprocess
from pathlib import Path

original = Path(os.environ['TABLE1_ORIGINAL_RUN'])
retry = Path(os.environ['TABLE1_RETRY_RUN'])
assert retry.parent == original.parent and retry.name.startswith('exact_retry_20260911_')
assert not retry.exists(), 'Never reuse a submission directory'
plan = json.loads(Path(os.environ['TABLE1_PLAN']).read_text())
assert len(plan['cells']) == 210 and plan['total_rollouts'] == 840
states = subprocess.check_output([
    'sacct', '-X', '-n', '-P', '-j', '17306820',
    '--format=JobID,State,ExitCode'], text=True)
state_rows = {r.split('|')[0]: r.split('|')[1:3] for r in states.splitlines() if r}
failed = {}
for i, controller in ((110, 'navdp'), (158, 'vint')):
    assert state_rows['17306820_%d' % i] == ['FAILED', '1:0']
    cell = plan['cells'][i]
    assert cell['index'] == i and cell['dataset'] == 'mp3d' and cell['controller'] == controller
    p = original / 'evaluation' / ('task_%03d' % i) / 'archive_receipt.json'
    a = json.loads(p.read_text())
    assert a['completed'] is False and a['exit_code'] == 1
    assert a['all_member_hashes_readback_verified'] is True
    digest = hashlib.sha256()
    with Path(a['archive']).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b''):
            digest.update(chunk)
    assert digest.hexdigest() == a['archive_sha256']
    failed[str(i)] = dict(cell=cell, receipt=str(p),
        receipt_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
        archive=a['archive'], archive_sha256=a['archive_sha256'])

retry.mkdir()
(retry / 'evaluation').mkdir()
combined = retry / 'combined_evaluation'
combined.mkdir()
for cell in plan['cells']:
    name = 'task_%03d' % cell['index']
    source = retry / 'evaluation' if cell['index'] in (110, 158) else original / 'evaluation'
    (combined / name).symlink_to(source / name, target_is_directory=True)
with (retry / 'recovery_manifest.json').open('x') as f:
    json.dump(dict(original_run=str(original), retry_indices=[110, 158],
        excluded_node=None, diagnosis='JPEG digest mismatch; both failures on ga033; root cause unresolved',
        mitigation='same runtime, same four-arm cell, normal A100 scheduler placement; all checks retained',
        scheduler_note='test-only rejected --exclude=ga033; no exclusion or node pinning requested',
        original_failures=failed, original_runtime_sha256=os.environ['EXPECTED_RUNTIME_SHA'],
        plan_sha256=os.environ['EXPECTED_PLAN_SHA'],
        original_outputs_modified=False, scientific_parameters_changed=False,
        combined_evaluation=str(combined)), f, indent=2)
print('PRESERVED both failed archives; staged exact 110,158 recovery', flush=True)
PY

source "${REPAIRED_BUNDLE}/MemNavData/slurm_safe_submit.sh"
TABLE1_EXPORT="ALL,REPAIRED_BUNDLE=${REPAIRED_BUNDLE},TABLE1_RUN=${TABLE1_RUN},TABLE1_PLAN=${TABLE1_PLAN},TABLE1_EVAL_ROOT=${TABLE1_EVAL_ROOT},EXPECTED_RUNTIME_SHA=${EXPECTED_RUNTIME_SHA},EXPECTED_PLAN_SHA=${EXPECTED_PLAN_SHA}"
TABLE1_SUMMARY_EXPORT="ALL,REPAIRED_BUNDLE=${REPAIRED_BUNDLE},TABLE1_RUN=${TABLE1_RUN},TABLE1_PLAN=${TABLE1_PLAN},TABLE1_EVAL_ROOT=${TABLE1_RETRY_RUN}/combined_evaluation,EXPECTED_RUNTIME_SHA=${EXPECTED_RUNTIME_SHA},EXPECTED_PLAN_SHA=${EXPECTED_PLAN_SHA}"
TABLE1_GPU=(--partition=a100_tandon --account=torch_pr_769_tandon_advanced --qos=gpu48
            --gres=gpu:1 --cpus-per-task=10 --mem=96G --time=01:00:00 --array=110,158%2)
TABLE1_CPU=(--partition=cpu_short --account=torch_pr_769_tandon_advanced
            --cpus-per-task=2 --mem=8G --time=00:30:00)
TABLE1_EVAL_TEMPLATE=${REPAIRED_BUNDLE}/MemNavData/slurm_table1_repaired_eval.sbatch
TABLE1_SUMMARY_TEMPLATE=${REPAIRED_BUNDLE}/MemNavData/slurm_table1_repaired_summary.sbatch
safe_sbatch --lint-fatal --test-only "${TABLE1_GPU[@]}" --export="${TABLE1_EXPORT}" "${TABLE1_EVAL_TEMPLATE}"
safe_sbatch --lint-fatal --test-only "${TABLE1_CPU[@]}" --export="${TABLE1_SUMMARY_EXPORT}" "${TABLE1_SUMMARY_TEMPLATE}"

TABLE1_RETRY_JOB=$(safe_sbatch --lint-fatal --parsable "${TABLE1_GPU[@]}" \
  --job-name=table1_exact_retry --export="${TABLE1_EXPORT}" "${TABLE1_EVAL_TEMPLATE}")
export TABLE1_RETRY_JOB
"${TABLE1_PY}" - <<'PY'
import json, os, time
from pathlib import Path
with (Path(os.environ['TABLE1_RETRY_RUN']) / 'retry_submission.json').open('x') as f:
    json.dump(dict(job_id=os.environ['TABLE1_RETRY_JOB'], submitted_at=time.time(),
        indices=[110,158], partition='a100_tandon', exclude=None, time_limit='01:00:00',
        runtime_sha256=os.environ['EXPECTED_RUNTIME_SHA'], plan_sha256=os.environ['EXPECTED_PLAN_SHA']), f, indent=2)
PY
TABLE1_RETRY_SUMMARY=$(safe_sbatch --lint-fatal --parsable "${TABLE1_CPU[@]}" \
  --job-name=table1_retry_summary --dependency="afterany:17306820,afterok:${TABLE1_RETRY_JOB}" \
  --kill-on-invalid-dep=yes --export="${TABLE1_SUMMARY_EXPORT}" "${TABLE1_SUMMARY_TEMPLATE}")
export TABLE1_RETRY_SUMMARY
"${TABLE1_PY}" - <<'PY'
import json, os, time
from pathlib import Path
row = dict(retry_job=os.environ['TABLE1_RETRY_JOB'], summary_job=os.environ['TABLE1_RETRY_SUMMARY'],
    submitted_at=time.time(), original_array='17306820',
    summary=str(Path(os.environ['TABLE1_RETRY_RUN']) / 'combined_evaluation/paired_summary.json'))
with (Path(os.environ['TABLE1_RETRY_RUN']) / 'submission.json').open('x') as f:
    json.dump(row, f, indent=2)
print(json.dumps(row, indent=2))
PY
scontrol show job "${TABLE1_RETRY_JOB}"
scontrol show job "${TABLE1_RETRY_SUMMARY}"
