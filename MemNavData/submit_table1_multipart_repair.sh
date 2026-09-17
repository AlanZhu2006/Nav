#!/usr/bin/env bash
# Exact failed-cell replay after CPU reproduction of the multipart defect.
set -euo pipefail
: "${REPAIRED_BUNDLE:?}" "${EXPECTED_RUNTIME_SHA:?}" "${TABLE1_RUN:?}"
[[ "$(id -un)" == yz11502 ]]
export TABLE1_ORIGINAL_RUN=/scratch/yz11502/Research/Nav-axis-uturn-results/table1_repaired_20260910/run_7b6bffa77891ef4f
export TABLE1_PLAN=${TABLE1_ORIGINAL_RUN}/plan.json
export EXPECTED_PLAN_SHA=0964f52738e22f2d16b099196605ef2e9681f57afce20fbdb99b417d8b535622
export TABLE1_EVAL_ROOT=${TABLE1_RUN}/evaluation
TABLE1_PY=/scratch/lg154/conda-envs/memnav/bin/python
[[ "$(sha256sum "${REPAIRED_BUNDLE}/SOURCE_BUNDLE.sha256" | awk '{print $1}')" == "${EXPECTED_RUNTIME_SHA}" ]]
[[ "$(sha256sum "${TABLE1_PLAN}" | awk '{print $1}')" == "${EXPECTED_PLAN_SHA}" ]]
(cd "${REPAIRED_BUNDLE}" && sha256sum -c --quiet SOURCE_BUNDLE.sha256)
"${TABLE1_PY}" - <<'PY'
import json, os
from pathlib import Path
original, run = map(Path, (os.environ['TABLE1_ORIGINAL_RUN'], os.environ['TABLE1_RUN']))
assert run.parent == original.parent and run.name.startswith('multipart_retry_20260911_')
assert not (run/'evaluation').exists() and not (run/'submission.json').exists()
audit=json.loads((run/'preflight/multipart_reproduction.json').read_text())
assert audit['verified'] and [r['index'] for r in audit['cases']]==[110,158]
assert all(r['exact_bytes_after_repair'] for r in audit['cases'])
plan=json.loads(Path(os.environ['TABLE1_PLAN']).read_text())
assert len(plan['cells'])==210 and plan['total_rollouts']==840
for cell in plan['cells']:
    receipt=json.loads((original/'evaluation'/('task_%03d'%cell['index'])/'archive_receipt.json').read_text())
    assert receipt['completed'] == (cell['index'] not in (110,158))
(run/'evaluation').mkdir()
combined=run/'combined_evaluation';combined.mkdir()
for cell in plan['cells']:
    name='task_%03d'%cell['index']
    (combined/name).symlink_to((run if cell['index'] in (110,158) else original)/'evaluation'/name, target_is_directory=True)
with (run/'recovery_manifest.json').open('x') as f:
    json.dump(dict(original_run=str(original),retry_indices=[110,158],
        original_runtime_sha256='7b6bffa77891ef4f041508dc7dc31b40fb56b7e1885334a9e180a1f06d76e099',
        repaired_runtime_sha256=os.environ['EXPECTED_RUNTIME_SHA'],
        plan_sha256=os.environ['EXPECTED_PLAN_SHA'],
        scope='multipart parser CRLF preservation only; all four arms per failed cell',
        original_outputs_modified=False,scientific_parameters_changed=False),f,indent=2)
PY
source "${REPAIRED_BUNDLE}/MemNavData/slurm_safe_submit.sh"
TABLE1_EXPORT="ALL,REPAIRED_BUNDLE=${REPAIRED_BUNDLE},TABLE1_RUN=${TABLE1_RUN},TABLE1_PLAN=${TABLE1_PLAN},TABLE1_EVAL_ROOT=${TABLE1_EVAL_ROOT},EXPECTED_RUNTIME_SHA=${EXPECTED_RUNTIME_SHA},EXPECTED_PLAN_SHA=${EXPECTED_PLAN_SHA}"
TABLE1_SUMMARY_EXPORT="ALL,REPAIRED_BUNDLE=${REPAIRED_BUNDLE},TABLE1_RUN=${TABLE1_RUN},TABLE1_PLAN=${TABLE1_PLAN},TABLE1_EVAL_ROOT=${TABLE1_RUN}/combined_evaluation,EXPECTED_RUNTIME_SHA=${EXPECTED_RUNTIME_SHA},EXPECTED_PLAN_SHA=${EXPECTED_PLAN_SHA}"
TABLE1_GPU=(--partition=a100_tandon --account=torch_pr_769_tandon_advanced --qos=gpu48 --gres=gpu:1 --cpus-per-task=10 --mem=96G --time=01:00:00 --array=110,158%2)
TABLE1_CPU=(--partition=cpu_short --account=torch_pr_769_tandon_advanced --cpus-per-task=2 --mem=8G --time=00:30:00)
safe_sbatch --lint-fatal --test-only "${TABLE1_GPU[@]}" --export="${TABLE1_EXPORT}" "${REPAIRED_BUNDLE}/MemNavData/slurm_table1_repaired_eval.sbatch"
safe_sbatch --lint-fatal --test-only "${TABLE1_CPU[@]}" --export="${TABLE1_SUMMARY_EXPORT}" "${REPAIRED_BUNDLE}/MemNavData/slurm_table1_repaired_summary.sbatch"
TABLE1_REPAIR_JOB=$(safe_sbatch --lint-fatal --parsable "${TABLE1_GPU[@]}" --job-name=table1_multipart_fix --export="${TABLE1_EXPORT}" "${REPAIRED_BUNDLE}/MemNavData/slurm_table1_repaired_eval.sbatch")
export TABLE1_REPAIR_JOB
printf '%s\n' "${TABLE1_REPAIR_JOB}" >"${TABLE1_RUN}/retry_job.txt"
TABLE1_SUMMARY_JOB=$(safe_sbatch --lint-fatal --parsable "${TABLE1_CPU[@]}" --job-name=table1_fixed_summary --dependency="afterok:${TABLE1_REPAIR_JOB}" --kill-on-invalid-dep=yes --export="${TABLE1_SUMMARY_EXPORT}" "${REPAIRED_BUNDLE}/MemNavData/slurm_table1_repaired_summary.sbatch")
export TABLE1_SUMMARY_JOB
"${TABLE1_PY}" - <<'PY'
import json,os,time
from pathlib import Path
row=dict(retry_job=os.environ['TABLE1_REPAIR_JOB'],summary_job=os.environ['TABLE1_SUMMARY_JOB'],
    submitted_at=time.time(),indices=[110,158],runtime_sha256=os.environ['EXPECTED_RUNTIME_SHA'],
    plan_sha256=os.environ['EXPECTED_PLAN_SHA'],partition='a100_tandon',time_limit='01:00:00')
with (Path(os.environ['TABLE1_RUN'])/'submission.json').open('x') as f:json.dump(row,f,indent=2)
print(json.dumps(row,indent=2))
PY
scontrol show job "${TABLE1_REPAIR_JOB}"
scontrol show job "${TABLE1_SUMMARY_JOB}"
