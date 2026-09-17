#!/usr/bin/env bash
# Preserve completed A0. Replace only the three not-started pilot jobs.
set -euo pipefail
: "${REPAIRED_BUNDLE:?}" "${EXPECTED_RUNTIME_SHA:?}" "${TABLE2_RUN:?}" "${TABLE2_PLAN:?}"
[[ "$(id -un)" == yz11502 ]]
export EXPECTED_PLAN_SHA=e745b30588a47e2ac92cb8e51c3ba7459a49341690c2172d7b27de02b293bbc5
export TABLE2_WORK_ROOT=/tmp/memnav_table2_pilot_20260911_v1
export PYTHONPATH=${REPAIRED_BUNDLE}:${REPAIRED_BUNDLE}/MemNavData
TABLE2_CONTINUATION=${TABLE2_RUN}/multipart_continuation_8b559e821e8c9df6
export TABLE2_CONTINUATION
[[ "$(sha256sum "${REPAIRED_BUNDLE}/SOURCE_BUNDLE.sha256" | awk '{print $1}')" == "${EXPECTED_RUNTIME_SHA}" ]]
[[ "$(sha256sum "${TABLE2_PLAN}" | awk '{print $1}')" == "${EXPECTED_PLAN_SHA}" ]]
[[ ! -e "${TABLE2_CONTINUATION}" ]]
/scratch/lg154/conda-envs/memnav/bin/python - <<'PY'
import json,os,subprocess
from pathlib import Path
run=Path(os.environ['TABLE2_RUN'])
v=json.loads((run/'preflight/multipart_8b559e821e8c9df6/verification.json').read_text())
assert v['verified'] and v['runtime_sha256']==os.environ['EXPECTED_RUNTIME_SHA']
assert v['plan_sha256']==os.environ['EXPECTED_PLAN_SHA']
a=json.loads((run/'A/task_000/archive_receipt.json').read_text())
assert a['completed'] and a['verification']['verified']
states=subprocess.check_output(['sacct','-X','-n','-P','-j','17347455,17347456,17347457',
    '--format=JobID,State,ElapsedRaw'],text=True)
rows=[s.split('|') for s in states.splitlines() if s.strip()]
assert len(rows)>=3 and all(r[1]=='PENDING' and int(r[2])==0 for r in rows), states
assert not list((run/'B').glob('task_*')) and not (run/'c_population.json').exists()
folder=Path(os.environ['TABLE2_CONTINUATION']);folder.mkdir()
with (folder/'preserved_state.json').open('x') as f:
    json.dump(dict(original_jobs=[17347455,17347456,17347457],states=rows,
        completed_A0_archive_sha256=a['archive_sha256'],
        plan_unchanged=True,original_outputs_modified=False),f,indent=2)
PY
source "${REPAIRED_BUNDLE}/MemNavData/slurm_safe_submit.sh"
TABLE2_GPU=(--partition=a100_tandon --account=torch_pr_769_tandon_advanced --qos=gpu48 --gres=gpu:1 --cpus-per-task=10 --mem=96G --time=01:00:00)
TABLE2_CPU=(--partition=cpu_short --account=torch_pr_769_tandon_advanced --cpus-per-task=2 --mem=8G --time=00:30:00)
TABLE2_EXPORT="ALL,REPAIRED_BUNDLE=${REPAIRED_BUNDLE},TABLE2_PLAN=${TABLE2_PLAN},TABLE2_RUN=${TABLE2_RUN},TABLE2_WORK_ROOT=${TABLE2_WORK_ROOT},EXPECTED_RUNTIME_SHA=${EXPECTED_RUNTIME_SHA},EXPECTED_PLAN_SHA=${EXPECTED_PLAN_SHA}"
TABLE2_EVAL=${REPAIRED_BUNDLE}/MemNavData/slurm_table2_mixed_eval.sbatch
TABLE2_REDUCER=${REPAIRED_BUNDLE}/MemNavData/slurm_table2_mixed_reduce.sbatch
safe_sbatch --lint-fatal --test-only "${TABLE2_GPU[@]}" --array=1-7%2 --export="${TABLE2_EXPORT},TABLE2_STAGE=A" "${TABLE2_EVAL}"
safe_sbatch --lint-fatal --test-only "${TABLE2_CPU[@]}" --export="${TABLE2_EXPORT},TABLE2_REDUCE=select_c" "${TABLE2_REDUCER}"
scancel --state=PENDING 17347455 17347456 17347457
/scratch/lg154/conda-envs/memnav/bin/python - <<'PY'
import subprocess
r=subprocess.run(['squeue','-h','-j','17347455,17347456,17347457','-o','%T'],
                 text=True,capture_output=True)
assert not r.stdout.strip(), 'An old task started meanwhile; preserve it and do not duplicate it'
PY
TABLE2_A=$(safe_sbatch --lint-fatal --parsable "${TABLE2_GPU[@]}" --array=1-7%2 --job-name=table2_multipart_fix --export="${TABLE2_EXPORT},TABLE2_STAGE=A" "${TABLE2_EVAL}")
printf '%s\n' "${TABLE2_A}" >"${TABLE2_CONTINUATION}/a_job.txt"
TABLE2_B=$(safe_sbatch --lint-fatal --parsable "${TABLE2_GPU[@]}" --array=0-15%2 --job-name=table2_multipart_fix --dependency="afterok:${TABLE2_A}" --kill-on-invalid-dep=yes --export="${TABLE2_EXPORT},TABLE2_STAGE=B" "${TABLE2_EVAL}")
printf '%s\n' "${TABLE2_B}" >"${TABLE2_CONTINUATION}/b_job.txt"
TABLE2_SELECT=$(safe_sbatch --lint-fatal --parsable "${TABLE2_CPU[@]}" --dependency="afterok:${TABLE2_B}" --kill-on-invalid-dep=yes --export="${TABLE2_EXPORT},TABLE2_REDUCE=select_c" "${TABLE2_REDUCER}")
export TABLE2_A TABLE2_B TABLE2_SELECT
/scratch/lg154/conda-envs/memnav/bin/python - <<'PY'
import json,os,time
from pathlib import Path
row={k:os.environ[k] for k in ('TABLE2_A','TABLE2_B','TABLE2_SELECT','EXPECTED_RUNTIME_SHA','EXPECTED_PLAN_SHA')}
row.update(submitted_at=time.time(),preserved_A0='17347454_0',reason='multipart CRLF repair; identical scientific plan')
with (Path(os.environ['TABLE2_CONTINUATION'])/'submission.json').open('x') as f:json.dump(row,f,indent=2)
print(json.dumps(row,indent=2))
PY
scontrol show job "${TABLE2_A}"
scontrol show job "${TABLE2_B}"
scontrol show job "${TABLE2_SELECT}"
