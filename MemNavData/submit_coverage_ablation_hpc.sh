#!/usr/bin/env bash
# Submit once, after the exact-container preflight. Smoke success means valid
# execution/receipts, never successful navigation or a favorable method effect.
set -euo pipefail
: "${REPAIRED_BUNDLE:?}" "${COVERAGE_RUN:?}" "${COVERAGE_PLAN:?}" "${COVERAGE_EVAL_ROOT:?}"
: "${EXPECTED_RUNTIME_SHA:?}"
[[ "$(id -un)" == yz11502 ]]
export EXPECTED_PLAN_SHA=$(sha256sum "${COVERAGE_PLAN}" | awk '{print $1}')
[[ ! -e "${COVERAGE_RUN}/submission.json" && ! -e "${COVERAGE_RUN}/gate_job.txt" ]]
python - <<'PY'
import json,os
from pathlib import Path
root=Path(os.environ['COVERAGE_RUN'])
pre=json.loads((root/'preflight_success.json').read_text())
assert pre['verified'] and pre['runtime_sha256']==os.environ['EXPECTED_RUNTIME_SHA']
assert pre['plan_sha256']==os.environ['EXPECTED_PLAN_SHA']
plan=json.loads(Path(os.environ['COVERAGE_PLAN']).read_text())
assert plan['smoke_indices']==[10,71]
assert plan['remaining_indices']==[i for i in range(159) if i not in (10,71)]
PY
source "${REPAIRED_BUNDLE}/MemNavData/slurm_safe_submit.sh"
COVERAGE_EXPORT="ALL,REPAIRED_BUNDLE=${REPAIRED_BUNDLE},COVERAGE_RUN=${COVERAGE_RUN},COVERAGE_PLAN=${COVERAGE_PLAN},COVERAGE_EVAL_ROOT=${COVERAGE_EVAL_ROOT},EXPECTED_RUNTIME_SHA=${EXPECTED_RUNTIME_SHA},EXPECTED_PLAN_SHA=${EXPECTED_PLAN_SHA}"
COVERAGE_TEMPLATE=${REPAIRED_BUNDLE}/MemNavData/slurm_coverage_ablation_eval.sbatch
COVERAGE_SUMMARY_TEMPLATE=${REPAIRED_BUNDLE}/MemNavData/slurm_coverage_ablation_summary.sbatch
COVERAGE_GPU=(--partition=a100_tandon --account=torch_pr_769_tandon_advanced --qos=gpu48
              --gres=gpu:1 --cpus-per-task=10 --mem=96G --time=01:00:00)
safe_sbatch --lint-fatal --test-only "${COVERAGE_GPU[@]}" --array=10,71%2 \
  --export="${COVERAGE_EXPORT}" "${COVERAGE_TEMPLATE}"
safe_sbatch --lint-fatal --test-only --partition=cpu_short --account=torch_pr_769_tandon_advanced \
  --time=00:30:00 --export="${COVERAGE_EXPORT}" "${COVERAGE_SUMMARY_TEMPLATE}"
COVERAGE_GATE=$(safe_sbatch --lint-fatal --parsable "${COVERAGE_GPU[@]}" --array=10,71%2 \
  --export="${COVERAGE_EXPORT}" "${COVERAGE_TEMPLATE}")
printf '%s\n' "${COVERAGE_GATE}" >"${COVERAGE_RUN}/gate_job.txt"
COVERAGE_EVAL=$(safe_sbatch --lint-fatal --parsable "${COVERAGE_GPU[@]}" --array=0-9,11-70,72-158%4 \
  --dependency="afterok:${COVERAGE_GATE}" --kill-on-invalid-dep=yes \
  --export="${COVERAGE_EXPORT}" "${COVERAGE_TEMPLATE}")
printf '%s\n' "${COVERAGE_EVAL}" >"${COVERAGE_RUN}/evaluation_job.txt"
COVERAGE_SUMMARY=$(safe_sbatch --lint-fatal --parsable --partition=cpu_short \
  --account=torch_pr_769_tandon_advanced --time=00:30:00 \
  --dependency="afterany:${COVERAGE_GATE}:${COVERAGE_EVAL}" --kill-on-invalid-dep=yes \
  --export="${COVERAGE_EXPORT}" "${COVERAGE_SUMMARY_TEMPLATE}")
printf '%s\n' "${COVERAGE_SUMMARY}" >"${COVERAGE_RUN}/summary_job.txt"
export COVERAGE_GATE COVERAGE_EVAL COVERAGE_SUMMARY
python - <<'PY'
import json,os,time
from pathlib import Path
keys=('REPAIRED_BUNDLE','COVERAGE_RUN','COVERAGE_PLAN','COVERAGE_EVAL_ROOT','EXPECTED_RUNTIME_SHA',
      'EXPECTED_PLAN_SHA','COVERAGE_GATE','COVERAGE_EVAL','COVERAGE_SUMMARY')
receipt={k:os.environ[k] for k in keys}
receipt.update(submitted_at=time.time(),partition='a100_tandon',time_limit='01:00:00',
    gate_indices=[10,71],remaining_indices=[i for i in range(159) if i not in (10,71)],
    queries=159,rollouts=636,scope='consumed-population four-arm coverage-only ablation')
with (Path(os.environ['COVERAGE_RUN'])/'submission.json').open('x') as f:
 json.dump(receipt,f,indent=2)
print(json.dumps(receipt,indent=2))
PY
scontrol show job "${COVERAGE_GATE}"
scontrol show job "${COVERAGE_EVAL}"
scontrol show job "${COVERAGE_SUMMARY}"
