#!/usr/bin/env bash
# Runtime validity gates, never success-rate gates.
set -euo pipefail
: "${REPAIRED_BUNDLE:?}" "${TABLE1_RUN:?}" "${TABLE1_PLAN:?}" "${TABLE1_EVAL_ROOT:?}" "${EXPECTED_RUNTIME_SHA:?}"
[[ "$(id -un)" == yz11502 ]]
export EXPECTED_PLAN_SHA=$(sha256sum "${TABLE1_PLAN}" | awk '{print $1}')
[[ ! -e "${TABLE1_RUN}/submission.json" && ! -e "${TABLE1_RUN}/gate_job.txt" ]]
python - <<'PY'
import json,os
from pathlib import Path
root=Path(os.environ['TABLE1_RUN'])
pre=json.loads((root/'preflight/verification.json').read_text())
assert pre['verified'] and pre['runtime_sha256']==os.environ['EXPECTED_RUNTIME_SHA']
assert pre['plan_sha256']==os.environ['EXPECTED_PLAN_SHA']
assert pre['gpu_gate_indices']==[0,28,56,84,126,168]
local=json.loads((root/'local_gate.json').read_text())
assert local['verified'] and local['controllers']==['navdp','vint','nomad']
assert local['rollouts']==12 and local['runtime_sha256']==os.environ['EXPECTED_RUNTIME_SHA']
plan=json.loads(Path(os.environ['TABLE1_PLAN']).read_text())
assert not plan['local'] and len(plan['cells'])==210 and plan['total_rollouts']==840
PY
source "${REPAIRED_BUNDLE}/MemNavData/slurm_safe_submit.sh"
TABLE1_EXPORT="ALL,REPAIRED_BUNDLE=${REPAIRED_BUNDLE},TABLE1_RUN=${TABLE1_RUN},TABLE1_PLAN=${TABLE1_PLAN},TABLE1_EVAL_ROOT=${TABLE1_EVAL_ROOT},EXPECTED_RUNTIME_SHA=${EXPECTED_RUNTIME_SHA},EXPECTED_PLAN_SHA=${EXPECTED_PLAN_SHA}"
TABLE1_TEMPLATE=${REPAIRED_BUNDLE}/MemNavData/slurm_table1_repaired_eval.sbatch
TABLE1_SUMMARY_TEMPLATE=${REPAIRED_BUNDLE}/MemNavData/slurm_table1_repaired_summary.sbatch
TABLE1_GPU=(--partition=a100_tandon --account=torch_pr_769_tandon_advanced --qos=gpu48
            --gres=gpu:1 --cpus-per-task=10 --mem=96G --time=01:00:00)
safe_sbatch --lint-fatal --test-only "${TABLE1_GPU[@]}" --array=0,28,56,84,126,168%4 \
  --export="${TABLE1_EXPORT}" "${TABLE1_TEMPLATE}"
safe_sbatch --lint-fatal --test-only --partition=cpu_short --account=torch_pr_769_tandon_advanced \
  --time=00:30:00 --export="${TABLE1_EXPORT}" "${TABLE1_SUMMARY_TEMPLATE}"
TABLE1_GATE=$(safe_sbatch --lint-fatal --parsable "${TABLE1_GPU[@]}" --array=0,28,56,84,126,168%4 \
  --export="${TABLE1_EXPORT}" "${TABLE1_TEMPLATE}")
printf '%s\n' "${TABLE1_GATE}" >"${TABLE1_RUN}/gate_job.txt"
TABLE1_EVAL=$(safe_sbatch --lint-fatal --parsable "${TABLE1_GPU[@]}" \
  --array=1-27,29-55,57-83,85-125,127-167,169-209%4 \
  --dependency="afterok:${TABLE1_GATE}" --kill-on-invalid-dep=yes \
  --export="${TABLE1_EXPORT}" "${TABLE1_TEMPLATE}")
printf '%s\n' "${TABLE1_EVAL}" >"${TABLE1_RUN}/evaluation_job.txt"
TABLE1_SUMMARY=$(safe_sbatch --lint-fatal --parsable --partition=cpu_short \
  --account=torch_pr_769_tandon_advanced --time=00:30:00 \
  --dependency="afterany:${TABLE1_GATE}:${TABLE1_EVAL}" --kill-on-invalid-dep=yes \
  --export="${TABLE1_EXPORT}" "${TABLE1_SUMMARY_TEMPLATE}")
printf '%s\n' "${TABLE1_SUMMARY}" >"${TABLE1_RUN}/summary_job.txt"
export TABLE1_GATE TABLE1_EVAL TABLE1_SUMMARY
python - <<'PY'
import json,os,time
from pathlib import Path
keys=('REPAIRED_BUNDLE','TABLE1_RUN','TABLE1_PLAN','TABLE1_EVAL_ROOT','EXPECTED_RUNTIME_SHA',
      'EXPECTED_PLAN_SHA','TABLE1_GATE','TABLE1_EVAL','TABLE1_SUMMARY')
receipt={k:os.environ[k] for k in keys}
receipt.update(submitted_at=time.time(),partition='a100_tandon',time_limit='01:00:00',
    gate_indices=[0,28,56,84,126,168],cells=210,rollouts=840,
    scope='Original Table-I query histories; repaired executor and RGB controller interfaces')
with (Path(os.environ['TABLE1_RUN'])/'submission.json').open('x') as f:
 json.dump(receipt,f,indent=2)
print(json.dumps(receipt,indent=2))
PY
scontrol show job "${TABLE1_GATE}"
scontrol show job "${TABLE1_EVAL}"
scontrol show job "${TABLE1_SUMMARY}"
