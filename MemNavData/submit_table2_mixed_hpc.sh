#!/usr/bin/env bash
# Finite pilot only. Dependent arrays never expand the frozen source budget.
set -euo pipefail
: "${REPAIRED_BUNDLE:?}" "${TABLE2_PLAN:?}" "${TABLE2_RUN:?}" "${EXPECTED_RUNTIME_SHA:?}"
[[ "$(id -un)" == yz11502 ]]
export EXPECTED_PLAN_SHA=$(sha256sum "${TABLE2_PLAN}" | awk '{print $1}')
export PYTHONPATH=${REPAIRED_BUNDLE}:${REPAIRED_BUNDLE}/MemNavData
export TABLE2_WORK_ROOT=$(/scratch/lg154/conda-envs/memnav/bin/python -c \
  'import json,os; print(json.load(open(os.environ["TABLE2_PLAN"]))["work_root"])')
source "${REPAIRED_BUNDLE}/MemNavData/slurm_safe_submit.sh"
TABLE2_GPU=(--partition=a100_tandon --account=torch_pr_769_tandon_advanced --qos=gpu48
            --gres=gpu:1 --cpus-per-task=10 --mem=96G --time=01:00:00)
TABLE2_CPU=(--partition=cpu_short --account=torch_pr_769_tandon_advanced
            --cpus-per-task=2 --mem=8G --time=00:30:00)
TABLE2_EXPORT="ALL,REPAIRED_BUNDLE=${REPAIRED_BUNDLE},TABLE2_PLAN=${TABLE2_PLAN},TABLE2_RUN=${TABLE2_RUN},TABLE2_WORK_ROOT=${TABLE2_WORK_ROOT},EXPECTED_RUNTIME_SHA=${EXPECTED_RUNTIME_SHA},EXPECTED_PLAN_SHA=${EXPECTED_PLAN_SHA}"
TABLE2_EVAL=${REPAIRED_BUNDLE}/MemNavData/slurm_table2_mixed_eval.sbatch
TABLE2_REDUCER=${REPAIRED_BUNDLE}/MemNavData/slurm_table2_mixed_reduce.sbatch
if [[ "${1:-initial}" == initial ]]; then
  [[ ! -e "${TABLE2_RUN}/submission.json" && ! -e "${TABLE2_RUN}/a_gate_job.txt" ]]
  /scratch/lg154/conda-envs/memnav/bin/python - <<'PY'
import json, os
from pathlib import Path
from MemNavData.table2_mixed_hpc import validate_plan
p=validate_plan(json.loads(Path(os.environ['TABLE2_PLAN']).read_text()))
v=json.loads((Path(os.environ['TABLE2_RUN'])/'preflight/table2/verification.json').read_text())
assert len(p['sources'])==8 and not p['formal_result']
assert v['verified'] and v['plan_sha256']==os.environ['EXPECTED_PLAN_SHA']
assert v['runtime_sha256']==os.environ['EXPECTED_RUNTIME_SHA']
PY
  safe_sbatch --lint-fatal --test-only "${TABLE2_GPU[@]}" --array=0 \
    --export="${TABLE2_EXPORT},TABLE2_STAGE=A" "${TABLE2_EVAL}"
  safe_sbatch --lint-fatal --test-only "${TABLE2_CPU[@]}" \
    --export="${TABLE2_EXPORT},TABLE2_REDUCE=select_c" "${TABLE2_REDUCER}"
  TABLE2_A_GATE=$(safe_sbatch --lint-fatal --parsable "${TABLE2_GPU[@]}" --array=0 \
    --export="${TABLE2_EXPORT},TABLE2_STAGE=A" "${TABLE2_EVAL}")
  printf '%s\n' "${TABLE2_A_GATE}" >"${TABLE2_RUN}/a_gate_job.txt"
  TABLE2_A=$(safe_sbatch --lint-fatal --parsable "${TABLE2_GPU[@]}" --array=1-7%2 \
    --dependency="afterok:${TABLE2_A_GATE}" --kill-on-invalid-dep=yes \
    --export="${TABLE2_EXPORT},TABLE2_STAGE=A" "${TABLE2_EVAL}")
  printf '%s\n' "${TABLE2_A}" >"${TABLE2_RUN}/a_job.txt"
  TABLE2_B=$(safe_sbatch --lint-fatal --parsable "${TABLE2_GPU[@]}" --array=0-15%2 \
    --dependency="afterok:${TABLE2_A_GATE}:${TABLE2_A}" --kill-on-invalid-dep=yes \
    --export="${TABLE2_EXPORT},TABLE2_STAGE=B" "${TABLE2_EVAL}")
  printf '%s\n' "${TABLE2_B}" >"${TABLE2_RUN}/b_job.txt"
  TABLE2_SELECT=$(safe_sbatch --lint-fatal --parsable "${TABLE2_CPU[@]}" \
    --dependency="afterok:${TABLE2_B}" --kill-on-invalid-dep=yes \
    --export="${TABLE2_EXPORT},TABLE2_REDUCE=select_c" "${TABLE2_REDUCER}")
  printf '%s\n' "${TABLE2_SELECT}" >"${TABLE2_RUN}/c_selection_job.txt"
  export TABLE2_A_GATE TABLE2_A TABLE2_B TABLE2_SELECT
  /scratch/lg154/conda-envs/memnav/bin/python - <<'PY'
import json, os, time
from pathlib import Path
keys=('TABLE2_A_GATE','TABLE2_A','TABLE2_B','TABLE2_SELECT','REPAIRED_BUNDLE',
      'TABLE2_PLAN','EXPECTED_PLAN_SHA','EXPECTED_RUNTIME_SHA','TABLE2_WORK_ROOT')
r={k:os.environ[k] for k in keys}
r.update(submitted_at=time.time(),formal_result=False,sources=8,scenes=4,
         gpu_partition='a100_tandon',time_limit='01:00:00',array_concurrency=2)
with (Path(os.environ['TABLE2_RUN'])/'submission.json').open('x') as f: json.dump(r,f,indent=2)
print(json.dumps(r,indent=2))
PY
  scontrol show job "${TABLE2_A_GATE}"
  scontrol show job "${TABLE2_A}"
  scontrol show job "${TABLE2_B}"
  scontrol show job "${TABLE2_SELECT}"
elif [[ "$1" == c ]]; then
  [[ ! -e "${TABLE2_RUN}/c_submission.json" && ! -e "${TABLE2_RUN}/c_job.txt" ]]
  TABLE2_C_COUNT=$(/scratch/lg154/conda-envs/memnav/bin/python -c \
    'import os; from pathlib import Path; from MemNavData.table2_mixed_hpc import population_at; print(population_at(Path(os.environ["TABLE2_PLAN"]),Path(os.environ["TABLE2_RUN"]))["c_query_tasks"])')
  if (( TABLE2_C_COUNT > 0 )); then
    TABLE2_C=$(safe_sbatch --lint-fatal --parsable "${TABLE2_GPU[@]}" \
      --array="0-$((TABLE2_C_COUNT-1))%2" \
      --export="${TABLE2_EXPORT},TABLE2_STAGE=C" "${TABLE2_EVAL}")
    printf '%s\n' "${TABLE2_C}" >"${TABLE2_RUN}/c_job.txt"
    TABLE2_SUMMARY=$(safe_sbatch --lint-fatal --parsable "${TABLE2_CPU[@]}" \
      --dependency="afterok:${TABLE2_C}" --kill-on-invalid-dep=yes \
      --export="${TABLE2_EXPORT},TABLE2_REDUCE=summary" "${TABLE2_REDUCER}")
  else
    TABLE2_C=none
    TABLE2_SUMMARY=$(safe_sbatch --lint-fatal --parsable "${TABLE2_CPU[@]}" \
      --export="${TABLE2_EXPORT},TABLE2_REDUCE=summary" "${TABLE2_REDUCER}")
  fi
  export TABLE2_C TABLE2_C_COUNT TABLE2_SUMMARY
  /scratch/lg154/conda-envs/memnav/bin/python - <<'PY'
import json,os
from pathlib import Path
r={k:os.environ[k] for k in ('TABLE2_C','TABLE2_C_COUNT','TABLE2_SUMMARY')}
with (Path(os.environ['TABLE2_RUN'])/'c_submission.json').open('x') as f: json.dump(r,f,indent=2)
print(json.dumps(r,indent=2))
PY
else
  echo "Use initial or c" >&2
  exit 2
fi
