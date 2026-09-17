#!/usr/bin/env bash
# Fixed full population; reducers launch only already-declared downstream queries.
set -euo pipefail
: "${REPAIRED_BUNDLE:?}" "${TABLE2_PLAN:?}" "${TABLE2_RUN:?}" "${EXPECTED_RUNTIME_SHA:?}"
[[ "$(id -un)" == yz11502 ]]
TABLE2_PY=/scratch/lg154/conda-envs/memnav/bin/python
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export PYTHONPATH=${REPAIRED_BUNDLE}:${REPAIRED_BUNDLE}/MemNavData
export EXPECTED_PLAN_SHA=$(sha256sum "${TABLE2_PLAN}" | awk '{print $1}')
export TABLE2_WORK_ROOT=$("${TABLE2_PY}" -c 'import json,os; print(json.load(open(os.environ["TABLE2_PLAN"]))["work_root"])')
source "${REPAIRED_BUNDLE}/MemNavData/slurm_safe_submit.sh"
TABLE2_GPU=(--partition=a100_tandon --account=torch_pr_769_tandon_advanced --qos=gpu48
            --gres=gpu:1 --cpus-per-task=10 --mem=96G --time=01:00:00)
TABLE2_CPU=(--partition=cpu_short --account=torch_pr_769_tandon_advanced
            --cpus-per-task=2 --mem=8G --time=00:30:00)
TABLE2_EXPORT="ALL,REPAIRED_BUNDLE=${REPAIRED_BUNDLE},TABLE2_PLAN=${TABLE2_PLAN},TABLE2_RUN=${TABLE2_RUN},TABLE2_WORK_ROOT=${TABLE2_WORK_ROOT},EXPECTED_RUNTIME_SHA=${EXPECTED_RUNTIME_SHA},EXPECTED_PLAN_SHA=${EXPECTED_PLAN_SHA}"
TABLE2_EVAL=${REPAIRED_BUNDLE}/MemNavData/slurm_table2_mixed_eval.sbatch
TABLE2_REDUCER=${REPAIRED_BUNDLE}/MemNavData/slurm_table2_full_reduce.sbatch
TABLE2_MODE=${1:-initial}
TABLE2_CONCURRENCY=$("${TABLE2_PY}" - <<'PY'
import json,os
from MemNavData.table2_sampling_profiles import FORWARD_SCHEMA
with open(os.environ['TABLE2_PLAN']) as stream:
    plan=json.load(stream)
print(4 if plan['schema']==FORWARD_SCHEMA else 2)
PY
)
export TABLE2_CONCURRENCY
if [[ "${TABLE2_MODE}" == initial ]]; then
  TABLE2_STAGE=A
  TABLE2_NEXT=select_b
  "${TABLE2_PY}" - <<'PY'
import json,os
from pathlib import Path
from MemNavData.table2_mixed_hpc import validate_plan,formal
from MemNavData.table2_mixed_local import dump
p=validate_plan(json.loads(Path(os.environ['TABLE2_PLAN']).read_text()))
v=json.loads((Path(os.environ['TABLE2_RUN'])/'preflight/table2/verification.json').read_text())
assert formal(p) and len(p['sources'])==144 and p['source_scenes']==18
assert v['verified'] and v['plan_sha256']==os.environ['EXPECTED_PLAN_SHA']
assert v['runtime_sha256']==os.environ['EXPECTED_RUNTIME_SHA']
indices=list(range(len(p['sources'])))
gate=next(s['index'] for s in p['sources'] if s['a_selection'])
indices.remove(gate)
dump(Path(os.environ['TABLE2_RUN'])/'A_launch_indices.json',dict(indices=[gate]+indices))
PY
elif [[ "${TABLE2_MODE}" == b ]]; then
  TABLE2_STAGE=B
  TABLE2_NEXT=select_c
  "${TABLE2_PY}" "${REPAIRED_BUNDLE}/MemNavData/table2_formal_population.py" empty-b \
    --plan "${TABLE2_PLAN}" --run "${TABLE2_RUN}" --out "${TABLE2_RUN}/B_launch_indices.json"
elif [[ "${TABLE2_MODE}" == c ]]; then
  TABLE2_STAGE=C
  TABLE2_NEXT=summary
  "${TABLE2_PY}" - <<'PY'
import os
from pathlib import Path
from MemNavData.table2_mixed_hpc import population_at
from MemNavData.table2_mixed_local import dump
r=Path(os.environ['TABLE2_RUN'])
p=population_at(Path(os.environ['TABLE2_PLAN']),r)
dump(r/'C_launch_indices.json',dict(indices=list(range(p['c_query_tasks']))))
PY
else
  echo "Use initial, b or c" >&2
  exit 2
fi
export TABLE2_STAGE TABLE2_NEXT
# Keep the v2 resource addendum: A may use four workers; B/C stay at two.
if [[ "${TABLE2_STAGE}" != A ]]; then
  TABLE2_CONCURRENCY=2
fi
[[ ! -e "${TABLE2_RUN}/${TABLE2_STAGE}_submission.json" ]]
mapfile -t TABLE2_ARRAYS < <("${TABLE2_PY}" - <<'PY'
import os,json
from pathlib import Path
x=json.loads((Path(os.environ['TABLE2_RUN'])/(os.environ['TABLE2_STAGE']+'_launch_indices.json')).read_text())['indices']
print(x[0] if x else 'none')
print(','.join(map(str,x[1:])) if len(x)>1 else 'none')
PY
)
TABLE2_GATE=none
TABLE2_REST=none
TABLE2_DEPS=()
if [[ "${TABLE2_ARRAYS[0]}" != none ]]; then
  safe_sbatch --lint-fatal --test-only "${TABLE2_GPU[@]}" --array="${TABLE2_ARRAYS[0]}" \
    --export="${TABLE2_EXPORT},TABLE2_STAGE=${TABLE2_STAGE}" "${TABLE2_EVAL}"
  TABLE2_GATE=$(safe_sbatch --lint-fatal --parsable "${TABLE2_GPU[@]}" \
    --job-name="table2_full_${TABLE2_STAGE}_gate" --array="${TABLE2_ARRAYS[0]}" \
    --export="${TABLE2_EXPORT},TABLE2_STAGE=${TABLE2_STAGE}" "${TABLE2_EVAL}")
  TABLE2_DEPS=(--dependency="afterany:${TABLE2_GATE}")
  if [[ "${TABLE2_ARRAYS[1]}" != none ]]; then
    TABLE2_REST=$(safe_sbatch --lint-fatal --parsable "${TABLE2_GPU[@]}" \
      --job-name="table2_full_${TABLE2_STAGE}" --array="${TABLE2_ARRAYS[1]}%${TABLE2_CONCURRENCY}" \
      --dependency="afterok:${TABLE2_GATE}" --kill-on-invalid-dep=yes \
      --export="${TABLE2_EXPORT},TABLE2_STAGE=${TABLE2_STAGE}" "${TABLE2_EVAL}")
    TABLE2_DEPS=(--dependency="afterany:${TABLE2_GATE}:${TABLE2_REST}")
  fi
fi
TABLE2_REDUCE_JOB=$(safe_sbatch --lint-fatal --parsable "${TABLE2_CPU[@]}" "${TABLE2_DEPS[@]}" \
  --export="${TABLE2_EXPORT},TABLE2_REDUCE=${TABLE2_NEXT}" "${TABLE2_REDUCER}")
export TABLE2_GATE TABLE2_REST TABLE2_REDUCE_JOB
"${TABLE2_PY}" - <<'PY'
import os,time
from pathlib import Path
from MemNavData.table2_mixed_local import dump
keys=('TABLE2_STAGE','TABLE2_GATE','TABLE2_REST','TABLE2_REDUCE_JOB','TABLE2_NEXT','EXPECTED_PLAN_SHA','EXPECTED_RUNTIME_SHA')
r={k:os.environ[k] for k in keys}
r.update(submitted_at=time.time(),time_limit='01:00:00',array_concurrency=int(os.environ['TABLE2_CONCURRENCY']))
dump(Path(os.environ['TABLE2_RUN'])/(os.environ['TABLE2_STAGE']+'_submission.json'),r)
print(r)
PY
