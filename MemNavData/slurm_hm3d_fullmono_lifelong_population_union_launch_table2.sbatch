#!/usr/bin/env bash
#SBATCH --job-name=h3LifeUnionT2
#SBATCH --output=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/%x_%j.out
#SBATCH --error=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/%x_%j.err
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --partition=cpu_short
#SBATCH --account=torch_pr_769_tandon_advanced

set -euo pipefail
umask 0022
: "${TASK_ROOT:?}" "${TASK_RECEIPT:?}" "${EXPECTED_TASK_RECEIPT_SHA:?}"
: "${BASE_SOURCE_ROOT:?}" "${BASE_RECEIPT:?}" "${EXPECTED_BASE_RECEIPT_SHA:?}"
: "${TABLE2_SERVER_SOURCE_ROOT:?}" "${TABLE2_SERVER_SOURCE_RECEIPT:?}"
: "${EXPECTED_TABLE2_SERVER_SOURCE_RECEIPT_SHA:?}"
: "${RUN_ROOT:?}" "${PARENT_ROOT:?}" "${PROTOCOL:?}"
: "${EXPECTED_PROTOCOL_SHA:?}" "${PARENT_MANIFEST:?}"

for spec in \
  "${TASK_RECEIPT}:${EXPECTED_TASK_RECEIPT_SHA}:${TASK_ROOT}" \
  "${BASE_RECEIPT}:${EXPECTED_BASE_RECEIPT_SHA}:${BASE_SOURCE_ROOT}" \
  "${TABLE2_SERVER_SOURCE_RECEIPT}:${EXPECTED_TABLE2_SERVER_SOURCE_RECEIPT_SHA}:${TABLE2_SERVER_SOURCE_ROOT}"; do
  IFS=: read -r receipt expected root <<<"${spec}"
  [[ "$(sha256sum "${receipt}" | awk '{print $1}')" == "${expected}" ]]
  (cd "${root}" && sha256sum -c --quiet "${receipt}")
done
[[ "$(sha256sum "${PROTOCOL}" | awk '{print $1}')" == \
   "${EXPECTED_PROTOCOL_SHA}" ]]
(cd "${RUN_ROOT}" && sha256sum -c --quiet \
  independent_natural_b_expansion_population_verification.json.sha256)

py=/scratch/lg154/conda-envs/memnav/bin/python
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPYCACHEPREFIX=${SLURM_TMPDIR:-/tmp}/h3_life_union_${SLURM_JOB_ID}
export PYTHONPATH=${TASK_ROOT}:${TASK_ROOT}/MemNavData
union_root=${RUN_ROOT}/table2_source_union
base_protocol=${TASK_ROOT}/MemNavData/hm3d_table2_leg3_mixed_role_protocol_20260829.json
"${py}" -u \
  "${TASK_ROOT}/MemNavData/merge_hm3d_fullmono_lifelong_populations.py" \
  --expansion-protocol "${PROTOCOL}" --expansion-run "${RUN_ROOT}" \
  --base-table2-protocol "${base_protocol}" --out "${union_root}"
union_verification=${union_root}/independent_population_union_verification.json
"${py}" -u \
  "${TASK_ROOT}/MemNavData/independent_verify_hm3d_fullmono_lifelong_population_union.py" \
  --root "${union_root}" --out "${union_verification}"
(cd "${union_root}" && sha256sum -c --quiet \
  independent_population_union_verification.json.sha256)
readarray -t gate < <("${py}" - "${union_verification}" <<'PY'
import json,sys
p=json.load(open(sys.argv[1]))
assert p['verified'] is True and p['result_blind'] is True
assert p['leg3_query_navigation_outcomes_read'] is False
assert p['target_met'] is True
print(p['union_supported_histories'])
print(p['union_scene_clusters'])
print(p['table2_protocol_sha256'])
PY
)
[[ "${#gate[@]}" -eq 3 && "${gate[0]}" -ge 40 && "${gate[1]}" -ge 15 ]]

table2_root=${RUN_ROOT}/table2_leg3_power
smoke_root=${RUN_ROOT}/table2_leg3_power_smoke
[[ ! -e "${table2_root}" && ! -e "${smoke_root}" ]]
mkdir -p "${table2_root}/construction/fragments" "${table2_root}/logs" \
  "${smoke_root}/construction/fragments" "${smoke_root}/logs"
table2_protocol=${union_root}/hm3d_table2_leg3_power_protocol.json
construct=${TASK_ROOT}/MemNavData/slurm_hm3d_table2_leg3_construct.sbatch
analysis=${TASK_ROOT}/MemNavData/slurm_hm3d_table2_leg3_analysis.sbatch
policy_launch=${TASK_ROOT}/MemNavData/slurm_hm3d_table2_leg3_power_policy_launch.sbatch
safe=${TASK_ROOT}/MemNavData/slurm_safe_submit.sh
common="ALL,TASK_ROOT=${TASK_ROOT},TASK_RECEIPT=${TASK_RECEIPT},EXPECTED_TASK_RECEIPT_SHA=${EXPECTED_TASK_RECEIPT_SHA},BASE_SOURCE_ROOT=${BASE_SOURCE_ROOT},BASE_RECEIPT=${BASE_RECEIPT},EXPECTED_BASE_RECEIPT_SHA=${EXPECTED_BASE_RECEIPT_SHA},SERVER_SOURCE_ROOT=${TABLE2_SERVER_SOURCE_ROOT},SERVER_SOURCE_RECEIPT=${TABLE2_SERVER_SOURCE_RECEIPT},EXPECTED_SERVER_SOURCE_RECEIPT_SHA=${EXPECTED_TABLE2_SERVER_SOURCE_RECEIPT_SHA},SOURCE_RUN_ROOT=${union_root},RUN_ROOT=${table2_root},PROTOCOL=${table2_protocol},TABLE2_RUN_ROOT=${table2_root},PARENT_MANIFEST=${PARENT_MANIFEST}"
source "${safe}"
last=$((gate[0] - 1))
safe_sbatch --lint-fatal --test-only --qos=gpu48 --time=00:45:00 \
  --array=0 --export="${common},RUN_ROOT=${smoke_root}" "${construct}" >/dev/null
safe_sbatch --lint-fatal --test-only --qos=gpu48 --time=00:45:00 \
  --array="0-${last}%4" --export="${common}" "${construct}" >/dev/null
safe_sbatch --lint-fatal --test-only --partition=cpu_short \
  --export="${common},MODE=finalize" "${analysis}" >/dev/null
safe_sbatch --lint-fatal --test-only --partition=cpu_short \
  --export="${common},MODE=verify" "${analysis}" >/dev/null
safe_sbatch --lint-fatal --test-only --partition=cpu_short \
  --export="${common}" "${policy_launch}" >/dev/null

s_raw=$(safe_sbatch --lint-fatal --parsable --qos=gpu48 --time=00:45:00 \
  --array=0 --export="${common},RUN_ROOT=${smoke_root}" "${construct}")
s_job=$(printf '%s\n' "${s_raw}" | tr -d '\r' | awk -F';' '/^[0-9]+(;|$)/ {print $1; exit}')
[[ "${s_job}" =~ ^[0-9]+$ ]]
c_raw=$(safe_sbatch --lint-fatal --parsable --qos=gpu48 --time=00:45:00 \
  --array="0-${last}%4" --dependency="afterok:${s_job}" \
  --kill-on-invalid-dep=yes --export="${common}" "${construct}")
c_job=$(printf '%s\n' "${c_raw}" | tr -d '\r' | awk -F';' '/^[0-9]+(;|$)/ {print $1; exit}')
[[ "${c_job}" =~ ^[0-9]+$ ]]
f_raw=$(safe_sbatch --lint-fatal --parsable --partition=cpu_short \
  --dependency="afterok:${c_job}" --kill-on-invalid-dep=yes \
  --export="${common},MODE=finalize" "${analysis}")
f_job=$(printf '%s\n' "${f_raw}" | tr -d '\r' | awk -F';' '/^[0-9]+(;|$)/ {print $1; exit}')
[[ "${f_job}" =~ ^[0-9]+$ ]]
v_raw=$(safe_sbatch --lint-fatal --parsable --partition=cpu_short \
  --dependency="afterok:${f_job}" --kill-on-invalid-dep=yes \
  --export="${common},MODE=verify" "${analysis}")
v_job=$(printf '%s\n' "${v_raw}" | tr -d '\r' | awk -F';' '/^[0-9]+(;|$)/ {print $1; exit}')
[[ "${v_job}" =~ ^[0-9]+$ ]]
p_raw=$(safe_sbatch --lint-fatal --parsable --partition=cpu_short \
  --dependency="afterok:${v_job}" --kill-on-invalid-dep=yes \
  --export="${common}" "${policy_launch}")
p_job=$(printf '%s\n' "${p_raw}" | tr -d '\r' | awk -F';' '/^[0-9]+(;|$)/ {print $1; exit}')
[[ "${p_job}" =~ ^[0-9]+$ ]]

receipt=${RUN_ROOT}/deferred_submission/table2_power_construction.json
mkdir -p "${receipt%/*}"
"${py}" - "${receipt}" "${gate[0]}" "${gate[1]}" "${s_job}" \
  "${c_job}" "${f_job}" "${v_job}" "${p_job}" <<'PY'
import json,sys
path,histories,scenes,smoke,build,finalize,verify,policy=sys.argv[1:]
p={'schema_version':'hm3d_table2_power_construction_submission_v1_20260830',
   'source_histories':int(histories),'source_scene_clusters':int(scenes),
   'construction_smoke_job':int(smoke),'construction_array_job':int(build),
   'construction_finalize_job':int(finalize),
   'construction_independent_verify_job':int(verify),
   'policy_deferred_launcher_job':int(policy),
   'leg3_policy_outcomes_read_at_submission':False}
open(path,'x').write(json.dumps(p,indent=2,sort_keys=True)+'\n')
PY
sha256sum "${receipt}" >"${receipt}.sha256"
chmod -R a-w "${union_root}" "${receipt}" "${receipt}.sha256"
