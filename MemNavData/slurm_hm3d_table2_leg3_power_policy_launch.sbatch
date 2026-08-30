#!/usr/bin/env bash
#SBATCH --job-name=h3T2PolicyLaunch
#SBATCH --output=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/%x_%j.out
#SBATCH --error=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/%x_%j.err
#SBATCH --time=00:15:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --partition=cpu_short
#SBATCH --account=torch_pr_769_tandon_advanced

set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
: "${TASK_ROOT:?}" "${TASK_RECEIPT:?}" "${EXPECTED_TASK_RECEIPT_SHA:?}"
: "${BASE_SOURCE_ROOT:?}" "${BASE_RECEIPT:?}" "${EXPECTED_BASE_RECEIPT_SHA:?}"
: "${SERVER_SOURCE_ROOT:?}" "${SERVER_SOURCE_RECEIPT:?}"
: "${EXPECTED_SERVER_SOURCE_RECEIPT_SHA:?}"
: "${SOURCE_RUN_ROOT:?}" "${TABLE2_RUN_ROOT:?}" "${PROTOCOL:?}"
: "${PARENT_MANIFEST:?}"
POLICY_GPU_PARTITION=${POLICY_GPU_PARTITION:-a100_tandon}
POLICY_LAUNCHER_ROOT=${POLICY_LAUNCHER_ROOT:-${TASK_ROOT}}
POLICY_LAUNCHER_RECEIPT=${POLICY_LAUNCHER_RECEIPT:-${TASK_RECEIPT}}
EXPECTED_POLICY_LAUNCHER_RECEIPT_SHA=${EXPECTED_POLICY_LAUNCHER_RECEIPT_SHA:-${EXPECTED_TASK_RECEIPT_SHA}}
[[ "${POLICY_GPU_PARTITION}" == a100_tandon ]] || {
  echo "ABORT: Table-2 formal policy evaluation is pinned to a100_tandon" >&2
  exit 2
}
[[ "$(sha256sum "${POLICY_LAUNCHER_RECEIPT}" | awk '{print $1}')" == \
   "${EXPECTED_POLICY_LAUNCHER_RECEIPT_SHA}" ]]
(cd "${POLICY_LAUNCHER_ROOT}" && sha256sum -c --quiet \
  "${POLICY_LAUNCHER_RECEIPT}")
[[ "$(sha256sum "${TASK_RECEIPT}" | awk '{print $1}')" == \
   "${EXPECTED_TASK_RECEIPT_SHA}" ]]
(cd "${TASK_ROOT}" && sha256sum -c --quiet "${TASK_RECEIPT}")
verification=${TABLE2_RUN_ROOT}/hm3d_table2_leg3_construction_verification.json
(cd "${TABLE2_RUN_ROOT}" && sha256sum -c --quiet \
  hm3d_table2_leg3_construction_verification.json.sha256)
bench=${TABLE2_RUN_ROOT}/population/natural_direction
py=/scratch/lg154/conda-envs/memnav/bin/python
readarray -t gate < <("${py}" - "${verification}" "${bench}/manifest.json" \
  "${PARENT_MANIFEST}" <<'PY'
import hashlib,json,sys
v=json.load(open(sys.argv[1])); manifest=sys.argv[2]
assert v['verified'] is True and v['construction_only'] is True
assert v['policy_outcomes_read'] is False
assert v['formal_policy_evaluation_authorized'] is True
assert hashlib.sha256(open(manifest,'rb').read()).hexdigest()==v['benchmark_manifest_sha256']
p=json.load(open(sys.argv[3])); assert isinstance(p['scenes'],list) and p['scenes']
print(v['histories']); print(v['scene_clusters'])
print(hashlib.sha256(open(sys.argv[1],'rb').read()).hexdigest())
print(len(p['scenes']))
PY
)
[[ "${#gate[@]}" -eq 4 && "${gate[0]}" -ge 16 && "${gate[1]}" -ge 10 ]]
last=$((gate[3] - 1))
formal_root=${TABLE2_RUN_ROOT}/policy
mkdir -p "${formal_root}/smoke" "${formal_root}/formal" \
  "${RUN_ROOT:-${TABLE2_RUN_ROOT}}/deferred_submission"
pair=${TASK_ROOT}/MemNavData/slurm_hm3d_table2_leg3_navdp_pair.sbatch
analysis=${TASK_ROOT}/MemNavData/slurm_hm3d_table2_leg3_navdp_analysis.sbatch
safe=${TASK_ROOT}/MemNavData/slurm_safe_submit.sh
common="ALL,TASK_ROOT=${TASK_ROOT},TASK_RECEIPT=${TASK_RECEIPT},EXPECTED_TASK_RECEIPT_SHA=${EXPECTED_TASK_RECEIPT_SHA},BASE_SOURCE_ROOT=${BASE_SOURCE_ROOT},BASE_RECEIPT=${BASE_RECEIPT},EXPECTED_BASE_RECEIPT_SHA=${EXPECTED_BASE_RECEIPT_SHA},SERVER_SOURCE_ROOT=${SERVER_SOURCE_ROOT},SERVER_SOURCE_RECEIPT=${SERVER_SOURCE_RECEIPT},EXPECTED_SERVER_SOURCE_RECEIPT_SHA=${EXPECTED_SERVER_SOURCE_RECEIPT_SHA},SOURCE_RUN_ROOT=${SOURCE_RUN_ROOT},BENCH_ROOT=${bench},PARENT_MANIFEST=${PARENT_MANIFEST},PROTOCOL=${PROTOCOL},CONSTRUCTION_VERIFICATION=${verification},EXPECTED_CONSTRUCTION_VERIFICATION_SHA=${gate[2]},FORMAL_RUN_ROOT=${formal_root},ROLE_PAIR_SCOPE=paper_heldout"
source "${safe}"
safe_sbatch --lint-fatal --test-only --partition="${POLICY_GPU_PARTITION}" \
  --qos=gpu48 --time=01:00:00 \
  --array=0 --export="${common},PHASE=smoke" "${pair}" >/dev/null
safe_sbatch --lint-fatal --test-only --partition="${POLICY_GPU_PARTITION}" \
  --qos=gpu48 --time=01:00:00 \
  --array="0-${last}%4" --export="${common},PHASE=formal" "${pair}" >/dev/null
safe_sbatch --lint-fatal --test-only --partition=cpu_short \
  --export="${common},MODE=aggregate" "${analysis}" >/dev/null

s_raw=$(safe_sbatch --lint-fatal --parsable \
  --partition="${POLICY_GPU_PARTITION}" --qos=gpu48 --time=01:00:00 \
  --array=0 --export="${common},PHASE=smoke" "${pair}")
s_job=$(printf '%s\n' "${s_raw}" | tr -d '\r' | awk -F';' '/^[0-9]+(;|$)/ {print $1; exit}')
[[ "${s_job}" =~ ^[0-9]+$ ]]
f_raw=$(safe_sbatch --lint-fatal --parsable \
  --partition="${POLICY_GPU_PARTITION}" --qos=gpu48 --time=01:00:00 \
  --array="0-${last}%4" --dependency="afterok:${s_job}" \
  --kill-on-invalid-dep=yes --export="${common},PHASE=formal" "${pair}")
f_job=$(printf '%s\n' "${f_raw}" | tr -d '\r' | awk -F';' '/^[0-9]+(;|$)/ {print $1; exit}')
[[ "${f_job}" =~ ^[0-9]+$ ]]
a_raw=$(safe_sbatch --lint-fatal --parsable --partition=cpu_short \
  --dependency="afterok:${f_job}" --kill-on-invalid-dep=yes \
  --export="${common},MODE=aggregate" "${analysis}")
a_job=$(printf '%s\n' "${a_raw}" | tr -d '\r' | awk -F';' '/^[0-9]+(;|$)/ {print $1; exit}')
[[ "${a_job}" =~ ^[0-9]+$ ]]
v_raw=$(safe_sbatch --lint-fatal --parsable --partition=cpu_short \
  --dependency="afterok:${a_job}" --kill-on-invalid-dep=yes \
  --export="${common},MODE=verify" "${analysis}")
v_job=$(printf '%s\n' "${v_raw}" | tr -d '\r' | awk -F';' '/^[0-9]+(;|$)/ {print $1; exit}')
[[ "${v_job}" =~ ^[0-9]+$ ]]

receipt=${TABLE2_RUN_ROOT}/policy_submission.json
"${py}" - "${receipt}" "${gate[0]}" "${gate[1]}" "${s_job}" \
  "${f_job}" "${a_job}" "${v_job}" <<'PY'
import json,sys
path,histories,scenes,smoke,formal,aggregate,verify=sys.argv[1:]
p={'schema_version':'hm3d_table2_power_policy_submission_v1_20260830',
   'leg3_histories':int(histories),'scene_clusters':int(scenes),
   'policy_gpu_partition':'a100_tandon',
   'smoke_job':int(smoke),'formal_scene_array_job':int(formal),
   'aggregate_job':int(aggregate),'independent_verify_job':int(verify),
   'partial_policy_outcomes_read_at_submission':False,
   'runtime_role_visibility':'none'}
open(path,'x').write(json.dumps(p,indent=2,sort_keys=True)+'\n')
PY
sha256sum "${receipt}" >"${receipt}.sha256"
chmod a-w "${receipt}" "${receipt}.sha256"
