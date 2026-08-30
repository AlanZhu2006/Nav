#!/usr/bin/env bash
#SBATCH --job-name=h3T3ARepairDone
#SBATCH --output=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/%x_%j.out
#SBATCH --error=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/%x_%j.err
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --partition=cpu_short
#SBATCH --account=torch_pr_769_tandon_advanced

set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
: "${WRAPPER_ROOT:?}" "${WRAPPER_RECEIPT:?}" "${EXPECTED_WRAPPER_RECEIPT_SHA:?}"
: "${RUN_ROOT:?}" "${TABLE3_PLAN:?}" "${EXPECTED_TABLE3_PLAN_SHA:?}"
: "${REPAIR_PLAN:?}" "${REPAIR_ARRAY_JOB:?}"
: "${REPAIR_CONTRACT:?}" "${EXPECTED_REPAIR_CONTRACT_SHA:?}"
[[ "$(sha256sum "${WRAPPER_RECEIPT}" | awk '{print $1}')" == \
   "${EXPECTED_WRAPPER_RECEIPT_SHA}" ]]
(cd "${WRAPPER_ROOT}" && sha256sum -c --quiet "${WRAPPER_RECEIPT}")
[[ "$(sha256sum "${REPAIR_CONTRACT}" | awk '{print $1}')" == \
   "${EXPECTED_REPAIR_CONTRACT_SHA}" ]]
repair_root=$(dirname "${REPAIR_PLAN}")
out=${repair_root}/completion.json
test ! -e "${out}"
/scratch/lg154/conda-envs/memnav/bin/python - "${RUN_ROOT}" "${TABLE3_PLAN}" \
  "${EXPECTED_TABLE3_PLAN_SHA}" "${REPAIR_PLAN}" "${REPAIR_ARRAY_JOB}" \
  "${out}" <<'PY'
import hashlib,json,sys
from pathlib import Path
run=Path(sys.argv[1]); plan_path=Path(sys.argv[2]); expected=sys.argv[3]
repair_path=Path(sys.argv[4]); job=int(sys.argv[5]); out=Path(sys.argv[6])
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8<<20),b''): h.update(block)
    return h.hexdigest()
assert sha(plan_path)==expected
plan=json.load(plan_path.open()); repair=json.load(repair_path.open())
assert len(plan['episodes'])==125
assert repair['candidate_plan_sha256']==expected
assert repair['navigation_outcomes_read'] is False
assert repair['fallback_completion_allowed'] is False
verification_path=repair_path.parent/'independent_verification.json'
verification_sidecar=verification_path.with_name(
    'independent_verification.json.sha256')
assert verification_path.is_file() and verification_sidecar.is_file()
assert verification_sidecar.read_text().split()==[
    sha(verification_path),'independent_verification.json']
verification=json.load(verification_path.open())
assert verification['verified'] is True
assert verification['navigation_outcomes_read'] is False
digests=[]
for i,row in enumerate(plan['episodes']):
    assert int(row['history_index'])==i
    label=f"{i:03d}_{row['scene']}_episode_{row['episode']}"
    completion=run/'factual_a'/label/'completion.json'
    sidecar=completion.with_name('completion.json.sha256')
    assert completion.is_file() and sidecar.is_file()
    digest=sha(completion)
    assert sidecar.read_text().split()==[digest,'completion.json']
    # Deliberately do not deserialize the outcome-bearing completion payload.
    digests.append({'history_index':i,'completion_sha256':digest})
payload={
 'schema_version':'hm3d_table3_actual_mono_a_transport_repair_completion_v1_20260830',
 'status':'complete','candidate_plan_sha256':expected,
 'repair_plan_sha256':sha(repair_path),'repair_array_job':job,
 'candidate_count':125,'completion_receipts_verified':125,
 'completion_receipt_digests':digests,
 'completion_payloads_deserialized':False,'navigation_outcomes_read':False,
 'query_policy_outcomes_read':False,'scientific_thresholds_changed':False,
 'fallback_completion_allowed':False,
}
out.write_text(json.dumps(payload,indent=2,sort_keys=True)+'\n')
out.with_name('completion.json.sha256').write_text(
    f"{sha(out)}  completion.json\n")
PY
chmod -R a-w "${repair_root}"
