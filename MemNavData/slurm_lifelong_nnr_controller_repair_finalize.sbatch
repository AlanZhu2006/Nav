#!/usr/bin/env bash
#SBATCH --job-name=lifeCtlRepairSum
#SBATCH --output=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/%x_%j.out
#SBATCH --error=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/%x_%j.err
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=cpu_short
#SBATCH --account=torch_pr_769_tandon_advanced
#SBATCH --export=ALL

# Post-audit the exact repair set, merge it with the 245 untouched originals,
# and independently verify all five complete 18-episode three-arm studies.
set -euo pipefail
umask 0022
export PYTHONDONTWRITEBYTECODE=1
: "${SOURCE_ROOT:?set immutable task source root}"
: "${SOURCE_RECEIPT:?set source receipt}"
: "${EXPECTED_SOURCE_RECEIPT_SHA:?set source receipt SHA}"
: "${ORIGINAL_RUN_ROOT:?set navdp-line population source}"
: "${REPAIR_ROOT:?set repair root}"

PY=/scratch/lg154/conda-envs/memnav/bin/python
MANIFEST=${SOURCE_ROOT}/MemNavData/lifelong_nnr_controller_repair_manifest_20260824.json
POPULATION=${ORIGINAL_RUN_ROOT}/factual_b_support/population.json
FORMAL=${REPAIR_ROOT}/formal
AUDIT=${REPAIR_ROOT}/post_repair_audit.json
[[ ! -e "${FORMAL}" && ! -e "${AUDIT}" && ! -e "${REPAIR_ROOT}/VERIFIED" ]]
[[ "$(sha256sum "${SOURCE_RECEIPT}" | awk '{print $1}')" == \
   "${EXPECTED_SOURCE_RECEIPT_SHA}" ]]
(cd "${SOURCE_ROOT}" && sha256sum -c --quiet "${SOURCE_RECEIPT}")
export PYTHONPATH=${SOURCE_ROOT}:${SOURCE_ROOT}/MemNavData
export PYTHONPYCACHEPREFIX=${SLURM_TMPDIR:-/tmp}/life_ctlrepair_sum_${SLURM_JOB_ID}

"${PY}" -u \
  "${SOURCE_ROOT}/MemNavData/audit_lifelong_nnr_controller_repair.py" \
  --manifest "${MANIFEST}" --population "${POPULATION}" \
  --repair-root "${REPAIR_ROOT}" --out "${AUDIT}"

mkdir -p "${FORMAL}"
while IFS=$'\t' read -r controller original; do
  controller_root=${FORMAL}/${controller}
  mkdir -p "${controller_root}"
  "${PY}" -u \
    "${SOURCE_ROOT}/MemNavData/aggregate_lifelong_nnr_expansion.py" \
    --population "${POPULATION}" \
    --evaluation-root "${original}/evaluation" \
    --repair-evaluation-root "${REPAIR_ROOT}/${controller}/evaluation" \
    --repair-manifest "${MANIFEST}" \
    --repair-controller "${controller}" \
    --scopes all_prior,initial_leg_only,forced_reject_native \
    --out "${controller_root}/aggregate"
  "${PY}" -u \
    "${SOURCE_ROOT}/MemNavData/independent_verify_shared_online_lifelong_nnr.py" \
    --all_prior "${controller_root}/aggregate/all_prior" \
    --initial_leg_only "${controller_root}/aggregate/initial_leg_only" \
    --forced_reject_native "${controller_root}/aggregate/forced_reject_native" \
    --population "${POPULATION}" \
    --out "${controller_root}/independent_verification.json"
  "${PY}" - "${controller_root}/independent_verification.json" <<'PY'
import json,sys
payload=json.load(open(sys.argv[1]))
if payload.get("verified") is not True:
    raise SystemExit("independent verification is false")
if int(payload.get("episodes", -1)) != 18:
    raise SystemExit("verified population is not 18")
if payload.get("forced_reject_native", {}).get("zero_takeover_decisions", 0) < 1:
    raise SystemExit("forced arm recorded no audited decisions")
PY
  sha256sum "${controller_root}/independent_verification.json" > \
    "${controller_root}/independent_verification.json.sha256"
done < <("${PY}" - "${MANIFEST}" <<'PY'
import json,sys
payload=json.load(open(sys.argv[1]))
for key,value in sorted(payload["original_controller_runs"].items()):
    print(f"{key}\t{value}")
PY
)

"${PY}" - "${FORMAL}" "${AUDIT}" \
  "${REPAIR_ROOT}/formal_summary.json" <<'PY'
import json,sys
from pathlib import Path
formal=Path(sys.argv[1]); audit=json.load(open(sys.argv[2])); out=Path(sys.argv[3])
controllers={}
for root in sorted(path for path in formal.iterdir() if path.is_dir()):
    verified=json.load(open(root/"independent_verification.json"))
    inputs=json.load(open(root/"aggregate/aggregate_inputs.json"))
    controllers[root.name]={
        "episodes": verified["episodes"],
        "scenes": verified["scenes"],
        "repair_arms_consumed": inputs["repair_arms_consumed"],
        "paired_joint": verified["paired_joint"],
        "paired_B2": verified["paired_B2"],
        "forced_reject_native": verified["forced_reject_native"],
        "paired_joint_all_prior_vs_forced": verified[
            "paired_joint_all_prior_vs_forced"],
        "paired_joint_initial_vs_forced": verified[
            "paired_joint_initial_vs_forced"],
    }
payload={
    "schema":"lifelong_nnr_controller_repair_formal_summary_v1_20260824",
    "verified": all(row["episodes"] == 18 for row in controllers.values()),
    "post_repair_audit_sha256": __import__("hashlib").sha256(
        Path(sys.argv[2]).read_bytes()).hexdigest(),
    "original_complete_arms": audit["original_complete_arms"],
    "repair_arms": audit["repair_arms"],
    "total_complete_arms": audit["original_complete_arms"]+audit["repair_arms"],
    "controllers": controllers,
}
if len(controllers) != 5 or payload["total_complete_arms"] != 270:
    raise SystemExit("formal controller matrix is incomplete")
out.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n")
print(json.dumps(payload,indent=2,sort_keys=True))
PY
sha256sum "${AUDIT}" "${REPAIR_ROOT}/formal_summary.json" > \
  "${REPAIR_ROOT}/FINAL_RECEIPTS.sha256"
printf 'verified=true\ncomplete_arms=270\n' > "${REPAIR_ROOT}/VERIFIED"
chmod -R a-w "${FORMAL}"
chmod a-w "${AUDIT}" "${REPAIR_ROOT}/formal_summary.json" \
  "${REPAIR_ROOT}/FINAL_RECEIPTS.sha256" "${REPAIR_ROOT}/VERIFIED"
echo "COMPLETE formal=${REPAIR_ROOT}/formal_summary.json"
