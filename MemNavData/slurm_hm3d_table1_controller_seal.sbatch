#!/usr/bin/env bash
#SBATCH --job-name=h3T1Seal
#SBATCH --output=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/%x_%j.out
#SBATCH --error=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/%x_%j.err
#SBATCH --time=00:20:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --partition=cpu_short
#SBATCH --account=torch_pr_769_tandon_advanced

set -euo pipefail
umask 0022
export PYTHONDONTWRITEBYTECODE=1
: "${FORMAL_RUN_ROOT:?}" "${CONSTRUCTION_VERIFICATION:?}"
: "${EXPECTED_CONSTRUCTION_VERIFICATION_SHA:?}"
DATASET=${DATASET:-HM3D}
CLAIM_SCOPE=${CLAIM_SCOPE:-}
if [[ -z "${CLAIM_SCOPE}" ]]; then
  CLAIM_SCOPE="fresh-query scene-overlap within-controller paired evaluation"
fi
PY=/scratch/lg154/conda-envs/memnav/bin/python
NAV=${FORMAL_RUN_ROOT}/formal/navdp/navdp_table1_independent_verification.json
VINT=${VINT_VERIFICATION_OVERRIDE:-${FORMAL_RUN_ROOT}/formal/vint/vint_table1_independent_verification.json}
EXPECTED_VINT_VERIFICATION_SHA=${EXPECTED_VINT_VERIFICATION_SHA:-}
dataset_lower=$(printf '%s' "${DATASET}" | tr '[:upper:]' '[:lower:]')
OUT=${FORMAL_RUN_ROOT}/${dataset_lower}_table1_controller_portability_receipt.json
[[ "${DATASET}" == HM3D || "${DATASET}" == MP3D ]]
[[ ! -e "${OUT}" ]]
[[ "$(sha256sum "${CONSTRUCTION_VERIFICATION}" | awk '{print $1}')" == \
   "${EXPECTED_CONSTRUCTION_VERIFICATION_SHA}" ]]
if [[ -n "${EXPECTED_VINT_VERIFICATION_SHA}" ]]; then
  [[ "$(sha256sum "${VINT}" | awk '{print $1}')" == \
     "${EXPECTED_VINT_VERIFICATION_SHA}" ]]
fi
"${PY}" - "${CONSTRUCTION_VERIFICATION}" "${NAV}" "${VINT}" \
  "${OUT}" "${DATASET}" "${CLAIM_SCOPE}" <<'PY'
import hashlib,json,os,sys
construction_path,nav_path,vint_path,out_path,dataset,claim_scope=sys.argv[1:]
def load(path):
 with open(path) as handle: return json.load(handle)
def digest(path):
 return hashlib.sha256(open(path,"rb").read()).hexdigest()
construction=load(construction_path); nav=load(nav_path); vint=load(vint_path)
if construction.get("verified") is not True or construction.get("formal_policy_evaluation_authorized") is not True:
 raise SystemExit("construction gate invalid")
if nav.get("verified") is not True or nav.get("authorized") is not True:
 raise SystemExit("NavDP verifier invalid")
if vint.get("verified") is not True:
 raise SystemExit("ViNT verifier invalid")
if nav.get("dataset","HM3D") != dataset or vint.get("dataset","HM3D") != dataset:
 raise SystemExit("controller verifier dataset changed")
expected=(int(construction["histories"]),int(construction["scene_clusters"]))
if (int(nav["histories"]),int(nav["scene_clusters"])) != expected:
 raise SystemExit("NavDP population changed")
if (int(vint["raw_histories"]),int(vint["scene_clusters"])) != expected:
 raise SystemExit("ViNT population changed")
payload={
 "schema_version":dataset.lower()+"_table1_controller_portability_receipt_v1_20260829",
 "verified":True,
 "dataset":dataset,
 "claim_scope":claim_scope,
 "cross_controller_absolute_sr_is_not_a_paired_estimand":True,
 "histories":expected[0],"scene_clusters":expected[1],
 "construction_verification_sha256":digest(construction_path),
 "navdp_independent_verification_sha256":digest(nav_path),
 "navdp_independent_verification_path":nav_path,
 "vint_independent_verification_sha256":digest(vint_path),
 "vint_independent_verification_path":vint_path,
 "runtime_role_visibility":"none",
 "navdp_depth":"causal_monocular_sidecar",
 "vint_cec_bearing_execution":"first_certified_bounded",
}
descriptor=os.open(out_path,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o444)
with os.fdopen(descriptor,"w") as handle:
 json.dump(payload,handle,indent=2,sort_keys=True); handle.write("\n")
print(json.dumps(payload,sort_keys=True))
PY
sha256sum "${OUT}" >"${OUT}.sha256"
chmod a-w "${OUT}" "${OUT}.sha256"
echo "COMPLETE ${OUT}"
