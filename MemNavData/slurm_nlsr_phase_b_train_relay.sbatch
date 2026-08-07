#!/usr/bin/env bash
# After both artifact audits complete, freeze their content hashes and submit
# the formal trainer with those hashes as independent launcher pins.
#SBATCH --job-name=nlsr_pB_relay
#SBATCH --account=torch_pr_769_tandon_advanced
#SBATCH --partition=cpu_short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/%x_%j.out
#SBATCH --error=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/%x_%j.err

set -euo pipefail
umask 0022

: "${REPO_ROOT:?export the exact clean Phase-B checkout}"
: "${EXPECTED_COMMIT:?export its full commit SHA}"
: "${EXPECTED_LAUNCHER_SHA:?export the committed relay launcher SHA256}"
: "${RUN_TAG:?export the audited NLSR run tag}"
: "${TRAIN_AUDIT_JOB_ID:?export train audit job ID}"
: "${DEVELOPMENT_AUDIT_JOB_ID:?export development audit job ID}"

RESULT_BASE="${RESULT_BASE:-/scratch/yz11502/Research/Nav-axis-uturn-results/nlsr_gapfill_20260807}"
RUN_ROOT="${RESULT_BASE}/${RUN_TAG}"
FEATURE_ROOT="${RUN_ROOT}/phase_b_lingbot_features_${EXPECTED_COMMIT:0:12}"
TRAIN_AUDIT="${FEATURE_ROOT}/train_top2_gap16/lingbot_native_training_artifact_audit.json"
DEVELOPMENT_AUDIT="${FEATURE_ROOT}/development_top8_gap16/lingbot_native_training_artifact_audit.json"
TRAINER_REL="MemNavData/slurm_nlsr_phase_b_train.sbatch"
RELAY_REL="MemNavData/slurm_nlsr_phase_b_train_relay.sbatch"
TRAINER="${REPO_ROOT}/${TRAINER_REL}"
CONTRACT_ROOT="${RUN_ROOT}/phase_b_training_launch_${EXPECTED_COMMIT:0:12}"
CONTRACT="${CONTRACT_ROOT}/launch_contract.json"

fail() { echo "ABORT: $*" >&2; exit 2; }
[[ "${TRAIN_AUDIT_JOB_ID}" =~ ^[0-9]+$ && \
   "${DEVELOPMENT_AUDIT_JOB_ID}" =~ ^[0-9]+$ ]] || fail "invalid audit job IDs"
[[ "$(git -C "${REPO_ROOT}" rev-parse HEAD)" == "${EXPECTED_COMMIT}" ]] || \
  fail "relay checkout commit changed"
[[ -z "$(git -C "${REPO_ROOT}" status --porcelain)" ]] || fail "checkout is not clean"
[[ "$(sha256sum "$0" | awk '{print $1}')" == "${EXPECTED_LAUNCHER_SHA}" ]] || \
  fail "spooled relay launcher changed"
for relative in "${RELAY_REL}" "${TRAINER_REL}"; do
  actual="$(sha256sum "${REPO_ROOT}/${relative}" | awk '{print $1}')"
  committed="$(git -C "${REPO_ROOT}" show "${EXPECTED_COMMIT}:${relative}" | \
    sha256sum | awk '{print $1}')"
  [[ "${actual}" == "${committed}" ]] || fail "${relative} differs from commit"
done
[[ ! -e "${CONTRACT}" && ! -e "${CONTRACT}.sha256" ]] || \
  fail "training launch contract already exists"

for audit in "${TRAIN_AUDIT}" "${DEVELOPMENT_AUDIT}"; do
  [[ -f "${audit}" && ! -L "${audit}" && \
     -f "${audit}.sha256" && ! -L "${audit}.sha256" ]] || \
    fail "missing physical audit pair: ${audit}"
done
TRAIN_AUDIT_SHA="$(sha256sum "${TRAIN_AUDIT}" | awk '{print $1}')"
DEVELOPMENT_AUDIT_SHA="$(sha256sum "${DEVELOPMENT_AUDIT}" | awk '{print $1}')"
[[ "$(<"${TRAIN_AUDIT}.sha256")" == \
   "${TRAIN_AUDIT_SHA}  $(basename "${TRAIN_AUDIT}")" ]] || fail "train audit sidecar mismatch"
[[ "$(<"${DEVELOPMENT_AUDIT}.sha256")" == \
   "${DEVELOPMENT_AUDIT_SHA}  $(basename "${DEVELOPMENT_AUDIT}")" ]] || \
  fail "development audit sidecar mismatch"

python3 - "${TRAIN_AUDIT}" "${DEVELOPMENT_AUDIT}" <<'PY'
import json
from pathlib import Path
import sys
for expected_role, name in zip(("train", "development"), sys.argv[1:]):
    audit = json.loads(Path(name).read_text(encoding="utf-8"))
    if audit.get("training_artifact_approved") is not True:
        raise SystemExit(f"{expected_role} artifact is not approved")
    if audit.get("role") != expected_role:
        raise SystemExit(f"{expected_role} audit role changed")
    if audit.get("upstream_receipts", {}).get("approved") is not True:
        raise SystemExit(f"{expected_role} upstream receipts are not approved")
    if audit.get("external_causal_scale", {}).get("approved") is not True:
        raise SystemExit(f"{expected_role} external scale is not approved")
PY

TRAINER_LAUNCHER_SHA="$(sha256sum "${TRAINER}" | awk '{print $1}')"
exports="ALL,REPO_ROOT=${REPO_ROOT},EXPECTED_COMMIT=${EXPECTED_COMMIT}"
exports+=",EXPECTED_LAUNCHER_SHA=${TRAINER_LAUNCHER_SHA},RUN_TAG=${RUN_TAG}"
exports+=",EXPECTED_TRAIN_AUDIT_SHA=${TRAIN_AUDIT_SHA}"
exports+=",EXPECTED_DEVELOPMENT_AUDIT_SHA=${DEVELOPMENT_AUDIT_SHA}"
sbatch --test-only --export="${exports}" "${TRAINER}"
TRAIN_JOB_ID="$(sbatch --parsable --export="${exports}" "${TRAINER}")"
TRAIN_JOB_ID="${TRAIN_JOB_ID%%;*}"
[[ "${TRAIN_JOB_ID}" =~ ^[0-9]+$ ]] || fail "unexpected trainer submission result"

mkdir -p "${CONTRACT_ROOT}"
python3 - "${CONTRACT}" "${EXPECTED_COMMIT}" "${EXPECTED_LAUNCHER_SHA}" \
  "${TRAINER_LAUNCHER_SHA}" "${TRAIN_AUDIT}" "${TRAIN_AUDIT_SHA}" \
  "${DEVELOPMENT_AUDIT}" "${DEVELOPMENT_AUDIT_SHA}" \
  "${TRAIN_AUDIT_JOB_ID}" "${DEVELOPMENT_AUDIT_JOB_ID}" "${TRAIN_JOB_ID}" <<'PY'
import hashlib
import json
import os
from pathlib import Path
import sys
path = Path(sys.argv[1])
value = {
    "schema_version": "nlsr_phase_b_training_launch_contract_v1",
    "status": "formal_trainer_submitted",
    "source_commit": sys.argv[2],
    "relay_launcher_sha256": sys.argv[3],
    "trainer_launcher_sha256": sys.argv[4],
    "train_audit": {"path": sys.argv[5], "sha256": sys.argv[6],
                    "job_id": int(sys.argv[9])},
    "development_audit": {"path": sys.argv[7], "sha256": sys.argv[8],
                          "job_id": int(sys.argv[10])},
    "trainer_job_id": int(sys.argv[11]),
}
payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
digest = hashlib.sha256(payload).hexdigest()
temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
temporary.write_bytes(payload)
os.replace(temporary, path)
Path(f"{path}.sha256").write_text(f"{digest}  {path.name}\n", encoding="ascii")
print(json.dumps({"trainer_job_id": int(sys.argv[11]),
                  "launch_contract_sha256": digest}, sort_keys=True))
PY

echo "submitted formal Phase-B trainer job=${TRAIN_JOB_ID} train_audit=${TRAIN_AUDIT_SHA} development_audit=${DEVELOPMENT_AUDIT_SHA}"
