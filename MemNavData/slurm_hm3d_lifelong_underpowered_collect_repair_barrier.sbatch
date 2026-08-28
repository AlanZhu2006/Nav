#!/usr/bin/env bash
#SBATCH --job-name=h3CrepairGate
#SBATCH --output=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/%x_%j.out
#SBATCH --error=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/%x_%j.err
#SBATCH --time=00:20:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --partition=cpu_short
#SBATCH --account=torch_pr_769_tandon_advanced
#SBATCH --export=ALL

set -euo pipefail
umask 0022
export PYTHONDONTWRITEBYTECODE=1
: "${AMENDMENT_ROOT:?}" "${AMENDMENT_RECEIPT:?}"
: "${EXPECTED_AMENDMENT_RECEIPT_SHA:?}" "${REPAIR_PROTOCOL:?}"
: "${RUN_ROOT:?}" "${REPAIR_ROOT:?}"

[[ "$(sha256sum "${AMENDMENT_RECEIPT}" | awk '{print $1}')" == \
   "${EXPECTED_AMENDMENT_RECEIPT_SHA}" ]]
(cd "${AMENDMENT_ROOT}" && sha256sum -c --quiet "${AMENDMENT_RECEIPT}")
audit=${AMENDMENT_ROOT}/MemNavData/audit_hm3d_lifelong_underpowered_collect_repair.py
ready=${REPAIR_ROOT}/ready_to_seal_audit.json
[[ -f "${REPAIR_ROOT}/pre_archive_audit.json" && ! -e "${ready}" ]]
/scratch/lg154/conda-envs/memnav/bin/python "${audit}" \
  --protocol "${REPAIR_PROTOCOL}" --run-root "${RUN_ROOT}" \
  --phase ready_to_seal --out "${ready}" >/dev/null
/scratch/lg154/conda-envs/memnav/bin/python - \
  "${REPAIR_ROOT}/pre_archive_audit.json" "${ready}" <<'PY'
import json,sys
before=json.load(open(sys.argv[1]))
after=json.load(open(sys.argv[2]))
for label,files in before["completed_required_file_sha256"].items():
 if after["completed_required_file_sha256"].get(label)!=files:
  raise SystemExit(f"preserved completed output changed: {label}")
if after["completed_indices_verified"]!=list(range(22)):
 raise SystemExit("repair set is not complete")
PY
sha256sum "${ready}" >"${ready}.sha256"
printf 'READY_TO_SEAL=1\nREPAIRED_INDICES=0,1,7,9,11,13\n'
