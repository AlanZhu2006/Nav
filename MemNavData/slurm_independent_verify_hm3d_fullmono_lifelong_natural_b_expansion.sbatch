#!/usr/bin/env bash
#SBATCH --job-name=h3lifeNatBXVer
#SBATCH --output=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/%x_%j.out
#SBATCH --error=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/%x_%j.err
#SBATCH --time=00:20:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --partition=cpu_short
#SBATCH --account=torch_pr_769_tandon_advanced

set -euo pipefail
: "${AUDIT_ROOT:?}" "${RUN_ROOT:?}" "${CURRENT_V4_RUN_ROOT:?}"
: "${AUDIT_RECEIPT:?}" "${EXPECTED_AUDIT_RECEIPT_SHA:?}"
[[ "$(sha256sum "${AUDIT_RECEIPT}" | awk '{print $1}')" == \
   "${EXPECTED_AUDIT_RECEIPT_SHA}" ]]
(cd "${AUDIT_ROOT}" && sha256sum -c --quiet "${AUDIT_RECEIPT}")

PY=/scratch/lg154/conda-envs/habitat/bin/python
pycache=${SLURM_TMPDIR:-/tmp}/h3life_natbx_verify_${SLURM_JOB_ID}
mkdir -p "${pycache}"
out=${RUN_ROOT}/independent_expansion_audit_verification.json
env PYTHONDONTWRITEBYTECODE=1 PYTHONPYCACHEPREFIX="${pycache}" \
  "${PY}" -u \
    "${AUDIT_ROOT}/MemNavData/independent_verify_hm3d_fullmono_lifelong_natural_b_expansion.py" \
    --audit-root "${RUN_ROOT}/expansion_audit/scenes" \
    --summary "${RUN_ROOT}/expansion_audit/summary.json" \
    --expansion-protocol "${AUDIT_ROOT}/MemNavData/hm3d_fullmono_lifelong_natural_b_expansion_audit_protocol_20260828.json" \
    --original-manifest "${CURRENT_V4_RUN_ROOT}/ab_population/role_pairs/manifest.json" \
    --expected-scenes 54 --out "${out}"
(cd "${RUN_ROOT}" && \
  sha256sum -c independent_expansion_audit_verification.json.sha256 >/dev/null)
chmod a-w "${out}" "${out}.sha256"
