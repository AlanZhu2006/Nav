#!/usr/bin/env bash
#SBATCH --job-name=h3T2Meeting
#SBATCH --output=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/%x_%j.out
#SBATCH --error=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/%x_%j.err
#SBATCH --time=00:20:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --partition=cpu_short
#SBATCH --account=torch_pr_769_tandon_advanced

set -euo pipefail
umask 0022
export PYTHONDONTWRITEBYTECODE=1

: "${TASK_ROOT:?}" "${TASK_RECEIPT:?}" "${EXPECTED_TASK_RECEIPT_SHA:?}"
: "${POLICY_LAUNCHER_ROOT:?}" "${POLICY_LAUNCHER_RECEIPT:?}"
: "${EXPECTED_POLICY_LAUNCHER_RECEIPT_SHA:?}"
: "${SOURCE_RUN_ROOT:?}" "${TABLE2_RUN_ROOT:?}" "${PARENT_MANIFEST:?}"
: "${CONSTRUCTION_VERIFICATION:?}" "${FORMAL_RUN_ROOT:?}"

[[ "$(sha256sum "${TASK_RECEIPT}" | awk '{print $1}')" == \
   "${EXPECTED_TASK_RECEIPT_SHA}" ]]
(cd "${TASK_ROOT}" && sha256sum -c --quiet "${TASK_RECEIPT}")
[[ "$(sha256sum "${POLICY_LAUNCHER_RECEIPT}" | awk '{print $1}')" == \
   "${EXPECTED_POLICY_LAUNCHER_RECEIPT_SHA}" ]]
(cd "${POLICY_LAUNCHER_ROOT}" && sha256sum -c --quiet \
  "${POLICY_LAUNCHER_RECEIPT}")

policy=${FORMAL_RUN_ROOT}/formal/navdp/navdp_table2_leg3_independent_verification.json
test -f "${policy}" -a -f "${policy}.sha256"
out_root=${FORMAL_RUN_ROOT}/meeting_result
out=${out_root}/hm3d_table2_meeting_result_independent_verification.json
test ! -e "${out_root}"
mkdir -p "${out_root}"
export PYTHONPATH=${POLICY_LAUNCHER_ROOT}:${POLICY_LAUNCHER_ROOT}/MemNavData:${TASK_ROOT}:${TASK_ROOT}/MemNavData
/scratch/lg154/conda-envs/memnav/bin/python -u \
  "${POLICY_LAUNCHER_ROOT}/MemNavData/independent_verify_hm3d_table2_meeting_result.py" \
  --parent-manifest "${PARENT_MANIFEST}" \
  --source-union-root "${SOURCE_RUN_ROOT}" \
  --construction-verification "${CONSTRUCTION_VERIFICATION}" \
  --policy-verification "${policy}" --out "${out}"
sha256sum "${out}" >"${out}.sha256"
printf '%s\n' verified=true >"${out_root}/SEALED"
chmod -R a-w "${out_root}"
