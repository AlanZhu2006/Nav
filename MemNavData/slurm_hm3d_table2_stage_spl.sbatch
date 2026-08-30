#!/usr/bin/env bash
#SBATCH --job-name=h3T2StageSPL
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
: "${MEETING_VERIFICATION:?}" "${PARENT_MANIFEST:?}"
: "${SOURCE_UNION_ROOT:?}" "${FORMAL_RUN_ROOT:?}"

[[ "$(sha256sum "${TASK_RECEIPT}" | awk '{print $1}')" == \
   "${EXPECTED_TASK_RECEIPT_SHA}" ]]
(cd "${TASK_ROOT}" && sha256sum -c --quiet "${TASK_RECEIPT}")
test -f "${MEETING_VERIFICATION}" -a -f "${MEETING_VERIFICATION}.sha256"

out_root=${FORMAL_RUN_ROOT}/meeting_result_stage_spl_v1
out=${out_root}/hm3d_table2_stage_spl_independent_verification.json
test ! -e "${out_root}"
mkdir -p "${out_root}"

/scratch/lg154/conda-envs/memnav/bin/python -u \
  "${TASK_ROOT}/MemNavData/independent_verify_hm3d_table2_stage_spl.py" \
  --meeting-verification "${MEETING_VERIFICATION}" \
  --parent-manifest "${PARENT_MANIFEST}" \
  --source-union-root "${SOURCE_UNION_ROOT}" \
  --out "${out}"
sha256sum "${out}" >"${out}.sha256"
printf '%s\n' verified=true >"${out_root}/SEALED"
chmod -R a-w "${out_root}"
