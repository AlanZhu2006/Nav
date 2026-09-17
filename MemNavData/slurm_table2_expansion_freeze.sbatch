#!/usr/bin/env bash
#SBATCH --job-name=table2_expand_freeze
#SBATCH --output=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/%x_%j.out
#SBATCH --error=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/%x_%j.err
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --partition=cpu_short
#SBATCH --account=torch_pr_769_tandon_advanced
set -euo pipefail
: "${REPAIRED_BUNDLE:?}" "${TABLE2_CONTINUOUS_RUN:?}" "${EXPECTED_BOOTSTRAP_SHA:?}"
: "${TABLE2_EXPANSION_FINAL_BUNDLE:?}"
[[ "$(id -un)" == yz11502 ]]
export PYTHONDONTWRITEBYTECODE=1
TABLE2_EXPANSION_TMP=$(mktemp -d "${SLURM_TMPDIR:-/tmp}/table2_expansion_freeze_${SLURM_JOB_ID}_XXXXXX")
export TMPDIR=${TABLE2_EXPANSION_TMP} LIBFFI_TMPDIR=${TABLE2_EXPANSION_TMP}
cd "${REPAIRED_BUNDLE}"
[[ "$(sha256sum SOURCE_BUNDLE.sha256 | awk '{print $1}')" == "${EXPECTED_BOOTSTRAP_SHA}" ]]
sha256sum -c --quiet SOURCE_BUNDLE.sha256
source MemNavData/repaired_hm3d_hpc_env.sh
exec "${REPAIRED_MEM_PY}" -u -m MemNavData.table2_continuous_expansion freeze \
  --bundle "${REPAIRED_BUNDLE}" --run "${TABLE2_CONTINUOUS_RUN}" \
  --final "${TABLE2_EXPANSION_FINAL_BUNDLE}" --submit
