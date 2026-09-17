#!/usr/bin/env bash
#SBATCH --job-name=cec_covis_seal
#SBATCH --output=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/%x_%j.out
#SBATCH --error=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/%x_%j.err
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --partition=cpu_short
#SBATCH --account=torch_pr_769_tandon_advanced
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
: "${COVIS_ADDON:?}" "${COVIS_RUN:?}" "${EXPECTED_COVIS_SHA:?}"
[[ "$(sha256sum "${COVIS_ADDON}/SOURCE_BUNDLE.sha256" | awk '{print $1}')" == "${EXPECTED_COVIS_SHA}" ]]
(cd "${COVIS_ADDON}" && sha256sum -c --quiet SOURCE_BUNDLE.sha256)
# afterany is safe: seal refuses missing or partial histories. It never reports SR.
/scratch/lg154/conda-envs/habitat/bin/python \
  "${COVIS_ADDON}/build_hm3d_covisibility_repaired.py" seal --root "${COVIS_RUN}/construction"
