#!/usr/bin/env bash
#SBATCH --job-name=table2_final_verify
#SBATCH --output=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/%x_%j.out
#SBATCH --error=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/%x_%j.err
#SBATCH --partition=cpu_short
#SBATCH --account=torch_pr_769_tandon_advanced
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
: "${TABLE2_POST_RUN:?}" "${TABLE2_POST_BUNDLE:?}" "${TABLE2_POST_STAGE:?}"
: "${TABLE2_POST_PLAN_SHA:?}" "${TABLE2_POST_CODE_SHA:?}"
[[ "$(id -un)" == yz11502 ]]
cd "${TABLE2_POST_RUN}"
[[ "$(sha256sum POSTPROCESSING.sha256 | awk '{print $1}')" == "${TABLE2_POST_CODE_SHA}" ]]
sha256sum -c --quiet POSTPROCESSING.sha256
exec /scratch/lg154/conda-envs/memnav/bin/python -u table2_final_postprocess.py \
  --root "${TABLE2_POST_RUN}" --bundle "${TABLE2_POST_BUNDLE}" --stage "${TABLE2_POST_STAGE}"
