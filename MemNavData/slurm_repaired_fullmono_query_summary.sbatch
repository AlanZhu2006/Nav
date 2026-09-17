#!/usr/bin/env bash
#SBATCH --job-name=gem_actual_a_query_summary
#SBATCH --output=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/%x_%j.out
#SBATCH --error=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/%x_%j.err
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --partition=cpu_short
#SBATCH --account=torch_pr_769_tandon_advanced
#SBATCH --qos=cpu48
set -euo pipefail
: "${GEM_SUMMARY_ADDON:?}" "${GEM_SUMMARY_ADDON_SHA:?}"
: "${GEM_POPULATION:?}" "${GEM_POPULATION_SHA:?}" "${FULLMONO_EVAL_ROOT:?}"
[[ "$(id -un)" == yz11502 ]]
[[ "$(sha256sum "${GEM_SUMMARY_ADDON}/SOURCE_BUNDLE.sha256" | awk '{print $1}')" == "${GEM_SUMMARY_ADDON_SHA}" ]]
(cd "${GEM_SUMMARY_ADDON}" && sha256sum -c --quiet SOURCE_BUNDLE.sha256)
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
GEM_SUMMARY_TMP=$(mktemp -d "${SLURM_TMPDIR:-/tmp}/gem_role_summary_${SLURM_JOB_ID}_XXXXXX")
export TMPDIR=${GEM_SUMMARY_TMP} LIBFFI_TMPDIR=${GEM_SUMMARY_TMP} PYTHONPATH=${GEM_SUMMARY_ADDON}
exec singularity exec -B /scratch/lg154 -B /scratch/yz11502 \
  /share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif \
  /scratch/lg154/conda-envs/habitat/bin/python "${GEM_SUMMARY_ADDON}/summarize_repaired_fullmono_queries.py" \
  --population "${GEM_POPULATION}" --population-sha256 "${GEM_POPULATION_SHA}" \
  --evaluation-root "${FULLMONO_EVAL_ROOT}" --output "${FULLMONO_EVAL_ROOT}/paired_summary.json"
