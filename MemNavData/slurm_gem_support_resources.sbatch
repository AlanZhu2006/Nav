#!/usr/bin/env bash
#SBATCH --job-name=gem_support_cost
#SBATCH --nodes=1
#SBATCH --ntasks=1
set -euo pipefail
bash "${REPAIRED_BUNDLE}/MemNavData/run_gem_support_resources_hpc.sh"
