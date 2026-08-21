#!/usr/bin/env bash
#SBATCH --job-name=cecCtlEnv
#SBATCH --output=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/%x_%j.out
#SBATCH --error=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/%x_%j.err
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --partition=cpu_short
#SBATCH --account=torch_pr_769_tandon_advanced
#SBATCH --export=ALL

set -euo pipefail
: "${SOURCE_ROOT:?set immutable source root}"
: "${SOURCE_RECEIPT:?set source receipt}"
: "${EXPECTED_SOURCE_RECEIPT_SHA:?set source receipt SHA}"
: "${ENV_ROOT:?set final environment root}"

[[ "$(sha256sum "${SOURCE_RECEIPT}" | awk '{print $1}')" == \
   "${EXPECTED_SOURCE_RECEIPT_SHA}" ]] || {
  echo "ABORT: source receipt changed" >&2; exit 2;
}
(cd "${SOURCE_ROOT}" && sha256sum -c --quiet "${SOURCE_RECEIPT}")

env ENV_ROOT="${ENV_ROOT}" \
  SETUP_SOURCE="${SOURCE_ROOT}/MemNavData/setup_controller_portability_hpc_envs.sh" \
  MEMNAV_PY=/scratch/lg154/conda-envs/memnav/bin/python \
  bash "${SOURCE_ROOT}/MemNavData/setup_controller_portability_hpc_envs.sh"
