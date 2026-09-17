#!/usr/bin/env bash
#SBATCH --job-name=gem_actual_a_launch
#SBATCH --output=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/%x_%j.out
#SBATCH --error=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/%x_%j.err
#SBATCH --time=00:15:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --partition=cpu_short
#SBATCH --account=torch_pr_769_tandon_advanced
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
: "${A_LAUNCH_ADDON:?}" "${A_LAUNCH_SHA:?}" "${DESIGN_ADDON:?}" "${DESIGN_RUN:?}"
: "${CORE_ADDON:?}" "${CORE_RUN:?}" "${A_LAUNCH_OUT:?}"
[[ "$(sha256sum "${A_LAUNCH_ADDON}/SOURCE_BUNDLE.sha256" | awk '{print $1}')" == "${A_LAUNCH_SHA}" ]]
(cd "${A_LAUNCH_ADDON}" && sha256sum -c --quiet SOURCE_BUNDLE.sha256)
export PYTHONPATH=${A_LAUNCH_ADDON}
PY=/scratch/lg154/conda-envs/habitat/bin/python
"${PY}" -m unittest test_repaired_fullmono_deferred_launch
exec "${PY}" "${A_LAUNCH_ADDON}/launch_repaired_fullmono_a_after_probe.py" \
  --probe "${DESIGN_RUN}" --probe-addon "${DESIGN_ADDON}" --collector "${CORE_ADDON}" \
  --run-root "${CORE_RUN}" --out "${A_LAUNCH_OUT}"
