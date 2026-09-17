#!/usr/bin/env bash
#SBATCH --job-name=nav_storage_inventory
#SBATCH --output=/scratch/yz11502/Research/Nav-axis-uturn-results/maintenance/cold_buffers_20260909/storage_inventory_%j.out
#SBATCH --error=/scratch/yz11502/Research/Nav-axis-uturn-results/maintenance/cold_buffers_20260909/storage_inventory_%j.err
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --partition=cpu_short
#SBATCH --account=torch_pr_769_tandon_advanced
set -uo pipefail
export PYTHONDONTWRITEBYTECODE=1
[[ "$(id -un)" == yz11502 ]] || exit 1
# Read-only census of this project's old outputs and code snapshots. Do not
# follow symlinks into datasets, shared environments or another user's files.
for inventory_path in \
  /scratch/yz11502/Research/Nav-axis-uturn-results \
  /scratch/yz11502/Research/Nav-axis-uturn-source-bundles \
  /scratch/yz11502/Research/Nav-graph-blind \
  /scratch/yz11502/Research/Nav-axis-uturn; do
  [[ -d "$inventory_path" && ! -L "$inventory_path" ]] || continue
  du --one-file-system --block-size=1 --max-depth=1 "$inventory_path"
done
