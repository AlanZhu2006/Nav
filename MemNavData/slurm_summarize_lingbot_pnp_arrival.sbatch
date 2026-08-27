#!/usr/bin/env bash
#SBATCH --job-name=pnp_arr_sum
#SBATCH --account=torch_pr_769_tandon_advanced
#SBATCH --partition=cpu_short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/%x_%j.out
#SBATCH --error=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/%x_%j.err

set -euo pipefail
umask 0022
: "${SOURCE_BUNDLE:?SOURCE_BUNDLE is required}"
: "${EXPECTED_BUNDLE_MANIFEST_SHA:?bundle SHA is required}"
: "${RUN_ROOT:?RUN_ROOT is required}"
: "${SHARD_COUNT:?SHARD_COUNT is required}"

manifest="${SOURCE_BUNDLE}/source_bundle_manifest.json"
[[ "$(sha256sum "${manifest}" | awk '{print $1}')" == \
   "${EXPECTED_BUNDLE_MANIFEST_SHA}" ]] || { echo "bundle changed" >&2; exit 2; }
out="${RUN_ROOT}/merged"
[[ ! -e "${out}" ]] || { echo "merged output already exists" >&2; exit 2; }
PYTHONPATH="${SOURCE_BUNDLE}${PYTHONPATH:+:${PYTHONPATH}}" \
  /scratch/lg154/conda-envs/memnav/bin/python \
  "${SOURCE_BUNDLE}/MemNavData/summarize_lingbot_pnp_arrival.py" \
  --states-csv "${SOURCE_BUNDLE}/inputs/states.csv" \
  --shard-root "${RUN_ROOT}/shards" --expected-shards "${SHARD_COUNT}" \
  --out-dir "${out}"
echo "PNP_ARRIVAL_MERGED=${out}"
