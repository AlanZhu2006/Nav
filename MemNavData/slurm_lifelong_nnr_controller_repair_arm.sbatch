#!/usr/bin/env bash
#SBATCH --job-name=lifeCtlRepair
#SBATCH --output=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/%x_%A_%a.out
#SBATCH --error=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/%x_%A_%a.err
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --partition=a100_tandon
#SBATCH --qos=gpu48
#SBATCH --gres=gpu:1
#SBATCH --account=torch_pr_769_tandon_advanced
#SBATCH --export=ALL

# Run exactly one entry from the outcome-blind 25-arm repair manifest.  The
# actual evaluator remains the original controller arm script; this wrapper
# only maps a compact repair-array index to its frozen population/controller/
# scope identity.  A100-only and array concurrency one retain strict RGB replay
# while avoiding the co-located H100 render mismatch seen in the first run.
set -euo pipefail
umask 0022
export PYTHONDONTWRITEBYTECODE=1
: "${SLURM_ARRAY_TASK_ID:?submit as repair array 0-24}"
: "${SOURCE_ROOT:?set immutable task source root}"
: "${REPAIR_ROOT:?set unique immutable repair root}"
: "${ORIGINAL_RUN_ROOT:?set navdp-line sealed population source}"

MANIFEST=${SOURCE_ROOT}/MemNavData/lifelong_nnr_controller_repair_manifest_20260824.json
PY=/scratch/lg154/conda-envs/memnav/bin/python
REPAIR_TASK_ID=${SLURM_ARRAY_TASK_ID}

readarray -t repair < <("${PY}" - "${MANIFEST}" "${REPAIR_TASK_ID}" <<'PY'
import json,sys
payload=json.load(open(sys.argv[1]))
if payload.get("schema") != "lifelong_nnr_controller_repair_manifest_v1_20260824":
    raise SystemExit("repair manifest schema changed")
index=int(sys.argv[2]); entries=payload["entries"]
if not 0 <= index < len(entries):
    raise SystemExit("repair array index escaped manifest")
row=entries[index]
for value in (
    row["controller"], row["scope"], str(row["population_index"]),
    row["scene"], row["episode"], row["reason"],
):
    print(value)
PY
)
[[ "${#repair[@]}" -eq 6 ]] || {
  echo "ABORT: incomplete repair manifest row" >&2
  exit 2
}

export CONTROLLER=${repair[0]}
export LIFELONG_HISTORY_SCOPE=${repair[1]}
export SLURM_ARRAY_TASK_ID=${repair[2]}
SCENE=${repair[3]}
EPISODE=${repair[4]}
REASON=${repair[5]}
export RUN_ROOT=${REPAIR_ROOT}/${CONTROLLER}

echo "repair_task=${REPAIR_TASK_ID} population_index=${SLURM_ARRAY_TASK_ID} controller=${CONTROLLER} scope=${LIFELONG_HISTORY_SCOPE} scene=${SCENE} episode=${EPISODE} frozen_reason=${REASON} population_root=${ORIGINAL_RUN_ROOT}"
bash "${SOURCE_ROOT}/MemNavData/slurm_lifelong_nnr_controller_arm.sbatch"
echo "COMPLETE repair_task=${REPAIR_TASK_ID}"
