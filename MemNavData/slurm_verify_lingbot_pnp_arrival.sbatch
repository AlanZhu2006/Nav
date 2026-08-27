#!/usr/bin/env bash
#SBATCH --job-name=pnp_arr_verify
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
: "${VERIFY_BUNDLE:?VERIFY_BUNDLE is required}"
: "${EXPECTED_VERIFY_MANIFEST_SHA:?verifier bundle SHA is required}"
: "${RUN_ROOT:?RUN_ROOT is required}"

manifest="${VERIFY_BUNDLE}/source_bundle_manifest.json"
[[ "$(sha256sum "${manifest}" | awk '{print $1}')" == \
   "${EXPECTED_VERIFY_MANIFEST_SHA}" ]] || { echo "verifier bundle changed" >&2; exit 2; }
/scratch/lg154/conda-envs/memnav/bin/python - "${VERIFY_BUNDLE}" "${manifest}" <<'PY'
import hashlib,json,sys
from pathlib import Path
root=Path(sys.argv[1]); payload=json.load(open(sys.argv[2]))
for relative,expected in payload["files"].items():
    path=root/relative
    if not path.is_file() or path.is_symlink():
        raise SystemExit(f"invalid verifier member: {relative}")
    if hashlib.sha256(path.read_bytes()).hexdigest()!=expected:
        raise SystemExit(f"changed verifier member: {relative}")
PY
export PYTHONPATH="${VERIFY_BUNDLE}${PYTHONPATH:+:${PYTHONPATH}}"
/scratch/lg154/conda-envs/memnav/bin/python -m unittest \
  MemNavData.test_verify_lingbot_pnp_arrival -v
out="${RUN_ROOT}/independent_verification.json"
[[ ! -e "${out}" ]] || { echo "verification output already exists" >&2; exit 2; }
/scratch/lg154/conda-envs/memnav/bin/python \
  "${VERIFY_BUNDLE}/MemNavData/verify_lingbot_pnp_arrival.py" \
  --merged-root "${RUN_ROOT}/merged" \
  --states-csv "${VERIFY_BUNDLE}/inputs/states.csv" --out "${out}"
echo "PNP_ARRIVAL_INDEPENDENT_VERIFICATION=${out}"
