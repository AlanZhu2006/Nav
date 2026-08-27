#!/usr/bin/env bash
# Extract the frozen factual native A/B-success prefix population.
#SBATCH --job-name=nnr_extract
#SBATCH --output=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/%x_%j.out
#SBATCH --error=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/%x_%j.err
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=cpu_short
#SBATCH --account=torch_pr_769_tandon_advanced
#SBATCH --export=ALL

set -euo pipefail
umask 0022
: "${SOURCE_ROOT:?set immutable task source root}"
: "${SOURCE_RECEIPT:?set source receipt}"
: "${EXPECTED_SOURCE_RECEIPT_SHA:?set receipt SHA}"
: "${RUN_ROOT:?set run root}"
: "${GENERATION_ROOT:?set strict-v4 generation root}"
: "${NATIVE_EVAL_ROOT:?set native evaluation root}"
: "${NAVDP_CHECKPOINT:?set pinned NavDP checkpoint}"
: "${EXPECTED_NAVDP_CHECKPOINT_SHA:?set checkpoint SHA}"

PY=/scratch/lg154/conda-envs/memnav/bin/python
OUT=${RUN_ROOT}/prepared/traces
export PYTHONPYCACHEPREFIX=${SLURM_TMPDIR:-/tmp}/nnr_extract_pycache_${SLURM_JOB_ID}
mkdir -p "${RUN_ROOT}/prepared" \
  "${PYTHONPYCACHEPREFIX}" \
  /scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs
fail() { echo "ABORT: $*" >&2; exit 2; }
[[ "$(sha256sum "${SOURCE_RECEIPT}" | awk '{print $1}')" == \
   "${EXPECTED_SOURCE_RECEIPT_SHA}" ]] || fail "source receipt changed"
(cd "${SOURCE_ROOT}" && sha256sum -c "${SOURCE_RECEIPT}") >/dev/null || \
  fail "source bundle changed"
[[ "$(sha256sum "${NAVDP_CHECKPOINT}" | awk '{print $1}')" == \
   "${EXPECTED_NAVDP_CHECKPOINT_SHA}" ]] || fail "NavDP checkpoint changed"
[[ -d "${GENERATION_ROOT}" && -d "${NATIVE_EVAL_ROOT}" ]] || \
  fail "qw2440 strict-v4 source is unavailable"
[[ ! -e "${OUT}" ]] || fail "trace output exists"

export PYTHONPATH=${SOURCE_ROOT}:${SOURCE_ROOT}/MemNavData${PYTHONPATH:+:${PYTHONPATH}}
"${PY}" -m py_compile \
  "${SOURCE_ROOT}/MemNavData/extract_native_shared_ab_traces.py" \
  "${SOURCE_ROOT}/MemNavData/deterministic_eval_protocol.py"
"${PY}" -u "${SOURCE_ROOT}/MemNavData/extract_native_shared_ab_traces.py" \
  --generation-root "${GENERATION_ROOT}" \
  --native-eval-root "${NATIVE_EVAL_ROOT}" \
  --navdp-checkpoint "${NAVDP_CHECKPOINT}" --out "${OUT}"

"${PY}" - "${OUT}/manifest.json" <<'PY'
import json,sys
m=json.load(open(sys.argv[1]))
assert m["schema_version"]=="native_shared_ab_extraction_v1_20260813"
assert m["replay_is_observation_only"] is True
assert m["source_memory_observer_present"] is False
assert len(m["accepted"])==22, len(m["accepted"])
assert len({r["scene"] for r in m["accepted"]})==8
PY
(cd "${OUT}" && sha256sum manifest.json > manifest.json.sha256)
touch "${RUN_ROOT}/prepared/EXTRACTED"
echo "COMPLETE ${OUT}"
