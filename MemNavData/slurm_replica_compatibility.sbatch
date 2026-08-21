#!/usr/bin/env bash
#SBATCH --job-name=replica_compat
#SBATCH --output=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/%x_%j.out
#SBATCH --error=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/%x_%j.err
#SBATCH --time=00:45:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=l40s_public,h100_tandon,h200_public,a100_tandon
#SBATCH --qos=gpu48
#SBATCH --gres=gpu:1
#SBATCH --account=torch_pr_769_tandon_advanced
#SBATCH --export=ALL

set -euo pipefail
umask 0022
: "${SOURCE_ROOT:?set immutable source root}"
: "${SOURCE_RECEIPT:?set source receipt}"
: "${EXPECTED_SOURCE_RECEIPT_SHA:?set source receipt SHA}"
: "${RUN_ROOT:?set run root}"
EXPECTED_COMPLETE_SCENES=${EXPECTED_COMPLETE_SCENES:-8}
MINIMUM_ELIGIBLE_SCENES=${MINIMUM_ELIGIBLE_SCENES:-6}

HAB_PY=/scratch/lg154/conda-envs/habitat/bin/python
BASE_SIF=/share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif
fail() { echo "ABORT: $*" >&2; exit 2; }
[[ "${EXPECTED_COMPLETE_SCENES}" =~ ^[1-9][0-9]*$ ]] || \
  fail "invalid complete-scene count"
[[ "${MINIMUM_ELIGIBLE_SCENES}" =~ ^[1-9][0-9]*$ ]] || \
  fail "invalid eligible-scene count"

[[ "$(sha256sum "${SOURCE_RECEIPT}" | awk '{print $1}')" == \
  "${EXPECTED_SOURCE_RECEIPT_SHA}" ]] || fail "source receipt changed"
(cd "${SOURCE_ROOT}" && sha256sum -c "${SOURCE_RECEIPT}") >/dev/null || \
  fail "source bundle changed"
[[ -f "${BASE_SIF}" && ! -e "${RUN_ROOT}" ]] || \
  fail "container missing or run root already exists"
mkdir -p "${RUN_ROOT}"
cp "${SOURCE_RECEIPT}" "${RUN_ROOT}/SOURCE_BUNDLE.sha256"
sha256sum "${SOURCE_ROOT}/source_bundle_manifest.json" \
  > "${RUN_ROOT}/source_bundle_manifest.json.sha256"

export PYTHONPYCACHEPREFIX=${SLURM_TMPDIR:-/tmp}/replica_compat_${SLURM_JOB_ID}
export PYTHONPATH=${SOURCE_ROOT}/MemNavData${PYTHONPATH:+:${PYTHONPATH}}
singularity exec --nv -B /scratch/lg154 -B /scratch/yz11502 \
  "${BASE_SIF}" env PYTHONPATH="${PYTHONPATH}" \
  PYTHONPYCACHEPREFIX="${PYTHONPYCACHEPREFIX}" \
  "${HAB_PY}" -m py_compile \
  "${SOURCE_ROOT}/MemNavData/audit_replica_habitat_compatibility.py" \
  "${SOURCE_ROOT}/MemNavData/generate_twoleg.py"
singularity exec --nv -B /scratch/lg154 -B /scratch/yz11502 \
  "${BASE_SIF}" env PYTHONPATH="${PYTHONPATH}" \
  PYTHONPYCACHEPREFIX="${PYTHONPYCACHEPREFIX}" \
  "${HAB_PY}" -u \
  "${SOURCE_ROOT}/MemNavData/audit_replica_habitat_compatibility.py" \
  --replica-root "${SOURCE_ROOT}/replica_v1" \
  --out "${RUN_ROOT}/compatibility.json" \
  --expected-complete-scenes "${EXPECTED_COMPLETE_SCENES}" \
  --minimum-eligible-scenes "${MINIMUM_ELIGIBLE_SCENES}" \
  --minimum-geodesic-m 2.0 --seed 20260814
sha256sum "${RUN_ROOT}/compatibility.json" \
  > "${RUN_ROOT}/compatibility.json.sha256"
touch "${RUN_ROOT}/COMPLETE"
