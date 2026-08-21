#!/usr/bin/env bash
# Construct and causally seal the 40-episode benchmark before any arm runs.
#SBATCH --job-name=drev_prep
#SBATCH --output=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/%x_%j.out
#SBATCH --error=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/%x_%j.err
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=l40s_public,h100_tandon,h200_public,a100_tandon
#SBATCH --qos=gpu48
#SBATCH --gres=gpu:1
#SBATCH --account=torch_pr_769_tandon_advanced
#SBATCH --export=ALL

set -euo pipefail
umask 0022
: "${SOURCE_ROOT:?set immutable task source root}"
: "${SOURCE_RECEIPT:?set task source receipt}"
: "${EXPECTED_SOURCE_RECEIPT_SHA:?set task source receipt SHA}"
: "${BASE_SOURCE_ROOT:?set immutable base source root}"
: "${BASE_SOURCE_RECEIPT_SHA:?set immutable base receipt SHA}"
: "${RUN_ROOT:?set unique run root}"
: "${UPSTREAM_RUN_ROOT:?set immutable fresh160 run root}"
: "${EXPECTED_UPSTREAM_MANIFEST_SHA:?set fresh160 manifest SHA}"

HAB_PY=/scratch/lg154/conda-envs/habitat/bin/python
PREPARER=${SOURCE_ROOT}/MemNavData/prepare_shared_online_double_revisit_fresh.py
OUT=${RUN_ROOT}/prepared
BASE_RECEIPT=${BASE_SOURCE_ROOT}/SOURCE_BUNDLE.sha256
export PYTHONPYCACHEPREFIX=${SLURM_TMPDIR:-/tmp}/drev_prep_pycache_${SLURM_JOB_ID}
export PYTHONPATH=${SOURCE_ROOT}:${SOURCE_ROOT}/MemNavData${PYTHONPATH:+:${PYTHONPATH}}
mkdir -p "${PYTHONPYCACHEPREFIX}" \
  /scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs

fail() { echo "ABORT: $*" >&2; exit 2; }
[[ "$(sha256sum "${SOURCE_RECEIPT}" | awk '{print $1}')" == \
   "${EXPECTED_SOURCE_RECEIPT_SHA}" ]] || fail "task source receipt changed"
(cd "${SOURCE_ROOT}" && sha256sum -c "${SOURCE_RECEIPT}") >/dev/null || \
  fail "task source bundle changed"
[[ "$(sha256sum "${BASE_RECEIPT}" | awk '{print $1}')" == \
   "${BASE_SOURCE_RECEIPT_SHA}" ]] || fail "base source receipt changed"
[[ "$(sha256sum "${UPSTREAM_RUN_ROOT}/data_manifest.json" | awk '{print $1}')" == \
   "${EXPECTED_UPSTREAM_MANIFEST_SHA}" ]] || fail "fresh160 manifest changed"
[[ ! -e "${OUT}" ]] || fail "preparation output already exists"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

"${HAB_PY}" -m py_compile \
  "${PREPARER}" \
  "${SOURCE_ROOT}/MemNavData/build_shared_online_double_revisit_v2.py" \
  "${SOURCE_ROOT}/MemNavData/eval_shared_online_double_revisit.py"
"${HAB_PY}" -m unittest \
  MemNavData.test_prepare_shared_online_double_revisit_fresh \
  MemNavData.test_multigoal_policy_contract \
  MemNavData.test_shared_online_double_revisit_runtime \
  MemNavData.test_navdp_goal_switch

"${HAB_PY}" -u "${PREPARER}" \
  --original-run-root "${UPSTREAM_RUN_ROOT}" \
  --expected-source-manifest-sha "${EXPECTED_UPSTREAM_MANIFEST_SHA}" \
  --out "${OUT}" \
  --target-count 40 \
  --expected-goal-a-successes 120 \
  --minimum-selected-scenes 15

[[ ! -e "${OUT}/INCOMPLETE" && -f "${OUT}/SEALED" ]] || \
  fail "preparation did not seal"
(cd "${OUT}/benchmark" && sha256sum -c manifest.json.sha256) >/dev/null
echo "COMPLETE ${OUT}"
