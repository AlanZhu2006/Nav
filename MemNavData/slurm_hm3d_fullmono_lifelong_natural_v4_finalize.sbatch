#!/usr/bin/env bash
#SBATCH --job-name=h3lifeV4Seal
#SBATCH --output=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/%x_%j.out
#SBATCH --error=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/%x_%j.err
#SBATCH --time=00:20:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --partition=cpu_short
#SBATCH --account=torch_pr_769_tandon_advanced

set -euo pipefail
: "${V4_ROOT:?}" "${AUDIT_SOURCE_ROOT:?}" "${SOURCE_TASK_ROOT:?}"
: "${BASE_SOURCE_ROOT:?}" "${RUN_ROOT:?}" "${PARENT_ROOT:?}"
: "${PROTOCOL:?}" "${V4_RECEIPT:?}" "${EXPECTED_V4_RECEIPT_SHA:?}"
: "${AUDIT_SOURCE_RECEIPT:?}" "${EXPECTED_AUDIT_SOURCE_RECEIPT_SHA:?}"
: "${SOURCE_TASK_RECEIPT:?}" "${EXPECTED_SOURCE_TASK_RECEIPT_SHA:?}"
: "${BASE_RECEIPT:?}" "${EXPECTED_BASE_RECEIPT_SHA:?}"
[[ "$(sha256sum "${V4_RECEIPT}" | awk '{print $1}')" == \
   "${EXPECTED_V4_RECEIPT_SHA}" ]]
(cd "${V4_ROOT}" && sha256sum -c --quiet "${V4_RECEIPT}")
[[ "$(sha256sum "${AUDIT_SOURCE_RECEIPT}" | awk '{print $1}')" == \
   "${EXPECTED_AUDIT_SOURCE_RECEIPT_SHA}" ]]
(cd "${AUDIT_SOURCE_ROOT}" && sha256sum -c --quiet "${AUDIT_SOURCE_RECEIPT}")
[[ "$(sha256sum "${SOURCE_TASK_RECEIPT}" | awk '{print $1}')" == \
   "${EXPECTED_SOURCE_TASK_RECEIPT_SHA}" ]]
(cd "${SOURCE_TASK_ROOT}" && sha256sum -c --quiet "${SOURCE_TASK_RECEIPT}")
[[ "$(sha256sum "${BASE_RECEIPT}" | awk '{print $1}')" == \
   "${EXPECTED_BASE_RECEIPT_SHA}" ]]
(cd "${BASE_SOURCE_ROOT}" && sha256sum -c --quiet "${BASE_RECEIPT}")

[[ "$(find "${RUN_ROOT}/construct_ab/scenes" -mindepth 2 -maxdepth 2 \
  -name completion.json -type f | wc -l)" -eq 54 ]]
PY=/scratch/lg154/conda-envs/memnav/bin/python
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPYCACHEPREFIX=${SLURM_TMPDIR:-/tmp}/h3life_v4_seal_${SLURM_JOB_ID}
# The protocol is itself stored in the immutable v4 source bundle.  Put that
# directory ahead of the older source-task bundle so the v4 schema parser used
# during materialization is also used during this renderer-free seal.
PROTOCOL_CODE_ROOT=$(dirname "${PROTOCOL}")
export PYTHONPATH=${V4_ROOT}/MemNavData:${PROTOCOL_CODE_ROOT}:${AUDIT_SOURCE_ROOT}/MemNavData:${SOURCE_TASK_ROOT}:${SOURCE_TASK_ROOT}/MemNavData:${BASE_SOURCE_ROOT}:${BASE_SOURCE_ROOT}/MemNavData
"${PY}" -u \
  "${V4_ROOT}/MemNavData/finalize_hm3d_fullmono_lifelong_ab.py" \
  --parent-root "${PARENT_ROOT}" --protocol "${PROTOCOL}" \
  --construction-root "${RUN_ROOT}/construct_ab/scenes" \
  --out "${RUN_ROOT}/ab_population"
"${PY}" - "${RUN_ROOT}" <<'PY'
import json, sys
from pathlib import Path
root=Path(sys.argv[1])
receipt=json.load(open(root/'ab_population/population_receipt.json'))
assert receipt['source_materialized_A_histories'] == 130
assert receipt['constructible_AB_C_histories'] == 99
assert receipt['constructible_scene_clusters'] == 35
assert receipt['construction_target_met'] is True
assert receipt['factual_B_authorized'] is True
completions=[json.load(open(p)) for p in sorted(
    (root/'construct_ab/scenes').glob('*/completion.json'))]
assert sum(x['materialized_recipient_histories'] for x in completions) == 61
assert all(x['audit_candidate_count_reproduced'] is True for x in completions)
assert all(x['pairwise_separation_recomputed'] is True for x in completions)
assert all(x['query_policy_outcomes_read'] is False for x in completions)
assert all(x['navigation_outcomes_read'] is False for x in completions)
PY
(cd "${RUN_ROOT}/ab_population" && \
  sha256sum -c --quiet population_receipt.json.sha256 && \
  sha256sum -c --quiet BENCHMARK_FILES.sha256)
chmod -R a-w "${RUN_ROOT}/construct_ab" "${RUN_ROOT}/ab_population"
