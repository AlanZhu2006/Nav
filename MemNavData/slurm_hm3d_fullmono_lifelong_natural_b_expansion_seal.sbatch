#!/usr/bin/env bash
#SBATCH --job-name=h3NatBxSeal
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
: "${TASK_ROOT:?}" "${RUN_ROOT:?}" "${PARENT_ROOT:?}" "${PROTOCOL:?}"
: "${TASK_RECEIPT:?}" "${EXPECTED_TASK_RECEIPT_SHA:?}"
[[ "$(sha256sum "${TASK_RECEIPT}" | awk '{print $1}')" == \
   "${EXPECTED_TASK_RECEIPT_SHA}" ]]
(cd "${TASK_ROOT}" && sha256sum -c --quiet "${TASK_RECEIPT}")
[[ "$(find "${RUN_ROOT}/construct_ab/scenes" -mindepth 2 -maxdepth 2 \
  -name completion.json -type f | wc -l)" -eq 54 ]]
py=/scratch/lg154/conda-envs/memnav/bin/python
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPYCACHEPREFIX=${SLURM_TMPDIR:-/tmp}/h3_natbx_seal_${SLURM_JOB_ID}
export PYTHONPATH=${TASK_ROOT}:${TASK_ROOT}/MemNavData
"${py}" -u "${TASK_ROOT}/MemNavData/finalize_hm3d_fullmono_lifelong_ab.py" \
  --parent-root "${PARENT_ROOT}" --protocol "${PROTOCOL}" \
  --construction-root "${RUN_ROOT}/construct_ab/scenes" \
  --out "${RUN_ROOT}/ab_population"
"${py}" - "${RUN_ROOT}" <<'PY'
import json,sys
from pathlib import Path
r=Path(sys.argv[1]); p=json.load(open(r/'ab_population/population_receipt.json'))
assert p['constructible_AB_C_histories'] == 84
assert p['constructible_scene_clusters'] == 30
assert p['construction_target_met'] is True
assert p['factual_B_authorized'] is True
rows=[json.load(open(x)) for x in sorted((r/'construct_ab/scenes').glob('*/completion.json'))]
assert sum(x['materialized_recipient_histories'] for x in rows) == 53
assert sum(x['constructible_AB_C_histories'] for x in rows) == 84
assert all(x['audit_candidate_count_reproduced'] is True for x in rows)
assert all(x['separation_against_original_and_expansion_recomputed'] is True for x in rows)
assert all(x['query_policy_outcomes_read'] is False for x in rows)
assert all(x['navigation_outcomes_read'] is False for x in rows)
PY
(cd "${RUN_ROOT}/ab_population" && \
  sha256sum -c --quiet population_receipt.json.sha256 && \
  sha256sum -c --quiet BENCHMARK_FILES.sha256)
chmod -R a-w "${RUN_ROOT}/construct_ab" "${RUN_ROOT}/ab_population"
