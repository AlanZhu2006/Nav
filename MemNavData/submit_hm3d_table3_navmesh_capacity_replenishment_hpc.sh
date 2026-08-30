#!/usr/bin/env bash
# Result-blind capacity replenishment only: no renderer, policy, or outcome.
set -euo pipefail

ROOT=${ROOT:-$(git rev-parse --show-toplevel)}
cd "${ROOT}"
manifest=MemNavData/hm3d_table3_combined_assets_20260830.json
protocol=MemNavData/hm3d_table3_navmesh_capacity_replenishment_protocol_20260831.json
[[ -f "${manifest}" && -f "${protocol}" ]]
manifest_sha=$(sha256sum "${manifest}" | awk '{print $1}')
recorded=$(/home/asus/miniconda3/envs/memnav/bin/python - "${protocol}" <<'PY'
import json,sys
p=json.load(open(sys.argv[1]))
assert p['authority_boundary']['query_policy_outcomes_read'] is False
assert p['authority_boundary']['navigation_outcomes_read'] is False
assert p['authority_boundary']['this_audit_authorizes_policy_evaluation'] is False
assert p['sampling']['points_per_scene']==512
assert p['sampling']['base_seed']==20260831
print(p['parent']['manifest_sha256'])
PY
)
[[ "${manifest_sha}" == "${recorded}" ]]
exec env \
  PARENT_MANIFEST_LOCAL="${manifest}" \
  PARENT_MANIFEST="${manifest}" \
  EXPECTED_PARENT_MANIFEST_SHA="${manifest_sha}" \
  PROTOCOL_REL="${protocol}" \
  SCENE_COUNT=100 \
  CONCURRENCY="${CONCURRENCY:-12}" \
  BUNDLE_PREFIX=hm3d_table3_capacity_replenishment \
  RUN_PREFIX=capacity_replenishment \
  LOCAL_SUBMISSION_RECEIPT=MemNavData/HM3D_TABLE3_NAVMESH_CAPACITY_REPLENISHMENT_SUBMISSION_20260831.json \
  REMOTE_RESULTS=/scratch/yz11502/Research/Nav-axis-uturn-results/hm3d_table3_navmesh_capacity_replenishment_20260831 \
  bash MemNavData/submit_hm3d_table3_navmesh_capacity_hpc.sh
