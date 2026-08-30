#!/usr/bin/env bash
set -euo pipefail
ROOT=${ROOT:-$(git rev-parse --show-toplevel)}
cd "${ROOT}"
manifest=MemNavData/hm3d_table3_combined_assets_20260830.json
protocol=MemNavData/hm3d_table3_navmesh_capacity_100scene_protocol_20260830.json
[[ -f "${manifest}" && -f "${protocol}" ]]
manifest_sha=$(sha256sum "${manifest}" | awk '{print $1}')
recorded=$(python - "${protocol}" <<'PY'
import json,sys
print(json.load(open(sys.argv[1]))['parent']['manifest_sha256'])
PY
)
[[ "${manifest_sha}" == "${recorded}" ]]
exec env \
  PARENT_MANIFEST_LOCAL="${manifest}" \
  PARENT_MANIFEST="${manifest}" \
  EXPECTED_PARENT_MANIFEST_SHA="${manifest_sha}" \
  PROTOCOL_REL="${protocol}" \
  SCENE_COUNT=100 \
  BUNDLE_PREFIX=hm3d_table3_capacity100 \
  RUN_PREFIX=capacity100 \
  LOCAL_SUBMISSION_RECEIPT=MemNavData/HM3D_TABLE3_NAVMESH_CAPACITY_100SCENE_SUBMISSION_20260830.json \
  REMOTE_RESULTS=/scratch/yz11502/Research/Nav-axis-uturn-results/hm3d_table3_navmesh_capacity_100scene_20260830 \
  bash MemNavData/submit_hm3d_table3_navmesh_capacity_hpc.sh
