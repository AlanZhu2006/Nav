#!/usr/bin/env bash
# CPU-only regression checks. This does not start a model server or a rollout.
set -euo pipefail
SIM_PROJECT_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
SIM_AUDIT_PY=${SIM_AUDIT_PY:-/home/asus/miniconda3/envs/memnav/bin/python}
cd -- "${SIM_PROJECT_ROOT}"
export PYTHONPATH="${SIM_PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

"${SIM_AUDIT_PY}" -m pytest -q \
  MemNavData/test_bounded_pursuit.py \
  MemNavData/test_habitat_executor_audit.py \
  MemNavData/test_navdp_base_rgb.py \
  MemNavData/test_lingbot_depth_raster.py \
  MemNavData/test_navdp_depth_raster_server.py \
  MemNavData/test_navdp_front_goal_adapter.py \
  MemNavData/test_archived_zero_reference.py \
  MemNavData/test_lingbot_pnp_localization.py \
  MemNavData/test_simulation_authority_boundary.py \
  MemNavData/test_navdp_memory_replay.py \
  MemNavData/test_navdp_goal_switch.py \
  MemNavData/test_navdp_native_first_audit_server.py \
  MemNavData/test_navdp_monocular_transaction.py \
  MemNavData/test_monocular_depth_runtime.py \
  MemNavData/test_causal_metric_scale_receipt.py \
  MemNavData/test_first40_local_pose_scale.py \
  MemNavData/test_shared_online_role_pair_contract.py \
  MemNavData/test_cec_authority_receipt.py \
  MemNavData/test_final14_spl_replay.py "$@"

# Tests cover the isolated candidate and explicitly reproduce legacy defects.
# Passing is not evidence that production defaults or old SR have been fixed.
