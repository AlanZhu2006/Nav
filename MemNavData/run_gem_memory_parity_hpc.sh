#!/usr/bin/env bash
set -euo pipefail
[[ "$(id -un)" == yz11502 ]]
source "${REPAIRED_BUNDLE}/MemNavData/repaired_hm3d_hpc_env.sh"
cd "${REPAIRED_BUNDLE}"
sha256sum -c --quiet GEM_SOURCE.sha256
export LINGBOT_REPO=${REPAIRED_LINGBOT_REPO}
export LINGBOT_WEIGHTS=${REPAIRED_LINGBOT_REPO}/weights/lingbot-map-long.pt
export MEMNAV_WINDOW=64 MEMNAV_NUM_SCALE=8 MEMNAV_MAX_FRAME_NUM=4096
export MEMNAV_GROUND_SCALE_MAX=6.0 MEMNAV_GATE_FUSION=complementary
export MEMNAV_AUX_POSE_CALIBRATION=empirical MEMNAV_COLLISION_SELECT=1
export MEMNAV_REPORT_TO=none PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
exec "${REPAIRED_MEM_PY}" MemNavData/verify_gem_episodic_gpu.py \
  --inputs "${GEM_PARITY_INPUTS}" --out "${GEM_PARITY_OUT}" \
  --checkpoint "${REPAIRED_MEM_CKPT}" --lightglue "${REPAIRED_LIGHTGLUE}" \
  --dependencies "${REPAIRED_DEPENDENCIES}"
