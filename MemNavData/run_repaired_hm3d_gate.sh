#!/usr/bin/env bash
# Runs inside the same GPU container as all model and Habitat processes.
set -euo pipefail
: "${REPAIRED_BUNDLE:?}" "${REPAIRED_RUN:?}" "${EXPECTED_REPAIR_SHA:?}"
source "${REPAIRED_BUNDLE}/MemNavData/repaired_hm3d_hpc_env.sh"
REPAIRED_TASK_TMP=$(mktemp -d "${SLURM_TMPDIR:-/tmp}/cec_repaired_${SLURM_JOB_ID:-gate}_XXXXXX")
export TMPDIR=${REPAIRED_TASK_TMP}
export LIBFFI_TMPDIR=${REPAIRED_TASK_TMP}
ffmpeg -hide_banner -loglevel error -f lavfi -i color=size=32x32:rate=1 \
  -frames:v 1 -c:v libx264 -f null -
[[ "$(id -un)" == yz11502 ]]
[[ "$(sha256sum "${REPAIRED_SOURCE_RECEIPT}" | awk '{print $1}')" == "${EXPECTED_REPAIR_SHA}" ]]
(cd "${REPAIRED_BUNDLE}" && sha256sum -c --quiet SOURCE_BUNDLE.sha256)
REPAIRED_DEP_BUNDLE=/scratch/yz11502/Research/Nav-axis-uturn-source-bundles/final14_mono_factorial_5690569a4373f2d2
[[ "$(sha256sum "${REPAIRED_DEP_BUNDLE}/source_inputs.sha256" | awk '{print $1}')" == 5690569a4373f2d2768671418f0c604c4a03aa4b0ffe01baf70b288af03ba216 ]]
(cd "${REPAIRED_DEP_BUNDLE}" && sha256sum -c --quiet source_inputs.sha256)
[[ "$(sha256sum "${REPAIRED_HAB_EXTRA}/SOURCE_DEPENDENCY.sha256" | awk '{print $1}')" == 43c80ec16d3363f516843f2897233f80a28ea18647f386ff7964e0a5a954567d ]]
(cd "${REPAIRED_HAB_EXTRA}" && sha256sum -c --quiet SOURCE_DEPENDENCY.sha256)
[[ ! -e "${REPAIRED_RUN}" ]]
mkdir -p "${REPAIRED_RUN}/preflight"
cd "${REPAIRED_BUNDLE}"
for kind in habitat memnav navdp; do
  if [[ "${kind}" == habitat ]]; then
    env PYTHONPATH="${PYTHONPATH}" "${REPAIRED_HAB_PY}" \
      MemNavData/preflight_repaired_fullmono.py "${kind}" >"${REPAIRED_RUN}/preflight/${kind}.log" 2>&1
  else
    env PYTHONPATH="${REPAIRED_BUNDLE}/NavDP/baselines/${kind}:${PYTHONPATH}" "${REPAIRED_MEM_PY}" \
      MemNavData/preflight_repaired_fullmono.py "${kind}" >"${REPAIRED_RUN}/preflight/${kind}.log" 2>&1
  fi
done
[[ "$(sha256sum "${REPAIRED_MEM_CKPT}" | awk '{print $1}')" == 9b7a5811ff0aea212503f58b45258ba4f66b06420f87c350946aead39db6fdb7 ]]
[[ "$(sha256sum "${REPAIRED_NAV_CKPT}" | awk '{print $1}')" == 3bb3ad4ab241e857bb57a4021cc6aab76d5263e81fbf80298d579053ef011947 ]]
[[ "$(sha256sum "${REPAIRED_LINGBOT_REPO}/weights/lingbot-map-long.pt" | awk '{print $1}')" == 832bc82cbae0bc9bbe946ef5ee1f7226abd8c0e183ccf8beddbb3d133576f409 ]]
"${REPAIRED_MEM_PY}" MemNavData/build_repaired_hm3d_gate_sources.py \
  /scratch/yz11502/Research/Nav-axis-uturn-results/hm3d_fresh_fullmono_mixed_role_20260820/formal_20260820T143609Z_e6dd44c6/sealed_inputs/parent_manifest.json \
  "${REPAIRED_GATE_PROTOCOL:-${REPAIRED_BUNDLE}/MemNavData/REPAIRED_HM3D_GATE_PROTOCOL_20260908.md}" \
  "${REPAIRED_RUN}/sources.json" --phase "${REPAIRED_GATE_PHASE:-initial}" \
  --shard "${REPAIRED_GATE_SHARD:-0}"
source "${REPAIRED_BUNDLE}/MemNavData/slurm_port_pair.sh"
claim_slurm_tcp_port_pair repaired_hm3d 12000 6000
trap release_slurm_tcp_port_pair EXIT
export REPAIRED_BUFFER_ROOT=${REPAIRED_TASK_TMP}/buffer
export REPAIRED_RUNTIME_ROOT=${REPAIRED_TASK_TMP}/runtime
nvidia-smi --query-gpu=name,uuid,driver_version,memory.total --format=csv >"${REPAIRED_RUN}/preflight/gpu.csv"
# Verify an actual render before loading the two models.  No policy is run.
"${REPAIRED_HAB_PY}" - "${REPAIRED_RUN}/sources.json" <<'PY'
import json,sys
from generate_twoleg import make_sim
for source in json.load(open(sys.argv[1]))['sources']:
    sim=make_sim(source['asset'], '', agent_radius=.30)
    try:
        images=sim.get_sensor_observations()
        assert images['color'].shape[:2] == (270,480)
        assert images['depth'].shape[:2] == (270,480)
    finally:
        sim.close()
print('Both source assets render in the exact runtime')
PY
"${REPAIRED_MEM_PY}" -u -m MemNavData.run_repaired_fullmono_local local \
  --source-manifest "${REPAIRED_RUN}/sources.json" --out "${REPAIRED_RUN}/integration" \
  --memnav-port "${MEMNAV_PORT}" --navdp-port "${NAVDP_PORT}"
# No downstream formal job is launched by this gate.
