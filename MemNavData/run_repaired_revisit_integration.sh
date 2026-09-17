#!/usr/bin/env bash
# A thin orchestrator over the already verified frozen runtime.
set -euo pipefail
: "${REVISIT_ADDON:?}" "${REVISIT_RUN:?}" "${EXPECTED_ADDON_SHA:?}"
export REPAIRED_BUNDLE=/scratch/yz11502/Research/Nav-axis-uturn-source-bundles/repaired_fullmono_c8cf8c60e7efd55f
source "${REPAIRED_BUNDLE}/MemNavData/repaired_hm3d_hpc_env.sh"
REVISIT_TASK_TMP=$(mktemp -d "${SLURM_TMPDIR:-/tmp}/cec_revisit_${SLURM_JOB_ID:-preflight}_XXXXXX")
export TMPDIR=${REVISIT_TASK_TMP} LIBFFI_TMPDIR=${REVISIT_TASK_TMP}
export PYTHONFAULTHANDLER=1
[[ "$(id -un)" == yz11502 ]]
[[ "$(sha256sum "${REPAIRED_SOURCE_RECEIPT}" | awk '{print $1}')" == c8cf8c60e7efd55f1b360bb5a2d5fd92850dc87d6a310b05e77f0a9239cdfc08 ]]
[[ "$(sha256sum "${REVISIT_ADDON}/SOURCE_BUNDLE.sha256" | awk '{print $1}')" == "${EXPECTED_ADDON_SHA}" ]]
(cd "${REPAIRED_BUNDLE}" && sha256sum -c --quiet SOURCE_BUNDLE.sha256)
(cd "${REVISIT_ADDON}" && sha256sum -c --quiet SOURCE_BUNDLE.sha256)
REVISIT_DEPS=/scratch/yz11502/Research/Nav-axis-uturn-source-bundles/final14_mono_factorial_5690569a4373f2d2
[[ "$(sha256sum "${REVISIT_DEPS}/source_inputs.sha256" | awk '{print $1}')" == 5690569a4373f2d2768671418f0c604c4a03aa4b0ffe01baf70b288af03ba216 ]]
(cd "${REVISIT_DEPS}" && sha256sum -c --quiet source_inputs.sha256)
[[ "$(sha256sum "${REPAIRED_HAB_EXTRA}/SOURCE_DEPENDENCY.sha256" | awk '{print $1}')" == 43c80ec16d3363f516843f2897233f80a28ea18647f386ff7964e0a5a954567d ]]
(cd "${REPAIRED_HAB_EXTRA}" && sha256sum -c --quiet SOURCE_DEPENDENCY.sha256)
[[ ! -e "${REVISIT_RUN}" ]]
mkdir -p "${REVISIT_RUN}/preflight"
cd "${REPAIRED_BUNDLE}"
for kind in habitat memnav navdp; do
  if [[ "${kind}" == habitat ]]; then
    "${REPAIRED_HAB_PY}" MemNavData/preflight_repaired_fullmono.py "${kind}" \
      >"${REVISIT_RUN}/preflight/${kind}.log" 2>&1
  else
    env PYTHONPATH="${REPAIRED_BUNDLE}/NavDP/baselines/${kind}:${PYTHONPATH}" \
      "${REPAIRED_MEM_PY}" MemNavData/preflight_repaired_fullmono.py "${kind}" \
      >"${REVISIT_RUN}/preflight/${kind}.log" 2>&1
  fi
done
"${REPAIRED_HAB_PY}" "${REVISIT_ADDON}/test_repaired_revisit_integration.py" -v \
  >"${REVISIT_RUN}/preflight/unit_tests.log" 2>&1
"${REPAIRED_HAB_PY}" "${REVISIT_ADDON}/repaired_revisit_integration.py" dry-run \
  --out "${REVISIT_TASK_TMP}/dry_run_must_not_exist" >"${REVISIT_RUN}/preflight/cli.log" 2>&1
[[ "$(sha256sum "${REPAIRED_MEM_CKPT}" | awk '{print $1}')" == 9b7a5811ff0aea212503f58b45258ba4f66b06420f87c350946aead39db6fdb7 ]]
[[ "$(sha256sum "${REPAIRED_NAV_CKPT}" | awk '{print $1}')" == 3bb3ad4ab241e857bb57a4021cc6aab76d5263e81fbf80298d579053ef011947 ]]
[[ "$(sha256sum "${REPAIRED_LINGBOT_REPO}/weights/lingbot-map-long.pt" | awk '{print $1}')" == 832bc82cbae0bc9bbe946ef5ee1f7226abd8c0e183ccf8beddbb3d133576f409 ]]
ffmpeg -hide_banner -loglevel error -f lavfi -i color=size=32x32:rate=1 -frames:v 1 -c:v libx264 -f null -
source "${REPAIRED_BUNDLE}/MemNavData/slurm_port_pair.sh"
claim_slurm_tcp_port_pair repaired_revisit 12000 6000
trap release_slurm_tcp_port_pair EXIT
export REPAIRED_BUFFER_ROOT=${REVISIT_TASK_TMP}/buffer
export REPAIRED_RUNTIME_ROOT=${REVISIT_TASK_TMP}/runtime
nvidia-smi --query-gpu=name,uuid,driver_version,memory.total --format=csv >"${REVISIT_RUN}/preflight/gpu.csv"
# Restoring both exact goals happens before starting either model or any query.
"${REPAIRED_MEM_PY}" -u "${REVISIT_ADDON}/repaired_revisit_integration.py" run \
  --out "${REVISIT_RUN}/integration" --memnav-port "${MEMNAV_PORT}" --navdp-port "${NAVDP_PORT}"
