#!/usr/bin/env bash
set -euo pipefail
: "${COVIS_ADDON:?}" "${COVIS_EVAL_ROOT:?}" "${COVIS_POPULATION:?}" "${EXPECTED_ADDON_SHA:?}"
: "${SLURM_ARRAY_TASK_ID:?}"
export REPAIRED_BUNDLE=/scratch/yz11502/Research/Nav-axis-uturn-source-bundles/repaired_fullmono_c8cf8c60e7efd55f
source "${REPAIRED_BUNDLE}/MemNavData/repaired_hm3d_hpc_env.sh"
COVIS_TMP=$(mktemp -d "${SLURM_TMPDIR:-/tmp}/covis_eval_${SLURM_JOB_ID}_XXXXXX")
export TMPDIR=${COVIS_TMP} LIBFFI_TMPDIR=${COVIS_TMP} PYTHONFAULTHANDLER=1
COVIS_DURABLE=$(printf '%s/task_%03d' "${COVIS_EVAL_ROOT}" "${SLURM_ARRAY_TASK_ID}")
[[ "$(id -un)" == yz11502 ]]
[[ ! -e "${COVIS_DURABLE}" ]]
mkdir -p "${COVIS_DURABLE}/preflight"
[[ "$(sha256sum "${REPAIRED_SOURCE_RECEIPT}" | awk '{print $1}')" == c8cf8c60e7efd55f1b360bb5a2d5fd92850dc87d6a310b05e77f0a9239cdfc08 ]]
[[ "$(sha256sum "${COVIS_ADDON}/SOURCE_BUNDLE.sha256" | awk '{print $1}')" == "${EXPECTED_ADDON_SHA}" ]]
(cd "${REPAIRED_BUNDLE}" && sha256sum -c --quiet SOURCE_BUNDLE.sha256)
(cd "${COVIS_ADDON}" && sha256sum -c --quiet SOURCE_BUNDLE.sha256)
COVIS_DEPS=/scratch/yz11502/Research/Nav-axis-uturn-source-bundles/final14_mono_factorial_5690569a4373f2d2
[[ "$(sha256sum "${COVIS_DEPS}/source_inputs.sha256" | awk '{print $1}')" == 5690569a4373f2d2768671418f0c604c4a03aa4b0ffe01baf70b288af03ba216 ]]
(cd "${COVIS_DEPS}" && sha256sum -c --quiet source_inputs.sha256)
[[ "$(sha256sum "${REPAIRED_HAB_EXTRA}/SOURCE_DEPENDENCY.sha256" | awk '{print $1}')" == 43c80ec16d3363f516843f2897233f80a28ea18647f386ff7964e0a5a954567d ]]
(cd "${REPAIRED_HAB_EXTRA}" && sha256sum -c --quiet SOURCE_DEPENDENCY.sha256)
[[ "$(sha256sum "${REPAIRED_MEM_CKPT}" | awk '{print $1}')" == 9b7a5811ff0aea212503f58b45258ba4f66b06420f87c350946aead39db6fdb7 ]]
[[ "$(sha256sum "${REPAIRED_NAV_CKPT}" | awk '{print $1}')" == 3bb3ad4ab241e857bb57a4021cc6aab76d5263e81fbf80298d579053ef011947 ]]
[[ "$(sha256sum "${REPAIRED_LINGBOT_REPO}/weights/lingbot-map-long.pt" | awk '{print $1}')" == 832bc82cbae0bc9bbe946ef5ee1f7226abd8c0e183ccf8beddbb3d133576f409 ]]
cd "${REPAIRED_BUNDLE}"
for kind in habitat memnav navdp; do
  if [[ "${kind}" == habitat ]]; then
    "${REPAIRED_HAB_PY}" MemNavData/preflight_repaired_fullmono.py "${kind}" >"${COVIS_DURABLE}/preflight/${kind}.log" 2>&1
  else
    env PYTHONPATH="${REPAIRED_BUNDLE}/NavDP/baselines/${kind}:${PYTHONPATH}" \
      "${REPAIRED_MEM_PY}" MemNavData/preflight_repaired_fullmono.py "${kind}" >"${COVIS_DURABLE}/preflight/${kind}.log" 2>&1
  fi
done
source "${REPAIRED_BUNDLE}/MemNavData/slurm_port_pair.sh"
claim_slurm_tcp_port_pair covis_repaired 12000 6000
trap release_slurm_tcp_port_pair EXIT
nvidia-smi --query-gpu=name,uuid,driver_version,memory.total --format=csv >"${COVIS_DURABLE}/preflight/gpu.csv"
# A bounded process deadline reserves time to close private servers and preserve
# partial evidence within the one-hour allocation. It does not alter tick budget.
set +e
timeout --signal=INT --kill-after=45s 48m "${REPAIRED_MEM_PY}" -u "${COVIS_ADDON}/repaired_covisibility_eval.py" run \
  --population "${COVIS_POPULATION}" --index "${SLURM_ARRAY_TASK_ID}" \
  --out "${COVIS_TMP}/task" --durable "${COVIS_DURABLE}" --memnav-port "${MEMNAV_PORT}" --navdp-port "${NAVDP_PORT}"
COVIS_EXIT=$?
set -e
if [[ -d "${COVIS_TMP}/task" ]]; then
  "${REPAIRED_MEM_PY}" "${COVIS_ADDON}/covisibility_task_archive.py" --out "${COVIS_TMP}/task" \
    --durable "${COVIS_DURABLE}" --exit-code "${COVIS_EXIT}" >"${COVIS_DURABLE}/archive.log" 2>&1
fi
exit "${COVIS_EXIT}"
