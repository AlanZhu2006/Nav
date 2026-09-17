#!/usr/bin/env bash
set -euo pipefail
: "${REPAIRED_BUNDLE:?}" "${COVERAGE_EVAL_ROOT:?}" "${COVERAGE_PLAN:?}"
: "${EXPECTED_RUNTIME_SHA:?}" "${EXPECTED_PLAN_SHA:?}" "${SLURM_ARRAY_TASK_ID:?}"
[[ "$(id -un)" == yz11502 ]]
source "${REPAIRED_BUNDLE}/MemNavData/repaired_hm3d_hpc_env.sh"
COVERAGE_TMP=$(mktemp -d "${SLURM_TMPDIR:-/tmp}/coverage_${SLURM_JOB_ID}_XXXXXX")
export TMPDIR=${COVERAGE_TMP} LIBFFI_TMPDIR=${COVERAGE_TMP} PYTHONFAULTHANDLER=1
COVERAGE_DURABLE=$(printf '%s/task_%03d' "${COVERAGE_EVAL_ROOT}" "${SLURM_ARRAY_TASK_ID}")
[[ ! -e "${COVERAGE_DURABLE}" ]]
mkdir -p "${COVERAGE_DURABLE}/preflight"
[[ "$(sha256sum "${REPAIRED_SOURCE_RECEIPT}" | awk '{print $1}')" == "${EXPECTED_RUNTIME_SHA}" ]]
[[ "$(sha256sum "${COVERAGE_PLAN}" | awk '{print $1}')" == "${EXPECTED_PLAN_SHA}" ]]
(cd "${REPAIRED_BUNDLE}" && sha256sum -c --quiet SOURCE_BUNDLE.sha256)
COVERAGE_DEPS=/scratch/yz11502/Research/Nav-axis-uturn-source-bundles/final14_mono_factorial_5690569a4373f2d2
[[ "$(sha256sum "${COVERAGE_DEPS}/source_inputs.sha256" | awk '{print $1}')" == 5690569a4373f2d2768671418f0c604c4a03aa4b0ffe01baf70b288af03ba216 ]]
(cd "${COVERAGE_DEPS}" && sha256sum -c --quiet source_inputs.sha256)
[[ "$(sha256sum "${REPAIRED_HAB_EXTRA}/SOURCE_DEPENDENCY.sha256" | awk '{print $1}')" == 43c80ec16d3363f516843f2897233f80a28ea18647f386ff7964e0a5a954567d ]]
(cd "${REPAIRED_HAB_EXTRA}" && sha256sum -c --quiet SOURCE_DEPENDENCY.sha256)
[[ "$(sha256sum "${REPAIRED_MEM_CKPT}" | awk '{print $1}')" == 9b7a5811ff0aea212503f58b45258ba4f66b06420f87c350946aead39db6fdb7 ]]
[[ "$(sha256sum "${REPAIRED_NAV_CKPT}" | awk '{print $1}')" == 3bb3ad4ab241e857bb57a4021cc6aab76d5263e81fbf80298d579053ef011947 ]]
[[ "$(sha256sum "${REPAIRED_LINGBOT_REPO}/weights/lingbot-map-long.pt" | awk '{print $1}')" == 832bc82cbae0bc9bbe946ef5ee1f7226abd8c0e183ccf8beddbb3d133576f409 ]]
cd "${REPAIRED_BUNDLE}"
for kind in habitat memnav navdp; do
  if [[ "${kind}" == habitat ]]; then
    "${REPAIRED_HAB_PY}" MemNavData/preflight_repaired_fullmono.py "${kind}" >"${COVERAGE_DURABLE}/preflight/${kind}.log" 2>&1
  else
    env PYTHONPATH="${REPAIRED_BUNDLE}/NavDP/baselines/${kind}:${PYTHONPATH}" \
      "${REPAIRED_MEM_PY}" MemNavData/preflight_repaired_fullmono.py "${kind}" >"${COVERAGE_DURABLE}/preflight/${kind}.log" 2>&1
  fi
done
source "${REPAIRED_BUNDLE}/MemNavData/slurm_port_pair.sh"
claim_slurm_tcp_port_pair coverage_ablation 12000 6000
trap release_slurm_tcp_port_pair EXIT
nvidia-smi --query-gpu=name,uuid,driver_version,memory.total --format=csv >"${COVERAGE_DURABLE}/preflight/gpu.csv"
set +e
timeout --signal=INT --kill-after=45s 48m "${REPAIRED_MEM_PY}" -u MemNavData/coverage_ablation_hpc.py run \
  --plan "${COVERAGE_PLAN}" --index "${SLURM_ARRAY_TASK_ID}" --out "${COVERAGE_TMP}/task" \
  --durable "${COVERAGE_DURABLE}" --memnav-port "${MEMNAV_PORT}" --navdp-port "${NAVDP_PORT}"
COVERAGE_EXIT=$?
set -e
if [[ -d "${COVERAGE_TMP}/task" ]]; then
  "${REPAIRED_MEM_PY}" MemNavData/covisibility_task_archive.py --out "${COVERAGE_TMP}/task" \
    --durable "${COVERAGE_DURABLE}" --exit-code "${COVERAGE_EXIT}" >"${COVERAGE_DURABLE}/archive.log" 2>&1
fi
exit "${COVERAGE_EXIT}"
