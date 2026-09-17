#!/usr/bin/env bash
set -euo pipefail
: "${REPAIRED_BUNDLE:?}" "${TABLE1_RUN:?}" "${TABLE1_PLAN:?}" "${TABLE1_EVAL_ROOT:?}"
: "${EXPECTED_RUNTIME_SHA:?}" "${EXPECTED_PLAN_SHA:?}" "${SLURM_ARRAY_TASK_ID:?}"
[[ "$(id -un)" == yz11502 ]]
source "${REPAIRED_BUNDLE}/MemNavData/repaired_hm3d_hpc_env.sh"
export TABLE1_IMAGE_PY=/scratch/yz11502/Research/Nav-axis-uturn-envs/controller_portability_a9ec7146bce7_v1/vint/bin/python
export TABLE1_CHECKPOINTS=/scratch/yz11502/Research/Nav-axis-uturn-checkpoints/controller_portability_50387aa89be8
TABLE1_TMP=$(mktemp -d "${SLURM_TMPDIR:-/tmp}/table1_${SLURM_JOB_ID}_XXXXXX")
export TMPDIR=${TABLE1_TMP} LIBFFI_TMPDIR=${TABLE1_TMP} PYTHONFAULTHANDLER=1
TABLE1_DURABLE=$(printf '%s/task_%03d' "${TABLE1_EVAL_ROOT}" "${SLURM_ARRAY_TASK_ID}")
[[ ! -e "${TABLE1_DURABLE}" ]]
mkdir -p "${TABLE1_DURABLE}/preflight"
[[ "$(sha256sum "${REPAIRED_SOURCE_RECEIPT}" | awk '{print $1}')" == "${EXPECTED_RUNTIME_SHA}" ]]
[[ "$(sha256sum "${TABLE1_PLAN}" | awk '{print $1}')" == "${EXPECTED_PLAN_SHA}" ]]
(cd "${REPAIRED_BUNDLE}" && sha256sum -c --quiet SOURCE_BUNDLE.sha256)
TABLE1_DEPS=/scratch/yz11502/Research/Nav-axis-uturn-source-bundles/final14_mono_factorial_5690569a4373f2d2
[[ "$(sha256sum "${TABLE1_DEPS}/source_inputs.sha256" | awk '{print $1}')" == 5690569a4373f2d2768671418f0c604c4a03aa4b0ffe01baf70b288af03ba216 ]]
(cd "${TABLE1_DEPS}" && sha256sum -c --quiet source_inputs.sha256)
[[ "$(sha256sum "${REPAIRED_HAB_EXTRA}/SOURCE_DEPENDENCY.sha256" | awk '{print $1}')" == 43c80ec16d3363f516843f2897233f80a28ea18647f386ff7964e0a5a954567d ]]
(cd "${REPAIRED_HAB_EXTRA}" && sha256sum -c --quiet SOURCE_DEPENDENCY.sha256)
[[ "$(sha256sum "${REPAIRED_MEM_CKPT}" | awk '{print $1}')" == 9b7a5811ff0aea212503f58b45258ba4f66b06420f87c350946aead39db6fdb7 ]]
[[ "$(sha256sum "${REPAIRED_NAV_CKPT}" | awk '{print $1}')" == 3bb3ad4ab241e857bb57a4021cc6aab76d5263e81fbf80298d579053ef011947 ]]
[[ "$(sha256sum "${REPAIRED_LINGBOT_REPO}/weights/lingbot-map-long.pt" | awk '{print $1}')" == 832bc82cbae0bc9bbe946ef5ee1f7226abd8c0e183ccf8beddbb3d133576f409 ]]
[[ "$(sha256sum "${TABLE1_CHECKPOINTS}/vint.pth" | awk '{print $1}')" == 155fd72de2e98ae0e2fef9404072e1aefa79dae5f7f2411d4bcf7e384b83aa1f ]]
[[ "$(sha256sum "${TABLE1_CHECKPOINTS}/nomad.pth" | awk '{print $1}')" == 70f79b8262527e20e56ced64a3e3d7ef91855bc9e7c3fa348d78edcb83c6a333 ]]
cd "${REPAIRED_BUNDLE}"
for kind in habitat memnav navdp; do
  if [[ "${kind}" == habitat ]]; then
    "${REPAIRED_HAB_PY}" MemNavData/preflight_repaired_fullmono.py "${kind}" >"${TABLE1_DURABLE}/preflight/${kind}.log" 2>&1
  else
    env PYTHONPATH="${REPAIRED_BUNDLE}/NavDP/baselines/${kind}:${PYTHONPATH}" \
      "${REPAIRED_MEM_PY}" MemNavData/preflight_repaired_fullmono.py "${kind}" >"${TABLE1_DURABLE}/preflight/${kind}.log" 2>&1
  fi
done
source "${REPAIRED_BUNDLE}/MemNavData/slurm_port_pair.sh"
claim_slurm_tcp_port_pair table1_repaired 12000 6000
trap release_slurm_tcp_port_pair EXIT
nvidia-smi --query-gpu=name,uuid,driver_version,memory.total --format=csv >"${TABLE1_DURABLE}/preflight/gpu.csv"
set +e
timeout --signal=INT --kill-after=45s 48m "${REPAIRED_MEM_PY}" -u MemNavData/table1_repaired_eval.py run \
  --plan "${TABLE1_PLAN}" --index "${SLURM_ARRAY_TASK_ID}" --out "${TABLE1_TMP}/task" \
  --mem-port "${MEMNAV_PORT}" --nav-port "${NAVDP_PORT}"
TABLE1_EXIT=$?
set -e
if [[ -d "${TABLE1_TMP}/task" ]]; then
  "${REPAIRED_MEM_PY}" MemNavData/covisibility_task_archive.py --out "${TABLE1_TMP}/task" \
    --durable "${TABLE1_DURABLE}" --exit-code "${TABLE1_EXIT}" >"${TABLE1_DURABLE}/archive.log" 2>&1
fi
exit "${TABLE1_EXIT}"
