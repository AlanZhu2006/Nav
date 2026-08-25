#!/usr/bin/env bash
# Bundle and submit a controller-portability lifelong SR study: the same
# frozen 18-episode factual-B population and CEC certificate pipeline as the
# memnav-native three-arm expansion, but the accepted branch is executed by
# CONTROLLER (vint, iplanner, ...) through cec_controller_portability_hub.py.
# Unlike the navdp forced-arm addendum, all three arms (all_prior,
# initial_leg_only, forced_reject_native) run fresh here -- there is no
# pre-existing verified pair for a non-navdp controller to join.
set -euo pipefail
umask 0022

LOCAL_ROOT=$(git rev-parse --show-toplevel)
CONTROLLER=${CONTROLLER:?set CONTROLLER=vint|iplanner|...}
SHARED_C_REPAIR=${SHARED_C_REPAIR:-0}
REMOTE_HOST=${REMOTE_HOST:-alantorch}
REMOTE_BUNDLE_BASE=${REMOTE_BUNDLE_BASE:-/scratch/yz11502/Research/Nav-axis-uturn-source-bundles}
REMOTE_RESULT_BASE=${REMOTE_RESULT_BASE:-/scratch/yz11502/Research/Nav-axis-uturn-results/lifelong_nnr_controller_20260822}
RUN_TAG=${RUN_TAG:-lifelong_nnr_${CONTROLLER}_$(date -u +%Y%m%dT%H%M%SZ)}
RUN_ROOT=${RUN_ROOT:-${REMOTE_RESULT_BASE}/${RUN_TAG}}
NNR_ROOT=${NNR_ROOT:-/scratch/yz11502/Research/Nav-axis-uturn-results/shared_online_nnr_20260814/shared_online_nnr_strict_v2_20260814}
ORIGINAL_RUN_ROOT=${ORIGINAL_RUN_ROOT:-/scratch/yz11502/Research/Nav-axis-uturn-results/lifelong_nnr_20260821/lifelong_nnr_runtime_repair_20260821T083000Z}
BASE_SOURCE_ROOT=${BASE_SOURCE_ROOT:-/scratch/yz11502/Research/Nav-axis-uturn-source-bundles/certified_relocalization_closed_loop_d3bd281fc374cc80}
BASE_SOURCE_RECEIPT_SHA=${BASE_SOURCE_RECEIPT_SHA:-74001a9e0150c38c599a206fa0f4dd5e1279b9bed5d167119f4d14cb77995e98}
DEPENDENCY_RECEIPT=${DEPENDENCY_RECEIPT:-/scratch/yz11502/Research/Nav-axis-uturn-results/shared_online_double_revisit_fresh_20260813/double_revisit_fresh40_20260813T200121Z/dependency_receipt.json}
EXPECTED_DEPENDENCY_RECEIPT_SHA=${EXPECTED_DEPENDENCY_RECEIPT_SHA:-4eb0ca6479a26f8e04f85a31d906cee4e68b1785f66cfd3ac23bf65424d36e5e}
PORTABILITY_ENV_ROOT=${PORTABILITY_ENV_ROOT:-/scratch/yz11502/Research/Nav-axis-uturn-envs/controller_portability_a9ec7146bce7_v1}
PORTABILITY_CHECKPOINT_ROOT=${PORTABILITY_CHECKPOINT_ROOT:-/scratch/yz11502/Research/Nav-axis-uturn-checkpoints/controller_portability_50387aa89be8}
MEMNAV_PY=${MEMNAV_PY:-/home/asus/miniconda3/envs/memnav/bin/python}
HAB_PY=${HAB_PY:-/home/asus/miniconda3/envs/habitat/bin/python}
EVAL_CONCURRENCY=${EVAL_CONCURRENCY:-2}
DRY_RUN=${DRY_RUN:-0}
SAFE_PARTITIONS=h100_tandon,a100_tandon
TRACE_REPLAY_NODE=${TRACE_REPLAY_NODE:-gh001}
SSH_CONTROL_PATH=${SSH_CONTROL_PATH:-$(
  ssh -G "${REMOTE_HOST}" 2>/dev/null |
    awk '$1=="controlpath"{value=$2} END{print value}'
)}

case "${CONTROLLER}" in
  vint|gnm|nomad|iplanner|viplanner) ;;
  *) echo "ABORT: unsupported CONTROLLER ${CONTROLLER}" >&2; exit 2 ;;
esac

navdp_runtime_support=(
  NavDP/baselines/navdp/depth_anything/depth_anything_v2/dinov2.py
  NavDP/baselines/navdp/depth_anything/depth_anything_v2/dpt.py
  NavDP/baselines/navdp/depth_anything/depth_anything_v2/dinov2_layers/__init__.py
  NavDP/baselines/navdp/depth_anything/depth_anything_v2/dinov2_layers/attention.py
  NavDP/baselines/navdp/depth_anything/depth_anything_v2/dinov2_layers/block.py
  NavDP/baselines/navdp/depth_anything/depth_anything_v2/dinov2_layers/drop_path.py
  NavDP/baselines/navdp/depth_anything/depth_anything_v2/dinov2_layers/layer_scale.py
  NavDP/baselines/navdp/depth_anything/depth_anything_v2/dinov2_layers/mlp.py
  NavDP/baselines/navdp/depth_anything/depth_anything_v2/dinov2_layers/patch_embed.py
  NavDP/baselines/navdp/depth_anything/depth_anything_v2/dinov2_layers/swiglu_ffn.py
  NavDP/baselines/navdp/depth_anything/depth_anything_v2/util/blocks.py
  NavDP/baselines/navdp/depth_anything/depth_anything_v2/util/transform.py
)

remote() {
  timeout 180 ssh -n -tt -o BatchMode=yes -o ControlMaster=no \
    -S "${SSH_CONTROL_PATH}" "${REMOTE_HOST}" "$@" | tr -d '\r'
}
fail() { echo "ABORT: $*" >&2; exit 2; }
[[ -S "${SSH_CONTROL_PATH}" ]] || fail "authoritative SSH master missing"
timeout 15 ssh -O check -S "${SSH_CONTROL_PATH}" "${REMOTE_HOST}" \
  >/dev/null 2>&1 || fail "authoritative SSH master is not responsive"
[[ "${RUN_TAG}" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]] || fail "invalid run tag"
[[ "${EVAL_CONCURRENCY}" =~ ^[1-9][0-9]*$ ]] || fail "invalid eval concurrency"
[[ "${DRY_RUN}" =~ ^[01]$ ]] || fail "DRY_RUN must be 0 or 1"
[[ "${SHARED_C_REPAIR}" =~ ^[01]$ ]] || fail "SHARED_C_REPAIR must be 0 or 1"
[[ "${TRACE_REPLAY_NODE}" =~ ^[A-Za-z0-9.-]+$ ]] || fail "invalid replay node"

# --- local gate: tests, lint, syntax --------------------------------------
export PYTHONPATH=${LOCAL_ROOT}:${LOCAL_ROOT}/MemNavData${PYTHONPATH:+:${PYTHONPATH}}
"${HAB_PY}" -m py_compile \
  "${LOCAL_ROOT}/MemNavData/eval_shared_online_lifelong_nnr.py" \
  "${LOCAL_ROOT}/MemNavData/collect_lifelong_shared_c.py" \
  "${LOCAL_ROOT}/MemNavData/eval_lifelong_shared_c_b2.py" \
  "${LOCAL_ROOT}/MemNavData/eval_2leg_habitat.py"
"${MEMNAV_PY}" -m py_compile \
  "${LOCAL_ROOT}/MemNavData/aggregate_lifelong_nnr_expansion.py" \
  "${LOCAL_ROOT}/MemNavData/independent_verify_shared_online_lifelong_nnr.py" \
  "${LOCAL_ROOT}/MemNavData/lifelong_shared_c_contract.py" \
  "${LOCAL_ROOT}/MemNavData/finalize_lifelong_shared_c_population.py" \
  "${LOCAL_ROOT}/MemNavData/aggregate_lifelong_shared_c_b2.py" \
  "${LOCAL_ROOT}/MemNavData/independent_verify_lifelong_shared_c_b2.py" \
  "${LOCAL_ROOT}/MemNavData/cec_controller_portability_hub.py" \
  "${LOCAL_ROOT}/MemNavData/controller_portability_proxy.py" \
  "${LOCAL_ROOT}/MemNavData/controller_portability_contract.py" \
  "${LOCAL_ROOT}/NavDP/baselines/memnav/policy_agent.py"
"${MEMNAV_PY}" -m pytest -q \
  "${LOCAL_ROOT}/MemNavData/test_cec_controller_portability_hub.py" \
  "${LOCAL_ROOT}/MemNavData/test_controller_portability_contract.py" \
  "${LOCAL_ROOT}/MemNavData/test_controller_portability_proxy.py" \
  "${LOCAL_ROOT}/MemNavData/test_lifelong_forced_reject_contract.py" \
  "${LOCAL_ROOT}/MemNavData/test_lifelong_shared_c_contract.py" \
  "${LOCAL_ROOT}/MemNavData/test_policy_agent_graph.py"
bash -n \
  "${LOCAL_ROOT}/MemNavData/run_cec_controller_portability_smoke_local.sh" \
  "${LOCAL_ROOT}/MemNavData/slurm_lifelong_nnr_controller_arm.sbatch" \
  "${LOCAL_ROOT}/MemNavData/slurm_lifelong_nnr_controller_aggregate.sbatch" \
  "${LOCAL_ROOT}/MemNavData/slurm_lifelong_nnr_controller_verify.sbatch" \
  "${LOCAL_ROOT}/MemNavData/slurm_lifelong_shared_c_arm.sbatch" \
  "${LOCAL_ROOT}/MemNavData/slurm_lifelong_shared_c_finalize.sbatch" \
  "${LOCAL_ROOT}/MemNavData/slurm_lifelong_shared_c_aggregate.sbatch" \
  "${LOCAL_ROOT}/MemNavData/slurm_lifelong_shared_c_verify.sbatch"

source "${LOCAL_ROOT}/MemNavData/slurm_safe_submit.sh"
for template in \
  "${LOCAL_ROOT}/MemNavData/slurm_lifelong_nnr_controller_arm.sbatch" \
  "${LOCAL_ROOT}/MemNavData/slurm_lifelong_nnr_controller_aggregate.sbatch" \
  "${LOCAL_ROOT}/MemNavData/slurm_lifelong_nnr_controller_verify.sbatch" \
  "${LOCAL_ROOT}/MemNavData/slurm_lifelong_shared_c_arm.sbatch" \
  "${LOCAL_ROOT}/MemNavData/slurm_lifelong_shared_c_finalize.sbatch" \
  "${LOCAL_ROOT}/MemNavData/slurm_lifelong_shared_c_aggregate.sbatch" \
  "${LOCAL_ROOT}/MemNavData/slurm_lifelong_shared_c_verify.sbatch"; do
  lint_sbatch_template "${template}" || fail "sbatch lint failed: ${template}"
done

# --- staging ---------------------------------------------------------------
STAGING=$(mktemp -d)
trap 'rm -rf -- "${STAGING}"' EXIT
mkdir -p "${STAGING}/MemNavData" \
  "${STAGING}/NavDP/baselines/memnav" "${STAGING}/NavDP/baselines/navdp" \
  "${STAGING}/NavDP/baselines/${CONTROLLER}"
while IFS= read -r -d '' path; do
  cp --preserve=mode,timestamps "${path}" \
    "${STAGING}/MemNavData/$(basename "${path}")"
done < <(find "${LOCAL_ROOT}/MemNavData" -maxdepth 1 -type f -name '*.py' -print0)
for name in \
  run_cec_controller_portability_smoke_local.sh \
  bundle_selftest.sh \
  slurm_lifelong_nnr_controller_arm.sbatch \
  slurm_lifelong_nnr_controller_aggregate.sbatch \
  slurm_lifelong_nnr_controller_verify.sbatch \
  slurm_lifelong_shared_c_arm.sbatch \
  slurm_lifelong_shared_c_finalize.sbatch \
  slurm_lifelong_shared_c_aggregate.sbatch \
  slurm_lifelong_shared_c_verify.sbatch; do
  cp --preserve=mode,timestamps "${LOCAL_ROOT}/MemNavData/${name}" \
    "${STAGING}/MemNavData/${name}"
done
for component in memnav navdp "${CONTROLLER}"; do
  while IFS= read -r -d '' path; do
    cp --preserve=mode,timestamps "${path}" \
      "${STAGING}/NavDP/baselines/${component}/$(basename "${path}")"
  done < <(find "${LOCAL_ROOT}/NavDP/baselines/${component}" \
    -maxdepth 1 -type f -name '*.py' -print0)
done
if [[ -d "${LOCAL_ROOT}/NavDP/baselines/${CONTROLLER}/configs" ]]; then
  mkdir -p "${STAGING}/NavDP/baselines/${CONTROLLER}/configs"
  while IFS= read -r -d '' path; do
    cp --preserve=mode,timestamps "${path}" \
      "${STAGING}/NavDP/baselines/${CONTROLLER}/configs/$(basename "${path}")"
  done < <(find "${LOCAL_ROOT}/NavDP/baselines/${CONTROLLER}/configs" \
    -maxdepth 1 -type f -print0)
fi
for relative in "${navdp_runtime_support[@]}"; do
  mkdir -p "${STAGING}/$(dirname "${relative}")"
  cp --preserve=mode,timestamps "${LOCAL_ROOT}/${relative}" \
    "${STAGING}/${relative}"
done

# --- staged-bundle selftest under node conditions --------------------------
SELFTEST_ENTRIES=$(mktemp)
cat > "${SELFTEST_ENTRIES}" <<ENTRIES
${MEMNAV_PY} import MemNavData.cec_controller_portability_hub
${MEMNAV_PY} import MemNavData.aggregate_lifelong_nnr_expansion
${MEMNAV_PY} import MemNavData.independent_verify_shared_online_lifelong_nnr
${MEMNAV_PY} import MemNavData.monocular_depth_runtime
${MEMNAV_PY} import MemNavData.controller_portability_contract
ENTRIES
SELFTEST_BUNDLE_SUBPATHS=MemNavData \
  bash "${LOCAL_ROOT}/MemNavData/bundle_selftest.sh" \
  "${STAGING}" "${SELFTEST_ENTRIES}" || fail "staged bundle selftest failed"
rm -f "${SELFTEST_ENTRIES}"
(
  cd "${STAGING}/NavDP/baselines/navdp"
  PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="${STAGING}" \
    "${MEMNAV_PY}" - <<'PY'
from depth_anything.depth_anything_v2.dpt import DepthAnythingV2
import policy_backbone
assert policy_backbone.DepthAnythingV2 is DepthAnythingV2
print("staged NavDP runtime import: ok")
PY
)

# --- bundle manifest + upload (frozen-chain convention, unchanged) ----------
LOCAL_HEAD=$(git -C "${LOCAL_ROOT}" rev-parse HEAD)
"${MEMNAV_PY}" - "${STAGING}" "${LOCAL_HEAD}" "${NNR_ROOT}" \
  "${ORIGINAL_RUN_ROOT}" "${CONTROLLER}" "${SHARED_C_REPAIR}" <<'PY'
import hashlib,json,sys
from pathlib import Path
root=Path(sys.argv[1]); files={}
for path in sorted(root.rglob("*")):
    if path.is_symlink(): raise SystemExit(f"bundle symlink: {path}")
    if path.is_file() and path.name not in {"source_bundle_manifest.json","SOURCE_BUNDLE.sha256"}:
        files[path.relative_to(root).as_posix()]=hashlib.sha256(path.read_bytes()).hexdigest()
payload={
 "schema":"lifelong_nnr_controller_bundle_v1_20260822",
 "local_git_head_context":sys.argv[2],
 "source_nnr_root":sys.argv[3],
 "original_navdp_run_root":sys.argv[4],
 "controller":sys.argv[5],
 "shared_c_repair":bool(int(sys.argv[6])),
 "arms":["all_prior","initial_leg_only","forced_reject_native"],
 "arm_semantics":"identical CEC certificate pipeline; accepted branch executed by CONTROLLER via cec_controller_portability_hub.py",
 "claim_scope":"controller-portability lifelong mechanism baseline; consumed NNR scenes",
 "files":files,
}
(root/"source_bundle_manifest.json").write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n")
PY
(
  cd "${STAGING}"
  find . -type f ! -name SOURCE_BUNDLE.sha256 -print0 | sort -z | \
    xargs -0 sha256sum > SOURCE_BUNDLE.sha256
  sha256sum -c SOURCE_BUNDLE.sha256 >/dev/null
)
SOURCE_RECEIPT_SHA=$(sha256sum "${STAGING}/SOURCE_BUNDLE.sha256" | awk '{print $1}')
BUNDLE_MANIFEST_SHA=$(sha256sum "${STAGING}/source_bundle_manifest.json" | awk '{print $1}')
REMOTE_BUNDLE=${REMOTE_BUNDLE_BASE}/lifelong_ctl_${CONTROLLER}_${BUNDLE_MANIFEST_SHA:0:16}
REMOTE_STAGING=${REMOTE_BUNDLE}.partial-$$

if [[ "${DRY_RUN}" == 1 ]]; then
  echo "DRY_RUN_RUN_ROOT=${RUN_ROOT}"
  echo "DRY_RUN_REMOTE_BUNDLE=${REMOTE_BUNDLE}"
  echo "DRY_RUN_SOURCE_RECEIPT_SHA=${SOURCE_RECEIPT_SHA}"
  exit 0
fi

remote "test -d '${NNR_ROOT}' && test -f '${NNR_ROOT}/prepared/benchmark/SEALED'"
remote "test -f '${ORIGINAL_RUN_ROOT}/factual_b_support/SEALED'"
remote "test \"\$(sha256sum '${BASE_SOURCE_ROOT}/SOURCE_BUNDLE.sha256' | awk '{print \$1}')\" = '${BASE_SOURCE_RECEIPT_SHA}'"
remote "test \"\$(sha256sum '${DEPENDENCY_RECEIPT}' | awk '{print \$1}')\" = '${EXPECTED_DEPENDENCY_RECEIPT_SHA}'"
remote "test -r '${PORTABILITY_ENV_ROOT}/environment_receipt.json'"
remote "cd '${PORTABILITY_CHECKPOINT_ROOT}' && sha256sum -c --quiet CHECKPOINTS.sha256"
if remote "test -d '${REMOTE_BUNDLE}' && test \"\$(sha256sum '${REMOTE_BUNDLE}/SOURCE_BUNDLE.sha256' | awk '{print \$1}')\" = '${SOURCE_RECEIPT_SHA}' && cd '${REMOTE_BUNDLE}' && sha256sum -c SOURCE_BUNDLE.sha256 >/dev/null"; then
  echo "Reusing verified bundle ${REMOTE_BUNDLE}"
else
  remote "test ! -e '${REMOTE_BUNDLE}' && mkdir -p '${REMOTE_STAGING}'"
  rsync -a --chmod=Du=rwx,Dgo=rx,Fu=rw,Fgo=r \
    -e "ssh -o BatchMode=yes -o ControlMaster=no -S ${SSH_CONTROL_PATH}" \
    "${STAGING}/" "${REMOTE_HOST}:${REMOTE_STAGING}/"
  remote "test ! -e '${REMOTE_BUNDLE}' && cd '${REMOTE_STAGING}' && sha256sum -c SOURCE_BUNDLE.sha256 >/dev/null && chmod -R a-w '${REMOTE_STAGING}' && mv '${REMOTE_STAGING}' '${REMOTE_BUNDLE}'"
fi

REMOTE_ENTRIES=${REMOTE_BUNDLE}.selftest-entries-$$
remote "cat > '${REMOTE_ENTRIES}' <<'ENTRIES'
/scratch/lg154/conda-envs/memnav/bin/python import MemNavData.cec_controller_portability_hub
/scratch/lg154/conda-envs/memnav/bin/python import MemNavData.aggregate_lifelong_nnr_expansion
/scratch/lg154/conda-envs/memnav/bin/python import MemNavData.independent_verify_shared_online_lifelong_nnr
/scratch/lg154/conda-envs/memnav/bin/python import MemNavData.lifelong_shared_c_contract
/scratch/lg154/conda-envs/memnav/bin/python import MemNavData.aggregate_lifelong_shared_c_b2
/scratch/lg154/conda-envs/memnav/bin/python import MemNavData.independent_verify_lifelong_shared_c_b2
/scratch/lg154/conda-envs/memnav/bin/python import MemNavData.monocular_depth_runtime
ENTRIES
SELFTEST_BUNDLE_SUBPATHS=MemNavData bash '${REMOTE_BUNDLE}/MemNavData/bundle_selftest.sh' '${REMOTE_BUNDLE}' '${REMOTE_ENTRIES}' && rm -f '${REMOTE_ENTRIES}'" \
  || fail "remote login-node bundle selftest failed"

remote "test ! -e '${RUN_ROOT}' && mkdir -p '${RUN_ROOT}'"

SOURCE_RECEIPT=${REMOTE_BUNDLE}/SOURCE_BUNDLE.sha256
common="ALL,SOURCE_ROOT=${REMOTE_BUNDLE},SOURCE_RECEIPT=${SOURCE_RECEIPT},EXPECTED_SOURCE_RECEIPT_SHA=${SOURCE_RECEIPT_SHA},NNR_ROOT=${NNR_ROOT},ORIGINAL_RUN_ROOT=${ORIGINAL_RUN_ROOT},BASE_SOURCE_ROOT=${BASE_SOURCE_ROOT},BASE_SOURCE_RECEIPT_SHA=${BASE_SOURCE_RECEIPT_SHA},DEPENDENCY_RECEIPT=${DEPENDENCY_RECEIPT},EXPECTED_DEPENDENCY_RECEIPT_SHA=${EXPECTED_DEPENDENCY_RECEIPT_SHA},PORTABILITY_ENV_ROOT=${PORTABILITY_ENV_ROOT},PORTABILITY_CHECKPOINT_ROOT=${PORTABILITY_CHECKPOINT_ROOT},RUN_ROOT=${RUN_ROOT},CONTROLLER=${CONTROLLER}"
ARM=${REMOTE_BUNDLE}/MemNavData/slurm_lifelong_nnr_controller_arm.sbatch
AGGREGATE=${REMOTE_BUNDLE}/MemNavData/slurm_lifelong_nnr_controller_aggregate.sbatch
VERIFY=${REMOTE_BUNDLE}/MemNavData/slurm_lifelong_nnr_controller_verify.sbatch

if [[ "${SHARED_C_REPAIR}" == 1 ]]; then
  SHARED_ARM=${REMOTE_BUNDLE}/MemNavData/slurm_lifelong_shared_c_arm.sbatch
  SHARED_FINALIZE=${REMOTE_BUNDLE}/MemNavData/slurm_lifelong_shared_c_finalize.sbatch
  SHARED_AGGREGATE=${REMOTE_BUNDLE}/MemNavData/slurm_lifelong_shared_c_aggregate.sbatch
  SHARED_VERIFY=${REMOTE_BUNDLE}/MemNavData/slurm_lifelong_shared_c_verify.sbatch
  remote "sbatch --test-only --partition='${SAFE_PARTITIONS}' --nodelist='${TRACE_REPLAY_NODE}' --array=0 --export='${common},STAGE=collect' '${SHARED_ARM}' >/dev/null"
  collect_raw=$(remote "sbatch --parsable --partition='${SAFE_PARTITIONS}' --nodelist='${TRACE_REPLAY_NODE}' --array=0-17%${EVAL_CONCURRENCY} --export='${common},STAGE=collect' '${SHARED_ARM}'")
  collect_id=${collect_raw%%;*}
  [[ "${collect_id}" =~ ^[0-9]+$ ]] || fail "bad shared-C collection array"
  finalize_raw=$(remote "sbatch --parsable --dependency=afterany:${collect_id} --export='${common}' '${SHARED_FINALIZE}'")
  finalize_id=${finalize_raw%%;*}
  [[ "${finalize_id}" =~ ^[0-9]+$ ]] || fail "bad shared-C finalizer"
  eval_raw=$(remote "sbatch --parsable --partition='${SAFE_PARTITIONS}' --nodelist='${TRACE_REPLAY_NODE}' --dependency=afterok:${finalize_id} --kill-on-invalid-dep=yes --array=0-17%${EVAL_CONCURRENCY} --export='${common},STAGE=evaluate' '${SHARED_ARM}'")
  eval_id=${eval_raw%%;*}
  [[ "${eval_id}" =~ ^[0-9]+$ ]] || fail "bad shared-C evaluation array"
  aggregate_raw=$(remote "sbatch --parsable --dependency=afterany:${eval_id} --export='${common}' '${SHARED_AGGREGATE}'")
  aggregate_id=${aggregate_raw%%;*}
  [[ "${aggregate_id}" =~ ^[0-9]+$ ]] || fail "bad shared-C aggregate"
  verify_raw=$(remote "sbatch --parsable --dependency=afterok:${aggregate_id} --kill-on-invalid-dep=yes --export='${common}' '${SHARED_VERIFY}'")
  verify_id=${verify_raw%%;*}
  [[ "${verify_id}" =~ ^[0-9]+$ ]] || fail "bad shared-C verifier"
  echo "RUN_ROOT=${RUN_ROOT}"
  echo "SOURCE_BUNDLE=${REMOTE_BUNDLE}"
  echo "SOURCE_RECEIPT_SHA=${SOURCE_RECEIPT_SHA}"
  echo "controller=${CONTROLLER} shared_C_collect=${collect_id} seal=${finalize_id} B2_pair=${eval_id} aggregate=${aggregate_id} verify=${verify_id}"
  exit 0
fi

arm_ids=()
for scope in all_prior initial_leg_only forced_reject_native; do
  remote "sbatch --test-only --partition='${SAFE_PARTITIONS}' --array=0 --export='${common},LIFELONG_HISTORY_SCOPE=${scope}' '${ARM}' >/dev/null"
  raw=$(remote "sbatch --parsable --partition='${SAFE_PARTITIONS}' --array=0-18%${EVAL_CONCURRENCY} --export='${common},LIFELONG_HISTORY_SCOPE=${scope}' '${ARM}'")
  id=${raw%%;*}
  [[ "${id}" =~ ^[0-9]+$ ]] || fail "bad arm array for scope ${scope}"
  arm_ids+=("${id}")
done
dependency=$(IFS=,; echo "afterany:${arm_ids[*]}")
aggregate_raw=$(remote "sbatch --parsable --dependency='${dependency}' --export='${common}' '${AGGREGATE}'")
aggregate_id=${aggregate_raw%%;*}
[[ "${aggregate_id}" =~ ^[0-9]+$ ]] || fail "bad aggregate job"
verify_raw=$(remote "sbatch --parsable --dependency=afterok:${aggregate_id} --kill-on-invalid-dep=yes --export='${common}' '${VERIFY}'")
verify_id=${verify_raw%%;*}
[[ "${verify_id}" =~ ^[0-9]+$ ]] || fail "bad verification job"

echo "RUN_ROOT=${RUN_ROOT}"
echo "SOURCE_BUNDLE=${REMOTE_BUNDLE}"
echo "SOURCE_RECEIPT_SHA=${SOURCE_RECEIPT_SHA}"
echo "controller=${CONTROLLER} arms(all_prior,initial_leg_only,forced_reject_native)=${arm_ids[*]} aggregate=${aggregate_id} verify=${verify_id}"
