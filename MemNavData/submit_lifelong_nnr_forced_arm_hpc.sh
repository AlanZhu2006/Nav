#!/usr/bin/env bash
# Bundle and submit the forced_reject_native (shared-native baseline) addendum
# for the completed lifelong NNR paired expansion.  Only the new arm runs;
# the verified two-arm results are consumed read-only by the aggregate.
set -euo pipefail
umask 0022

LOCAL_ROOT=$(git rev-parse --show-toplevel)
REMOTE_HOST=${REMOTE_HOST:-alantorch}
REMOTE_BUNDLE_BASE=${REMOTE_BUNDLE_BASE:-/scratch/yz11502/Research/Nav-axis-uturn-source-bundles}
REMOTE_RESULT_BASE=${REMOTE_RESULT_BASE:-/scratch/yz11502/Research/Nav-axis-uturn-results/lifelong_nnr_20260821}
RUN_TAG=${RUN_TAG:-lifelong_nnr_forced_$(date -u +%Y%m%dT%H%M%SZ)}
RUN_ROOT=${RUN_ROOT:-${REMOTE_RESULT_BASE}/${RUN_TAG}}
ORIGINAL_RUN_ROOT=${ORIGINAL_RUN_ROOT:-${REMOTE_RESULT_BASE}/lifelong_nnr_runtime_repair_20260821T083000Z}
NNR_ROOT=${NNR_ROOT:-/scratch/yz11502/Research/Nav-axis-uturn-results/shared_online_nnr_20260814/shared_online_nnr_strict_v2_20260814}
BASE_SOURCE_ROOT=${BASE_SOURCE_ROOT:-/scratch/yz11502/Research/Nav-axis-uturn-source-bundles/certified_relocalization_closed_loop_d3bd281fc374cc80}
BASE_SOURCE_RECEIPT_SHA=${BASE_SOURCE_RECEIPT_SHA:-74001a9e0150c38c599a206fa0f4dd5e1279b9bed5d167119f4d14cb77995e98}
DEPENDENCY_RECEIPT=${DEPENDENCY_RECEIPT:-/scratch/yz11502/Research/Nav-axis-uturn-results/shared_online_double_revisit_fresh_20260813/double_revisit_fresh40_20260813T200121Z/dependency_receipt.json}
EXPECTED_DEPENDENCY_RECEIPT_SHA=${EXPECTED_DEPENDENCY_RECEIPT_SHA:-4eb0ca6479a26f8e04f85a31d906cee4e68b1785f66cfd3ac23bf65424d36e5e}
MEMNAV_PY=${MEMNAV_PY:-/home/asus/miniconda3/envs/memnav/bin/python}
HAB_PY=${HAB_PY:-/home/asus/miniconda3/envs/habitat/bin/python}
EVAL_CONCURRENCY=${EVAL_CONCURRENCY:-2}
DRY_RUN=${DRY_RUN:-0}
SAFE_PARTITIONS=h100_tandon,a100_tandon
SSH_CONTROL_PATH=${SSH_CONTROL_PATH:-$(
  ssh -G "${REMOTE_HOST}" 2>/dev/null |
    awk '$1=="controlpath"{value=$2} END{print value}'
)}

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
  ssh -o BatchMode=yes -o ControlMaster=no -S "${SSH_CONTROL_PATH}" \
    "${REMOTE_HOST}" "$@"
}
fail() { echo "ABORT: $*" >&2; exit 2; }
[[ -S "${SSH_CONTROL_PATH}" ]] || fail "authoritative SSH master missing"
[[ "${RUN_TAG}" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]] || fail "invalid run tag"
[[ "${EVAL_CONCURRENCY}" =~ ^[1-9][0-9]*$ ]] || fail "invalid eval concurrency"
[[ "${DRY_RUN}" =~ ^[01]$ ]] || fail "DRY_RUN must be 0 or 1"

# --- local gate: tests, lint, syntax --------------------------------------
export PYTHONPATH=${LOCAL_ROOT}:${LOCAL_ROOT}/MemNavData${PYTHONPATH:+:${PYTHONPATH}}
"${HAB_PY}" -m py_compile \
  "${LOCAL_ROOT}/MemNavData/eval_shared_online_lifelong_nnr.py"
"${MEMNAV_PY}" -m py_compile \
  "${LOCAL_ROOT}/MemNavData/aggregate_lifelong_nnr_expansion.py" \
  "${LOCAL_ROOT}/MemNavData/independent_verify_shared_online_lifelong_nnr.py" \
  "${LOCAL_ROOT}/MemNavData/cec_controller_portability_hub.py" \
  "${LOCAL_ROOT}/NavDP/baselines/memnav/policy_agent.py"
"${MEMNAV_PY}" -m unittest \
  MemNavData.test_cec_controller_portability_hub \
  MemNavData.test_lifelong_forced_reject_contract \
  MemNavData.test_controller_portability_contract \
  MemNavData.test_policy_agent_graph
bash -n \
  "${LOCAL_ROOT}/MemNavData/run_cec_controller_portability_smoke_local.sh" \
  "${LOCAL_ROOT}/MemNavData/slurm_lifelong_nnr_forced_arm.sbatch" \
  "${LOCAL_ROOT}/MemNavData/slurm_lifelong_nnr_forced_aggregate.sbatch" \
  "${LOCAL_ROOT}/MemNavData/slurm_lifelong_nnr_forced_verify.sbatch"

source "${LOCAL_ROOT}/MemNavData/slurm_safe_submit.sh"
for template in \
  "${LOCAL_ROOT}/MemNavData/slurm_lifelong_nnr_forced_arm.sbatch" \
  "${LOCAL_ROOT}/MemNavData/slurm_lifelong_nnr_forced_aggregate.sbatch" \
  "${LOCAL_ROOT}/MemNavData/slurm_lifelong_nnr_forced_verify.sbatch"; do
  lint_sbatch_template "${template}" || fail "sbatch lint failed: ${template}"
done

# --- staging ---------------------------------------------------------------
STAGING=$(mktemp -d)
trap 'rm -rf -- "${STAGING}"' EXIT
mkdir -p "${STAGING}/MemNavData" \
  "${STAGING}/NavDP/baselines/memnav" "${STAGING}/NavDP/baselines/navdp"
while IFS= read -r -d '' path; do
  cp --preserve=mode,timestamps "${path}" \
    "${STAGING}/MemNavData/$(basename "${path}")"
done < <(find "${LOCAL_ROOT}/MemNavData" -maxdepth 1 -type f -name '*.py' -print0)
for name in \
  LIFELONG_5LEG_PROTOCOL_20260821.md \
  LIFELONG_NNR_RUNTIME_REPAIR_20260821.md \
  run_cec_controller_portability_smoke_local.sh \
  bundle_selftest.sh \
  slurm_lifelong_nnr_forced_arm.sbatch \
  slurm_lifelong_nnr_forced_aggregate.sbatch \
  slurm_lifelong_nnr_forced_verify.sbatch; do
  cp --preserve=mode,timestamps "${LOCAL_ROOT}/MemNavData/${name}" \
    "${STAGING}/MemNavData/${name}"
done
for component in memnav navdp; do
  while IFS= read -r -d '' path; do
    cp --preserve=mode,timestamps "${path}" \
      "${STAGING}/NavDP/baselines/${component}/$(basename "${path}")"
  done < <(find "${LOCAL_ROOT}/NavDP/baselines/${component}" \
    -maxdepth 1 -type f -name '*.py' -print0)
done
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
  "${ORIGINAL_RUN_ROOT}" <<'PY'
import hashlib,json,sys
from pathlib import Path
root=Path(sys.argv[1]); files={}
for path in sorted(root.rglob("*")):
    if path.is_symlink(): raise SystemExit(f"bundle symlink: {path}")
    if path.is_file() and path.name not in {"source_bundle_manifest.json","SOURCE_BUNDLE.sha256"}:
        files[path.relative_to(root).as_posix()]=hashlib.sha256(path.read_bytes()).hexdigest()
payload={
 "schema":"lifelong_nnr_forced_arm_bundle_v1_20260822",
 "local_git_head_context":sys.argv[2],
 "source_nnr_root":sys.argv[3],
 "original_two_arm_run_root":sys.argv[4],
 "addendum_arm":"forced_reject_native",
 "arm_semantics":"identical memory recording and receipts; CEC never holds takeover authority; every action is the shared mono-NavDP fallback",
 "existing_arms_rerun":False,
 "claim_scope":"internal lifelong mechanism baseline; consumed NNR scenes",
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
REMOTE_BUNDLE=${REMOTE_BUNDLE_BASE}/lifelong_forced_${BUNDLE_MANIFEST_SHA:0:16}
REMOTE_STAGING=${REMOTE_BUNDLE}.partial-$$

if [[ "${DRY_RUN}" == 1 ]]; then
  echo "DRY_RUN_RUN_ROOT=${RUN_ROOT}"
  echo "DRY_RUN_REMOTE_BUNDLE=${REMOTE_BUNDLE}"
  echo "DRY_RUN_SOURCE_RECEIPT_SHA=${SOURCE_RECEIPT_SHA}"
  exit 0
fi

remote "test -d '${NNR_ROOT}' && test -f '${NNR_ROOT}/prepared/benchmark/SEALED'"
remote "test -f '${ORIGINAL_RUN_ROOT}/VERIFIED'"
remote "test -f '${ORIGINAL_RUN_ROOT}/factual_b_support/SEALED'"
remote "test \"\$(sha256sum '${BASE_SOURCE_ROOT}/SOURCE_BUNDLE.sha256' | awk '{print \$1}')\" = '${BASE_SOURCE_RECEIPT_SHA}'"
remote "test \"\$(sha256sum '${DEPENDENCY_RECEIPT}' | awk '{print \$1}')\" = '${EXPECTED_DEPENDENCY_RECEIPT_SHA}'"
if remote "test -d '${REMOTE_BUNDLE}' && test \"\$(sha256sum '${REMOTE_BUNDLE}/SOURCE_BUNDLE.sha256' | awk '{print \$1}')\" = '${SOURCE_RECEIPT_SHA}' && cd '${REMOTE_BUNDLE}' && sha256sum -c SOURCE_BUNDLE.sha256 >/dev/null"; then
  echo "Reusing verified bundle ${REMOTE_BUNDLE}"
else
  remote "test ! -e '${REMOTE_BUNDLE}' && mkdir -p '${REMOTE_STAGING}'"
  rsync -a --chmod=Du=rwx,Dgo=rx,Fu=rw,Fgo=r \
    -e "ssh -o BatchMode=yes -o ControlMaster=no -S ${SSH_CONTROL_PATH}" \
    "${STAGING}/" "${REMOTE_HOST}:${REMOTE_STAGING}/"
  remote "test ! -e '${REMOTE_BUNDLE}' && cd '${REMOTE_STAGING}' && sha256sum -c SOURCE_BUNDLE.sha256 >/dev/null && chmod -R a-w '${REMOTE_STAGING}' && mv '${REMOTE_STAGING}' '${REMOTE_BUNDLE}'"
fi

# remote login-node selftest with the cluster interpreters
REMOTE_ENTRIES=${REMOTE_BUNDLE}.selftest-entries-$$
remote "cat > '${REMOTE_ENTRIES}' <<'ENTRIES'
/scratch/lg154/conda-envs/memnav/bin/python import MemNavData.cec_controller_portability_hub
/scratch/lg154/conda-envs/memnav/bin/python import MemNavData.aggregate_lifelong_nnr_expansion
/scratch/lg154/conda-envs/memnav/bin/python import MemNavData.independent_verify_shared_online_lifelong_nnr
/scratch/lg154/conda-envs/memnav/bin/python import MemNavData.monocular_depth_runtime
ENTRIES
SELFTEST_BUNDLE_SUBPATHS=MemNavData bash '${REMOTE_BUNDLE}/MemNavData/bundle_selftest.sh' '${REMOTE_BUNDLE}' '${REMOTE_ENTRIES}' && rm -f '${REMOTE_ENTRIES}'" \
  || fail "remote login-node bundle selftest failed"

remote "test ! -e '${RUN_ROOT}' && mkdir -p '${RUN_ROOT}'"

SOURCE_RECEIPT=${REMOTE_BUNDLE}/SOURCE_BUNDLE.sha256
exports="ALL,SOURCE_ROOT=${REMOTE_BUNDLE},SOURCE_RECEIPT=${SOURCE_RECEIPT},EXPECTED_SOURCE_RECEIPT_SHA=${SOURCE_RECEIPT_SHA},NNR_ROOT=${NNR_ROOT},RUN_ROOT=${RUN_ROOT},ORIGINAL_RUN_ROOT=${ORIGINAL_RUN_ROOT},BASE_SOURCE_ROOT=${BASE_SOURCE_ROOT},BASE_SOURCE_RECEIPT_SHA=${BASE_SOURCE_RECEIPT_SHA},DEPENDENCY_RECEIPT=${DEPENDENCY_RECEIPT},EXPECTED_DEPENDENCY_RECEIPT_SHA=${EXPECTED_DEPENDENCY_RECEIPT_SHA}"
ARM=${REMOTE_BUNDLE}/MemNavData/slurm_lifelong_nnr_forced_arm.sbatch
AGGREGATE=${REMOTE_BUNDLE}/MemNavData/slurm_lifelong_nnr_forced_aggregate.sbatch
VERIFY=${REMOTE_BUNDLE}/MemNavData/slurm_lifelong_nnr_forced_verify.sbatch

remote "sbatch --test-only --partition='${SAFE_PARTITIONS}' --array=0 --export='${exports}' '${ARM}' >/dev/null"
arm_raw=$(remote "sbatch --parsable --partition='${SAFE_PARTITIONS}' --array=0-18%${EVAL_CONCURRENCY} --export='${exports}' '${ARM}'")
arm_id=${arm_raw%%;*}
[[ "${arm_id}" =~ ^[0-9]+$ ]] || fail "bad forced-arm array"
# afterany + completeness inside the aggregate: an afterok dependency can
# never release from an array containing a cancelled element.
aggregate_raw=$(remote "sbatch --parsable --dependency=afterany:${arm_id} --export='${exports}' '${AGGREGATE}'")
aggregate_id=${aggregate_raw%%;*}
[[ "${aggregate_id}" =~ ^[0-9]+$ ]] || fail "bad aggregate job"
verify_raw=$(remote "sbatch --parsable --dependency=afterok:${aggregate_id} --kill-on-invalid-dep=yes --export='${exports}' '${VERIFY}'")
verify_id=${verify_raw%%;*}
[[ "${verify_id}" =~ ^[0-9]+$ ]] || fail "bad verification job"

echo "RUN_ROOT=${RUN_ROOT}"
echo "SOURCE_BUNDLE=${REMOTE_BUNDLE}"
echo "SOURCE_RECEIPT_SHA=${SOURCE_RECEIPT_SHA}"
echo "arm=${arm_id} aggregate=${aggregate_id} verify=${verify_id}"
