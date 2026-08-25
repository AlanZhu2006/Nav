#!/usr/bin/env bash
# Submit only the 25 pre-registered missing controller-portability arms, then
# merge them with the 245 untouched originals and verify the full 270-arm
# matrix.  Repair tasks are A100-only, single-concurrency, and one hour each.
set -euo pipefail
umask 0022

LOCAL_ROOT=$(git rev-parse --show-toplevel)
REMOTE_HOST=${REMOTE_HOST:-alantorch}
REMOTE_BUNDLE_BASE=${REMOTE_BUNDLE_BASE:-/scratch/yz11502/Research/Nav-axis-uturn-source-bundles}
REMOTE_RESULT_BASE=${REMOTE_RESULT_BASE:-/scratch/yz11502/Research/Nav-axis-uturn-results/lifelong_nnr_controller_repair_20260824}
RUN_TAG=${RUN_TAG:-lifelong_nnr_controller_repair_$(date -u +%Y%m%dT%H%M%SZ)}
REPAIR_ROOT=${REPAIR_ROOT:-${REMOTE_RESULT_BASE}/${RUN_TAG}}
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
EVAL_CONCURRENCY=${EVAL_CONCURRENCY:-1}
DRY_RUN=${DRY_RUN:-0}
SAFE_GPU_PARTITION=a100_tandon
MANIFEST_REL=MemNavData/lifelong_nnr_controller_repair_manifest_20260824.json
SSH_CONTROL_PATH=${SSH_CONTROL_PATH:-$(
  ssh -G "${REMOTE_HOST}" 2>/dev/null |
    awk '$1=="controlpath"{value=$2} END{print value}'
)}

remote() {
  ssh -o BatchMode=yes -o ControlMaster=no -S "${SSH_CONTROL_PATH}" \
    "${REMOTE_HOST}" "$@"
}
fail() { echo "ABORT: $*" >&2; exit 2; }
[[ -S "${SSH_CONTROL_PATH}" ]] || fail "authoritative SSH master missing"
[[ "${RUN_TAG}" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]] || fail "invalid run tag"
[[ "${EVAL_CONCURRENCY}" == 1 ]] || fail "formal repair is frozen at concurrency one"
[[ "${DRY_RUN}" =~ ^[01]$ ]] || fail "DRY_RUN must be 0 or 1"

MANIFEST=${LOCAL_ROOT}/${MANIFEST_REL}
REPAIR_COUNT=$("${MEMNAV_PY}" - "${MANIFEST}" <<'PY'
import json,sys
payload=json.load(open(sys.argv[1]))
assert payload["schema"] == "lifelong_nnr_controller_repair_manifest_v1_20260824"
assert len(payload["entries"]) == payload["expected_missing"] == 25
print(len(payload["entries"]))
PY
)

# Local contract gate.
export PYTHONPATH=${LOCAL_ROOT}:${LOCAL_ROOT}/MemNavData${PYTHONPATH:+:${PYTHONPATH}}
"${HAB_PY}" -m py_compile \
  "${LOCAL_ROOT}/MemNavData/eval_shared_online_lifelong_nnr.py" \
  "${LOCAL_ROOT}/MemNavData/eval_2leg_habitat.py"
"${MEMNAV_PY}" -m py_compile \
  "${LOCAL_ROOT}/MemNavData/audit_lifelong_nnr_controller_repair.py" \
  "${LOCAL_ROOT}/MemNavData/aggregate_lifelong_nnr_expansion.py" \
  "${LOCAL_ROOT}/MemNavData/independent_verify_shared_online_lifelong_nnr.py" \
  "${LOCAL_ROOT}/MemNavData/cec_controller_portability_hub.py" \
  "${LOCAL_ROOT}/MemNavData/controller_portability_proxy.py" \
  "${LOCAL_ROOT}/NavDP/baselines/memnav/policy_agent.py"
"${MEMNAV_PY}" -m pytest -q \
  "${LOCAL_ROOT}/MemNavData/test_cec_controller_portability_hub.py" \
  "${LOCAL_ROOT}/MemNavData/test_controller_portability_contract.py" \
  "${LOCAL_ROOT}/MemNavData/test_controller_portability_proxy.py" \
  "${LOCAL_ROOT}/MemNavData/test_lifelong_forced_reject_contract.py" \
  "${LOCAL_ROOT}/MemNavData/test_policy_agent_graph.py"
bash -n \
  "${LOCAL_ROOT}/MemNavData/run_cec_controller_portability_smoke_local.sh" \
  "${LOCAL_ROOT}/MemNavData/slurm_lifelong_nnr_controller_arm.sbatch" \
  "${LOCAL_ROOT}/MemNavData/slurm_lifelong_nnr_controller_repair_arm.sbatch" \
  "${LOCAL_ROOT}/MemNavData/slurm_lifelong_nnr_controller_repair_finalize.sbatch"
source "${LOCAL_ROOT}/MemNavData/slurm_safe_submit.sh"
lint_sbatch_template \
  "${LOCAL_ROOT}/MemNavData/slurm_lifelong_nnr_controller_repair_arm.sbatch" \
  || fail "repair-arm sbatch lint failed"
lint_sbatch_template \
  "${LOCAL_ROOT}/MemNavData/slurm_lifelong_nnr_controller_repair_finalize.sbatch" \
  || fail "repair-finalizer sbatch lint failed"

# Build one source bundle containing all five downstream controller adapters.
STAGING=$(mktemp -d)
trap 'rm -rf -- "${STAGING}"' EXIT
mkdir -p "${STAGING}/MemNavData" \
  "${STAGING}/NavDP/baselines/memnav" \
  "${STAGING}/NavDP/baselines/navdp"
while IFS= read -r -d '' path; do
  cp --preserve=mode,timestamps "${path}" \
    "${STAGING}/MemNavData/$(basename "${path}")"
done < <(find "${LOCAL_ROOT}/MemNavData" -maxdepth 1 -type f -name '*.py' -print0)
for name in \
  bundle_selftest.sh \
  run_cec_controller_portability_smoke_local.sh \
  slurm_lifelong_nnr_controller_arm.sbatch \
  slurm_lifelong_nnr_controller_repair_arm.sbatch \
  slurm_lifelong_nnr_controller_repair_finalize.sbatch \
  lifelong_nnr_controller_repair_manifest_20260824.json; do
  cp --preserve=mode,timestamps "${LOCAL_ROOT}/MemNavData/${name}" \
    "${STAGING}/MemNavData/${name}"
done
for component in memnav navdp vint gnm nomad iplanner viplanner; do
  mkdir -p "${STAGING}/NavDP/baselines/${component}"
  while IFS= read -r -d '' path; do
    cp --preserve=mode,timestamps "${path}" \
      "${STAGING}/NavDP/baselines/${component}/$(basename "${path}")"
  done < <(find "${LOCAL_ROOT}/NavDP/baselines/${component}" \
    -maxdepth 1 -type f -name '*.py' -print0)
  if [[ -d "${LOCAL_ROOT}/NavDP/baselines/${component}/configs" ]]; then
    mkdir -p "${STAGING}/NavDP/baselines/${component}/configs"
    while IFS= read -r -d '' path; do
      cp --preserve=mode,timestamps "${path}" \
        "${STAGING}/NavDP/baselines/${component}/configs/$(basename "${path}")"
    done < <(find "${LOCAL_ROOT}/NavDP/baselines/${component}/configs" \
      -maxdepth 1 -type f -print0)
  fi
done
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
for relative in "${navdp_runtime_support[@]}"; do
  mkdir -p "${STAGING}/$(dirname "${relative}")"
  cp --preserve=mode,timestamps "${LOCAL_ROOT}/${relative}" \
    "${STAGING}/${relative}"
done

SELFTEST_ENTRIES=$(mktemp)
cat > "${SELFTEST_ENTRIES}" <<ENTRIES
${MEMNAV_PY} import MemNavData.cec_controller_portability_hub
${MEMNAV_PY} import MemNavData.audit_lifelong_nnr_controller_repair
${MEMNAV_PY} import MemNavData.aggregate_lifelong_nnr_expansion
${MEMNAV_PY} import MemNavData.independent_verify_shared_online_lifelong_nnr
ENTRIES
SELFTEST_BUNDLE_SUBPATHS=MemNavData \
  bash "${LOCAL_ROOT}/MemNavData/bundle_selftest.sh" \
  "${STAGING}" "${SELFTEST_ENTRIES}" || fail "staged selftest failed"
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

LOCAL_HEAD=$(git -C "${LOCAL_ROOT}" rev-parse HEAD)
"${MEMNAV_PY}" - "${STAGING}" "${LOCAL_HEAD}" "${NNR_ROOT}" <<'PY'
import hashlib,json,sys
from pathlib import Path
root=Path(sys.argv[1]); files={}
for path in sorted(root.rglob("*")):
    if path.is_symlink(): raise SystemExit(f"bundle symlink: {path}")
    if path.is_file() and path.name not in {"source_bundle_manifest.json","SOURCE_BUNDLE.sha256"}:
        files[path.relative_to(root).as_posix()]=hashlib.sha256(path.read_bytes()).hexdigest()
payload={
 "schema":"lifelong_nnr_controller_repair_bundle_v1_20260824",
 "local_git_head_context":sys.argv[2],
 "source_nnr_root":sys.argv[3],
 "repair_selection_reads_navigation_outcomes":False,
 "gpu_protocol":"A100-only; one-hour; concurrency one",
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
REMOTE_BUNDLE=${REMOTE_BUNDLE_BASE}/lifelong_ctl_repair_${BUNDLE_MANIFEST_SHA:0:16}
REMOTE_STAGING=${REMOTE_BUNDLE}.partial-$$

if [[ "${DRY_RUN}" == 1 ]]; then
  echo "DRY_RUN_REPAIR_ROOT=${REPAIR_ROOT}"
  echo "DRY_RUN_REMOTE_BUNDLE=${REMOTE_BUNDLE}"
  echo "DRY_RUN_SOURCE_RECEIPT_SHA=${SOURCE_RECEIPT_SHA}"
  echo "DRY_RUN_REPAIR_COUNT=${REPAIR_COUNT}"
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

remote "test ! -e '${REPAIR_ROOT}' && mkdir -p '${REPAIR_ROOT}'"
SOURCE_RECEIPT=${REMOTE_BUNDLE}/SOURCE_BUNDLE.sha256
POPULATION=${ORIGINAL_RUN_ROOT}/factual_b_support/population.json
remote "PYTHONDONTWRITEBYTECODE=1 PYTHONPATH='${REMOTE_BUNDLE}:${REMOTE_BUNDLE}/MemNavData' /scratch/lg154/conda-envs/memnav/bin/python -u '${REMOTE_BUNDLE}/MemNavData/audit_lifelong_nnr_controller_repair.py' --manifest '${REMOTE_BUNDLE}/${MANIFEST_REL}' --population '${POPULATION}' --out '${REPAIR_ROOT}/pre_repair_audit.json' && chmod a-w '${REPAIR_ROOT}/pre_repair_audit.json'"

common="ALL,SOURCE_ROOT=${REMOTE_BUNDLE},SOURCE_RECEIPT=${SOURCE_RECEIPT},EXPECTED_SOURCE_RECEIPT_SHA=${SOURCE_RECEIPT_SHA},NNR_ROOT=${NNR_ROOT},ORIGINAL_RUN_ROOT=${ORIGINAL_RUN_ROOT},BASE_SOURCE_ROOT=${BASE_SOURCE_ROOT},BASE_SOURCE_RECEIPT_SHA=${BASE_SOURCE_RECEIPT_SHA},DEPENDENCY_RECEIPT=${DEPENDENCY_RECEIPT},EXPECTED_DEPENDENCY_RECEIPT_SHA=${EXPECTED_DEPENDENCY_RECEIPT_SHA},PORTABILITY_ENV_ROOT=${PORTABILITY_ENV_ROOT},PORTABILITY_CHECKPOINT_ROOT=${PORTABILITY_CHECKPOINT_ROOT},REPAIR_ROOT=${REPAIR_ROOT}"
ARM=${REMOTE_BUNDLE}/MemNavData/slurm_lifelong_nnr_controller_repair_arm.sbatch
FINALIZE=${REMOTE_BUNDLE}/MemNavData/slurm_lifelong_nnr_controller_repair_finalize.sbatch
remote "sbatch --test-only --partition='${SAFE_GPU_PARTITION}' --time=01:00:00 --array=0 --export='${common}' '${ARM}' >/dev/null"
raw=$(remote "sbatch --parsable --partition='${SAFE_GPU_PARTITION}' --time=01:00:00 --array=0-$((REPAIR_COUNT - 1))%1 --export='${common}' '${ARM}'")
repair_job=${raw%%;*}
[[ "${repair_job}" =~ ^[0-9]+$ ]] || fail "bad repair array job id"
final_raw=$(remote "sbatch --parsable --partition=cpu_short --dependency=afterany:${repair_job} --export='${common}' '${FINALIZE}'")
final_job=${final_raw%%;*}
[[ "${final_job}" =~ ^[0-9]+$ ]] || fail "bad finalizer job id"

echo "REPAIR_ROOT=${REPAIR_ROOT}"
echo "SOURCE_BUNDLE=${REMOTE_BUNDLE}"
echo "SOURCE_RECEIPT_SHA=${SOURCE_RECEIPT_SHA}"
echo "repair_array=${repair_job} tasks=${REPAIR_COUNT} concurrency=1 partition=${SAFE_GPU_PARTITION} time=01:00:00 finalizer=${final_job}"
