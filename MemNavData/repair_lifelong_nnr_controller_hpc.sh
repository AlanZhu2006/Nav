#!/usr/bin/env bash
# Targeted repair for a controller-portability lifelong SR run: resubmit only
# the specific (scope, index) cells that failed or were cancelled, into the
# SAME existing RUN_ROOT, then re-trigger aggregate+verify.  Never touches
# already-completed cells (the arm sbatch itself refuses to overwrite an
# existing ARM_ROOT).
set -euo pipefail
umask 0022

LOCAL_ROOT=$(git rev-parse --show-toplevel)
CONTROLLER=${CONTROLLER:?set CONTROLLER=vint|iplanner|gnm|nomad|viplanner}
RUN_ROOT=${RUN_ROOT:?set the existing controller run root to repair into}
REPAIR_all_prior=${REPAIR_all_prior:-}
REPAIR_initial_leg_only=${REPAIR_initial_leg_only:-}
REPAIR_forced_reject_native=${REPAIR_forced_reject_native:-}
REMOTE_HOST=${REMOTE_HOST:-alantorch}
REMOTE_BUNDLE_BASE=${REMOTE_BUNDLE_BASE:-/scratch/yz11502/Research/Nav-axis-uturn-source-bundles}
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
SAFE_PARTITIONS=h100_tandon,a100_tandon
SSH_CONTROL_PATH=${SSH_CONTROL_PATH:-$(
  ssh -G "${REMOTE_HOST}" 2>/dev/null |
    awk '$1=="controlpath"{value=$2} END{print value}'
)}

case "${CONTROLLER}" in
  vint|gnm|nomad|iplanner|viplanner) ;;
  *) echo "ABORT: unsupported CONTROLLER ${CONTROLLER}" >&2; exit 2 ;;
esac
if [[ -z "${REPAIR_all_prior}${REPAIR_initial_leg_only}${REPAIR_forced_reject_native}" ]]; then
  echo "ABORT: no repair indices requested for any scope" >&2; exit 2
fi

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
remote "test -d '${RUN_ROOT}'" || fail "RUN_ROOT does not exist: ${RUN_ROOT}"

export PYTHONPATH=${LOCAL_ROOT}:${LOCAL_ROOT}/MemNavData${PYTHONPATH:+:${PYTHONPATH}}
bash -n "${LOCAL_ROOT}/MemNavData/slurm_lifelong_nnr_controller_arm.sbatch" \
  "${LOCAL_ROOT}/MemNavData/slurm_lifelong_nnr_controller_aggregate.sbatch" \
  "${LOCAL_ROOT}/MemNavData/slurm_lifelong_nnr_controller_verify.sbatch"
source "${LOCAL_ROOT}/MemNavData/slurm_safe_submit.sh"
lint_sbatch_template "${LOCAL_ROOT}/MemNavData/slurm_lifelong_nnr_controller_arm.sbatch" || \
  fail "sbatch lint failed"

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
  slurm_lifelong_nnr_controller_verify.sbatch; do
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

LOCAL_HEAD=$(git -C "${LOCAL_ROOT}" rev-parse HEAD)
"${MEMNAV_PY}" - "${STAGING}" "${LOCAL_HEAD}" "${CONTROLLER}" <<'PY'
import hashlib,json,sys
from pathlib import Path
root=Path(sys.argv[1]); files={}
for path in sorted(root.rglob("*")):
    if path.is_symlink(): raise SystemExit(f"bundle symlink: {path}")
    if path.is_file() and path.name not in {"source_bundle_manifest.json","SOURCE_BUNDLE.sha256"}:
        files[path.relative_to(root).as_posix()]=hashlib.sha256(path.read_bytes()).hexdigest()
payload={
 "schema":"lifelong_nnr_controller_repair_bundle_v1_20260823",
 "local_git_head_context":sys.argv[2],
 "controller":sys.argv[3],
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
REMOTE_BUNDLE=${REMOTE_BUNDLE_BASE}/lifelong_ctl_repair_${CONTROLLER}_${BUNDLE_MANIFEST_SHA:0:16}
REMOTE_STAGING=${REMOTE_BUNDLE}.partial-$$
remote "test ! -e '${REMOTE_BUNDLE}' && mkdir -p '${REMOTE_STAGING}'"
rsync -a --chmod=Du=rwx,Dgo=rx,Fu=rw,Fgo=r \
  -e "ssh -o BatchMode=yes -o ControlMaster=no -S ${SSH_CONTROL_PATH}" \
  "${STAGING}/" "${REMOTE_HOST}:${REMOTE_STAGING}/"
remote "test ! -e '${REMOTE_BUNDLE}' && cd '${REMOTE_STAGING}' && sha256sum -c SOURCE_BUNDLE.sha256 >/dev/null && chmod -R a-w '${REMOTE_STAGING}' && mv '${REMOTE_STAGING}' '${REMOTE_BUNDLE}'"

common="ALL,SOURCE_ROOT=${REMOTE_BUNDLE},SOURCE_RECEIPT=${REMOTE_BUNDLE}/SOURCE_BUNDLE.sha256,EXPECTED_SOURCE_RECEIPT_SHA=${SOURCE_RECEIPT_SHA},NNR_ROOT=${NNR_ROOT},ORIGINAL_RUN_ROOT=${ORIGINAL_RUN_ROOT},BASE_SOURCE_ROOT=${BASE_SOURCE_ROOT},BASE_SOURCE_RECEIPT_SHA=${BASE_SOURCE_RECEIPT_SHA},DEPENDENCY_RECEIPT=${DEPENDENCY_RECEIPT},EXPECTED_DEPENDENCY_RECEIPT_SHA=${EXPECTED_DEPENDENCY_RECEIPT_SHA},PORTABILITY_ENV_ROOT=${PORTABILITY_ENV_ROOT},PORTABILITY_CHECKPOINT_ROOT=${PORTABILITY_CHECKPOINT_ROOT},RUN_ROOT=${RUN_ROOT},CONTROLLER=${CONTROLLER}"
ARM=${REMOTE_BUNDLE}/MemNavData/slurm_lifelong_nnr_controller_arm.sbatch
AGGREGATE=${REMOTE_BUNDLE}/MemNavData/slurm_lifelong_nnr_controller_aggregate.sbatch
VERIFY=${REMOTE_BUNDLE}/MemNavData/slurm_lifelong_nnr_controller_verify.sbatch

repair_ids=()
for scope in all_prior initial_leg_only forced_reject_native; do
  var="REPAIR_${scope}"
  indices=${!var}
  [[ -n "${indices}" ]] || continue
  raw=$(remote "sbatch --parsable --partition='${SAFE_PARTITIONS}' --array=${indices}%${EVAL_CONCURRENCY} --export='${common},LIFELONG_HISTORY_SCOPE=${scope}' '${ARM}'")
  id=${raw%%;*}
  [[ "${id}" =~ ^[0-9]+$ ]] || fail "bad repair array for scope ${scope}"
  echo "repair arm scope=${scope} indices=${indices} job=${id}"
  repair_ids+=("${id}")
done
dependency=$(IFS=,; echo "afterany:${repair_ids[*]}")
aggregate_raw=$(remote "sbatch --parsable --dependency='${dependency}' --export='${common}' '${AGGREGATE}'")
aggregate_id=${aggregate_raw%%;*}
verify_raw=$(remote "sbatch --parsable --dependency=afterok:${aggregate_id} --kill-on-invalid-dep=yes --export='${common}' '${VERIFY}'")
verify_id=${verify_raw%%;*}
echo "RUN_ROOT=${RUN_ROOT}"
echo "controller=${CONTROLLER} repair_arms=${repair_ids[*]} aggregate=${aggregate_id} verify=${verify_id}"
