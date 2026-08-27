#!/usr/bin/env bash
# Freeze, upload and submit the protocol-locked HM3D ViNT-native/ViNT+CEC
# pilot followed by the full 28-history population only after pilot verifier
# success.  Pilot navigation outcomes never select or modify the formal arm.
set -euo pipefail
umask 0022

LOCAL_ROOT=${LOCAL_ROOT:-$(git rev-parse --show-toplevel)}
REMOTE_HOST=${REMOTE_HOST:-alantorch}
REMOTE_BUNDLE_BASE=${REMOTE_BUNDLE_BASE:-/scratch/yz11502/Research/Nav-axis-uturn-source-bundles}
REMOTE_RESULT_BASE=${REMOTE_RESULT_BASE:-/scratch/yz11502/Research/Nav-axis-uturn-results/hm3d_vint_controller_native_cec_20260828}
RUN_TAG=${RUN_TAG:-formal_$(date -u +%Y%m%dT%H%M%SZ)}
RUN_ROOT=${RUN_ROOT:-${REMOTE_RESULT_BASE}/${RUN_TAG}}
FRESH_ROOT=${FRESH_ROOT:-/scratch/yz11502/Research/Nav-axis-uturn-results/hm3d_fresh_fullmono_mixed_role_20260820/formal_20260820T143609Z_e6dd44c6}
BASE_SOURCE_ROOT=${BASE_SOURCE_ROOT:-/scratch/yz11502/Research/Nav-axis-uturn-source-bundles/certified_relocalization_closed_loop_d3bd281fc374cc80}
BASE_SOURCE_RECEIPT_SHA=${BASE_SOURCE_RECEIPT_SHA:-74001a9e0150c38c599a206fa0f4dd5e1279b9bed5d167119f4d14cb77995e98}
DEPENDENCY_RECEIPT=${DEPENDENCY_RECEIPT:-/scratch/yz11502/Research/Nav-axis-uturn-results/shared_online_double_revisit_fresh_20260813/double_revisit_fresh40_20260813T200121Z/dependency_receipt.json}
EXPECTED_DEPENDENCY_RECEIPT_SHA=${EXPECTED_DEPENDENCY_RECEIPT_SHA:-4eb0ca6479a26f8e04f85a31d906cee4e68b1785f66cfd3ac23bf65424d36e5e}
PORTABILITY_ENV_ROOT=${PORTABILITY_ENV_ROOT:-/scratch/yz11502/Research/Nav-axis-uturn-envs/controller_portability_a9ec7146bce7_v1}
PORTABILITY_CHECKPOINT_ROOT=${PORTABILITY_CHECKPOINT_ROOT:-/scratch/yz11502/Research/Nav-axis-uturn-checkpoints/controller_portability_50387aa89be8}
PILOT_CONCURRENCY=${PILOT_CONCURRENCY:-2}
FORMAL_CONCURRENCY=${FORMAL_CONCURRENCY:-2}
DRY_RUN=${DRY_RUN:-0}
EXPORT_ARCHIVE=${EXPORT_ARCHIVE:-}
MEMNAV_PY=${MEMNAV_PY:-/home/asus/miniconda3/envs/memnav/bin/python}
HAB_PY=${HAB_PY:-/home/asus/miniconda3/envs/habitat/bin/python}
SSH_CONTROL_PATH=${SSH_CONTROL_PATH:-$(
  ssh -G "${REMOTE_HOST}" 2>/dev/null |
    awk '$1=="controlpath"{value=$2} END{print value}'
)}

fail() { echo "ABORT: $*" >&2; exit 2; }
remote() {
  timeout 180 ssh -n -T -o BatchMode=yes -o ControlMaster=no \
    -S "${SSH_CONTROL_PATH}" "${REMOTE_HOST}" "$@"
}
[[ -S "${SSH_CONTROL_PATH}" ]] || fail "authoritative SSH master missing"
timeout 15 ssh -O check -S "${SSH_CONTROL_PATH}" "${REMOTE_HOST}" \
  >/dev/null 2>&1 || fail "authoritative SSH master is not responsive"
[[ "${RUN_TAG}" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]] || fail "invalid run tag"
[[ "${PILOT_CONCURRENCY}" =~ ^[1-9][0-9]*$ ]] || fail "bad pilot concurrency"
[[ "${FORMAL_CONCURRENCY}" =~ ^[1-9][0-9]*$ ]] || fail "bad formal concurrency"
[[ "${DRY_RUN}" =~ ^[01]$ ]] || fail "DRY_RUN must be 0 or 1"

export PYTHONPATH=${LOCAL_ROOT}:${LOCAL_ROOT}/MemNavData${PYTHONPATH:+:${PYTHONPATH}}
"${MEMNAV_PY}" -m py_compile \
  "${LOCAL_ROOT}/MemNavData/cec_handoff_contract.py" \
  "${LOCAL_ROOT}/MemNavData/cec_controller_portability_hub.py" \
  "${LOCAL_ROOT}/MemNavData/controller_portability_contract.py" \
  "${LOCAL_ROOT}/MemNavData/controller_portability_proxy.py" \
  "${LOCAL_ROOT}/MemNavData/audit_vint_controller_native_pair.py" \
  "${LOCAL_ROOT}/MemNavData/aggregate_vint_controller_native_hm3d.py" \
  "${LOCAL_ROOT}/MemNavData/independent_verify_vint_controller_native_hm3d.py"
"${HAB_PY}" -m py_compile \
  "${LOCAL_ROOT}/MemNavData/eval_2leg_habitat.py" \
  "${LOCAL_ROOT}/MemNavData/eval_shared_online_role_pairs.py"
"${MEMNAV_PY}" -m pytest -q \
  "${LOCAL_ROOT}/MemNavData/test_cec_handoff_contract.py" \
  "${LOCAL_ROOT}/MemNavData/test_cec_controller_portability_hub.py" \
  "${LOCAL_ROOT}/MemNavData/test_controller_portability_contract.py" \
  "${LOCAL_ROOT}/MemNavData/test_controller_portability_proxy.py" \
  "${LOCAL_ROOT}/MemNavData/test_audit_vint_controller_native_pair.py" \
  "${LOCAL_ROOT}/MemNavData/test_aggregate_vint_controller_native_hm3d.py"
bash -n \
  "${LOCAL_ROOT}/MemNavData/run_cec_controller_portability_smoke_local.sh" \
  "${LOCAL_ROOT}/MemNavData/slurm_hm3d_vint_controller_native_pair.sbatch" \
  "${LOCAL_ROOT}/MemNavData/slurm_hm3d_vint_controller_native_analysis.sbatch"
source "${LOCAL_ROOT}/MemNavData/slurm_safe_submit.sh"
lint_sbatch_template \
  "${LOCAL_ROOT}/MemNavData/slurm_hm3d_vint_controller_native_pair.sbatch" || \
  fail "pair sbatch lint failed"
lint_sbatch_template \
  "${LOCAL_ROOT}/MemNavData/slurm_hm3d_vint_controller_native_analysis.sbatch" || \
  fail "analysis sbatch lint failed"

staging=$(mktemp -d)
trap 'rm -rf -- "${staging}"' EXIT
mkdir -p "${staging}/MemNavData"
while IFS= read -r -d '' path; do
  cp --preserve=mode,timestamps "${path}" \
    "${staging}/MemNavData/$(basename "${path}")"
done < <(find "${LOCAL_ROOT}/MemNavData" -maxdepth 1 -type f \
  -name '*.py' -print0)
for name in run_cec_controller_portability_smoke_local.sh bundle_selftest.sh \
  slurm_hm3d_vint_controller_native_pair.sbatch \
  slurm_hm3d_vint_controller_native_analysis.sbatch \
  HM3D_VINT_CONTROLLER_NATIVE_CEC_PROTOCOL_20260828.md \
  hm3d_vint_controller_native_cec_protocol_20260828.json; do
  cp --preserve=mode,timestamps "${LOCAL_ROOT}/MemNavData/${name}" \
    "${staging}/MemNavData/${name}"
done
for component in memnav navdp vint; do
  mkdir -p "${staging}/NavDP/baselines/${component}"
  while IFS= read -r -d '' path; do
    cp --preserve=mode,timestamps "${path}" \
      "${staging}/NavDP/baselines/${component}/$(basename "${path}")"
  done < <(find "${LOCAL_ROOT}/NavDP/baselines/${component}" \
    -maxdepth 1 -type f -name '*.py' -print0)
  if [[ -d "${LOCAL_ROOT}/NavDP/baselines/${component}/configs" ]]; then
    mkdir -p "${staging}/NavDP/baselines/${component}/configs"
    while IFS= read -r -d '' path; do
      cp --preserve=mode,timestamps "${path}" \
        "${staging}/NavDP/baselines/${component}/configs/$(basename "${path}")"
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
  mkdir -p "${staging}/$(dirname "${relative}")"
  cp --preserve=mode,timestamps "${LOCAL_ROOT}/${relative}" \
    "${staging}/${relative}"
done

local_head=$(git -C "${LOCAL_ROOT}" rev-parse HEAD)
"${MEMNAV_PY}" - "${staging}" "${local_head}" "${FRESH_ROOT}" <<'PY'
import hashlib,json,sys
from pathlib import Path
root=Path(sys.argv[1]); files={}
for path in sorted(root.rglob("*")):
 if path.is_symlink(): raise SystemExit("bundle symlink: "+str(path))
 if path.is_file() and path.name not in {"SOURCE_BUNDLE.sha256","source_bundle_manifest.json"}:
  files[path.relative_to(root).as_posix()]=hashlib.sha256(path.read_bytes()).hexdigest()
payload={
 "schema_version":"hm3d_vint_controller_native_bundle_v1_20260828",
 "local_git_head_context":sys.argv[2],"fresh_source_root":sys.argv[3],
 "pilot_history_indices":[0,7,14,21],"formal_history_indices":list(range(28)),
 "controller":"vint","reject_policy":"controller_native_exact",
 "paired_arms":["forced_reject_native","grant"],
 "pilot_performance_used_as_gate":False,"files":files,
}
(root/"source_bundle_manifest.json").write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n")
PY
(
  cd "${staging}"
  find . -type f ! -name SOURCE_BUNDLE.sha256 -print0 | sort -z | \
    xargs -0 sha256sum >SOURCE_BUNDLE.sha256
  sha256sum -c SOURCE_BUNDLE.sha256 >/dev/null
)
source_receipt_sha=$(sha256sum "${staging}/SOURCE_BUNDLE.sha256" | awk '{print $1}')
bundle_manifest_sha=$(sha256sum "${staging}/source_bundle_manifest.json" | awk '{print $1}')
remote_bundle=${REMOTE_BUNDLE_BASE}/hm3d_vint_cec_${bundle_manifest_sha:0:16}
remote_stage=${remote_bundle}.partial.$$
if [[ -n "${EXPORT_ARCHIVE}" ]]; then
  [[ "${EXPORT_ARCHIVE}" == /* ]] || fail "EXPORT_ARCHIVE must be absolute"
  [[ ! -e "${EXPORT_ARCHIVE}" ]] || fail "EXPORT_ARCHIVE already exists"
  tar -C "${staging}" -czf "${EXPORT_ARCHIVE}" .
fi
if [[ "${DRY_RUN}" == 1 ]]; then
  printf 'DRY_RUN_RUN_ROOT=%s\nDRY_RUN_REMOTE_BUNDLE=%s\nDRY_RUN_SOURCE_RECEIPT_SHA=%s\n' \
    "${RUN_ROOT}" "${remote_bundle}" "${source_receipt_sha}"
  exit 0
fi

remote "test \"\$(sha256sum '${FRESH_ROOT}/benchmarks/natural_direction/manifest.json' | awk '{print \$1}')\" = 'aada40d25d01e9385df3ffdcaf37847f471b63c7be785a704eade961346a50b0'"
remote "test \"\$(sha256sum '${FRESH_ROOT}/hm3d_fullmono_mixed_role_independent_verification.json' | awk '{print \$1}')\" = '3ae4b556eef9e8144f635495f65d58b177ceee8d98301327374967415cf8d2d8'"
remote "test \"\$(sha256sum '${BASE_SOURCE_ROOT}/SOURCE_BUNDLE.sha256' | awk '{print \$1}')\" = '${BASE_SOURCE_RECEIPT_SHA}'"
remote "test \"\$(sha256sum '${DEPENDENCY_RECEIPT}' | awk '{print \$1}')\" = '${EXPECTED_DEPENDENCY_RECEIPT_SHA}'"
remote "test -r '${PORTABILITY_ENV_ROOT}/environment_receipt.json' && cd '${PORTABILITY_CHECKPOINT_ROOT}' && sha256sum -c --quiet CHECKPOINTS.sha256"
if remote "test -d '${remote_bundle}' && test \"\$(sha256sum '${remote_bundle}/SOURCE_BUNDLE.sha256' | awk '{print \$1}')\" = '${source_receipt_sha}' && cd '${remote_bundle}' && sha256sum -c --quiet SOURCE_BUNDLE.sha256"; then
  echo "Reusing verified bundle ${remote_bundle}"
else
  remote "test ! -e '${remote_bundle}' && mkdir -p '${remote_stage}'"
  rsync -a --chmod=Du=rwx,Dgo=rx,Fu=rw,Fgo=r \
    -e "ssh -o BatchMode=yes -o ControlMaster=no -S ${SSH_CONTROL_PATH}" \
    "${staging}/" "${REMOTE_HOST}:${remote_stage}/"
  remote "cd '${remote_stage}' && sha256sum -c --quiet SOURCE_BUNDLE.sha256 && chmod -R a-w '${remote_stage}' && mv '${remote_stage}' '${remote_bundle}'"
fi

remote "test ! -e '${RUN_ROOT}' && mkdir -p '${RUN_ROOT}/sealed_inputs' '${RUN_ROOT}/pilot/evaluation' '${RUN_ROOT}/formal/evaluation' /scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs"
remote "cp '${remote_bundle}/MemNavData/HM3D_VINT_CONTROLLER_NATIVE_CEC_PROTOCOL_20260828.md' '${RUN_ROOT}/sealed_inputs/' && cp '${remote_bundle}/MemNavData/hm3d_vint_controller_native_cec_protocol_20260828.json' '${RUN_ROOT}/sealed_inputs/' && sha256sum '${FRESH_ROOT}/benchmarks/natural_direction/manifest.json' >'${RUN_ROOT}/sealed_inputs/benchmark_manifest.sha256' && chmod -R a-w '${RUN_ROOT}/sealed_inputs'"

source_receipt=${remote_bundle}/SOURCE_BUNDLE.sha256
common="ALL,SOURCE_ROOT=${remote_bundle},SOURCE_RECEIPT=${source_receipt},EXPECTED_SOURCE_RECEIPT_SHA=${source_receipt_sha},FRESH_ROOT=${FRESH_ROOT},RUN_ROOT=${RUN_ROOT},BASE_SOURCE_ROOT=${BASE_SOURCE_ROOT},BASE_SOURCE_RECEIPT_SHA=${BASE_SOURCE_RECEIPT_SHA},DEPENDENCY_RECEIPT=${DEPENDENCY_RECEIPT},EXPECTED_DEPENDENCY_RECEIPT_SHA=${EXPECTED_DEPENDENCY_RECEIPT_SHA},PORTABILITY_ENV_ROOT=${PORTABILITY_ENV_ROOT},PORTABILITY_CHECKPOINT_ROOT=${PORTABILITY_CHECKPOINT_ROOT}"
pair_sbatch=${remote_bundle}/MemNavData/slurm_hm3d_vint_controller_native_pair.sbatch
analysis_sbatch=${remote_bundle}/MemNavData/slurm_hm3d_vint_controller_native_analysis.sbatch
remote "sbatch --test-only --array=0 --export='${common},PHASE=pilot' '${pair_sbatch}' >/dev/null"
remote "sbatch --test-only --export='${common},MODE=pilot_aggregate' '${analysis_sbatch}' >/dev/null"

pilot_raw=$(remote "sbatch --parsable --array=0-3%${PILOT_CONCURRENCY} --export='${common},PHASE=pilot' '${pair_sbatch}'")
pilot_id=${pilot_raw%%;*}; [[ "${pilot_id}" =~ ^[0-9]+$ ]] || fail "bad pilot job"
pilot_aggregate_raw=$(remote "sbatch --parsable --dependency=afterok:${pilot_id} --kill-on-invalid-dep=yes --export='${common},MODE=pilot_aggregate' '${analysis_sbatch}'")
pilot_aggregate_id=${pilot_aggregate_raw%%;*}
pilot_verify_raw=$(remote "sbatch --parsable --dependency=afterok:${pilot_aggregate_id} --kill-on-invalid-dep=yes --export='${common},MODE=pilot_verify' '${analysis_sbatch}'")
pilot_verify_id=${pilot_verify_raw%%;*}
formal_raw=$(remote "sbatch --parsable --array=0-27%${FORMAL_CONCURRENCY} --dependency=afterok:${pilot_verify_id} --kill-on-invalid-dep=yes --export='${common},PHASE=formal' '${pair_sbatch}'")
formal_id=${formal_raw%%;*}
formal_aggregate_raw=$(remote "sbatch --parsable --dependency=afterok:${formal_id} --kill-on-invalid-dep=yes --export='${common},MODE=formal_aggregate' '${analysis_sbatch}'")
formal_aggregate_id=${formal_aggregate_raw%%;*}
formal_verify_raw=$(remote "sbatch --parsable --dependency=afterok:${formal_aggregate_id} --kill-on-invalid-dep=yes --export='${common},MODE=formal_verify' '${analysis_sbatch}'")
formal_verify_id=${formal_verify_raw%%;*}
for id in "${pilot_aggregate_id}" "${pilot_verify_id}" "${formal_id}" \
          "${formal_aggregate_id}" "${formal_verify_id}"; do
  [[ "${id}" =~ ^[0-9]+$ ]] || fail "invalid submitted job id"
done

printf 'RUN_ROOT=%s\nSOURCE_BUNDLE=%s\nSOURCE_RECEIPT_SHA=%s\nPILOT=%s\nPILOT_AGGREGATE=%s\nPILOT_VERIFY=%s\nFORMAL=%s\nFORMAL_AGGREGATE=%s\nFORMAL_VERIFY=%s\n' \
  "${RUN_ROOT}" "${remote_bundle}" "${source_receipt_sha}" \
  "${pilot_id}" "${pilot_aggregate_id}" "${pilot_verify_id}" \
  "${formal_id}" "${formal_aggregate_id}" "${formal_verify_id}"
