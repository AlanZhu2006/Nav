#!/usr/bin/env bash
# Submit NavDP and ViNT Table-1 pairs only after the construction verifier
# independently authorizes the frozen fresh-query, scene-overlap population.
set -euo pipefail
umask 0022

ROOT=${ROOT:-/home/asus/Research/Nav-graph-blind}
SSH_ALIAS=${SSH_ALIAS:-alantorch}
LOCAL_MEMNAV_PY=/home/asus/miniconda3/envs/memnav/bin/python
LOCAL_HAB_PY=/home/asus/miniconda3/envs/habitat/bin/python
REMOTE_BUNDLES=/scratch/yz11502/Research/Nav-axis-uturn-source-bundles
REMOTE_RESULTS=/scratch/yz11502/Research/Nav-axis-uturn-results/hm3d_table1_controller_portability_20260829
CONSTRUCTION_RUN=${CONSTRUCTION_RUN:-/scratch/yz11502/Research/Nav-axis-uturn-results/hm3d_table1_fresh_query_reserve_20260829/construction_20260828T212552Z_bb757914}
SOURCE_RUN_ROOT=/scratch/yz11502/Research/Nav-axis-uturn-results/hm3d_fresh_fullmono_mixed_role_20260820/formal_20260820T143609Z_e6dd44c6
NAVDP_BASE_SOURCE_ROOT=/scratch/yz11502/Research/Nav-axis-uturn-source-bundles/final14_mono_factorial_5690569a4373f2d2
NAVDP_BASE_RECEIPT=${NAVDP_BASE_SOURCE_ROOT}/source_inputs.sha256
EXPECTED_NAVDP_BASE_RECEIPT_SHA=5690569a4373f2d2768671418f0c604c4a03aa4b0ffe01baf70b288af03ba216
NAVDP_SERVER_SOURCE_ROOT=/scratch/yz11502/Research/Nav-axis-uturn-source-bundles/hm3d_fullmono_transaction_repair_67e1132783ce2cb1
NAVDP_SERVER_RECEIPT=${NAVDP_SERVER_SOURCE_ROOT}/SOURCE_BUNDLE.sha256
EXPECTED_NAVDP_SERVER_RECEIPT_SHA=05ce401aac8c2e7e31e8a8d820613d30b3a03856a35c8750085b93d5a1539a97
VINT_BASE_SOURCE_ROOT=/scratch/yz11502/Research/Nav-axis-uturn-source-bundles/certified_relocalization_closed_loop_d3bd281fc374cc80
VINT_BASE_RECEIPT_SHA=74001a9e0150c38c599a206fa0f4dd5e1279b9bed5d167119f4d14cb77995e98
DEPENDENCY_RECEIPT=/scratch/yz11502/Research/Nav-axis-uturn-results/shared_online_double_revisit_fresh_20260813/double_revisit_fresh40_20260813T200121Z/dependency_receipt.json
EXPECTED_DEPENDENCY_RECEIPT_SHA=4eb0ca6479a26f8e04f85a31d906cee4e68b1785f66cfd3ac23bf65424d36e5e
PORTABILITY_ENV_ROOT=/scratch/yz11502/Research/Nav-axis-uturn-envs/controller_portability_a9ec7146bce7_v1
PORTABILITY_CHECKPOINT_ROOT=/scratch/yz11502/Research/Nav-axis-uturn-checkpoints/controller_portability_50387aa89be8
HAB_REQUESTS_VENDOR=/scratch/lg154/conda-envs/habitat/lib/python3.9/site-packages/pip/_vendor
NAVDP_CONCURRENCY=${NAVDP_CONCURRENCY:-2}
VINT_CONCURRENCY=${VINT_CONCURRENCY:-2}
RUN_TAG=${RUN_TAG:-formal_$(date -u +%Y%m%dT%H%M%SZ)}
SSH_CONTROL_PATH=${SSH_CONTROL_PATH:-$(ssh -G "${SSH_ALIAS}" 2>/dev/null | awk '$1=="controlpath"{value=$2} END{print value}')}
cd "${ROOT}"

fail() { echo "ABORT: $*" >&2; exit 2; }
remote() {
  # Some calls run inside process substitution.  With a local controlling TTY,
  # an SSH slave that can read stdin is backgrounded and receives SIGTTIN.
  # The complete remote program is already passed as an argv string, so detach
  # stdin explicitly while retaining the proven shared-master PTY path.
  timeout 180 ssh -n -tt -o BatchMode=yes -o ControlMaster=no \
    -S "${SSH_CONTROL_PATH}" "${SSH_ALIAS}" "$@"
}
[[ -S "${SSH_CONTROL_PATH}" ]] || fail "authoritative SSH master missing"
for value in "${NAVDP_CONCURRENCY}" "${VINT_CONCURRENCY}"; do
  [[ "${value}" =~ ^[1-9][0-9]*$ ]] || fail "invalid concurrency ${value}"
done
[[ "${RUN_TAG}" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]] || fail "invalid run tag"

required=(
  MemNavData/HM3D_TABLE1_CONTROLLER_PORTABILITY_PROTOCOL_20260829.md
  MemNavData/HM3D_TABLE1_NAVDP_TRANSACTION_REPAIR_20260829.md
  MemNavData/hm3d_table1_controller_portability_protocol_20260829.json
  MemNavData/hm3d_fullmono_mixed_role_protocol_20260820.json
  MemNavData/hm3d_fullmono_mixed_role.py
  MemNavData/run_hm3d_fullmono_query_history.py
  MemNavData/run_hm3d_fullmono_server_scene.sh
  MemNavData/aggregate_hm3d_table1_navdp_pair.py
  MemNavData/independent_verify_hm3d_table1_navdp_pair.py
  MemNavData/aggregate_vint_controller_native_hm3d.py
  MemNavData/independent_verify_vint_controller_native_hm3d.py
  MemNavData/audit_vint_controller_native_pair.py
  MemNavData/run_cec_controller_portability_smoke_local.sh
  MemNavData/slurm_hm3d_table1_navdp_pair.sbatch
  MemNavData/slurm_hm3d_table1_navdp_analysis.sbatch
  MemNavData/slurm_hm3d_table1_vint_pair.sbatch
  MemNavData/slurm_hm3d_table1_vint_analysis.sbatch
  MemNavData/slurm_hm3d_table1_controller_seal.sbatch
  MemNavData/test_hm3d_table1_navdp_pair.py
  MemNavData/test_hm3d_table1_navdp_transport_contract.py
  MemNavData/test_aggregate_vint_controller_native_hm3d.py
  MemNavData/test_audit_vint_controller_native_pair.py
)
for path in "${required[@]}"; do
  [[ -f "${path}" && ! -L "${path}" ]] || fail "missing physical ${path}"
done

export PYTHONPATH=${ROOT}:${ROOT}/MemNavData${PYTHONPATH:+:${PYTHONPATH}}
"${LOCAL_MEMNAV_PY}" -m py_compile \
  MemNavData/aggregate_hm3d_table1_navdp_pair.py \
  MemNavData/independent_verify_hm3d_table1_navdp_pair.py \
  MemNavData/aggregate_vint_controller_native_hm3d.py \
  MemNavData/independent_verify_vint_controller_native_hm3d.py \
  MemNavData/audit_vint_controller_native_pair.py
"${LOCAL_MEMNAV_PY}" -m json.tool \
  MemNavData/hm3d_table1_controller_portability_protocol_20260829.json \
  >/dev/null
"${LOCAL_HAB_PY}" -m py_compile \
  MemNavData/run_hm3d_fullmono_query_history.py \
  MemNavData/eval_shared_online_role_pairs.py \
  MemNavData/eval_2leg_habitat.py
"${LOCAL_MEMNAV_PY}" -m pytest -q \
  MemNavData/test_hm3d_table1_navdp_pair.py \
  MemNavData/test_hm3d_table1_navdp_transport_contract.py \
  MemNavData/test_aggregate_vint_controller_native_hm3d.py \
  MemNavData/test_audit_vint_controller_native_pair.py \
  MemNavData/test_hm3d_fullmono_mixed_role.py \
  MemNavData/test_cec_handoff_contract.py \
  MemNavData/test_cec_controller_portability_hub.py \
  MemNavData/test_controller_portability_contract.py \
  MemNavData/test_controller_portability_proxy.py
bash -n \
  MemNavData/run_hm3d_fullmono_server_scene.sh \
  MemNavData/run_cec_controller_portability_smoke_local.sh \
  MemNavData/slurm_hm3d_table1_navdp_pair.sbatch \
  MemNavData/slurm_hm3d_table1_navdp_analysis.sbatch \
  MemNavData/slurm_hm3d_table1_vint_pair.sbatch \
  MemNavData/slurm_hm3d_table1_vint_analysis.sbatch \
  MemNavData/slurm_hm3d_table1_controller_seal.sbatch \
  MemNavData/submit_hm3d_table1_controller_portability_hpc.sh
source MemNavData/slurm_safe_submit.sh
for script in \
  MemNavData/slurm_hm3d_table1_navdp_pair.sbatch \
  MemNavData/slurm_hm3d_table1_navdp_analysis.sbatch \
  MemNavData/slurm_hm3d_table1_vint_pair.sbatch \
  MemNavData/slurm_hm3d_table1_vint_analysis.sbatch \
  MemNavData/slurm_hm3d_table1_controller_seal.sbatch; do
  lint_sbatch_template "${script}" || fail "sbatch lint failed: ${script}"
done

construction_verification=${CONSTRUCTION_RUN}/hm3d_table1_fresh_query_verification.json
bench_root=${CONSTRUCTION_RUN}/population/natural_direction
parent_manifest=${SOURCE_RUN_ROOT}/sealed_inputs/parent_manifest.json
run_root=${REMOTE_RESULTS}/${RUN_TAG}

echo '[gate] independently verified construction authorization'
readarray -t gate < <(remote "python - '${construction_verification}' '${bench_root}/manifest.json' <<'PY'
import hashlib,json,sys
verification=json.load(open(sys.argv[1])); manifest=sys.argv[2]
if verification.get('verified') is not True:
 raise SystemExit('construction verifier did not pass')
if verification.get('formal_policy_evaluation_authorized') is not True:
 raise SystemExit('construction power gate did not authorize evaluation')
digest=hashlib.sha256(open(manifest,'rb').read()).hexdigest()
if digest != verification.get('benchmark_manifest_sha256'):
 raise SystemExit('verified benchmark manifest changed')
print(verification['histories']); print(verification['scene_clusters'])
print(digest); print(hashlib.sha256(open(sys.argv[1],'rb').read()).hexdigest())
PY" | tr -d '\r')
[[ "${#gate[@]}" -eq 4 ]] || fail "construction gate receipt incomplete"
histories=${gate[0]}; scenes=${gate[1]}
manifest_sha=${gate[2]}; construction_verification_sha=${gate[3]}
[[ "${histories}" =~ ^[1-9][0-9]*$ && "${scenes}" =~ ^[1-9][0-9]*$ ]] || \
  fail "invalid frozen denominator"
vint_last=$((histories - 1))

staging=$(mktemp -d)
cleanup() { rm -rf -- "${staging}"; }
trap cleanup EXIT
mkdir -p "${staging}/MemNavData"
while IFS= read -r -d '' path; do
  cp --preserve=mode,timestamps "${path}" \
    "${staging}/MemNavData/$(basename "${path}")"
done < <(find "${ROOT}/MemNavData" -maxdepth 1 -type f -name '*.py' -print0)
for path in "${required[@]}"; do
  case "${path}" in
    *.py) ;;
    *) cp --preserve=mode,timestamps "${path}" \
         "${staging}/MemNavData/$(basename "${path}")" ;;
  esac
done
for component in memnav navdp vint; do
  mkdir -p "${staging}/NavDP/baselines/${component}"
  while IFS= read -r -d '' path; do
    cp --preserve=mode,timestamps "${path}" \
      "${staging}/NavDP/baselines/${component}/$(basename "${path}")"
  done < <(find "${ROOT}/NavDP/baselines/${component}" \
    -maxdepth 1 -type f -name '*.py' -print0)
  if [[ -d "${ROOT}/NavDP/baselines/${component}/configs" ]]; then
    mkdir -p "${staging}/NavDP/baselines/${component}/configs"
    cp -a "${ROOT}/NavDP/baselines/${component}/configs/." \
      "${staging}/NavDP/baselines/${component}/configs/"
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
  cp --preserve=mode,timestamps "${ROOT}/${relative}" "${staging}/${relative}"
done
local_head=$(git -C "${ROOT}" rev-parse HEAD)
"${LOCAL_MEMNAV_PY}" - "${staging}" "${local_head}" "${manifest_sha}" \
  "${construction_verification_sha}" "${histories}" "${scenes}" <<'PY'
import hashlib,json,sys
root,head,manifest,verification,histories,scenes=sys.argv[1:]
base=__import__('pathlib').Path(root); files={}
for path in sorted(base.rglob('*')):
 if path.is_symlink(): raise SystemExit('bundle symlink: '+str(path))
 if path.is_file() and path.name not in {'SOURCE_BUNDLE.sha256','source_bundle_manifest.json'}:
  files[path.relative_to(base).as_posix()]=hashlib.sha256(path.read_bytes()).hexdigest()
payload={
 'schema_version':'hm3d_table1_controller_portability_bundle_v1_20260829',
 'local_git_head_context':head,'benchmark_manifest_sha256':manifest,
 'construction_verification_sha256':verification,
 'histories':int(histories),'scene_clusters':int(scenes),
 'controllers':['navdp','vint'],'runtime_role_visibility':'none',
 'navdp_arms':['mono_native','mono_cec'],
 'vint_arms':['forced_reject_native','grant'],
 'vint_grant_bearing_alignment':'first_certified_bounded',
 'policy_outcomes_read_during_population_selection':False,'files':files,
}
(base/'source_bundle_manifest.json').write_text(json.dumps(payload,indent=2,sort_keys=True)+'\n')
PY
(
  cd "${staging}"
  find . -type f ! -name SOURCE_BUNDLE.sha256 -print0 | sort -z | \
    xargs -0 sha256sum >SOURCE_BUNDLE.sha256
  sha256sum -c SOURCE_BUNDLE.sha256 >/dev/null
)
task_receipt_sha=$(sha256sum "${staging}/SOURCE_BUNDLE.sha256" | awk '{print $1}')
bundle_key=${task_receipt_sha:0:16}
task_root=${REMOTE_BUNDLES}/hm3d_table1_controller_portability_${bundle_key}
task_stage=${task_root}.partial.$$

echo '[gate] remote dependency and identity audit'
remote_identity=$(remote 'id -un' | tr -d '\r')
[[ "${remote_identity}" == yz11502 ]] || fail "wrong remote identity"
remote "set -euo pipefail
test \"\$(sha256sum '${NAVDP_BASE_RECEIPT}' | awk '{print \$1}')\" = '${EXPECTED_NAVDP_BASE_RECEIPT_SHA}'
cd '${NAVDP_BASE_SOURCE_ROOT}' && sha256sum -c --quiet '${NAVDP_BASE_RECEIPT}'
test \"\$(sha256sum '${NAVDP_SERVER_RECEIPT}' | awk '{print \$1}')\" = '${EXPECTED_NAVDP_SERVER_RECEIPT_SHA}'
cd '${NAVDP_SERVER_SOURCE_ROOT}' && sha256sum -c --quiet '${NAVDP_SERVER_RECEIPT}'
test \"\$(sha256sum '${VINT_BASE_SOURCE_ROOT}/SOURCE_BUNDLE.sha256' | awk '{print \$1}')\" = '${VINT_BASE_RECEIPT_SHA}'
test \"\$(sha256sum '${DEPENDENCY_RECEIPT}' | awk '{print \$1}')\" = '${EXPECTED_DEPENDENCY_RECEIPT_SHA}'
test -r '${PORTABILITY_ENV_ROOT}/environment_receipt.json'
cd '${PORTABILITY_CHECKPOINT_ROOT}' && sha256sum -c --quiet CHECKPOINTS.sha256
test \"\$(sha256sum '${parent_manifest}' | awk '{print \$1}')\" = a96a0b96fab7b7b47709b36cb8eeb9410b42b09f095f87ef01304a68de716dd5"

echo '[stage] verify or upload immutable source bundle'
if remote "test -d '${task_root}'"; then
  remote "test \"\$(sha256sum '${task_root}/SOURCE_BUNDLE.sha256' | awk '{print \$1}')\" = '${task_receipt_sha}' && cd '${task_root}' && sha256sum -c --quiet SOURCE_BUNDLE.sha256"
else
  remote "test ! -e '${task_stage}' && mkdir -p '${task_stage}'"
  rsync -a --chmod=Du=rwx,Dgo=rx,Fu=rw,Fgo=r \
    -e "ssh -o BatchMode=yes -o ControlMaster=no -S ${SSH_CONTROL_PATH}" \
    "${staging}/" "${SSH_ALIAS}:${task_stage}/"
  remote "cd '${task_stage}' && sha256sum -c --quiet SOURCE_BUNDLE.sha256 && chmod -R a-w '${task_stage}' && mv '${task_stage}' '${task_root}'"
fi

task_receipt=${task_root}/SOURCE_BUNDLE.sha256
protocol=${task_root}/MemNavData/hm3d_fullmono_mixed_role_protocol_20260820.json
remote "test ! -e '${run_root}' && mkdir -p '${run_root}/sealed_inputs' '${run_root}/logs' '${run_root}/smoke' '${run_root}/formal' /scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs
cp '${construction_verification}' '${run_root}/sealed_inputs/'
cp '${CONSTRUCTION_RUN}/population/population_receipt.json' '${run_root}/sealed_inputs/'
cp '${task_root}/MemNavData/HM3D_TABLE1_CONTROLLER_PORTABILITY_PROTOCOL_20260829.md' '${run_root}/sealed_inputs/'
cp '${task_root}/MemNavData/hm3d_table1_controller_portability_protocol_20260829.json' '${run_root}/sealed_inputs/'
sha256sum '${bench_root}/manifest.json' '${construction_verification}' '${CONSTRUCTION_RUN}/population/population_receipt.json' >'${run_root}/sealed_inputs/experiment_inputs.sha256'
chmod -R a-w '${run_root}/sealed_inputs'"

common="ALL,TASK_ROOT=${task_root},TASK_RECEIPT=${task_receipt},EXPECTED_TASK_RECEIPT_SHA=${task_receipt_sha},FORMAL_RUN_ROOT=${run_root},BENCH_ROOT=${bench_root},CONSTRUCTION_VERIFICATION=${construction_verification},EXPECTED_CONSTRUCTION_VERIFICATION_SHA=${construction_verification_sha}"
nav_common="${common},BASE_SOURCE_ROOT=${NAVDP_BASE_SOURCE_ROOT},BASE_RECEIPT=${NAVDP_BASE_RECEIPT},EXPECTED_BASE_RECEIPT_SHA=${EXPECTED_NAVDP_BASE_RECEIPT_SHA},SERVER_SOURCE_ROOT=${NAVDP_SERVER_SOURCE_ROOT},SERVER_SOURCE_RECEIPT=${NAVDP_SERVER_RECEIPT},EXPECTED_SERVER_SOURCE_RECEIPT_SHA=${EXPECTED_NAVDP_SERVER_RECEIPT_SHA},SOURCE_RUN_ROOT=${SOURCE_RUN_ROOT},PARENT_MANIFEST=${parent_manifest},PROTOCOL=${protocol}"
vint_common="${common},BASE_SOURCE_ROOT=${VINT_BASE_SOURCE_ROOT},BASE_SOURCE_RECEIPT_SHA=${VINT_BASE_RECEIPT_SHA},DEPENDENCY_RECEIPT=${DEPENDENCY_RECEIPT},EXPECTED_DEPENDENCY_RECEIPT_SHA=${EXPECTED_DEPENDENCY_RECEIPT_SHA},PORTABILITY_ENV_ROOT=${PORTABILITY_ENV_ROOT},PORTABILITY_CHECKPOINT_ROOT=${PORTABILITY_CHECKPOINT_ROOT}"
nav_pair=${task_root}/MemNavData/slurm_hm3d_table1_navdp_pair.sbatch
nav_analysis=${task_root}/MemNavData/slurm_hm3d_table1_navdp_analysis.sbatch
vint_pair=${task_root}/MemNavData/slurm_hm3d_table1_vint_pair.sbatch
vint_analysis=${task_root}/MemNavData/slurm_hm3d_table1_vint_analysis.sbatch
seal=${task_root}/MemNavData/slurm_hm3d_table1_controller_seal.sbatch

echo '[gate] remote imports and Slurm test-only'
remote "test -r '${HAB_REQUESTS_VENDOR}/requests/__init__.py' && singularity exec --nv -B /scratch/lg154 -B /scratch/yz11502 /share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH='${task_root}:${task_root}/MemNavData:${NAVDP_SERVER_SOURCE_ROOT}:${NAVDP_SERVER_SOURCE_ROOT}/MemNavData:${NAVDP_BASE_SOURCE_ROOT}:${NAVDP_BASE_SOURCE_ROOT}/MemNavData:${HAB_REQUESTS_VENDOR}' /scratch/lg154/conda-envs/habitat/bin/python -c 'from pathlib import Path; [compile(Path(path).read_bytes(),path,\"exec\") for path in (\"${task_root}/MemNavData/eval_shared_online_role_pairs.py\",\"${task_root}/MemNavData/eval_2leg_habitat.py\",\"${NAVDP_SERVER_SOURCE_ROOT}/NavDP/baselines/memnav/memnav_server.py\",\"${NAVDP_SERVER_SOURCE_ROOT}/NavDP/baselines/navdp/navdp_server.py\")]' && grep -q 'def append_request_frame' '${NAVDP_SERVER_SOURCE_ROOT}/NavDP/baselines/memnav/memnav_server.py' && grep -q 'require_monocular_depth_transaction' '${NAVDP_SERVER_SOURCE_ROOT}/NavDP/baselines/navdp/navdp_server.py' && singularity exec --nv -B /scratch/lg154 -B /scratch/yz11502 /share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH='${task_root}:${task_root}/MemNavData:${NAVDP_SERVER_SOURCE_ROOT}:${NAVDP_SERVER_SOURCE_ROOT}/MemNavData:${NAVDP_BASE_SOURCE_ROOT}:${NAVDP_BASE_SOURCE_ROOT}/MemNavData:${HAB_REQUESTS_VENDOR}' /scratch/lg154/conda-envs/habitat/bin/python -c 'import MemNavData.run_hm3d_fullmono_query_history'"
remote "sbatch --test-only --array=0 --export='${nav_common},PHASE=smoke' '${nav_pair}' >/dev/null"
remote "sbatch --test-only --array=0-53%${NAVDP_CONCURRENCY} --export='${nav_common},PHASE=formal' '${nav_pair}' >/dev/null"
remote "sbatch --test-only --export='${common},MODE=aggregate' '${nav_analysis}' >/dev/null"
remote "sbatch --test-only --array=0 --export='${vint_common},PHASE=smoke' '${vint_pair}' >/dev/null"
remote "sbatch --test-only --array=0-${vint_last}%${VINT_CONCURRENCY} --export='${vint_common},PHASE=formal' '${vint_pair}' >/dev/null"
remote "sbatch --test-only --export='${common},MODE=aggregate' '${vint_analysis}' >/dev/null"
remote "sbatch --test-only --export='${common}' '${seal}' >/dev/null"

echo '[submit] result-blind smoke -> formal -> independent verification DAGs'
nav_smoke_raw=$(remote "sbatch --parsable --array=0 --export='${nav_common},PHASE=smoke' '${nav_pair}'" | tr -d '\r')
nav_smoke=${nav_smoke_raw%%;*}
vint_smoke_raw=$(remote "sbatch --parsable --array=0 --export='${vint_common},PHASE=smoke' '${vint_pair}'" | tr -d '\r')
vint_smoke=${vint_smoke_raw%%;*}
nav_formal_raw=$(remote "sbatch --parsable --array=0-53%${NAVDP_CONCURRENCY} --dependency=afterok:${nav_smoke} --kill-on-invalid-dep=yes --export='${nav_common},PHASE=formal' '${nav_pair}'" | tr -d '\r')
nav_formal=${nav_formal_raw%%;*}
vint_formal_raw=$(remote "sbatch --parsable --array=0-${vint_last}%${VINT_CONCURRENCY} --dependency=afterok:${vint_smoke} --kill-on-invalid-dep=yes --export='${vint_common},PHASE=formal' '${vint_pair}'" | tr -d '\r')
vint_formal=${vint_formal_raw%%;*}
nav_aggregate_raw=$(remote "sbatch --parsable --dependency=afterok:${nav_formal} --kill-on-invalid-dep=yes --export='${common},MODE=aggregate' '${nav_analysis}'" | tr -d '\r')
nav_aggregate=${nav_aggregate_raw%%;*}
nav_verify_raw=$(remote "sbatch --parsable --dependency=afterok:${nav_aggregate} --kill-on-invalid-dep=yes --export='${common},MODE=verify' '${nav_analysis}'" | tr -d '\r')
nav_verify=${nav_verify_raw%%;*}
vint_aggregate_raw=$(remote "sbatch --parsable --dependency=afterok:${vint_formal} --kill-on-invalid-dep=yes --export='${common},MODE=aggregate' '${vint_analysis}'" | tr -d '\r')
vint_aggregate=${vint_aggregate_raw%%;*}
vint_verify_raw=$(remote "sbatch --parsable --dependency=afterok:${vint_aggregate} --kill-on-invalid-dep=yes --export='${common},MODE=verify' '${vint_analysis}'" | tr -d '\r')
vint_verify=${vint_verify_raw%%;*}
seal_raw=$(remote "sbatch --parsable --dependency=afterok:${nav_verify}:${vint_verify} --kill-on-invalid-dep=yes --export='${common}' '${seal}'" | tr -d '\r')
seal_job=${seal_raw%%;*}
for id in "${nav_smoke}" "${vint_smoke}" "${nav_formal}" "${vint_formal}" \
          "${nav_aggregate}" "${nav_verify}" "${vint_aggregate}" \
          "${vint_verify}" "${seal_job}"; do
  [[ "${id}" =~ ^[0-9]+$ ]] || fail "invalid submitted job id: ${id}"
done

receipt=MemNavData/HM3D_TABLE1_CONTROLLER_PORTABILITY_SUBMISSION_20260829.json
[[ ! -e "${receipt}" ]] || fail "local submission receipt already exists"
"${LOCAL_MEMNAV_PY}" - "${receipt}" "${run_root}" "${task_root}" \
  "${task_receipt_sha}" "${construction_verification_sha}" \
  "${manifest_sha}" "${histories}" "${scenes}" "${nav_smoke}" \
  "${vint_smoke}" "${nav_formal}" "${vint_formal}" "${nav_aggregate}" \
  "${nav_verify}" "${vint_aggregate}" "${vint_verify}" "${seal_job}" <<'PY'
import json,sys
(path,run,bundle,bundle_sha,construction_sha,manifest_sha,histories,scenes,
 nav_smoke,vint_smoke,nav_formal,vint_formal,nav_aggregate,nav_verify,
 vint_aggregate,vint_verify,seal)=sys.argv[1:]
payload={
 'schema_version':'hm3d_table1_controller_portability_submission_v1_20260829',
 'scope':'fresh-query scene-overlap within-controller paired evaluation',
 'run_root':run,'task_bundle':bundle,'task_receipt_sha256':bundle_sha,
 'construction_verification_sha256':construction_sha,
 'benchmark_manifest_sha256':manifest_sha,'histories':int(histories),
 'scene_clusters':int(scenes),'partial_policy_outcomes_read_at_submission':False,
 'vint_grant_bearing_alignment':'first_certified_bounded',
 'jobs':{
  'navdp_smoke':int(nav_smoke),'vint_smoke':int(vint_smoke),
  'navdp_formal':int(nav_formal),'vint_formal':int(vint_formal),
  'navdp_aggregate':int(nav_aggregate),'navdp_verify':int(nav_verify),
  'vint_aggregate':int(vint_aggregate),'vint_verify':int(vint_verify),
  'controller_portability_seal':int(seal),
 },
}
open(path,'x').write(json.dumps(payload,indent=2,sort_keys=True)+'\n')
print(json.dumps(payload,indent=2,sort_keys=True))
PY
scp -q -o BatchMode=yes -o ControlMaster=no -o ControlPath="${SSH_CONTROL_PATH}" \
  "${ROOT}/${receipt}" "${SSH_ALIAS}:${run_root}/submission.json"
remote "sha256sum '${run_root}/submission.json' >'${run_root}/submission.json.sha256' && chmod a-w '${run_root}/submission.json' '${run_root}/submission.json.sha256'"
printf 'RUN_ROOT=%s\nTASK_ROOT=%s\nNAVDP_SMOKE=%s\nVINT_SMOKE=%s\nNAVDP_FORMAL=%s\nVINT_FORMAL=%s\nNAVDP_VERIFY=%s\nVINT_VERIFY=%s\nSEAL=%s\n' \
  "${run_root}" "${task_root}" "${nav_smoke}" "${vint_smoke}" \
  "${nav_formal}" "${vint_formal}" "${nav_verify}" "${vint_verify}" \
  "${seal_job}"
