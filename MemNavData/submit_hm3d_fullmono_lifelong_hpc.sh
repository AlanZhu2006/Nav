#!/usr/bin/env bash
# Freeze, audit, upload, and submit the actual-full-mono lifelong HM3D DAG.
set -euo pipefail
umask 0022

ROOT=${ROOT:-/home/asus/Research/Nav-graph-blind}
POWER_EXPANSION=${POWER_EXPANSION:-0}
DRY_RUN=${DRY_RUN:-0}
SSH_ALIAS=${SSH_ALIAS:-alantorch}
LOCAL_PY=${LOCAL_PY:-/home/asus/miniconda3/envs/memnav/bin/python}
LOCAL_HAB_PY=${LOCAL_HAB_PY:-/home/asus/miniconda3/envs/habitat/bin/python}
REMOTE_BUNDLES=/scratch/yz11502/Research/Nav-axis-uturn-source-bundles
REMOTE_RESULTS=/scratch/yz11502/Research/Nav-axis-uturn-results/hm3d_fullmono_lifelong_20260824
PARENT_ROOT=/scratch/yz11502/Research/Nav-axis-uturn-results/hm3d_fresh_fullmono_mixed_role_20260820/formal_20260820T143609Z_e6dd44c6
PARENT_MANIFEST=${PARENT_ROOT}/sealed_inputs/parent_manifest.json
PARENT_POPULATION_RECEIPT=${PARENT_ROOT}/benchmarks/population_receipt.json
EXPECTED_PARENT_MANIFEST_SHA=a96a0b96fab7b7b47709b36cb8eeb9410b42b09f095f87ef01304a68de716dd5
EXPECTED_PARENT_POPULATION_SHA=4dd6b8dcb759dff1c0835bef8e755e7291a5f049c9adec88b954d1fda62e30d5
BASE_SOURCE_ROOT=/scratch/yz11502/Research/Nav-axis-uturn-source-bundles/final14_mono_factorial_5690569a4373f2d2
BASE_RECEIPT=${BASE_SOURCE_ROOT}/source_inputs.sha256
EXPECTED_BASE_RECEIPT_SHA=5690569a4373f2d2768671418f0c604c4a03aa4b0ffe01baf70b288af03ba216
DEPENDENCY_RECEIPT_SOURCE=/scratch/yz11502/Research/Nav-axis-uturn-results/shared_online_double_revisit_fresh_20260813/double_revisit_fresh40_20260813T200121Z/dependency_receipt.json
EXPECTED_DEPENDENCY_RECEIPT_SHA=4eb0ca6479a26f8e04f85a31d906cee4e68b1785f66cfd3ac23bf65424d36e5e
HAB_REQUESTS_VENDOR=/scratch/lg154/conda-envs/habitat/lib/python3.9/site-packages/pip/_vendor
EXPECTED_HAB_REQUESTS_VERSION=2.32.4
EXPECTED_HAB_REQUESTS_INIT_BYTES=5057
EXPECTED_HAB_REQUESTS_INIT_SHA=1e507f1f386bcc6b5f0ff69a614c14875cd65cb67be7f6022f28adef9774573f
EXPECTED_HAB_REQUESTS_VERSION_BYTES=435
EXPECTED_HAB_REQUESTS_VERSION_SHA=143abaf3563712f063743a7952aa65319dbcb934d894cfc989bd2c015f8da577
if [[ "${POWER_EXPANSION}" == 1 ]]; then
  PROTOCOL_REL=MemNavData/hm3d_fullmono_lifelong_power_expansion_protocol_20260825.json
else
  PROTOCOL_REL=MemNavData/hm3d_fullmono_lifelong_protocol_20260824.json
fi
CONSTRUCT_CONCURRENCY=${CONSTRUCT_CONCURRENCY:-4}
B_CONCURRENCY=${B_CONCURRENCY:-4}
PREFIX_CONCURRENCY=${PREFIX_CONCURRENCY:-4}
EVAL_CONCURRENCY=${EVAL_CONCURRENCY:-4}
SSH_CONTROL_PATH=${SSH_CONTROL_PATH:-$(ssh -G "${SSH_ALIAS}" 2>/dev/null | awk '$1=="controlpath"{value=$2} END{print value}')}

cd "${ROOT}"
fail() { echo "ABORT: $*" >&2; exit 2; }
remote() {
  # `remote` is frequently called inside command substitution/pipelines.  A
  # forced PTY without `-n` can then receive SIGTTIN and stop while trying to
  # read the submitter's controlling terminal.  Keep the PTY (the documented
  # reliable path for this shared master), but make every scripted call
  # explicitly stdin-independent.
  timeout 180 ssh -n -tt -o BatchMode=yes -o ControlMaster=no \
    -S "${SSH_CONTROL_PATH}" "${SSH_ALIAS}" "$@"
}
job_id() {
  tr -d '\r' | awk -F';' '/^[0-9]+(;|$)/ {print $1; exit}'
}
upload_bundle() {
  local source=$1 destination=$2 attempt
  for attempt in 1 2 3; do
    if timeout 240 rsync -a --partial \
      --chmod=Du=rwx,Dgo=rx,Fu=rw,Fgo=r \
      -e "ssh -o BatchMode=yes -o ControlMaster=no -S ${SSH_CONTROL_PATH}" \
      "${source}/" "${SSH_ALIAS}:${destination}/"; then
      return 0
    fi
    echo "rsync attempt ${attempt} failed; retrying the same immutable stage" >&2
  done
  return 1
}

[[ -S "${SSH_CONTROL_PATH}" ]] || fail "authoritative shared SSH socket missing: ${SSH_CONTROL_PATH}"
[[ "${POWER_EXPANSION}" =~ ^[01]$ ]] || fail "POWER_EXPANSION must be 0 or 1"
[[ "${DRY_RUN}" =~ ^[01]$ ]] || fail "DRY_RUN must be 0 or 1"
timeout 15 ssh -O check -S "${SSH_CONTROL_PATH}" "${SSH_ALIAS}" >/dev/null 2>&1 \
  || fail "shared SSH socket has no responsive master: ${SSH_CONTROL_PATH}"
for value in "${CONSTRUCT_CONCURRENCY}" "${B_CONCURRENCY}" \
             "${PREFIX_CONCURRENCY}" "${EVAL_CONCURRENCY}"; do
  [[ "${value}" =~ ^[1-9][0-9]*$ ]] || fail "invalid concurrency ${value}"
done
[[ -x "${LOCAL_PY}" && -x "${LOCAL_HAB_PY}" ]] || fail "local environments missing"

required_shell=(
  MemNavData/run_hm3d_fullmono_server_scene.sh
  MemNavData/run_cec_controller_portability_smoke_local.sh
  MemNavData/slurm_hm3d_fullmono_lifelong_construct_ab.sbatch
  MemNavData/slurm_hm3d_fullmono_lifelong_finalize_ab.sbatch
  MemNavData/slurm_hm3d_fullmono_lifelong_collect_b.sbatch
  MemNavData/slurm_hm3d_fullmono_lifelong_construct_prefix.sbatch
  MemNavData/slurm_hm3d_fullmono_lifelong_finalize_population.sbatch
  MemNavData/slurm_hm3d_fullmono_lifelong_eval.sbatch
  MemNavData/slurm_hm3d_fullmono_lifelong_analysis.sbatch
  MemNavData/slurm_hm3d_fullmono_shared_c_arm.sbatch
  MemNavData/slurm_hm3d_fullmono_shared_c_finalize.sbatch
  MemNavData/slurm_hm3d_fullmono_shared_c_analysis.sbatch
  MemNavData/slurm_hm3d_fullmono_shared_c_deferred.sbatch
  MemNavData/submit_hm3d_fullmono_lifelong_hpc.sh
)
required_static=(
  "${PROTOCOL_REL}"
  MemNavData/hm3d_fullmono_lifelong_protocol_20260824.json
  MemNavData/HM3D_FULLMONO_LIFELONG_ACCUMULATION_PROTOCOL_20260824.md
)
for path in "${required_shell[@]}" "${required_static[@]}"; do
  [[ -f "${path}" && ! -L "${path}" ]] || fail "missing physical input ${path}"
done

"${LOCAL_PY}" -m json.tool "${PROTOCOL_REL}" >/dev/null
"${LOCAL_PY}" -m py_compile \
  MemNavData/hm3d_fullmono_lifelong.py \
  MemNavData/construct_hm3d_fullmono_lifelong_ab.py \
  MemNavData/finalize_hm3d_fullmono_lifelong_ab.py \
  MemNavData/collect_hm3d_fullmono_lifelong_b.py \
  MemNavData/construct_hm3d_fullmono_lifelong_prefix.py \
  MemNavData/finalize_hm3d_fullmono_lifelong_population.py \
  MemNavData/eval_hm3d_fullmono_lifelong.py \
  MemNavData/aggregate_hm3d_fullmono_lifelong.py \
  MemNavData/independent_verify_hm3d_fullmono_lifelong.py \
  MemNavData/lifelong_shared_c_contract.py \
  MemNavData/collect_hm3d_lifelong_shared_c.py \
  MemNavData/eval_hm3d_lifelong_shared_c_b2.py \
  MemNavData/finalize_lifelong_shared_c_population.py \
  MemNavData/aggregate_lifelong_shared_c_b2.py \
  MemNavData/independent_verify_lifelong_shared_c_b2.py \
  MemNavData/eval_shared_online_role_pairs.py \
  MemNavData/eval_3leg_habitat.py \
  MemNavData/cec_controller_portability_hub.py \
  NavDP/baselines/memnav/memnav_server.py \
  NavDP/baselines/navdp/navdp_server.py
PYTHONPATH="${ROOT}:${ROOT}/MemNavData" "${LOCAL_PY}" -m pytest -q \
  MemNavData/test_hm3d_fullmono_lifelong.py \
  MemNavData/test_lifelong_forced_reject_contract.py \
  MemNavData/test_lifelong_shared_c_contract.py \
  MemNavData/test_policy_agent_graph.py \
  MemNavData/test_controller_portability_contract.py \
  MemNavData/test_cec_controller_portability_hub.py
PYTHONPATH="${ROOT}:${ROOT}/MemNavData" "${LOCAL_HAB_PY}" -m unittest -q \
  MemNavData.test_final14_role_pair_construction \
  MemNavData.test_hm3d_fullmono_lifelong
bash -n "${required_shell[@]}"

staging=$(mktemp -d /tmp/h3life_bundle.XXXXXX)
cleanup() { rm -rf -- "${staging}"; }
trap cleanup EXIT
files=("${required_shell[@]}" "${required_static[@]}")
while IFS= read -r path; do files+=("${path}"); done < <(
  find MemNavData -maxdepth 1 -type f -name '*.py' -print | sort)
while IFS= read -r path; do files+=("${path}"); done < <(
  find NavDP/baselines/memnav NavDP/baselines/navdp -maxdepth 1 \
    -type f -name '*.py' -print | sort)
while IFS= read -r path; do files+=("${path}"); done < <(
  find NavDP/baselines/navdp/depth_anything/depth_anything_v2 -type f \
    ! -path '*/__pycache__/*' ! -name '*.pyc' -print | sort)
printf '%s\n' "${files[@]}" | sort -u >"${staging}/file_list.txt"
while IFS= read -r path; do
  [[ -f "${path}" && ! -L "${path}" ]] || fail "bundle input changed: ${path}"
  mkdir -p "${staging}/root/$(dirname "${path}")"
  cp -p -- "${path}" "${staging}/root/${path}"
done <"${staging}/file_list.txt"

(
  # Run from inside the staged root so an unbundled workspace module cannot
  # leak through Python's implicit current-working-directory entry.
  cd "${staging}/root"
  PYTHONPATH="${staging}/root:${staging}/root/MemNavData" \
    "${LOCAL_PY}" -m pytest -q \
      MemNavData/test_hm3d_fullmono_lifelong.py \
      MemNavData/test_lifelong_forced_reject_contract.py \
      MemNavData/test_lifelong_shared_c_contract.py \
      MemNavData/test_policy_agent_graph.py \
      MemNavData/test_controller_portability_contract.py \
      MemNavData/test_cec_controller_portability_hub.py
)
(
  cd "${staging}/root"
  find . -type f ! -name SOURCE_BUNDLE.sha256 -print0 | sort -z | \
    xargs -0 sha256sum >SOURCE_BUNDLE.sha256
  sha256sum -c --quiet SOURCE_BUNDLE.sha256
)
task_receipt_sha=$(sha256sum "${staging}/root/SOURCE_BUNDLE.sha256" | awk '{print $1}')
bundle_key=${task_receipt_sha:0:16}
task_root=${REMOTE_BUNDLES}/hm3d_fullmono_lifelong_${bundle_key}
task_stage=${task_root}.partial.$$
run_tag=formal_$(date -u +%Y%m%dT%H%M%SZ)_${bundle_key:0:8}
run_root=${REMOTE_RESULTS}/${run_tag}
smoke_root=${REMOTE_RESULTS}/${run_tag}_smoke

if [[ "${DRY_RUN}" == 1 ]]; then
  printf 'DRY_RUN_RUN_ROOT=%s\nDRY_RUN_SMOKE_ROOT=%s\nDRY_RUN_TASK_ROOT=%s\nDRY_RUN_SOURCE_RECEIPT_SHA=%s\n' \
    "${run_root}" "${smoke_root}" "${task_root}" "${task_receipt_sha}"
  exit 0
fi

remote_identity=$(remote 'id -un' | tr -d '\r')
[[ "${remote_identity}" == yz11502 ]] || fail "wrong remote identity: ${remote_identity}"
remote "set -euo pipefail
test \"\$(sha256sum '${PARENT_MANIFEST}' | awk '{print \$1}')\" = '${EXPECTED_PARENT_MANIFEST_SHA}'
test \"\$(sha256sum '${PARENT_POPULATION_RECEIPT}' | awk '{print \$1}')\" = '${EXPECTED_PARENT_POPULATION_SHA}'
test \"\$(sha256sum '${BASE_RECEIPT}' | awk '{print \$1}')\" = '${EXPECTED_BASE_RECEIPT_SHA}'
test \"\$(sha256sum '${DEPENDENCY_RECEIPT_SOURCE}' | awk '{print \$1}')\" = '${EXPECTED_DEPENDENCY_RECEIPT_SHA}'
cd '${BASE_SOURCE_ROOT}' && sha256sum -c --quiet '${BASE_RECEIPT}'
test -d '${PARENT_ROOT}/construction/scenes'
test -x /scratch/lg154/conda-envs/habitat/bin/python
test -x /scratch/lg154/conda-envs/memnav/bin/python
test -r /share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif
test -r '${HAB_REQUESTS_VENDOR}/requests/__init__.py'
test -r '${HAB_REQUESTS_VENDOR}/requests/__version__.py'
test \"\$(stat -c '%s' '${HAB_REQUESTS_VENDOR}/requests/__init__.py')\" = '${EXPECTED_HAB_REQUESTS_INIT_BYTES}'
test \"\$(sha256sum '${HAB_REQUESTS_VENDOR}/requests/__init__.py' | awk '{print \$1}')\" = '${EXPECTED_HAB_REQUESTS_INIT_SHA}'
test \"\$(stat -c '%s' '${HAB_REQUESTS_VENDOR}/requests/__version__.py')\" = '${EXPECTED_HAB_REQUESTS_VERSION_BYTES}'
test \"\$(sha256sum '${HAB_REQUESTS_VENDOR}/requests/__version__.py' | awk '{print \$1}')\" = '${EXPECTED_HAB_REQUESTS_VERSION_SHA}'
test \"\$(/scratch/lg154/conda-envs/memnav/bin/python -c \"import json; print(len(json.load(open('${PARENT_MANIFEST}'))['scenes']))\")\" = 54"

if remote "test -d '${task_root}'"; then
  remote "test \"\$(sha256sum '${task_root}/SOURCE_BUNDLE.sha256' | awk '{print \$1}')\" = '${task_receipt_sha}' && cd '${task_root}' && sha256sum -c --quiet SOURCE_BUNDLE.sha256"
else
  remote "test ! -e '${task_stage}' && mkdir -p '${task_stage}'"
  upload_bundle "${staging}/root" "${task_stage}" || fail "immutable bundle upload failed"
  remote "cd '${task_stage}' && sha256sum -c --quiet SOURCE_BUNDLE.sha256 && chmod -R a-w '${task_stage}' && mv '${task_stage}' '${task_root}'"
fi

remote "set -euo pipefail
test ! -e '${run_root}'
test ! -e '${smoke_root}'
mkdir -p '${run_root}/sealed_inputs' '${run_root}/logs' '${smoke_root}/logs' /scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs
cp '${PARENT_MANIFEST}' '${run_root}/sealed_inputs/parent_manifest.json'
cp '${PARENT_POPULATION_RECEIPT}' '${run_root}/sealed_inputs/parent_population_receipt.json'
cp '${DEPENDENCY_RECEIPT_SOURCE}' '${run_root}/sealed_inputs/dependency_receipt.json'
printf '%s  %s\n' '${EXPECTED_PARENT_MANIFEST_SHA}' parent_manifest.json >'${run_root}/sealed_inputs/parent_manifest.json.sha256'
printf '%s  %s\n' '${EXPECTED_PARENT_POPULATION_SHA}' parent_population_receipt.json >'${run_root}/sealed_inputs/parent_population_receipt.json.sha256'
printf '%s  %s\n' '${EXPECTED_DEPENDENCY_RECEIPT_SHA}' dependency_receipt.json >'${run_root}/sealed_inputs/dependency_receipt.json.sha256'
cd '${run_root}/sealed_inputs' && sha256sum -c --quiet *.sha256
chmod -R a-w '${run_root}/sealed_inputs'"

protocol=${task_root}/${PROTOCOL_REL}
task_receipt=${task_root}/SOURCE_BUNDLE.sha256
dependency_receipt=${run_root}/sealed_inputs/dependency_receipt.json

# Exact production-container import and argument-contract gate.
remote "singularity exec -B /scratch/lg154 -B /scratch/yz11502 /share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH='${task_root}:${task_root}/MemNavData:${BASE_SOURCE_ROOT}:${BASE_SOURCE_ROOT}/MemNavData:${HAB_REQUESTS_VENDOR}' /scratch/lg154/conda-envs/habitat/bin/python -c 'import requests,sys; assert requests.__version__ == sys.argv[1]' '${EXPECTED_HAB_REQUESTS_VERSION}'"
remote "singularity exec -B /scratch/lg154 -B /scratch/yz11502 /share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH='${task_root}:${task_root}/MemNavData:${BASE_SOURCE_ROOT}:${BASE_SOURCE_ROOT}/MemNavData:${HAB_REQUESTS_VENDOR}' /scratch/lg154/conda-envs/habitat/bin/python -m unittest -q MemNavData.test_final14_role_pair_construction MemNavData.test_hm3d_fullmono_lifelong MemNavData.test_lifelong_shared_c_contract"
remote "singularity exec -B /scratch/lg154 -B /scratch/yz11502 /share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH='${task_root}:${task_root}/MemNavData:${BASE_SOURCE_ROOT}:${BASE_SOURCE_ROOT}/MemNavData' /scratch/lg154/conda-envs/memnav/bin/python -m unittest -q MemNavData.test_policy_agent_graph MemNavData.test_lifelong_shared_c_contract"
remote "singularity exec -B /scratch/lg154 -B /scratch/yz11502 /share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH='${task_root}:${task_root}/MemNavData:${BASE_SOURCE_ROOT}:${BASE_SOURCE_ROOT}/MemNavData:${HAB_REQUESTS_VENDOR}' /scratch/lg154/conda-envs/habitat/bin/python '${task_root}/MemNavData/eval_hm3d_fullmono_lifelong.py' --episode_root /contract/dry/episodes --scene /contract/dry/scene.glb --contract_dry_run --server_backend cec_portability --navdp_depth_source monocular_sidecar --max_steps 600 --exec_horizon 8 --trajectory_selector server --leg1_mode shared_trace --leg1_goal_source own --deterministic_plan_seeds --terminal_uturn off --terminal_visual_refine off --retrieval_override off --hybrid_route phase --revisit_controller navdp_mixed --revisit_adapter legacy_metric --navdp_goal_switch_reset before_c --shared_leg1_trace_root /contract/dry/run --double_revisit_c_history initial_leg_only --shared_online_nnr_arm cec_portability --lifelong_history_scope all_prior"

common="ALL,TASK_ROOT=${task_root},BASE_SOURCE_ROOT=${BASE_SOURCE_ROOT},RUN_ROOT=${run_root},PARENT_ROOT=${PARENT_ROOT},PROTOCOL=${protocol},TASK_RECEIPT=${task_receipt},EXPECTED_TASK_RECEIPT_SHA=${task_receipt_sha},BASE_RECEIPT=${BASE_RECEIPT},EXPECTED_BASE_RECEIPT_SHA=${EXPECTED_BASE_RECEIPT_SHA},DEPENDENCY_RECEIPT=${dependency_receipt},EXPECTED_DEPENDENCY_RECEIPT_SHA=${EXPECTED_DEPENDENCY_RECEIPT_SHA},EXPECTED_HAB_REQUESTS_VERSION=${EXPECTED_HAB_REQUESTS_VERSION},EXPECTED_HAB_REQUESTS_INIT_BYTES=${EXPECTED_HAB_REQUESTS_INIT_BYTES},EXPECTED_HAB_REQUESTS_INIT_SHA=${EXPECTED_HAB_REQUESTS_INIT_SHA},EXPECTED_HAB_REQUESTS_VERSION_BYTES=${EXPECTED_HAB_REQUESTS_VERSION_BYTES},EXPECTED_HAB_REQUESTS_VERSION_SHA=${EXPECTED_HAB_REQUESTS_VERSION_SHA}"
build=${task_root}/MemNavData/slurm_hm3d_fullmono_lifelong_construct_ab.sbatch
seal_ab=${task_root}/MemNavData/slurm_hm3d_fullmono_lifelong_finalize_ab.sbatch
collect_b=${task_root}/MemNavData/slurm_hm3d_fullmono_lifelong_collect_b.sbatch
prefix=${task_root}/MemNavData/slurm_hm3d_fullmono_lifelong_construct_prefix.sbatch
seal_population=${task_root}/MemNavData/slurm_hm3d_fullmono_lifelong_finalize_population.sbatch
evaluate=${task_root}/MemNavData/slurm_hm3d_fullmono_lifelong_eval.sbatch
analysis=${task_root}/MemNavData/slurm_hm3d_fullmono_lifelong_analysis.sbatch
shared_c_arm=${task_root}/MemNavData/slurm_hm3d_fullmono_shared_c_arm.sbatch
shared_c_finalize=${task_root}/MemNavData/slurm_hm3d_fullmono_shared_c_finalize.sbatch
shared_c_analysis=${task_root}/MemNavData/slurm_hm3d_fullmono_shared_c_analysis.sbatch
shared_c_deferred=${task_root}/MemNavData/slurm_hm3d_fullmono_shared_c_deferred.sbatch

remote "sbatch --test-only --array=0 --export='${common}' '${build}' >/dev/null"
remote "sbatch --test-only --export='${common}' '${seal_ab}' >/dev/null"
remote "sbatch --test-only --array=0 --export='${common},MAX_STEPS=600' '${collect_b}' >/dev/null"
remote "sbatch --test-only --array=0 --export='${common}' '${prefix}' >/dev/null"
remote "sbatch --test-only --export='${common}' '${seal_population}' >/dev/null"
remote "sbatch --test-only --array=0 --export='${common},OUTPUT_ROOT=${smoke_root},MAX_STEPS=80' '${evaluate}' >/dev/null"
remote "sbatch --test-only --array=0 --export='${common},OUTPUT_ROOT=${run_root},MAX_STEPS=600' '${evaluate}' >/dev/null"
remote "sbatch --test-only --export='${common},MODE=aggregate' '${analysis}' >/dev/null"
remote "sbatch --test-only --export='${common},MODE=verify' '${analysis}' >/dev/null"
remote "sbatch --test-only --array=0 --export='${common},STAGE=collect' '${shared_c_arm}' >/dev/null"
remote "sbatch --test-only --export='${common}' '${shared_c_finalize}' >/dev/null"
remote "sbatch --test-only --array=0 --export='${common},STAGE=evaluate' '${shared_c_arm}' >/dev/null"
remote "sbatch --test-only --export='${common},MODE=aggregate' '${shared_c_analysis}' >/dev/null"
remote "sbatch --test-only --export='${common},MODE=verify' '${shared_c_analysis}' >/dev/null"
remote "sbatch --test-only --export='${common},DEFERRED_MODE=collect,DEFERRED_SCRIPT=${shared_c_deferred},EXPECTED_DEFERRED_SCRIPT_SHA=dry,EVAL_CONCURRENCY=${EVAL_CONCURRENCY},UPSTREAM_SEAL_JOB_ID=1,SMOKE_ROOT=${smoke_root}' '${shared_c_deferred}' >/dev/null"

build_time=
if [[ "${POWER_EXPANSION}" == 1 ]]; then build_time="--time=01:30:00"; fi
build_id=$(remote "sbatch --parsable --qos=gpu48 ${build_time} --array=0-53%${CONSTRUCT_CONCURRENCY} --export='${common}' '${build}'" | job_id)
[[ "${build_id}" =~ ^[0-9]+$ ]] || fail "bad A/B construction job id"
seal_ab_id=$(remote "sbatch --parsable --dependency=afterok:${build_id} --kill-on-invalid-dep=yes --export='${common}' '${seal_ab}'" | job_id)
[[ "${seal_ab_id}" =~ ^[0-9]+$ ]] || fail "bad A/B seal job id"
collect_time=
if [[ "${POWER_EXPANSION}" == 1 ]]; then collect_time="--time=03:00:00"; fi
collect_b_id=$(remote "sbatch --parsable --qos=gpu48 ${collect_time} --array=0-53%${B_CONCURRENCY} --dependency=afterok:${seal_ab_id} --kill-on-invalid-dep=yes --export='${common},MAX_STEPS=600' '${collect_b}'" | job_id)
[[ "${collect_b_id}" =~ ^[0-9]+$ ]] || fail "bad factual-B job id"
prefix_max=129
if [[ "${POWER_EXPANSION}" == 1 ]]; then prefix_max=259; fi
prefix_id=$(remote "sbatch --parsable --qos=gpu48 --array=0-${prefix_max}%${PREFIX_CONCURRENCY} --dependency=afterok:${collect_b_id} --kill-on-invalid-dep=yes --export='${common}' '${prefix}'" | job_id)
[[ "${prefix_id}" =~ ^[0-9]+$ ]] || fail "bad prefix construction job id"
seal_population_id=$(remote "sbatch --parsable --dependency=afterok:${prefix_id} --kill-on-invalid-dep=yes --export='${common}' '${seal_population}'" | job_id)
[[ "${seal_population_id}" =~ ^[0-9]+$ ]] || fail "bad population seal job id"
if [[ "${POWER_EXPANSION}" == 1 ]]; then
  deferred_sha=$(sha256sum "${ROOT}/MemNavData/slurm_hm3d_fullmono_shared_c_deferred.sbatch" | awk '{print $1}')
  deferred_id=$(remote "sbatch --parsable --dependency=afterok:${seal_population_id} --kill-on-invalid-dep=yes --export='${common},DEFERRED_MODE=collect,DEFERRED_SCRIPT=${shared_c_deferred},EXPECTED_DEFERRED_SCRIPT_SHA=${deferred_sha},EVAL_CONCURRENCY=${EVAL_CONCURRENCY},UPSTREAM_SEAL_JOB_ID=${seal_population_id},SMOKE_ROOT=${smoke_root}' '${shared_c_deferred}'" | job_id)
  [[ "${deferred_id}" =~ ^[0-9]+$ ]] || fail "bad shared-C deferred launcher id"
  shared_c_collect_id=0
  shared_c_seal_id=0
  smoke_id=0
  eval_id=0
  aggregate_id=0
  verify_id=0
else
  smoke_id=$(remote "sbatch --parsable --qos=gpu48 --array=0 --dependency=afterok:${seal_population_id} --kill-on-invalid-dep=yes --export='${common},OUTPUT_ROOT=${smoke_root},MAX_STEPS=80' '${evaluate}'" | job_id)
  [[ "${smoke_id}" =~ ^[0-9]+$ ]] || fail "bad remote smoke job id"
  eval_id=$(remote "sbatch --parsable --qos=gpu48 --array=0-129%${EVAL_CONCURRENCY} --dependency=afterok:${smoke_id} --kill-on-invalid-dep=yes --export='${common},OUTPUT_ROOT=${run_root},MAX_STEPS=600' '${evaluate}'" | job_id)
  [[ "${eval_id}" =~ ^[0-9]+$ ]] || fail "bad formal evaluation job id"
  aggregate_id=$(remote "sbatch --parsable --dependency=afterok:${eval_id} --kill-on-invalid-dep=yes --export='${common},MODE=aggregate' '${analysis}'" | job_id)
  [[ "${aggregate_id}" =~ ^[0-9]+$ ]] || fail "bad aggregate job id"
  verify_id=$(remote "sbatch --parsable --dependency=afterok:${aggregate_id} --kill-on-invalid-dep=yes --export='${common},MODE=verify' '${analysis}'" | job_id)
  [[ "${verify_id}" =~ ^[0-9]+$ ]] || fail "bad verifier job id"
  shared_c_collect_id=0
  shared_c_seal_id=0
  deferred_id=0
fi

receipt=MemNavData/HM3D_FULLMONO_LIFELONG_SUBMISSION_RECEIPT_${run_tag}.json
"${LOCAL_PY}" - "${receipt}" "${run_root}" "${smoke_root}" "${task_root}" \
  "${task_receipt_sha}" "${build_id}" "${seal_ab_id}" "${collect_b_id}" \
  "${prefix_id}" "${seal_population_id}" "${smoke_id}" "${eval_id}" \
  "${aggregate_id}" "${verify_id}" "${POWER_EXPANSION}" \
  "${shared_c_collect_id}" "${shared_c_seal_id}" "${deferred_id}" <<'PY'
import json,sys
(path,run,smoke,bundle,sha,build,seal_ab,collect_b,prefix,seal_population,
 smoke_job,evaluation,aggregate,verify,power,shared_collect,shared_seal,deferred)=sys.argv[1:]
payload={
  "schema_version":"hm3d_fullmono_lifelong_submission_v2_20260825" if int(power) else "hm3d_fullmono_lifelong_submission_v1_20260824",
  "scope":"result-blind actual-full-mono lifelong power expansion" if int(power) else "consumed-scene actual-full-mono lifelong accumulation confirmation",
  "run_root":run,"smoke_root":smoke,"task_bundle":bundle,
  "task_receipt_sha256":sha,"parent_scene_count":54,
  "maximum_AB_histories":260 if int(power) else 130,
  "maximum_paired_evaluation_tasks":260 if int(power) else 130,
  "maximum_logical_query_arms":780 if int(power) else 390,
  "primary_pair_same_loaded_GPU_process":True,
  "query_outcomes_read_at_submission":False,
  "fresh_scene_generalization_claim":False,
  "jobs":{
    "construct_AB_array":int(build),"seal_AB_population":int(seal_ab),
    "collect_factual_B_array":int(collect_b),
    "construct_actual_prefix_array":int(prefix),
    "seal_query_population":int(seal_population),
    "remote_true_stack_smoke":int(smoke_job),
    "collect_shared_C_array":int(shared_collect),
    "seal_shared_C_population":int(shared_seal),
    "deferred_shared_C_launcher":int(deferred),
    "formal_three_arm_evaluation":int(evaluation),
    "aggregate":int(aggregate),"independent_verification":int(verify),
  },
}
open(path,"x").write(json.dumps(payload,indent=2,sort_keys=True)+"\n")
print(json.dumps(payload,indent=2,sort_keys=True))
PY
timeout 120 scp -q -o BatchMode=yes -o ControlMaster=no \
  -o ControlPath="${SSH_CONTROL_PATH}" "${receipt}" \
  "${SSH_ALIAS}:${run_root}/submission.json" || fail "submission receipt upload failed"
remote "sha256sum '${run_root}/submission.json' >'${run_root}/submission.json.sha256'"
if [[ "${POWER_EXPANSION}" == 1 ]]; then
  queue_ids=${build_id},${seal_ab_id},${collect_b_id},${prefix_id},${seal_population_id},${deferred_id}
else
  queue_ids=${build_id},${seal_ab_id},${collect_b_id},${prefix_id},${seal_population_id},${smoke_id},${eval_id},${aggregate_id},${verify_id}
fi
remote "squeue -j '${queue_ids}' -o '%.18i %.22j %.2t %.12M %.28R'"
printf 'RUN_ROOT=%s\nSMOKE_ROOT=%s\nTASK_ROOT=%s\nBUILD=%s\nSEAL_AB=%s\nCOLLECT_B=%s\nPREFIX=%s\nSEAL_POPULATION=%s\nSMOKE=%s\nEVAL=%s\nAGGREGATE=%s\nVERIFY=%s\n' \
  "${run_root}" "${smoke_root}" "${task_root}" "${build_id}" \
  "${seal_ab_id}" "${collect_b_id}" "${prefix_id}" \
  "${seal_population_id}" "${smoke_id}" "${eval_id}" \
  "${aggregate_id}" "${verify_id}"
if [[ "${POWER_EXPANSION}" == 1 ]]; then
  printf 'DEFERRED_SHARED_C_LAUNCHER=%s\n' "${deferred_id}"
fi
