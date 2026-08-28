#!/usr/bin/env bash
# Freeze and submit construction only. No controller rollout is in this DAG.
set -euo pipefail
umask 0022

ROOT=${ROOT:-/home/asus/Research/Nav-graph-blind}
SSH_ALIAS=${SSH_ALIAS:-alantorch}
LOCAL_MEMNAV_PY=/home/asus/miniconda3/envs/memnav/bin/python
LOCAL_HAB_PY=/home/asus/miniconda3/envs/habitat/bin/python
REMOTE_BUNDLES=/scratch/yz11502/Research/Nav-axis-uturn-source-bundles
REMOTE_RESULTS=/scratch/yz11502/Research/Nav-axis-uturn-results/hm3d_table1_fresh_query_reserve_20260829
SOURCE_RUN_ROOT=/scratch/yz11502/Research/Nav-axis-uturn-results/hm3d_fresh_fullmono_mixed_role_20260820/formal_20260820T143609Z_e6dd44c6
BASE_SOURCE_ROOT=/scratch/yz11502/Research/Nav-axis-uturn-source-bundles/final14_mono_factorial_5690569a4373f2d2
BASE_RECEIPT=${BASE_SOURCE_ROOT}/source_inputs.sha256
EXPECTED_BASE_RECEIPT_SHA=5690569a4373f2d2768671418f0c604c4a03aa4b0ffe01baf70b288af03ba216
CONSTRUCT_CONCURRENCY=${CONSTRUCT_CONCURRENCY:-2}
SSH_CONTROL_PATH=${SSH_CONTROL_PATH:-$(ssh -G "${SSH_ALIAS}" 2>/dev/null | awk '$1=="controlpath"{value=$2} END{print value}')}
cd "${ROOT}"

fail() { echo "ABORT: $*" >&2; exit 2; }
remote() {
  ssh -tt -o BatchMode=yes -o ControlMaster=no \
    -S "${SSH_CONTROL_PATH}" "${SSH_ALIAS}" "$@"
}
[[ -S "${SSH_CONTROL_PATH}" ]] || fail "authoritative SSH master missing"
[[ "${CONSTRUCT_CONCURRENCY}" =~ ^[1-9][0-9]*$ ]] || \
  fail "invalid construction concurrency"

files=(
  MemNavData/HM3D_TABLE1_FRESH_QUERY_RESERVE_PROTOCOL_20260829.md
  MemNavData/hm3d_table1_fresh_query_reserve_protocol_20260829.json
  MemNavData/hm3d_table1_fresh_query_contract.py
  MemNavData/build_hm3d_table1_fresh_query_scene.py
  MemNavData/build_final14_role_pair_scene.py
  MemNavData/final14_role_pair_contract.py
  MemNavData/shared_online_role_pair_contract.py
  MemNavData/audit_shared_online_role_pairs.py
  MemNavData/finalize_hm3d_table1_fresh_query_reserve.py
  MemNavData/verify_hm3d_table1_fresh_query_reserve.py
  MemNavData/test_hm3d_table1_fresh_query_reserve.py
  MemNavData/slurm_hm3d_table1_fresh_query_construct.sbatch
  MemNavData/slurm_hm3d_table1_fresh_query_analysis.sbatch
  MemNavData/submit_hm3d_table1_fresh_query_reserve_hpc.sh
)
for file in "${files[@]}"; do
  [[ -f "${file}" && ! -L "${file}" ]] || fail "missing physical ${file}"
done

"${LOCAL_MEMNAV_PY}" -m json.tool \
  MemNavData/hm3d_table1_fresh_query_reserve_protocol_20260829.json >/dev/null
"${LOCAL_MEMNAV_PY}" -m py_compile \
  MemNavData/hm3d_table1_fresh_query_contract.py \
  MemNavData/finalize_hm3d_table1_fresh_query_reserve.py \
  MemNavData/verify_hm3d_table1_fresh_query_reserve.py
PYTHONPATH="${ROOT}:${ROOT}/MemNavData" "${LOCAL_MEMNAV_PY}" -m unittest \
  MemNavData.test_hm3d_table1_fresh_query_reserve
"${LOCAL_HAB_PY}" -m py_compile \
  MemNavData/build_final14_role_pair_scene.py \
  MemNavData/build_hm3d_table1_fresh_query_scene.py
PYTHONPATH="${ROOT}:${ROOT}/MemNavData" "${LOCAL_HAB_PY}" -m unittest \
  MemNavData.test_final14_role_pair_construction
bash -n \
  MemNavData/slurm_hm3d_table1_fresh_query_construct.sbatch \
  MemNavData/slurm_hm3d_table1_fresh_query_analysis.sbatch \
  MemNavData/submit_hm3d_table1_fresh_query_reserve_hpc.sh

staging=$(mktemp -d)
cleanup() { rm -rf -- "${staging}"; }
trap cleanup EXIT
for file in "${files[@]}"; do
  mkdir -p "${staging}/$(dirname "${file}")"
  cp --preserve=mode,timestamps "${file}" "${staging}/${file}"
done
(
  cd "${staging}"
  find . -type f ! -name SOURCE_BUNDLE.sha256 -print0 | sort -z | \
    xargs -0 sha256sum >SOURCE_BUNDLE.sha256
  sha256sum -c SOURCE_BUNDLE.sha256 >/dev/null
)
task_receipt_sha=$(sha256sum "${staging}/SOURCE_BUNDLE.sha256" | awk '{print $1}')
bundle_key=${task_receipt_sha:0:16}
task_root=${REMOTE_BUNDLES}/hm3d_table1_fresh_query_${bundle_key}
task_stage=${task_root}.partial.$$
run_tag=construction_$(date -u +%Y%m%dT%H%M%SZ)_${bundle_key:0:8}
run_root=${REMOTE_RESULTS}/${run_tag}
smoke_root=${REMOTE_RESULTS}/${run_tag}_smoke

echo '[stage] verify shared SSH identity'
remote_identity=$(remote 'id -un' | tr -d '\r')
[[ "${remote_identity}" == yz11502 ]] || fail "wrong remote identity"
remote "set -euo pipefail
test \"\$(sha256sum '${BASE_RECEIPT}' | awk '{print \$1}')\" = '${EXPECTED_BASE_RECEIPT_SHA}'
cd '${BASE_SOURCE_ROOT}' && sha256sum -c --quiet '${BASE_RECEIPT}'
test \"\$(sha256sum '${SOURCE_RUN_ROOT}/sealed_inputs/parent_manifest.json' | awk '{print \$1}')\" = a96a0b96fab7b7b47709b36cb8eeb9410b42b09f095f87ef01304a68de716dd5
test \"\$(sha256sum '${SOURCE_RUN_ROOT}/benchmarks/natural_direction/manifest.json' | awk '{print \$1}')\" = aada40d25d01e9385df3ffdcaf37847f471b63c7be785a704eade961346a50b0
test -x /scratch/lg154/conda-envs/habitat/bin/python
test -x /scratch/lg154/conda-envs/memnav/bin/python
test -r /share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif"

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

protocol=${task_root}/MemNavData/hm3d_table1_fresh_query_reserve_protocol_20260829.json
task_receipt=${task_root}/SOURCE_BUNDLE.sha256
echo '[stage] create isolated construction roots and seal source receipts'
remote "set -euo pipefail
test ! -e '${run_root}' && test ! -e '${smoke_root}'
mkdir -p '${run_root}/construction/scenes' '${run_root}/logs' '${run_root}/sealed_inputs' '${smoke_root}/construction/scenes' '${smoke_root}/logs' /scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs
cp '${protocol}' '${run_root}/sealed_inputs/'
sha256sum '${SOURCE_RUN_ROOT}/sealed_inputs/parent_manifest.json' '${SOURCE_RUN_ROOT}/benchmarks/natural_direction/manifest.json' >'${run_root}/sealed_inputs/source_inputs.sha256'
chmod -R a-w '${run_root}/sealed_inputs'"

echo '[stage] remote Habitat cold-import and unit-test gate (may be quiet for several minutes)'
remote "singularity exec --nv -B /scratch/lg154 -B /scratch/yz11502 /share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH='${task_root}:${task_root}/MemNavData:${BASE_SOURCE_ROOT}:${BASE_SOURCE_ROOT}/MemNavData' /scratch/lg154/conda-envs/habitat/bin/python -c 'import build_final14_role_pair_scene,build_hm3d_table1_fresh_query_scene' && singularity exec --nv -B /scratch/lg154 -B /scratch/yz11502 /share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH='${task_root}:${task_root}/MemNavData:${BASE_SOURCE_ROOT}:${BASE_SOURCE_ROOT}/MemNavData' /scratch/lg154/conda-envs/habitat/bin/python -m unittest MemNavData.test_hm3d_table1_fresh_query_reserve"

common="ALL,TASK_ROOT=${task_root},TASK_RECEIPT=${task_receipt},EXPECTED_TASK_RECEIPT_SHA=${task_receipt_sha},BASE_SOURCE_ROOT=${BASE_SOURCE_ROOT},BASE_RECEIPT=${BASE_RECEIPT},EXPECTED_BASE_RECEIPT_SHA=${EXPECTED_BASE_RECEIPT_SHA},SOURCE_RUN_ROOT=${SOURCE_RUN_ROOT},RUN_ROOT=${run_root},PROTOCOL=${protocol}"
construct=${task_root}/MemNavData/slurm_hm3d_table1_fresh_query_construct.sbatch
analysis=${task_root}/MemNavData/slurm_hm3d_table1_fresh_query_analysis.sbatch
echo '[stage] Slurm test-only gates'
remote "sbatch --test-only --partition=h100_tandon,a100_tandon --account=torch_pr_769_tandon_advanced --qos=gpu48 --gres=gpu:1 --time=01:00:00 --array=0 --export='${common},RUN_ROOT=${smoke_root}' '${construct}' >/dev/null"
remote "sbatch --test-only --partition=h100_tandon,a100_tandon --account=torch_pr_769_tandon_advanced --qos=gpu48 --gres=gpu:1 --time=01:00:00 --array=0-53%${CONSTRUCT_CONCURRENCY} --export='${common}' '${construct}' >/dev/null"
remote "sbatch --test-only --partition=cpu_short --account=torch_pr_769_tandon_advanced --time=00:30:00 --export='${common},MODE=finalize' '${analysis}' >/dev/null"
remote "sbatch --test-only --partition=cpu_short --account=torch_pr_769_tandon_advanced --time=00:30:00 --export='${common},MODE=verify' '${analysis}' >/dev/null"

echo '[stage] submit construction-only dependency DAG'
smoke_raw=$(remote "sbatch --parsable --job-name=h3T1smoke --partition=h100_tandon,a100_tandon --account=torch_pr_769_tandon_advanced --qos=gpu48 --gres=gpu:1 --time=01:00:00 --array=0 --export='${common},RUN_ROOT=${smoke_root}' '${construct}'" | tr -d '\r')
smoke_id=${smoke_raw%%;*}; [[ "${smoke_id}" =~ ^[0-9]+$ ]] || fail "bad smoke job"
formal_raw=$(remote "sbatch --parsable --job-name=h3T1construct --partition=h100_tandon,a100_tandon --account=torch_pr_769_tandon_advanced --qos=gpu48 --gres=gpu:1 --time=01:00:00 --array=0-53%${CONSTRUCT_CONCURRENCY} --dependency=afterok:${smoke_id} --kill-on-invalid-dep=yes --export='${common}' '${construct}'" | tr -d '\r')
formal_id=${formal_raw%%;*}; [[ "${formal_id}" =~ ^[0-9]+$ ]] || fail "bad formal array"
finalize_raw=$(remote "sbatch --parsable --job-name=h3T1finalize --partition=cpu_short --account=torch_pr_769_tandon_advanced --time=00:30:00 --dependency=afterok:${formal_id} --kill-on-invalid-dep=yes --export='${common},MODE=finalize' '${analysis}'" | tr -d '\r')
finalize_id=${finalize_raw%%;*}; [[ "${finalize_id}" =~ ^[0-9]+$ ]] || fail "bad finalizer"
verify_raw=$(remote "sbatch --parsable --job-name=h3T1verify --partition=cpu_short --account=torch_pr_769_tandon_advanced --time=00:30:00 --dependency=afterok:${finalize_id} --kill-on-invalid-dep=yes --export='${common},MODE=verify' '${analysis}'" | tr -d '\r')
verify_id=${verify_raw%%;*}; [[ "${verify_id}" =~ ^[0-9]+$ ]] || fail "bad verifier"

receipt=MemNavData/HM3D_TABLE1_FRESH_QUERY_RESERVE_SUBMISSION_20260829.json
[[ ! -e "${receipt}" ]] || fail "local submission receipt already exists"
"${LOCAL_MEMNAV_PY}" - "${receipt}" "${run_root}" "${smoke_root}" \
  "${task_root}" "${task_receipt_sha}" "${smoke_id}" "${formal_id}" \
  "${finalize_id}" "${verify_id}" <<'PY'
import json,sys
(path,run,smoke,bundle,sha,smoke_job,formal,finalize,verify)=sys.argv[1:]
payload={
 "schema_version":"hm3d_table1_fresh_query_submission_v1_20260829",
 "scope":"construction-only power gate; no navigation rollout",
 "run_root":run,"smoke_root":smoke,"task_bundle":bundle,
 "task_receipt_sha256":sha,"query_policy_outcomes_read_at_submission":False,
 "jobs":{"construction_smoke":int(smoke_job),
         "construction_array":int(formal),"population_finalize":int(finalize),
         "independent_verification":int(verify)},
 "future_policy_evaluation_submitted":False,
}
open(path,"x").write(json.dumps(payload,indent=2,sort_keys=True)+"\n")
print(json.dumps(payload,indent=2,sort_keys=True))
PY
scp -q -o BatchMode=yes -o ControlMaster=no \
  -o ControlPath="${SSH_CONTROL_PATH}" \
  "${ROOT}/${receipt}" "${SSH_ALIAS}:${run_root}/submission.json"
remote "sha256sum '${run_root}/submission.json' >'${run_root}/submission.json.sha256' && chmod a-w '${run_root}/submission.json' '${run_root}/submission.json.sha256'"
printf 'RUN_ROOT=%s\nSMOKE_ROOT=%s\nTASK_ROOT=%s\nSMOKE=%s\nCONSTRUCTION=%s\nFINALIZE=%s\nVERIFY=%s\n' \
  "${run_root}" "${smoke_root}" "${task_root}" "${smoke_id}" \
  "${formal_id}" "${finalize_id}" "${verify_id}"
