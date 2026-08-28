#!/usr/bin/env bash
# Package, preflight, archive, and exact-submit the interrupted HM3D C/B2 chain.
set -euo pipefail
umask 0022

ROOT=${ROOT:-/home/asus/Research/Nav-graph-blind}
SSH_ALIAS=${SSH_ALIAS:-alantorch}
EXPECTED_SSH_USER=${EXPECTED_SSH_USER:-yz11502}
SUBMIT=${SUBMIT:-0}
LOCAL_PY=${LOCAL_PY:-/home/asus/miniconda3/envs/memnav/bin/python}
REMOTE_PY=/scratch/lg154/conda-envs/memnav/bin/python
HAB_PY=/scratch/lg154/conda-envs/habitat/bin/python
REMOTE_BUNDLES=/scratch/yz11502/Research/Nav-axis-uturn-source-bundles
RUN_ROOT=/scratch/yz11502/Research/Nav-axis-uturn-results/hm3d_fullmono_lifelong_natural_v4_20260827/formal_materialize_20260827T133704Z_d85fc50d
TASK_ROOT=/scratch/yz11502/Research/Nav-axis-uturn-source-bundles/hm3d_fullmono_lifelong_375f0b6879b2ff87
TASK_RECEIPT=${TASK_ROOT}/SOURCE_BUNDLE.sha256
EXPECTED_TASK_RECEIPT_SHA=375f0b6879b2ff87b7019dae4727880d1b03fd3185a1862e6239942a76b5bcc8
BASE_SOURCE_ROOT=/scratch/yz11502/Research/Nav-axis-uturn-source-bundles/final14_mono_factorial_5690569a4373f2d2
BASE_RECEIPT=${BASE_SOURCE_ROOT}/source_inputs.sha256
EXPECTED_BASE_RECEIPT_SHA=5690569a4373f2d2768671418f0c604c4a03aa4b0ffe01baf70b288af03ba216
DEPENDENCY_RECEIPT=/scratch/yz11502/Research/Nav-axis-uturn-results/shared_online_double_revisit_fresh_20260813/double_revisit_fresh40_20260813T200121Z/dependency_receipt.json
EXPECTED_DEPENDENCY_RECEIPT_SHA=4eb0ca6479a26f8e04f85a31d906cee4e68b1785f66cfd3ac23bf65424d36e5e
POPULATION_SHA=ec11c0dbc43a4abe585330c1ce52a8c14ad1d4b1da6fd8397e1d15592707a6d5
POPULATION_VERIFIER_SHA=d9ce97df4b0687969090e710ef719f6da56fc5d39a0535a7e8afd6c5d852499b
ORIGINAL_ARRAY_JOB=16505696
ARCHIVE_ROOT=${RUN_ROOT}/failed_attempts/shared_c_16505696_replay_contract_20260828
REPAIR_ROOT=${RUN_ROOT}/underpowered_collect_repair_20260828
SMOKE_ROOT=${RUN_ROOT}/underpowered_smoke_collect_repair_20260828
HAB_REQUESTS_VENDOR=/scratch/lg154/conda-envs/habitat/lib/python3.9/site-packages/pip/_vendor
SUBMISSION_RECEIPT=MemNavData/HM3D_LIFELONG_UNDERPOWERED_COLLECT_REPAIR_SUBMISSION_20260828.json

cd "${ROOT}"
fail() { echo "ABORT: $*" >&2; exit 2; }
[[ "${SUBMIT}" =~ ^[01]$ ]] || fail "SUBMIT must be 0 or 1"

SSH_CONTROL_PATH=${SSH_CONTROL_PATH:-$(
  ssh -G "${SSH_ALIAS}" 2>/dev/null |
    awk '$1=="controlpath"{value=$2} END{print value}'
)}
[[ -S "${SSH_CONTROL_PATH}" ]] || fail "authoritative shared SSH socket missing"
timeout 15 ssh -O check -S "${SSH_CONTROL_PATH}" "${SSH_ALIAS}" \
  >/dev/null 2>&1 || fail "authoritative shared SSH master is not responsive"
remote_user=$(timeout 20 ssh -n -T -o BatchMode=yes -o ControlMaster=no \
  -S "${SSH_CONTROL_PATH}" "${SSH_ALIAS}" 'id -un' 2>/dev/null || true)
[[ "${remote_user}" == "${EXPECTED_SSH_USER}" ]] || \
  fail "shared SSH identity is ${remote_user:-unavailable}"
remote() {
  timeout 240 ssh -n -T -o BatchMode=yes -o ControlMaster=no \
    -o ServerAliveInterval=15 -S "${SSH_CONTROL_PATH}" "${SSH_ALIAS}" "$@"
}
job_id() { tr -d '\r' | awk -F';' '/^[0-9]+(;|$)/ {print $1; exit}'; }

files=(
  MemNavData/hm3d_fullmono_lifelong_underpowered_amendment_20260828.json
  MemNavData/HM3D_FULLMONO_LIFELONG_UNDERPOWERED_AMENDMENT_20260828.md
  MemNavData/audit_hm3d_lifelong_underpowered_amendment.py
  MemNavData/test_audit_hm3d_lifelong_underpowered_amendment.py
  MemNavData/hm3d_lifelong_underpowered_collect_repair_20260828.json
  MemNavData/HM3D_LIFELONG_UNDERPOWERED_COLLECT_REPAIR_PROTOCOL_20260828.md
  MemNavData/audit_hm3d_lifelong_underpowered_collect_repair.py
  MemNavData/test_audit_hm3d_lifelong_underpowered_collect_repair.py
  MemNavData/hm3d_lifelong_node_affinity.py
  MemNavData/test_hm3d_lifelong_node_affinity.py
  MemNavData/navdp_replay_contract.py
  MemNavData/test_navdp_replay_contract.py
  MemNavData/eval_3leg_habitat.py
  MemNavData/collect_hm3d_lifelong_shared_c.py
  MemNavData/eval_hm3d_lifelong_shared_c_b2.py
  MemNavData/run_cec_controller_portability_smoke_local.sh
  MemNavData/slurm_hm3d_fullmono_shared_c_arm.sbatch
  MemNavData/slurm_hm3d_lifelong_underpowered_deferred.sbatch
  MemNavData/slurm_hm3d_lifelong_underpowered_collect_repair_barrier.sbatch
  MemNavData/slurm_safe_submit.sh
  MemNavData/test_slurm_safe_submit.sh
  MemNavData/slurm_port_pair.sh
  MemNavData/test_slurm_port_pair.sh
  MemNavData/archive_hm3d_lifelong_underpowered_collect_repair.sh
  MemNavData/submit_hm3d_lifelong_underpowered_collect_repair_hpc.sh
)
for path in "${files[@]}"; do
  [[ -f "${path}" && ! -L "${path}" ]] || fail "missing physical ${path}"
done

"${LOCAL_PY}" -m json.tool \
  MemNavData/hm3d_lifelong_underpowered_collect_repair_20260828.json >/dev/null
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="${ROOT}:${ROOT}/MemNavData" \
  "${LOCAL_PY}" -m pytest -q -p no:cacheprovider \
  MemNavData/test_navdp_replay_contract.py \
  MemNavData/test_hm3d_lifelong_node_affinity.py \
  MemNavData/test_audit_hm3d_lifelong_underpowered_collect_repair.py \
  MemNavData/test_audit_hm3d_lifelong_underpowered_amendment.py
bash MemNavData/test_slurm_safe_submit.sh
bash MemNavData/test_slurm_port_pair.sh
"${LOCAL_PY}" -m py_compile \
  MemNavData/navdp_replay_contract.py MemNavData/hm3d_lifelong_node_affinity.py \
  MemNavData/audit_hm3d_lifelong_underpowered_collect_repair.py \
  MemNavData/eval_3leg_habitat.py \
  MemNavData/collect_hm3d_lifelong_shared_c.py \
  MemNavData/eval_hm3d_lifelong_shared_c_b2.py
bash -n MemNavData/slurm_hm3d_fullmono_shared_c_arm.sbatch \
  MemNavData/slurm_hm3d_lifelong_underpowered_deferred.sbatch \
  MemNavData/slurm_hm3d_lifelong_underpowered_collect_repair_barrier.sbatch \
  MemNavData/slurm_safe_submit.sh MemNavData/test_slurm_safe_submit.sh \
  MemNavData/slurm_port_pair.sh MemNavData/test_slurm_port_pair.sh \
  MemNavData/archive_hm3d_lifelong_underpowered_collect_repair.sh \
  MemNavData/submit_hm3d_lifelong_underpowered_collect_repair_hpc.sh
(
  source MemNavData/slurm_safe_submit.sh
  lint_sbatch_template MemNavData/slurm_hm3d_fullmono_shared_c_arm.sbatch
  lint_sbatch_template MemNavData/slurm_hm3d_lifelong_underpowered_deferred.sbatch
)

stage=$(mktemp -d /tmp/h3_collect_repair.XXXXXX)
cleanup() { rm -rf -- "${stage}"; }
trap cleanup EXIT
for path in "${files[@]}"; do
  mkdir -p "${stage}/root/$(dirname "${path}")"
  install -m 0644 "${path}" "${stage}/root/${path}"
done
(
  cd "${stage}/root"
  find . -type f ! -name SOURCE_BUNDLE.sha256 -print0 | sort -z | \
    xargs -0 sha256sum >SOURCE_BUNDLE.sha256
  sha256sum -c --quiet SOURCE_BUNDLE.sha256
)
bundle_sha=$(sha256sum "${stage}/root/SOURCE_BUNDLE.sha256" | awk '{print $1}')
bundle=${REMOTE_BUNDLES}/hm3d_lifelong_collect_repair_${bundle_sha:0:16}
partial=${bundle}.partial.$$
bundle_receipt=${bundle}/SOURCE_BUNDLE.sha256
protocol=${bundle}/MemNavData/hm3d_lifelong_underpowered_collect_repair_20260828.json
amendment_protocol=${bundle}/MemNavData/hm3d_fullmono_lifelong_underpowered_amendment_20260828.json
arm=${bundle}/MemNavData/slurm_hm3d_fullmono_shared_c_arm.sbatch
deferred=${bundle}/MemNavData/slurm_hm3d_lifelong_underpowered_deferred.sbatch
barrier_template=${bundle}/MemNavData/slurm_hm3d_lifelong_underpowered_collect_repair_barrier.sbatch
archive_script=${bundle}/MemNavData/archive_hm3d_lifelong_underpowered_collect_repair.sh
finalize=${TASK_ROOT}/MemNavData/slurm_hm3d_fullmono_shared_c_finalize.sbatch
analysis=${TASK_ROOT}/MemNavData/slurm_hm3d_fullmono_shared_c_analysis.sbatch

remote "set -euo pipefail; \
  test \"\$(id -un)\" = '${EXPECTED_SSH_USER}'; \
  test \"\$(sha256sum '${TASK_RECEIPT}' | awk '{print \$1}')\" = '${EXPECTED_TASK_RECEIPT_SHA}'; \
  cd '${TASK_ROOT}' && sha256sum -c --quiet SOURCE_BUNDLE.sha256; \
  test \"\$(sha256sum '${BASE_RECEIPT}' | awk '{print \$1}')\" = '${EXPECTED_BASE_RECEIPT_SHA}'; \
  cd '${BASE_SOURCE_ROOT}' && sha256sum -c --quiet source_inputs.sha256; \
  test \"\$(sha256sum '${DEPENDENCY_RECEIPT}' | awk '{print \$1}')\" = '${EXPECTED_DEPENDENCY_RECEIPT_SHA}'; \
  test \"\$(sha256sum '${RUN_ROOT}/population/population.json' | awk '{print \$1}')\" = '${POPULATION_SHA}'; \
  test \"\$(sha256sum '${RUN_ROOT}/independent_natural_v4_population_verification.json' | awk '{print \$1}')\" = '${POPULATION_VERIFIER_SHA}'; \
  test ! -e '${RUN_ROOT}/shared_c_population'; \
  test ! -e '${RUN_ROOT}/shared_c_evaluation'; \
  test ! -e '${RUN_ROOT}/shared_c_aggregate'; \
  test ! -e '${RUN_ROOT}/shared_c_independent_verification.json'"

if remote "test -d '${bundle}'"; then
  remote "test \"\$(sha256sum '${bundle_receipt}' | awk '{print \$1}')\" = '${bundle_sha}' && cd '${bundle}' && sha256sum -c --quiet SOURCE_BUNDLE.sha256"
else
  remote "test ! -e '${partial}' && mkdir -p '${partial}'"
  timeout 240 rsync -a --partial \
    --chmod=Du=rwx,Dgo=rx,Fu=rw,Fgo=r \
    -e "ssh -o BatchMode=yes -o ControlMaster=no -S ${SSH_CONTROL_PATH}" \
    "${stage}/root/" "${SSH_ALIAS}:${partial}/"
  remote "cd '${partial}' && sha256sum -c --quiet SOURCE_BUNDLE.sha256 && chmod -R a-w '${partial}' && mv '${partial}' '${bundle}'"
fi

remote "PYTHONDONTWRITEBYTECODE=1 PYTHONPATH='${bundle}:${bundle}/MemNavData' '${REMOTE_PY}' -m pytest -q -p no:cacheprovider '${bundle}/MemNavData/test_navdp_replay_contract.py' '${bundle}/MemNavData/test_hm3d_lifelong_node_affinity.py' '${bundle}/MemNavData/test_audit_hm3d_lifelong_underpowered_collect_repair.py' '${bundle}/MemNavData/test_audit_hm3d_lifelong_underpowered_amendment.py'"
remote "ROOT='${bundle}' bash '${bundle}/MemNavData/test_slurm_port_pair.sh' && ROOT='${bundle}' bash '${bundle}/MemNavData/test_slurm_safe_submit.sh'"
remote "bash -n '${arm}' '${deferred}' '${barrier_template}' '${archive_script}' '${bundle}/MemNavData/slurm_safe_submit.sh'"
remote "singularity exec -B /scratch/lg154 -B /scratch/yz11502 /share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH='${bundle}/MemNavData:${bundle}:${TASK_ROOT}/MemNavData:${TASK_ROOT}:${HAB_REQUESTS_VENDOR}' '${HAB_PY}' -c 'import importlib.util,pathlib; root=pathlib.Path(\"${bundle}/MemNavData\").resolve(); names=(\"navdp_replay_contract\",\"eval_3leg_habitat\",\"collect_hm3d_lifelong_shared_c\",\"eval_hm3d_lifelong_shared_c_b2\"); assert all(pathlib.Path(importlib.util.find_spec(n).origin).resolve()==root/(n+\".py\") for n in names)'"
remote "singularity exec -B /scratch/lg154 -B /scratch/yz11502 /share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif env PYTHONDONTWRITEBYTECODE=1 '${HAB_PY}' -c 'import pathlib,sys; paths=[pathlib.Path(x) for x in sys.argv[1:]]; [compile(p.read_text(),str(p),\"exec\") for p in paths]' '${bundle}/MemNavData/navdp_replay_contract.py' '${bundle}/MemNavData/eval_3leg_habitat.py' '${bundle}/MemNavData/collect_hm3d_lifelong_shared_c.py' '${bundle}/MemNavData/eval_hm3d_lifelong_shared_c_b2.py'"
remote "PYTHONDONTWRITEBYTECODE=1 PYTHONPATH='${bundle}:${bundle}/MemNavData' '${REMOTE_PY}' '${bundle}/MemNavData/audit_hm3d_lifelong_underpowered_collect_repair.py' --protocol '${protocol}' --run-root '${RUN_ROOT}' --phase pre_archive >/dev/null"

common="ALL,AMENDMENT_ROOT=${bundle},AMENDMENT_RECEIPT=${bundle_receipt},EXPECTED_AMENDMENT_RECEIPT_SHA=${bundle_sha},TASK_ROOT=${TASK_ROOT},TASK_RECEIPT=${TASK_RECEIPT},EXPECTED_TASK_RECEIPT_SHA=${EXPECTED_TASK_RECEIPT_SHA},BASE_SOURCE_ROOT=${BASE_SOURCE_ROOT},BASE_RECEIPT=${BASE_RECEIPT},EXPECTED_BASE_RECEIPT_SHA=${EXPECTED_BASE_RECEIPT_SHA},RUN_ROOT=${RUN_ROOT},PROTOCOL=${amendment_protocol},DEPENDENCY_RECEIPT=${DEPENDENCY_RECEIPT},EXPECTED_DEPENDENCY_RECEIPT_SHA=${EXPECTED_DEPENDENCY_RECEIPT_SHA},EXPECTED_HAB_REQUESTS_VERSION=2.32.4,EXPECTED_HAB_REQUESTS_INIT_BYTES=5057,EXPECTED_HAB_REQUESTS_INIT_SHA=1e507f1f386bcc6b5f0ff69a614c14875cd65cb67be7f6022f28adef9774573f,EXPECTED_HAB_REQUESTS_VERSION_BYTES=435,EXPECTED_HAB_REQUESTS_VERSION_SHA=143abaf3563712f063743a7952aa65319dbcb934d894cfc989bd2c015f8da577"
continuation="${common},SMOKE_ROOT=${SMOKE_ROOT},SOURCE_POPULATION_SHA=${POPULATION_SHA},SOURCE_POPULATION_COUNT=22,SOURCE_SCENE_COUNT=15,EVAL_CONCURRENCY=2,GPU_TIME_LIMIT=01:00:00,UPSTREAM_POPULATION_SEAL_JOB_ID=16489720"
remote "source '${bundle}/MemNavData/slurm_safe_submit.sh'; safe_sbatch --lint-fatal --test-only --partition=h100_tandon --nodelist=gh005 --qos=gpu48 --time=01:00:00 --array=0 --export='${common},STAGE=collect,EXPECTED_REPLAY_NODE=gh005' '${arm}' >/dev/null; safe_sbatch --lint-fatal --test-only --partition=a100_tandon --nodelist=ga005 --qos=gpu48 --time=01:00:00 --array=7 --export='${common},STAGE=collect,EXPECTED_REPLAY_NODE=ga005' '${arm}' >/dev/null; safe_sbatch --lint-fatal --test-only --partition=cpu_short --export='${common},REPAIR_PROTOCOL=${protocol},REPAIR_ROOT=${REPAIR_ROOT}' '${barrier_template}' >/dev/null; safe_sbatch --lint-fatal --test-only --partition=cpu_short --export='${common}' '${finalize}' >/dev/null; safe_sbatch --lint-fatal --test-only --partition=cpu_short --export='${continuation},DEFERRED_MODE=resume' '${deferred}' >/dev/null"

if [[ "${SUBMIT}" == 0 ]]; then
  printf 'PREPARED_ONLY=1\nBUNDLE=%s\nBUNDLE_SHA=%s\n' "${bundle}" "${bundle_sha}"
  exit 0
fi

[[ ! -e "${SUBMISSION_RECEIPT}" ]] || fail "submission receipt already exists"
remote "test ! -e '${ARCHIVE_ROOT}' && test ! -e '${REPAIR_ROOT}' && test ! -e '${SMOKE_ROOT}'"
remote "env RUN_ROOT='${RUN_ROOT}' PROTOCOL='${protocol}' REPAIR_ROOT='${REPAIR_ROOT}' ARCHIVE_ROOT='${ARCHIVE_ROOT}' AMENDMENT_ROOT='${bundle}' ORIGINAL_ARRAY_JOB='${ORIGINAL_ARRAY_JOB}' REMOTE_PY='${REMOTE_PY}' bash '${archive_script}'"

source_remote="source '${bundle}/MemNavData/slurm_safe_submit.sh';"
submit_repair() {
  local index=$1 node=$2 partition=$3 dependency=${4:-} command
  command="${source_remote} safe_sbatch --lint-fatal --parsable --partition=${partition} --nodelist=${node} --qos=gpu48 --time=01:00:00 --array=${index}"
  if [[ -n "${dependency}" ]]; then command+=" --dependency=afterany:${dependency}"; fi
  command+=" --export='${common},STAGE=collect,EXPECTED_REPLAY_NODE=${node}' '${arm}'"
  remote "${command}" | job_id
}

repair_0=$(submit_repair 0 gh005 h100_tandon)
repair_1=$(submit_repair 1 gh001 h100_tandon)
repair_7=$(submit_repair 7 ga005 a100_tandon "${repair_0}")
repair_9=$(submit_repair 9 ga003 a100_tandon "${repair_1}")
repair_11=$(submit_repair 11 ga028 a100_tandon "${repair_7}")
repair_13=$(submit_repair 13 ga002 a100_tandon "${repair_9}")
for value in "${repair_0}" "${repair_1}" "${repair_7}" "${repair_9}" \
  "${repair_11}" "${repair_13}"; do
  [[ "${value}" =~ ^[0-9]+$ ]] || fail "bad repair job id"
done
barrier=$(remote "${source_remote} safe_sbatch --lint-fatal --parsable --partition=cpu_short --dependency=afterany:${repair_11}:${repair_13} --export='${common},REPAIR_PROTOCOL=${protocol},REPAIR_ROOT=${REPAIR_ROOT}' '${barrier_template}'" | job_id)
seal=$(remote "${source_remote} safe_sbatch --lint-fatal --parsable --partition=cpu_short --dependency=afterok:${barrier} --kill-on-invalid-dep=yes --export='${common}' '${finalize}'" | job_id)
resume=$(remote "${source_remote} safe_sbatch --lint-fatal --parsable --partition=cpu_short --dependency=afterok:${seal} --kill-on-invalid-dep=yes --export='${continuation},DEFERRED_MODE=resume' '${deferred}'" | job_id)
[[ "${barrier}" =~ ^[0-9]+$ && "${seal}" =~ ^[0-9]+$ \
   && "${resume}" =~ ^[0-9]+$ ]] || fail "bad barrier/seal/resume job id"

"${LOCAL_PY}" - "${SUBMISSION_RECEIPT}" "${bundle}" "${bundle_sha}" \
  "${RUN_ROOT}" "${ARCHIVE_ROOT}" "${REPAIR_ROOT}" "${SMOKE_ROOT}" \
  "${repair_0}" "${repair_1}" "${repair_7}" "${repair_9}" \
  "${repair_11}" "${repair_13}" "${barrier}" "${seal}" "${resume}" <<'PY'
import json,sys
(path,bundle,digest,run,archive,repair_root,smoke,
 j0,j1,j7,j9,j11,j13,barrier,seal,resume)=sys.argv[1:]
payload={
 "schema_version":"hm3d_lifelong_underpowered_collect_repair_submission_v1_20260828",
 "source_bundle":bundle,"source_bundle_receipt_sha256":digest,
 "run_root":run,"archive_root":archive,"repair_receipt_root":repair_root,
 "smoke_root":smoke,"original_array_job":16505696,
 "preserved_completed_indices":[2,3,4,5,6,8,10,12,14,15,16,17,18,19,20,21],
 "repair_jobs":[
  {"index":0,"node":"gh005","partition":"h100_tandon","lane":0,"job":int(j0)},
  {"index":1,"node":"gh001","partition":"h100_tandon","lane":1,"job":int(j1)},
  {"index":7,"node":"ga005","partition":"a100_tandon","lane":0,"job":int(j7)},
  {"index":9,"node":"ga003","partition":"a100_tandon","lane":1,"job":int(j9)},
  {"index":11,"node":"ga028","partition":"a100_tandon","lane":0,"job":int(j11)},
  {"index":13,"node":"ga002","partition":"a100_tandon","lane":1,"job":int(j13)},
 ],
 "repair_integrity_barrier_job":int(barrier),"factual_C_seal_job":int(seal),
 "node_affine_resume_launcher_job":int(resume),
 "maximum_concurrent_GPU_jobs":2,"outcome_blind_repair_selection":True,
 "underpowered":True,"powered_confirmation_claim":False,"submitted":True,
}
with open(path,"x") as f:
 json.dump(payload,f,indent=2,sort_keys=True); f.write("\n")
PY
sha256sum "${SUBMISSION_RECEIPT}" >"${SUBMISSION_RECEIPT}.sha256"
remote "test ! -e '${REPAIR_ROOT}/submission.json' && test ! -e '${REPAIR_ROOT}/submission.json.sha256'"
timeout 120 rsync -a \
  -e "ssh -o BatchMode=yes -o ControlMaster=no -S ${SSH_CONTROL_PATH}" \
  "${SUBMISSION_RECEIPT}" "${SSH_ALIAS}:${REPAIR_ROOT}/submission.json"
remote "sha256sum '${REPAIR_ROOT}/submission.json' >'${REPAIR_ROOT}/submission.json.sha256' && chmod a-w '${REPAIR_ROOT}/submission.json' '${REPAIR_ROOT}/submission.json.sha256'"
printf 'SUBMITTED=1\nREPAIR_JOBS=%s,%s,%s,%s,%s,%s\nBARRIER_JOB=%s\nSEAL_JOB=%s\nRESUME_JOB=%s\n' \
  "${repair_0}" "${repair_1}" "${repair_7}" "${repair_9}" \
  "${repair_11}" "${repair_13}" "${barrier}" "${seal}" "${resume}"
