#!/usr/bin/env bash
# Freeze, upload, and submit fresh-scene actual-online Full-Mono HM3D.
set -euo pipefail
umask 0022

ROOT=${ROOT:-/home/asus/Research/Nav-graph-blind}
SSH_ALIAS=${SSH_ALIAS:-alantorch}
LOCAL_PY=/home/asus/miniconda3/envs/memnav/bin/python
LOCAL_HAB_PY=/home/asus/miniconda3/envs/habitat/bin/python
REMOTE_BUNDLES=/scratch/yz11502/Research/Nav-axis-uturn-source-bundles
REMOTE_RESULTS=/scratch/yz11502/Research/Nav-axis-uturn-results/hm3d_fresh_fullmono_mixed_role_20260820
DATA_ROOT=/scratch/yz11502/Research/datasets/hm3d_fresh_fullmono_v0.2_20260820
BASE_SOURCE_ROOT=/scratch/yz11502/Research/Nav-axis-uturn-source-bundles/final14_mono_factorial_5690569a4373f2d2
BASE_RECEIPT=${BASE_SOURCE_ROOT}/source_inputs.sha256
EXPECTED_BASE_RECEIPT_SHA=5690569a4373f2d2768671418f0c604c4a03aa4b0ffe01baf70b288af03ba216
GEN_CONCURRENCY=${GEN_CONCURRENCY:-6}
COLLECT_CONCURRENCY=${COLLECT_CONCURRENCY:-4}
CONSTRUCT_CONCURRENCY=${CONSTRUCT_CONCURRENCY:-4}
EVAL_CONCURRENCY=${EVAL_CONCURRENCY:-4}
SSH_CONTROL_PATH=${SSH_CONTROL_PATH:-$(ssh -G "${SSH_ALIAS}" 2>/dev/null | awk '$1=="controlpath"{value=$2} END{print value}')}
cd "${ROOT}"

fail() { echo "ABORT: $*" >&2; exit 2; }
remote() {
  ssh -tt -o BatchMode=yes -o ControlMaster=no \
    -S "${SSH_CONTROL_PATH}" "${SSH_ALIAS}" "$@"
}
[[ -S "${SSH_CONTROL_PATH}" ]] || fail "authoritative SSH master missing"
for value in "${GEN_CONCURRENCY}" "${COLLECT_CONCURRENCY}" \
             "${CONSTRUCT_CONCURRENCY}" "${EVAL_CONCURRENCY}"; do
  [[ "${value}" =~ ^[1-9][0-9]*$ ]] || fail "invalid concurrency ${value}"
done

files=(
  MemNavData/HM3D_FRESH_FULLMONO_MIXED_ROLE_PROTOCOL_20260820.md
  MemNavData/hm3d_fresh_fullmono_mixed_role_protocol_20260820.json
  MemNavData/hm3d_fullmono_mixed_role_protocol_20260820.json
  MemNavData/hm3d_consumed_scene_audit_20260816.json
  MemNavData/hm3d_heldout_val10_revisit_protocol_20260816.json
  .diagnostics/datasets/goat-bench/hm3d_val_receipts/hm3d-val-habitat-v0.2.members.txt
  MemNavData/audit_hm3d_fresh_fullmono_selection.py
  MemNavData/test_audit_hm3d_fresh_fullmono_selection.py
  MemNavData/build_hm3d_fresh_fullmono_parent_manifest.py
  MemNavData/generate_twoleg.py
  MemNavData/slurm_hm3d_fresh_fullmono_prepare.sbatch
  MemNavData/slurm_hm3d_fresh_fullmono_generate.sbatch
  MemNavData/slurm_hm3d_fresh_fullmono_manifest.sbatch
  MemNavData/hm3d_fullmono_mixed_role.py
  MemNavData/test_hm3d_fullmono_mixed_role.py
  MemNavData/test_hm3d_fullmono_materialize.py
  MemNavData/audit_hm3d_fullmono_inputs.py
  MemNavData/collect_hm3d_fullmono_goal_a.py
  MemNavData/materialize_hm3d_fullmono_online_a.py
  MemNavData/construct_hm3d_fullmono_role_pairs.py
  MemNavData/finalize_hm3d_fullmono_mixed_role.py
  MemNavData/run_hm3d_fullmono_query_history.py
  MemNavData/summarize_hm3d_fullmono_mixed_role.py
  MemNavData/independent_verify_hm3d_fullmono_mixed_role.py
  MemNavData/run_hm3d_fullmono_server_scene.sh
  MemNavData/slurm_hm3d_fullmono_goal_a.sbatch
  MemNavData/slurm_hm3d_fullmono_construct.sbatch
  MemNavData/slurm_hm3d_fullmono_finalize.sbatch
  MemNavData/slurm_hm3d_fullmono_eval.sbatch
  MemNavData/slurm_hm3d_fullmono_analysis.sbatch
  MemNavData/submit_hm3d_fresh_fullmono_mixed_role_hpc.sh
)
for file in "${files[@]}"; do
  [[ -f "${file}" && ! -L "${file}" ]] || fail "missing physical ${file}"
done

"${LOCAL_PY}" -m json.tool \
  MemNavData/hm3d_fresh_fullmono_mixed_role_protocol_20260820.json >/dev/null
"${LOCAL_PY}" -m py_compile \
  MemNavData/audit_hm3d_fresh_fullmono_selection.py \
  MemNavData/build_hm3d_fresh_fullmono_parent_manifest.py \
  MemNavData/hm3d_fullmono_mixed_role.py \
  MemNavData/audit_hm3d_fullmono_inputs.py \
  MemNavData/collect_hm3d_fullmono_goal_a.py \
  MemNavData/materialize_hm3d_fullmono_online_a.py \
  MemNavData/construct_hm3d_fullmono_role_pairs.py \
  MemNavData/finalize_hm3d_fullmono_mixed_role.py \
  MemNavData/run_hm3d_fullmono_query_history.py \
  MemNavData/summarize_hm3d_fullmono_mixed_role.py \
  MemNavData/independent_verify_hm3d_fullmono_mixed_role.py
PYTHONPATH="${ROOT}:${ROOT}/MemNavData" "${LOCAL_PY}" -m unittest \
  MemNavData.test_audit_hm3d_fresh_fullmono_selection \
  MemNavData.test_hm3d_fullmono_mixed_role \
  MemNavData.test_final14_mono_factorial \
  MemNavData.test_mdtec_raw_depth_gate_d \
  MemNavData.test_monocular_depth_runtime \
  MemNavData.test_shared_online_role_pair_contract
PYTHONPATH="${ROOT}:${ROOT}/MemNavData" "${LOCAL_HAB_PY}" -m unittest \
  MemNavData.test_final14_role_pair_construction \
  MemNavData.test_hm3d_fullmono_materialize \
  MemNavData.test_hm3d_fullmono_mixed_role
bash -n \
  MemNavData/slurm_hm3d_fresh_fullmono_prepare.sbatch \
  MemNavData/slurm_hm3d_fresh_fullmono_generate.sbatch \
  MemNavData/slurm_hm3d_fresh_fullmono_manifest.sbatch \
  MemNavData/run_hm3d_fullmono_server_scene.sh \
  MemNavData/slurm_hm3d_fullmono_goal_a.sbatch \
  MemNavData/slurm_hm3d_fullmono_construct.sbatch \
  MemNavData/slurm_hm3d_fullmono_finalize.sbatch \
  MemNavData/slurm_hm3d_fullmono_eval.sbatch \
  MemNavData/slurm_hm3d_fullmono_analysis.sbatch \
  MemNavData/submit_hm3d_fresh_fullmono_mixed_role_hpc.sh

staging=$(mktemp -d)
cleanup() { rm -rf -- "${staging}"; }
trap cleanup EXIT
for file in "${files[@]}"; do
  mkdir -p "${staging}/$(dirname "${file}")"
  cp --preserve=mode,timestamps "${file}" "${staging}/${file}"
done
PYTHONPATH="${staging}:${staging}/MemNavData" "${LOCAL_PY}" \
  "${staging}/MemNavData/audit_hm3d_fresh_fullmono_selection.py" \
    --protocol "${staging}/MemNavData/hm3d_fresh_fullmono_mixed_role_protocol_20260820.json" \
    --member-list "${staging}/.diagnostics/datasets/goat-bench/hm3d_val_receipts/hm3d-val-habitat-v0.2.members.txt" \
    --prior-audit "${staging}/MemNavData/hm3d_consumed_scene_audit_20260816.json" \
    --heldout10-protocol "${staging}/MemNavData/hm3d_heldout_val10_revisit_protocol_20260816.json" \
    --out "${staging}/fresh_scene_selection_verification.json" >/dev/null
(
  cd "${staging}"
  find . -type f ! -name SOURCE_BUNDLE.sha256 -print0 | sort -z | \
    xargs -0 sha256sum >SOURCE_BUNDLE.sha256
  sha256sum -c SOURCE_BUNDLE.sha256 >/dev/null
)
task_receipt_sha=$(sha256sum "${staging}/SOURCE_BUNDLE.sha256" | awk '{print $1}')
bundle_key=${task_receipt_sha:0:16}
task_root=${REMOTE_BUNDLES}/hm3d_fresh_fullmono_mixed_role_${bundle_key}
task_stage=${task_root}.partial.$$
run_tag=formal_$(date -u +%Y%m%dT%H%M%SZ)_${bundle_key:0:8}
run_root=${REMOTE_RESULTS}/${run_tag}
smoke_root=${REMOTE_RESULTS}/${run_tag}_smoke

remote_identity=$(remote 'id -un' | tr -d '\r')
[[ "${remote_identity}" == yz11502 ]] || fail "wrong remote identity"
remote "set -euo pipefail
test \"\$(sha256sum '${BASE_RECEIPT}' | awk '{print \$1}')\" = '${EXPECTED_BASE_RECEIPT_SHA}'
cd '${BASE_SOURCE_ROOT}' && sha256sum -c --quiet '${BASE_RECEIPT}'
test -r /scratch/yz11502/Research/datasets/goat_bench_20260814/downloads/hm3d-val-habitat-v0.2.tar
test -x /scratch/lg154/conda-envs/habitat/bin/python
test -x /scratch/lg154/conda-envs/memnav/bin/python
test -r /share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif"

if remote "test -d '${task_root}'"; then
  remote "test \"\$(sha256sum '${task_root}/SOURCE_BUNDLE.sha256' | awk '{print \$1}')\" = '${task_receipt_sha}' && cd '${task_root}' && sha256sum -c --quiet SOURCE_BUNDLE.sha256"
else
  remote "test ! -e '${task_stage}' && mkdir -p '${task_stage}'"
  rsync -a --chmod=Du=rwx,Dgo=rx,Fu=rw,Fgo=r \
    -e "ssh -o BatchMode=yes -o ControlMaster=no -S ${SSH_CONTROL_PATH}" \
    "${staging}/" "${SSH_ALIAS}:${task_stage}/"
  remote "cd '${task_stage}' && sha256sum -c --quiet SOURCE_BUNDLE.sha256 && chmod -R a-w '${task_stage}' && mv '${task_stage}' '${task_root}'"
fi

remote "test ! -e '${run_root}' && test ! -e '${smoke_root}' && mkdir -p '${run_root}/sealed_inputs' '${run_root}/logs' '${smoke_root}/logs' /scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs"
protocol=${task_root}/MemNavData/hm3d_fresh_fullmono_mixed_role_protocol_20260820.json
parent_manifest=${run_root}/sealed_inputs/parent_manifest.json
bench_root=${run_root}/benchmarks/natural_direction
task_receipt=${task_root}/SOURCE_BUNDLE.sha256

remote "singularity exec --nv -B /scratch/lg154 -B /scratch/yz11502 /share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH='${task_root}:${task_root}/MemNavData:${BASE_SOURCE_ROOT}:${BASE_SOURCE_ROOT}/MemNavData' /scratch/lg154/conda-envs/habitat/bin/python -m unittest MemNavData.test_audit_hm3d_fresh_fullmono_selection MemNavData.test_final14_role_pair_construction MemNavData.test_hm3d_fullmono_materialize MemNavData.test_hm3d_fullmono_mixed_role"

common="ALL,TASK_ROOT=${task_root},BASE_SOURCE_ROOT=${BASE_SOURCE_ROOT},RUN_ROOT=${run_root},DATA_ROOT=${DATA_ROOT},PARENT_MANIFEST=${parent_manifest},PROTOCOL=${protocol},TASK_RECEIPT=${task_receipt},EXPECTED_TASK_RECEIPT_SHA=${task_receipt_sha},BASE_RECEIPT=${BASE_RECEIPT},EXPECTED_BASE_RECEIPT_SHA=${EXPECTED_BASE_RECEIPT_SHA}"
prep=${task_root}/MemNavData/slurm_hm3d_fresh_fullmono_prepare.sbatch
generate=${task_root}/MemNavData/slurm_hm3d_fresh_fullmono_generate.sbatch
parent=${task_root}/MemNavData/slurm_hm3d_fresh_fullmono_manifest.sbatch
collect=${task_root}/MemNavData/slurm_hm3d_fullmono_goal_a.sbatch
construct=${task_root}/MemNavData/slurm_hm3d_fullmono_construct.sbatch
finalize=${task_root}/MemNavData/slurm_hm3d_fullmono_finalize.sbatch
eval_script=${task_root}/MemNavData/slurm_hm3d_fullmono_eval.sbatch
analysis=${task_root}/MemNavData/slurm_hm3d_fullmono_analysis.sbatch

remote "sbatch --test-only --export='${common}' '${prep}' >/dev/null"
remote "sbatch --test-only --array=0 --export='${common}' '${generate}' >/dev/null"
remote "sbatch --test-only --export='${common}' '${parent}' >/dev/null"
remote "sbatch --test-only --array=0 --export='${common}' '${collect}' >/dev/null"
remote "sbatch --test-only --array=0 --export='${common}' '${construct}' >/dev/null"
remote "sbatch --test-only --export='${common}' '${finalize}' >/dev/null"
remote "sbatch --test-only --array=0 --export='${common},RUN_ROOT=${smoke_root},BENCH_ROOT=${bench_root},MODE=smoke,MAX_STEPS=80' '${eval_script}' >/dev/null"
remote "sbatch --test-only --array=0 --export='${common},BENCH_ROOT=${bench_root},MODE=eval,MAX_STEPS=600' '${eval_script}' >/dev/null"

prep_raw=$(remote "sbatch --parsable --export='${common}' '${prep}'" | tr -d '\r')
prep_id=${prep_raw%%;*}; [[ "${prep_id}" =~ ^[0-9]+$ ]] || fail "bad prep job"
gen_raw=$(remote "sbatch --parsable --qos=gpu48 --array=0-53%${GEN_CONCURRENCY} --dependency=afterok:${prep_id} --kill-on-invalid-dep=yes --export='${common}' '${generate}'" | tr -d '\r')
gen_id=${gen_raw%%;*}; [[ "${gen_id}" =~ ^[0-9]+$ ]] || fail "bad generation job"
parent_raw=$(remote "sbatch --parsable --dependency=afterok:${gen_id} --kill-on-invalid-dep=yes --export='${common}' '${parent}'" | tr -d '\r')
parent_id=${parent_raw%%;*}; [[ "${parent_id}" =~ ^[0-9]+$ ]] || fail "bad parent job"
collect_raw=$(remote "sbatch --parsable --qos=gpu48 --array=0-53%${COLLECT_CONCURRENCY} --dependency=afterok:${parent_id} --kill-on-invalid-dep=yes --export='${common}' '${collect}'" | tr -d '\r')
collect_id=${collect_raw%%;*}; [[ "${collect_id}" =~ ^[0-9]+$ ]] || fail "bad collect job"
construct_raw=$(remote "sbatch --parsable --qos=gpu48 --array=0-53%${CONSTRUCT_CONCURRENCY} --dependency=afterok:${collect_id} --kill-on-invalid-dep=yes --export='${common}' '${construct}'" | tr -d '\r')
construct_id=${construct_raw%%;*}; [[ "${construct_id}" =~ ^[0-9]+$ ]] || fail "bad construct job"
finalize_raw=$(remote "sbatch --parsable --dependency=afterok:${construct_id} --kill-on-invalid-dep=yes --export='${common}' '${finalize}'" | tr -d '\r')
finalize_id=${finalize_raw%%;*}; [[ "${finalize_id}" =~ ^[0-9]+$ ]] || fail "bad finalize job"
smoke_raw=$(remote "sbatch --parsable --qos=gpu48 --array=0 --dependency=afterok:${finalize_id} --kill-on-invalid-dep=yes --export='${common},RUN_ROOT=${smoke_root},BENCH_ROOT=${bench_root},MODE=smoke,MAX_STEPS=80' '${eval_script}'" | tr -d '\r')
smoke_id=${smoke_raw%%;*}; [[ "${smoke_id}" =~ ^[0-9]+$ ]] || fail "bad smoke job"
eval_raw=$(remote "sbatch --parsable --qos=gpu48 --array=0-53%${EVAL_CONCURRENCY} --dependency=afterok:${smoke_id} --kill-on-invalid-dep=yes --export='${common},BENCH_ROOT=${bench_root},MODE=eval,MAX_STEPS=600' '${eval_script}'" | tr -d '\r')
eval_id=${eval_raw%%;*}; [[ "${eval_id}" =~ ^[0-9]+$ ]] || fail "bad eval job"
summary_raw=$(remote "sbatch --parsable --dependency=afterok:${eval_id} --kill-on-invalid-dep=yes --export='${common},BENCH_ROOT=${bench_root},MODE=summary' '${analysis}'" | tr -d '\r')
summary_id=${summary_raw%%;*}; [[ "${summary_id}" =~ ^[0-9]+$ ]] || fail "bad summary job"
verify_raw=$(remote "sbatch --parsable --dependency=afterok:${summary_id} --kill-on-invalid-dep=yes --export='${common},BENCH_ROOT=${bench_root},MODE=verify' '${analysis}'" | tr -d '\r')
verify_id=${verify_raw%%;*}; [[ "${verify_id}" =~ ^[0-9]+$ ]] || fail "bad verify job"

receipt=MemNavData/HM3D_FRESH_FULLMONO_MIXED_ROLE_SUBMISSION_RECEIPT_20260820.json
[[ ! -e "${receipt}" ]] || fail "local receipt already exists"
"${LOCAL_PY}" - "${receipt}" "${run_root}" "${smoke_root}" "${task_root}" \
  "${task_receipt_sha}" "${DATA_ROOT}" "${prep_id}" "${gen_id}" \
  "${parent_id}" "${collect_id}" "${construct_id}" "${finalize_id}" \
  "${smoke_id}" "${eval_id}" "${summary_id}" "${verify_id}" <<'PY'
import json,sys
(path,run,smoke,bundle,sha,data,prep,generation,parent,collect,construct,
 finalize,smoke_job,evaluation,summary,verify)=sys.argv[1:]
payload={
 "schema_version":"hm3d_fresh_fullmono_submission_v1_20260820",
 "scope":"fresh-scene HM3D actual-online Full-Mono mixed-role confirmation",
 "run_root":run,"smoke_root":smoke,"task_bundle":bundle,
 "task_receipt_sha256":sha,"data_root":data,"reserve_scenes":54,
 "target_source_episodes":216,"query_outcomes_read_at_submission":False,
 "jobs":{"asset_prepare":int(prep),"source_generation_array":int(generation),
         "parent_manifest":int(parent),"goal_a_collection_array":int(collect),
         "construction_array":int(construct),"population_finalize":int(finalize),
         "query_smoke":int(smoke_job),"query_evaluation_array":int(evaluation),
         "summary":int(summary),"independent_verification":int(verify)},
}
open(path,"x").write(json.dumps(payload,indent=2,sort_keys=True)+"\n")
print(json.dumps(payload,indent=2,sort_keys=True))
PY
scp -q -o BatchMode=yes -o ControlMaster=no \
  -o ControlPath="${SSH_CONTROL_PATH}" \
  "${ROOT}/${receipt}" "${SSH_ALIAS}:${run_root}/submission.json"
remote "sha256sum '${run_root}/submission.json' >'${run_root}/submission.json.sha256'"
printf 'RUN_ROOT=%s\nSMOKE_ROOT=%s\nTASK_ROOT=%s\nPREP=%s\nGEN=%s\nPARENT=%s\nCOLLECT=%s\nCONSTRUCT=%s\nFINALIZE=%s\nSMOKE=%s\nEVAL=%s\nSUMMARY=%s\nVERIFY=%s\n' \
  "${run_root}" "${smoke_root}" "${task_root}" "${prep_id}" "${gen_id}" \
  "${parent_id}" "${collect_id}" "${construct_id}" "${finalize_id}" \
  "${smoke_id}" "${eval_id}" "${summary_id}" "${verify_id}"
