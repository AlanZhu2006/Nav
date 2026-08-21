#!/usr/bin/env bash
# Repair the reproducible HM3D mono-depth transport mismatch, then resume formal eval.
set -euo pipefail
umask 0022

ROOT=${ROOT:-$(git rev-parse --show-toplevel)}
SSH_ALIAS=${SSH_ALIAS:-alantorch}
SSH_CONTROL_PATH=${SSH_CONTROL_PATH:-$(
  ssh -G "${SSH_ALIAS}" 2>/dev/null |
    awk '$1=="controlpath"{value=$2} END{print value}'
)}
LOCAL_MEMNAV_PY=${LOCAL_MEMNAV_PY:-/home/asus/miniconda3/envs/memnav/bin/python}
LOCAL_HAB_PY=${LOCAL_HAB_PY:-/home/asus/miniconda3/envs/habitat/bin/python}
REMOTE_BUNDLES=${REMOTE_BUNDLES:-/scratch/yz11502/Research/Nav-axis-uturn-source-bundles}
RUN_ROOT=/scratch/yz11502/Research/Nav-axis-uturn-results/hm3d_fresh_fullmono_mixed_role_20260820/formal_20260820T143609Z_e6dd44c6
SMOKE_ROOT=${RUN_ROOT}_transaction_smoke
BASE_SOURCE_ROOT=/scratch/yz11502/Research/Nav-axis-uturn-source-bundles/final14_mono_factorial_5690569a4373f2d2
BASE_RECEIPT=${BASE_SOURCE_ROOT}/source_inputs.sha256
EXPECTED_BASE_RECEIPT_SHA=5690569a4373f2d2768671418f0c604c4a03aa4b0ffe01baf70b288af03ba216
PARENT_MANIFEST=${RUN_ROOT}/sealed_inputs/parent_manifest.json
EXPECTED_PARENT_MANIFEST_SHA=a96a0b96fab7b7b47709b36cb8eeb9410b42b09f095f87ef01304a68de716dd5
FAILED_REPAIR_ARRAY=16126593
REPAIR_INDEX=29
RUNTIME_ATTEMPT=repair_20260821_v2_transaction
DRY_RUN=${DRY_RUN:-0}
RECEIPT=${ROOT}/MemNavData/HM3D_FULLMONO_TRANSACTION_REPAIR_SUBMISSION_RECEIPT_20260821.json

fail() { echo "ABORT: $*" >&2; exit 2; }
remote() {
  ssh -o BatchMode=yes -o ControlMaster=no -S "${SSH_CONTROL_PATH}" \
    "${SSH_ALIAS}" "$@"
}
[[ -S "${SSH_CONTROL_PATH}" ]] || fail "authoritative SSH master missing"
[[ "${DRY_RUN}" =~ ^[01]$ ]] || fail "DRY_RUN must be 0 or 1"
[[ ! -e "${RECEIPT}" ]] || fail "local transaction repair receipt exists"
cd "${ROOT}"

files=(
  MemNavData/HM3D_FRESH_FULLMONO_MIXED_ROLE_PROTOCOL_20260820.md
  MemNavData/HM3D_FULLMONO_RUNTIME_REPAIR_20260821.md
  MemNavData/hm3d_fresh_fullmono_mixed_role_protocol_20260820.json
  MemNavData/hm3d_fullmono_mixed_role_protocol_20260820.json
  MemNavData/hm3d_consumed_scene_audit_20260816.json
  MemNavData/hm3d_heldout_val10_revisit_protocol_20260816.json
  .diagnostics/datasets/goat-bench/hm3d_val_receipts/hm3d-val-habitat-v0.2.members.txt
  MemNavData/deterministic_eval_protocol.py
  MemNavData/monocular_depth_runtime.py
  MemNavData/eval_2leg_habitat.py
  MemNavData/eval_shared_online_role_pairs.py
  MemNavData/audit_hm3d_fresh_fullmono_selection.py
  MemNavData/build_hm3d_fresh_fullmono_parent_manifest.py
  MemNavData/generate_twoleg.py
  MemNavData/final14_mono_factorial.py
  MemNavData/mdtec_raw_depth_gate_d.py
  MemNavData/hm3d_fullmono_mixed_role.py
  MemNavData/audit_hm3d_fullmono_inputs.py
  MemNavData/collect_hm3d_fullmono_goal_a.py
  MemNavData/materialize_online_a_traces.py
  MemNavData/materialize_hm3d_fullmono_online_a.py
  MemNavData/build_final14_role_pair_scene.py
  MemNavData/shared_online_role_pair_contract.py
  MemNavData/audit_shared_online_role_pairs.py
  MemNavData/construct_hm3d_fullmono_role_pairs.py
  MemNavData/finalize_hm3d_fullmono_mixed_role.py
  MemNavData/run_hm3d_fullmono_query_history.py
  MemNavData/summarize_hm3d_fullmono_mixed_role.py
  MemNavData/independent_verify_hm3d_fullmono_mixed_role.py
  MemNavData/hm3d_fullmono_repair_audit.py
  MemNavData/run_hm3d_fullmono_server_scene.sh
  MemNavData/slurm_hm3d_fullmono_goal_a_repair.sbatch
  MemNavData/slurm_hm3d_fullmono_repair_barrier.sbatch
  MemNavData/slurm_hm3d_fullmono_construct.sbatch
  MemNavData/slurm_hm3d_fullmono_finalize.sbatch
  MemNavData/slurm_hm3d_fullmono_eval.sbatch
  MemNavData/slurm_hm3d_fullmono_analysis.sbatch
  MemNavData/test_monocular_depth_runtime.py
  MemNavData/test_collect_hm3d_fullmono_goal_a.py
  MemNavData/test_hm3d_fullmono_repair_audit.py
  MemNavData/test_hm3d_fullmono_mixed_role.py
  MemNavData/test_hm3d_fullmono_materialize.py
  MemNavData/test_final14_role_pair_construction.py
  MemNavData/test_shared_online_role_pair_contract.py
  MemNavData/submit_hm3d_fullmono_transaction_repair_hpc.sh
  NavDP/baselines/navdp/deterministic_seed.py
  NavDP/baselines/navdp/navdp_server.py
  NavDP/baselines/navdp/policy_agent.py
  NavDP/baselines/navdp/policy_backbone.py
  NavDP/baselines/navdp/policy_network.py
  NavDP/baselines/memnav/memnav_server.py
  NavDP/baselines/memnav/policy_agent.py
  NavDP/baselines/memnav/policy_backbone.py
  NavDP/baselines/memnav/policy_network.py
  NavDP/baselines/memnav/pose_alignment.py
  NavDP/baselines/memnav/reverse_memory_graph.py
  NavDP/baselines/memnav/router_candidates.py
)
for file in "${files[@]}"; do
  [[ -f "${file}" && ! -L "${file}" ]] || fail "missing physical ${file}"
done

"${LOCAL_HAB_PY}" -m py_compile \
  MemNavData/monocular_depth_runtime.py \
  MemNavData/eval_2leg_habitat.py \
  MemNavData/collect_hm3d_fullmono_goal_a.py \
  MemNavData/run_hm3d_fullmono_query_history.py \
  NavDP/baselines/navdp/navdp_server.py \
  NavDP/baselines/memnav/memnav_server.py
PYTHONPATH="${ROOT}:${ROOT}/MemNavData" "${LOCAL_HAB_PY}" -m unittest \
  MemNavData.test_monocular_depth_runtime \
  MemNavData.test_collect_hm3d_fullmono_goal_a \
  MemNavData.test_hm3d_fullmono_repair_audit \
  MemNavData.test_hm3d_fullmono_materialize \
  MemNavData.test_hm3d_fullmono_mixed_role
bash -n \
  MemNavData/run_hm3d_fullmono_server_scene.sh \
  MemNavData/slurm_hm3d_fullmono_goal_a_repair.sbatch \
  MemNavData/slurm_hm3d_fullmono_repair_barrier.sbatch \
  MemNavData/slurm_hm3d_fullmono_construct.sbatch \
  MemNavData/slurm_hm3d_fullmono_finalize.sbatch \
  MemNavData/slurm_hm3d_fullmono_eval.sbatch \
  MemNavData/slurm_hm3d_fullmono_analysis.sbatch

staging=$(mktemp -d)
cleanup() { rm -rf -- "${staging}"; }
trap cleanup EXIT
for file in "${files[@]}"; do
  mkdir -p "${staging}/$(dirname "${file}")"
  cp --preserve=mode,timestamps "${file}" "${staging}/${file}"
done
(
  cd "${staging}"
  PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="${staging}:${staging}/MemNavData" \
    "${LOCAL_HAB_PY}" -m unittest \
      MemNavData.test_monocular_depth_runtime \
      MemNavData.test_collect_hm3d_fullmono_goal_a \
      MemNavData.test_hm3d_fullmono_repair_audit \
      MemNavData.test_hm3d_fullmono_materialize \
      MemNavData.test_hm3d_fullmono_mixed_role
)
head=$(git rev-parse HEAD)
"${LOCAL_MEMNAV_PY}" - "${staging}" "${head}" <<'PY'
import hashlib,json,sys
from pathlib import Path
root=Path(sys.argv[1]); files={}
for path in sorted(root.rglob("*")):
    if path.is_symlink(): raise SystemExit(f"bundle symlink: {path}")
    if path.is_file() and path.name not in {"source_bundle_manifest.json","SOURCE_BUNDLE.sha256"}:
        files[path.relative_to(root).as_posix()]=hashlib.sha256(path.read_bytes()).hexdigest()
payload={
 "schema":"hm3d_fullmono_transaction_repair_bundle_v2_20260821",
 "local_git_head_context":sys.argv[2],
 "repair_index":29,
 "change":"atomic SHA/frame-bound planning append and monocular depth receipt",
 "model_depth_or_navigation_output_change":False,
 "method_threshold_checkpoint_seed_budget_or_population_change":False,
 "query_outcomes_read":False,
 "files":files,
}
(root/"source_bundle_manifest.json").write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n")
PY
(
  cd "${staging}"
  find . -type f ! -name SOURCE_BUNDLE.sha256 -print0 | sort -z | \
    xargs -0 sha256sum >SOURCE_BUNDLE.sha256
  sha256sum -c SOURCE_BUNDLE.sha256 >/dev/null
)
task_receipt_sha=$(sha256sum "${staging}/SOURCE_BUNDLE.sha256" | awk '{print $1}')
manifest_sha=$(sha256sum "${staging}/source_bundle_manifest.json" | awk '{print $1}')
task_root=${REMOTE_BUNDLES}/hm3d_fullmono_transaction_repair_${manifest_sha:0:16}
task_stage=${task_root}.partial.$$
protocol=${task_root}/MemNavData/hm3d_fresh_fullmono_mixed_role_protocol_20260820.json
task_receipt=${task_root}/SOURCE_BUNDLE.sha256
bench_root=${RUN_ROOT}/benchmarks/natural_direction
repair_root=${RUN_ROOT}/transaction_repair
pre_inventory=${repair_root}/pre_transaction_inventory.json

if [[ "${DRY_RUN}" == 1 ]]; then
  echo "DRY_RUN_TASK_ROOT=${task_root}"
  echo "DRY_RUN_TASK_RECEIPT_SHA=${task_receipt_sha}"
  exit 0
fi

remote "set -euo pipefail
test \"\$(sha256sum '${PARENT_MANIFEST}' | awk '{print \$1}')\" = '${EXPECTED_PARENT_MANIFEST_SHA}'
test \"\$(sha256sum '${BASE_RECEIPT}' | awk '{print \$1}')\" = '${EXPECTED_BASE_RECEIPT_SHA}'
state=\$(sacct -j '${FAILED_REPAIR_ARRAY}_${REPAIR_INDEX}' -X -n -o State | xargs)
test \"\${state}\" = FAILED
test ! -e '${RUN_ROOT}/goal_a/scenes/29_fsQtJ8t3nTf/completion.json'
test -d '${RUN_ROOT}/goal_a/scenes/29_fsQtJ8t3nTf/episode_0003'
test -z \"\$(find '${RUN_ROOT}/goal_a/scenes/29_fsQtJ8t3nTf/episode_0003' -mindepth 1 -print -quit)\"
test -f '${RUN_ROOT}/goal_a/scenes/46_*/completion.json' 2>/dev/null || test \"\$(find '${RUN_ROOT}/goal_a/scenes' -maxdepth 2 -path '*/46_*/*completion.json' | wc -l)\" -eq 1
test \"\$(find '${RUN_ROOT}/goal_a/scenes' -maxdepth 2 -path '*/47_*/*completion.json' | wc -l)\" -eq 1
test \"\$(find '${RUN_ROOT}/goal_a/scenes' -maxdepth 2 -path '*/53_*/*completion.json' | wc -l)\" -eq 1
test ! -e '${RUN_ROOT}/construction'
test ! -e '${RUN_ROOT}/benchmarks'
test ! -e '${repair_root}'
test ! -e '${SMOKE_ROOT}'"

if remote "test -d '${task_root}' && test \"\$(sha256sum '${task_root}/SOURCE_BUNDLE.sha256' | awk '{print \$1}')\" = '${task_receipt_sha}' && cd '${task_root}' && sha256sum -c --quiet SOURCE_BUNDLE.sha256"; then
  echo "Reusing verified bundle ${task_root}"
else
  remote "test ! -e '${task_root}' && test ! -e '${task_stage}' && mkdir -p '${task_stage}'"
  rsync -a --chmod=Du=rwx,Dgo=rx,Fu=rw,Fgo=r \
    -e "ssh -o BatchMode=yes -o ControlMaster=no -S ${SSH_CONTROL_PATH}" \
    "${staging}/" "${SSH_ALIAS}:${task_stage}/"
  remote "cd '${task_stage}' && sha256sum -c --quiet SOURCE_BUNDLE.sha256 && chmod -R a-w '${task_stage}' && mv '${task_stage}' '${task_root}'"
fi

remote "set -euo pipefail
mkdir -p '${repair_root}'
singularity exec -B /scratch/lg154 -B /scratch/yz11502 /share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH='${task_root}:${task_root}/MemNavData:${BASE_SOURCE_ROOT}:${BASE_SOURCE_ROOT}/MemNavData' /scratch/lg154/conda-envs/habitat/bin/python -u '${task_root}/MemNavData/hm3d_fullmono_repair_audit.py' snapshot --run-root '${RUN_ROOT}' --protocol '${protocol}' --parent-manifest '${PARENT_MANIFEST}' --repair-indices '${REPAIR_INDEX}' --out '${pre_inventory}'
chmod a-w '${pre_inventory}' '${pre_inventory}.sha256'"

common="ALL,TASK_ROOT=${task_root},BASE_SOURCE_ROOT=${BASE_SOURCE_ROOT},RUN_ROOT=${RUN_ROOT},PARENT_MANIFEST=${PARENT_MANIFEST},PROTOCOL=${protocol},TASK_RECEIPT=${task_receipt},EXPECTED_TASK_RECEIPT_SHA=${task_receipt_sha},BASE_RECEIPT=${BASE_RECEIPT},EXPECTED_BASE_RECEIPT_SHA=${EXPECTED_BASE_RECEIPT_SHA}"
repair_script=${task_root}/MemNavData/slurm_hm3d_fullmono_goal_a_repair.sbatch
barrier_script=${task_root}/MemNavData/slurm_hm3d_fullmono_repair_barrier.sbatch
construct_script=${task_root}/MemNavData/slurm_hm3d_fullmono_construct.sbatch
finalize_script=${task_root}/MemNavData/slurm_hm3d_fullmono_finalize.sbatch
eval_script=${task_root}/MemNavData/slurm_hm3d_fullmono_eval.sbatch
analysis_script=${task_root}/MemNavData/slurm_hm3d_fullmono_analysis.sbatch

remote "sbatch --test-only --array=${REPAIR_INDEX} --export='${common},RUNTIME_ATTEMPT=${RUNTIME_ATTEMPT}' '${repair_script}' >/dev/null"
repair_raw=$(remote "sbatch --parsable --qos=gpu48 --array=${REPAIR_INDEX} --export='${common},RUNTIME_ATTEMPT=${RUNTIME_ATTEMPT}' '${repair_script}'")
repair_id=${repair_raw%%;*}; [[ "${repair_id}" =~ ^[0-9]+$ ]] || fail "bad repair job"
barrier_raw=$(remote "sbatch --parsable --dependency=afterok:${repair_id} --kill-on-invalid-dep=yes --export='${common},PRE_REPAIR_INVENTORY=${pre_inventory}' '${barrier_script}'")
barrier_id=${barrier_raw%%;*}; [[ "${barrier_id}" =~ ^[0-9]+$ ]] || fail "bad barrier job"
construct_raw=$(remote "sbatch --parsable --qos=gpu48 --array=0-53%2 --dependency=afterok:${barrier_id} --kill-on-invalid-dep=yes --export='${common}' '${construct_script}'")
construct_id=${construct_raw%%;*}; [[ "${construct_id}" =~ ^[0-9]+$ ]] || fail "bad construction job"
finalize_raw=$(remote "sbatch --parsable --dependency=afterok:${construct_id} --kill-on-invalid-dep=yes --export='${common}' '${finalize_script}'")
finalize_id=${finalize_raw%%;*}; [[ "${finalize_id}" =~ ^[0-9]+$ ]] || fail "bad finalize job"
smoke_raw=$(remote "sbatch --parsable --qos=gpu48 --array=0 --dependency=afterok:${finalize_id} --kill-on-invalid-dep=yes --export='${common},RUN_ROOT=${SMOKE_ROOT},BENCH_ROOT=${bench_root},MODE=smoke,MAX_STEPS=80' '${eval_script}'")
smoke_id=${smoke_raw%%;*}; [[ "${smoke_id}" =~ ^[0-9]+$ ]] || fail "bad smoke job"
eval_raw=$(remote "sbatch --parsable --qos=gpu48 --array=0-53%2 --dependency=afterok:${smoke_id} --kill-on-invalid-dep=yes --export='${common},BENCH_ROOT=${bench_root},MODE=eval,MAX_STEPS=600' '${eval_script}'")
eval_id=${eval_raw%%;*}; [[ "${eval_id}" =~ ^[0-9]+$ ]] || fail "bad eval job"
summary_raw=$(remote "sbatch --parsable --dependency=afterok:${eval_id} --kill-on-invalid-dep=yes --export='${common},BENCH_ROOT=${bench_root},MODE=summary' '${analysis_script}'")
summary_id=${summary_raw%%;*}; [[ "${summary_id}" =~ ^[0-9]+$ ]] || fail "bad summary job"
verify_raw=$(remote "sbatch --parsable --dependency=afterok:${summary_id} --kill-on-invalid-dep=yes --export='${common},BENCH_ROOT=${bench_root},MODE=verify' '${analysis_script}'")
verify_id=${verify_raw%%;*}; [[ "${verify_id}" =~ ^[0-9]+$ ]] || fail "bad verify job"

"${LOCAL_MEMNAV_PY}" - "${RECEIPT}" "${task_root}" "${task_receipt_sha}" \
  "${pre_inventory}" "${repair_id}" "${barrier_id}" "${construct_id}" \
  "${finalize_id}" "${smoke_id}" "${eval_id}" "${summary_id}" \
  "${verify_id}" <<'PY'
import json,sys
(path,bundle,sha,inventory,repair,barrier,construct,finalize,smoke,evaluation,summary,verify)=sys.argv[1:]
payload={
 "schema_version":"hm3d_fullmono_transaction_repair_submission_v2_20260821",
 "run_root":"/scratch/yz11502/Research/Nav-axis-uturn-results/hm3d_fresh_fullmono_mixed_role_20260820/formal_20260820T143609Z_e6dd44c6",
 "task_bundle":bundle,"task_receipt_sha256":sha,
 "pre_transaction_inventory":inventory,"repair_indices":[29],
 "superseded_repair_array":16126593,
 "jobs":{"goal_a_transaction_repair":int(repair),"collection_barrier":int(barrier),
         "construction_array":int(construct),"population_finalize":int(finalize),
         "query_smoke":int(smoke),"query_evaluation_array":int(evaluation),
         "summary":int(summary),"independent_verification":int(verify)},
 "scientific_guards":{"model_or_depth_output_changed":False,
   "method_threshold_checkpoint_seed_budget_or_population_changed":False,
   "query_outcomes_read_before_repair":False,
   "completed_episode_overwrite_allowed":False,
   "transport_is_sha_frame_and_depth_bound":True},
}
open(path,"x").write(json.dumps(payload,indent=2,sort_keys=True)+"\n")
print(json.dumps(payload,indent=2,sort_keys=True))
PY
scp -q -o BatchMode=yes -o ControlMaster=no \
  -o ControlPath="${SSH_CONTROL_PATH}" "${RECEIPT}" \
  "${SSH_ALIAS}:${repair_root}/submission.json"
remote "sha256sum '${repair_root}/submission.json' >'${repair_root}/submission.json.sha256'"
echo "REPAIR=${repair_id} BARRIER=${barrier_id} CONSTRUCT=${construct_id} FINALIZE=${finalize_id}"
echo "SMOKE=${smoke_id} EVAL=${eval_id} SUMMARY=${summary_id} VERIFY=${verify_id}"
