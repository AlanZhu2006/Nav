#!/usr/bin/env bash
# Submit an immutable additive repair and replacement downstream HM3D chain.
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
RUN_ROOT=${RUN_ROOT:-/scratch/yz11502/Research/Nav-axis-uturn-results/hm3d_fresh_fullmono_mixed_role_20260820/formal_20260820T143609Z_e6dd44c6}
SMOKE_ROOT=${SMOKE_ROOT:-${RUN_ROOT}_smoke}
BASE_SOURCE_ROOT=${BASE_SOURCE_ROOT:-/scratch/yz11502/Research/Nav-axis-uturn-source-bundles/final14_mono_factorial_5690569a4373f2d2}
BASE_RECEIPT=${BASE_SOURCE_ROOT}/source_inputs.sha256
EXPECTED_BASE_RECEIPT_SHA=${EXPECTED_BASE_RECEIPT_SHA:-5690569a4373f2d2768671418f0c604c4a03aa4b0ffe01baf70b288af03ba216}
ORIGINAL_TASK_ROOT=${ORIGINAL_TASK_ROOT:-/scratch/yz11502/Research/Nav-axis-uturn-source-bundles/hm3d_fresh_fullmono_mixed_role_e6dd44c66eb72d90}
EXPECTED_ORIGINAL_TASK_RECEIPT_SHA=${EXPECTED_ORIGINAL_TASK_RECEIPT_SHA:-e6dd44c66eb72d905f4db96db9f604c042cc0d89c77a16cd4dc92eabb99c8f01}
PARENT_MANIFEST=${PARENT_MANIFEST:-${RUN_ROOT}/sealed_inputs/parent_manifest.json}
EXPECTED_PARENT_MANIFEST_SHA=${EXPECTED_PARENT_MANIFEST_SHA:-a96a0b96fab7b7b47709b36cb8eeb9410b42b09f095f87ef01304a68de716dd5}
ORIGINAL_COLLECT_JOB=${ORIGINAL_COLLECT_JOB:-16120334}
REPAIR_INDICES=${REPAIR_INDICES:-29,46,47,53}
RUNTIME_ATTEMPT=${RUNTIME_ATTEMPT:-repair_20260821_v1}
CONCURRENCY=${CONCURRENCY:-2}
DRY_RUN=${DRY_RUN:-0}

fail() { echo "ABORT: $*" >&2; exit 2; }
remote() {
  ssh -o BatchMode=yes -o ControlMaster=no -S "${SSH_CONTROL_PATH}" \
    "${SSH_ALIAS}" "$@"
}
[[ -S "${SSH_CONTROL_PATH}" ]] || fail "authoritative SSH master missing"
[[ "${REPAIR_INDICES}" == "29,46,47,53" ]] || fail "frozen repair indices changed"
[[ "${RUNTIME_ATTEMPT}" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]] || \
  fail "invalid runtime attempt"
[[ "${CONCURRENCY}" =~ ^[1-9][0-9]*$ ]] || fail "invalid concurrency"
(( CONCURRENCY <= 2 )) || fail "repair concurrency exceeds observed QoS"
[[ "${DRY_RUN}" =~ ^[01]$ ]] || fail "DRY_RUN must be 0 or 1"
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
  MemNavData/test_collect_hm3d_fullmono_goal_a.py
  MemNavData/test_hm3d_fullmono_repair_audit.py
  MemNavData/test_hm3d_fullmono_mixed_role.py
  MemNavData/test_hm3d_fullmono_materialize.py
  MemNavData/test_final14_role_pair_construction.py
  MemNavData/test_shared_online_role_pair_contract.py
  MemNavData/submit_hm3d_fullmono_runtime_repair_hpc.sh
)
for file in "${files[@]}"; do
  [[ -f "${file}" && ! -L "${file}" ]] || fail "missing physical ${file}"
done

"${LOCAL_HAB_PY}" -m py_compile \
  MemNavData/collect_hm3d_fullmono_goal_a.py \
  MemNavData/hm3d_fullmono_repair_audit.py \
  MemNavData/construct_hm3d_fullmono_role_pairs.py \
  MemNavData/finalize_hm3d_fullmono_mixed_role.py
PYTHONPATH="${ROOT}:${ROOT}/MemNavData" "${LOCAL_HAB_PY}" -m unittest \
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
  PYTHONDONTWRITEBYTECODE=1 \
  PYTHONPATH="${staging}:${staging}/MemNavData" \
    "${LOCAL_HAB_PY}" -m unittest \
      MemNavData.test_collect_hm3d_fullmono_goal_a \
      MemNavData.test_hm3d_fullmono_repair_audit \
      MemNavData.test_hm3d_fullmono_materialize \
      MemNavData.test_hm3d_fullmono_mixed_role
)
head=$(git rev-parse HEAD)
"${LOCAL_MEMNAV_PY}" - "${staging}" "${head}" \
  "${EXPECTED_ORIGINAL_TASK_RECEIPT_SHA}" "${EXPECTED_PARENT_MANIFEST_SHA}" <<'PY'
import hashlib,json,sys
from pathlib import Path
root=Path(sys.argv[1]); files={}
for path in sorted(root.rglob("*")):
    if path.is_symlink(): raise SystemExit(f"bundle symlink: {path}")
    if path.is_file() and path.name not in {"source_bundle_manifest.json","SOURCE_BUNDLE.sha256"}:
        files[path.relative_to(root).as_posix()]=hashlib.sha256(path.read_bytes()).hexdigest()
payload={
 "schema":"hm3d_fullmono_runtime_repair_bundle_v1_20260821",
 "local_git_head_context":sys.argv[2],
 "superseded_task_receipt_sha256":sys.argv[3],
 "frozen_parent_manifest_sha256":sys.argv[4],
 "repair_indices":[29,46,47,53],
 "change":"additive Goal-A resume, pre-repair hash inventory, and full collection barrier",
 "method_or_threshold_change":False,
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
task_root=${REMOTE_BUNDLES}/hm3d_fullmono_runtime_repair_${manifest_sha:0:16}
task_stage=${task_root}.partial.$$
protocol=${task_root}/MemNavData/hm3d_fresh_fullmono_mixed_role_protocol_20260820.json
task_receipt=${task_root}/SOURCE_BUNDLE.sha256
bench_root=${RUN_ROOT}/benchmarks/natural_direction
pre_inventory=${RUN_ROOT}/repair/pre_repair_inventory.json

if [[ "${DRY_RUN}" == 1 ]]; then
  echo "DRY_RUN_TASK_ROOT=${task_root}"
  echo "DRY_RUN_TASK_RECEIPT_SHA=${task_receipt_sha}"
  echo "DRY_RUN_RUN_ROOT=${RUN_ROOT}"
  exit 0
fi

remote "set -euo pipefail
test \"\$(sha256sum '${ORIGINAL_TASK_ROOT}/SOURCE_BUNDLE.sha256' | awk '{print \$1}')\" = '${EXPECTED_ORIGINAL_TASK_RECEIPT_SHA}'
test \"\$(sha256sum '${PARENT_MANIFEST}' | awk '{print \$1}')\" = '${EXPECTED_PARENT_MANIFEST_SHA}'
test \"\$(sha256sum '${BASE_RECEIPT}' | awk '{print \$1}')\" = '${EXPECTED_BASE_RECEIPT_SHA}'
test ! -e '${pre_inventory}'
test ! -e '${RUN_ROOT}/benchmarks'
test ! -e '${RUN_ROOT}/construction'"
if remote "test -d '${task_root}' && test \"\$(sha256sum '${task_root}/SOURCE_BUNDLE.sha256' | awk '{print \$1}')\" = '${task_receipt_sha}' && cd '${task_root}' && sha256sum -c --quiet SOURCE_BUNDLE.sha256"; then
  echo "Reusing verified bundle ${task_root}"
else
  remote "test ! -e '${task_root}' && test ! -e '${task_stage}' && mkdir -p '${task_stage}'"
  rsync -a --chmod=Du=rwx,Dgo=rx,Fu=rw,Fgo=r \
    -e "ssh -o BatchMode=yes -o ControlMaster=no -S ${SSH_CONTROL_PATH}" \
    "${staging}/" "${SSH_ALIAS}:${task_stage}/"
  remote "cd '${task_stage}' && sha256sum -c --quiet SOURCE_BUNDLE.sha256 && chmod -R a-w '${task_stage}' && mv '${task_stage}' '${task_root}'"
fi

# Freeze the partial scene contents before any repair allocation starts.
remote "set -euo pipefail
mkdir -p '${RUN_ROOT}/repair'
singularity exec -B /scratch/lg154 -B /scratch/yz11502 /share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH='${task_root}:${task_root}/MemNavData:${BASE_SOURCE_ROOT}:${BASE_SOURCE_ROOT}/MemNavData' /scratch/lg154/conda-envs/habitat/bin/python -u '${task_root}/MemNavData/hm3d_fullmono_repair_audit.py' snapshot --run-root '${RUN_ROOT}' --protocol '${protocol}' --parent-manifest '${PARENT_MANIFEST}' --repair-indices '${REPAIR_INDICES}' --out '${pre_inventory}'
chmod a-w '${pre_inventory}' '${pre_inventory}.sha256'"

common="ALL,TASK_ROOT=${task_root},BASE_SOURCE_ROOT=${BASE_SOURCE_ROOT},RUN_ROOT=${RUN_ROOT},PARENT_MANIFEST=${PARENT_MANIFEST},PROTOCOL=${protocol},TASK_RECEIPT=${task_receipt},EXPECTED_TASK_RECEIPT_SHA=${task_receipt_sha},BASE_RECEIPT=${BASE_RECEIPT},EXPECTED_BASE_RECEIPT_SHA=${EXPECTED_BASE_RECEIPT_SHA}"
repair_script=${task_root}/MemNavData/slurm_hm3d_fullmono_goal_a_repair.sbatch
barrier_script=${task_root}/MemNavData/slurm_hm3d_fullmono_repair_barrier.sbatch
construct_script=${task_root}/MemNavData/slurm_hm3d_fullmono_construct.sbatch
finalize_script=${task_root}/MemNavData/slurm_hm3d_fullmono_finalize.sbatch
eval_script=${task_root}/MemNavData/slurm_hm3d_fullmono_eval.sbatch
analysis_script=${task_root}/MemNavData/slurm_hm3d_fullmono_analysis.sbatch

remote "sbatch --test-only --array=29 --export='${common},RUNTIME_ATTEMPT=${RUNTIME_ATTEMPT}' '${repair_script}' >/dev/null"
repair_raw=$(remote "sbatch --parsable --qos=gpu48 --array=${REPAIR_INDICES}%${CONCURRENCY} --export='${common},RUNTIME_ATTEMPT=${RUNTIME_ATTEMPT}' '${repair_script}'")
repair_id=${repair_raw%%;*}; [[ "${repair_id}" =~ ^[0-9]+$ ]] || fail "bad repair array"
barrier_dependency="afterany:${ORIGINAL_COLLECT_JOB},afterok:${repair_id}"
remote "sbatch --test-only --dependency='${barrier_dependency}' --kill-on-invalid-dep=yes --export='${common},PRE_REPAIR_INVENTORY=${pre_inventory}' '${barrier_script}' >/dev/null"
barrier_raw=$(remote "sbatch --parsable --dependency='${barrier_dependency}' --kill-on-invalid-dep=yes --export='${common},PRE_REPAIR_INVENTORY=${pre_inventory}' '${barrier_script}'")
barrier_id=${barrier_raw%%;*}; [[ "${barrier_id}" =~ ^[0-9]+$ ]] || fail "bad barrier job"
construct_raw=$(remote "sbatch --parsable --qos=gpu48 --array=0-53%${CONCURRENCY} --dependency=afterok:${barrier_id} --kill-on-invalid-dep=yes --export='${common}' '${construct_script}'")
construct_id=${construct_raw%%;*}; [[ "${construct_id}" =~ ^[0-9]+$ ]] || fail "bad construction array"
finalize_raw=$(remote "sbatch --parsable --dependency=afterok:${construct_id} --kill-on-invalid-dep=yes --export='${common}' '${finalize_script}'")
finalize_id=${finalize_raw%%;*}; [[ "${finalize_id}" =~ ^[0-9]+$ ]] || fail "bad finalize job"
smoke_raw=$(remote "sbatch --parsable --qos=gpu48 --array=0 --dependency=afterok:${finalize_id} --kill-on-invalid-dep=yes --export='${common},RUN_ROOT=${SMOKE_ROOT},BENCH_ROOT=${bench_root},MODE=smoke,MAX_STEPS=80' '${eval_script}'")
smoke_id=${smoke_raw%%;*}; [[ "${smoke_id}" =~ ^[0-9]+$ ]] || fail "bad smoke job"
eval_raw=$(remote "sbatch --parsable --qos=gpu48 --array=0-53%${CONCURRENCY} --dependency=afterok:${smoke_id} --kill-on-invalid-dep=yes --export='${common},BENCH_ROOT=${bench_root},MODE=eval,MAX_STEPS=600' '${eval_script}'")
eval_id=${eval_raw%%;*}; [[ "${eval_id}" =~ ^[0-9]+$ ]] || fail "bad eval array"
summary_raw=$(remote "sbatch --parsable --dependency=afterok:${eval_id} --kill-on-invalid-dep=yes --export='${common},BENCH_ROOT=${bench_root},MODE=summary' '${analysis_script}'")
summary_id=${summary_raw%%;*}; [[ "${summary_id}" =~ ^[0-9]+$ ]] || fail "bad summary job"
verify_raw=$(remote "sbatch --parsable --dependency=afterok:${summary_id} --kill-on-invalid-dep=yes --export='${common},BENCH_ROOT=${bench_root},MODE=verify' '${analysis_script}'")
verify_id=${verify_raw%%;*}; [[ "${verify_id}" =~ ^[0-9]+$ ]] || fail "bad verify job"

receipt=MemNavData/HM3D_FULLMONO_RUNTIME_REPAIR_SUBMISSION_RECEIPT_20260821.json
[[ ! -e "${receipt}" ]] || fail "local repair receipt already exists"
"${LOCAL_MEMNAV_PY}" - "${receipt}" "${RUN_ROOT}" "${SMOKE_ROOT}" \
  "${task_root}" "${task_receipt_sha}" "${pre_inventory}" \
  "${repair_id}" "${barrier_id}" "${construct_id}" "${finalize_id}" \
  "${smoke_id}" "${eval_id}" "${summary_id}" "${verify_id}" <<'PY'
import json,sys
(path,run,smoke,bundle,sha,inventory,repair,barrier,construct,finalize,
 smoke_job,evaluation,summary,verify)=sys.argv[1:]
payload={
 "schema_version":"hm3d_fullmono_runtime_repair_submission_v1_20260821",
 "run_root":run,"smoke_root":smoke,"task_bundle":bundle,
 "task_receipt_sha256":sha,"pre_repair_inventory":inventory,
 "repair_indices":[29,46,47,53],"concurrency":2,
 "original_collection_array":16120334,
 "superseded_downstream_jobs":[16120335,16120336,16120337,16120338,16120339,16120340],
 "jobs":{"goal_a_repair_array":int(repair),"collection_barrier":int(barrier),
         "construction_array":int(construct),"population_finalize":int(finalize),
         "query_smoke":int(smoke_job),"query_evaluation_array":int(evaluation),
         "summary":int(summary),"independent_verification":int(verify)},
 "scientific_guards":{"method_or_threshold_changed":False,
    "query_outcomes_read_before_repair":False,
    "completed_episode_overwrite_allowed":False},
}
open(path,"x").write(json.dumps(payload,indent=2,sort_keys=True)+"\n")
print(json.dumps(payload,indent=2,sort_keys=True))
PY
scp -q -o BatchMode=yes -o ControlMaster=no -o ControlPath="${SSH_CONTROL_PATH}" \
  "${receipt}" "${SSH_ALIAS}:${RUN_ROOT}/repair/submission.json"
remote "sha256sum '${RUN_ROOT}/repair/submission.json' >'${RUN_ROOT}/repair/submission.json.sha256'"

# These jobs never ran and can no longer release because the original Goal-A
# array contains failed/cancelled elements.  Cancel only after the replacement
# chain and its local receipt exist.
remote "scancel 16120335 16120336 16120337 16120338 16120339 16120340 || true"
echo "TASK_ROOT=${task_root}"
echo "REPAIR=${repair_id} BARRIER=${barrier_id} CONSTRUCT=${construct_id} FINALIZE=${finalize_id}"
echo "SMOKE=${smoke_id} EVAL=${eval_id} SUMMARY=${summary_id} VERIFY=${verify_id}"
