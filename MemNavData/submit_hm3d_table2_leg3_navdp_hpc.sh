#!/usr/bin/env bash
# Submit Table-2 Leg-3 NavDP native/CEC only after the independent construction
# verifier authorizes the frozen new-query population.
set -euo pipefail
umask 0022

ROOT=${ROOT:-/home/asus/Research/Nav-graph-blind}
SSH_ALIAS=${SSH_ALIAS:-alantorch}
LOCAL_MEMNAV_PY=/home/asus/miniconda3/envs/memnav/bin/python
LOCAL_HAB_PY=/home/asus/miniconda3/envs/habitat/bin/python
REMOTE_BUNDLES=/scratch/yz11502/Research/Nav-axis-uturn-source-bundles
REMOTE_RESULTS=/scratch/yz11502/Research/Nav-axis-uturn-results/hm3d_table2_leg3_mixed_role_20260829
CONSTRUCTION_RUN=${CONSTRUCTION_RUN:-${REMOTE_RESULTS}/construction_repair_20260829T064841Z_8e909a5b}
CONSTRUCTION_VERIFICATION=${CONSTRUCTION_RUN}/hm3d_table2_leg3_construction_verification.json
BENCH_ROOT=${CONSTRUCTION_RUN}/population/natural_direction
SOURCE_RUN_ROOT=/scratch/yz11502/Research/Nav-axis-uturn-results/hm3d_fullmono_lifelong_natural_v4_20260827/formal_materialize_20260827T133704Z_d85fc50d
PARENT_MANIFEST=/scratch/yz11502/Research/Nav-axis-uturn-results/hm3d_fresh_fullmono_mixed_role_20260820/formal_20260820T143609Z_e6dd44c6/sealed_inputs/parent_manifest.json
EXPECTED_PARENT_MANIFEST_SHA=a96a0b96fab7b7b47709b36cb8eeb9410b42b09f095f87ef01304a68de716dd5
BASE_SOURCE_ROOT=/scratch/yz11502/Research/Nav-axis-uturn-source-bundles/final14_mono_factorial_5690569a4373f2d2
BASE_RECEIPT=${BASE_SOURCE_ROOT}/source_inputs.sha256
EXPECTED_BASE_RECEIPT_SHA=5690569a4373f2d2768671418f0c604c4a03aa4b0ffe01baf70b288af03ba216
SERVER_SOURCE_ROOT=/scratch/yz11502/Research/Nav-axis-uturn-source-bundles/hm3d_table1_navdp_authority_transaction_718661db1733d5de
SERVER_SOURCE_RECEIPT=${SERVER_SOURCE_ROOT}/SOURCE_BUNDLE.sha256
EXPECTED_SERVER_SOURCE_RECEIPT_SHA=718661db1733d5de16cd86687eec880a8d02fc5ae5ca982e1ab7d5bde5e96f7d
CONCURRENCY=${CONCURRENCY:-2}
RUN_TAG=${RUN_TAG:-formal_$(date -u +%Y%m%dT%H%M%SZ)}
SSH_CONTROL_PATH=${SSH_CONTROL_PATH:-$(ssh -G "${SSH_ALIAS}" 2>/dev/null | awk '$1=="controlpath"{value=$2} END{print value}')}
cd "${ROOT}"

fail() { echo "ABORT: $*" >&2; exit 2; }
remote() {
  timeout 180 ssh -n -tt -o BatchMode=yes -o ControlMaster=no \
    -S "${SSH_CONTROL_PATH}" "${SSH_ALIAS}" "$@"
}
[[ -S "${SSH_CONTROL_PATH}" ]] || fail "authoritative SSH master missing"
[[ "${CONCURRENCY}" =~ ^[1-9][0-9]*$ ]] || fail "invalid concurrency"
[[ "${RUN_TAG}" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]] || fail "bad run tag"

files=(
  MemNavData/HM3D_TABLE2_LEG3_MIXED_ROLE_PROTOCOL_20260829.md
  MemNavData/hm3d_table2_leg3_mixed_role_protocol_20260829.json
  MemNavData/hm3d_fullmono_mixed_role.py
  MemNavData/final14_mono_factorial.py
  MemNavData/mdtec_raw_depth_gate_d.py
  MemNavData/run_hm3d_fullmono_query_history.py
  MemNavData/run_hm3d_fullmono_server_scene.sh
  MemNavData/run_final14_mono_factorial_episode.py
  MemNavData/eval_shared_online_role_pairs.py
  MemNavData/eval_2leg_habitat.py
  MemNavData/audit_shared_online_role_pairs.py
  MemNavData/shared_online_role_pair_contract.py
  MemNavData/shared_online_double_revisit_runtime.py
  MemNavData/generate_twoleg.py
  MemNavData/terminal_uturn.py
  MemNavData/visual_yaw_refinement.py
  MemNavData/deterministic_eval_protocol.py
  MemNavData/arrival_shadow.py
  MemNavData/bearing_diagnostics.py
  MemNavData/cec_bearing_alignment.py
  MemNavData/cec_authority_receipt.py
  MemNavData/navdp_goal_switch.py
  MemNavData/revisit_bearing_adapter.py
  MemNavData/revisit_action_shadow.py
  MemNavData/xnavdp_revisit_contract.py
  MemNavData/aggregate_hm3d_table1_navdp_pair.py
  MemNavData/independent_verify_hm3d_table1_navdp_pair.py
  MemNavData/slurm_hm3d_table2_leg3_navdp_pair.sbatch
  MemNavData/slurm_hm3d_table2_leg3_navdp_analysis.sbatch
  MemNavData/test_hm3d_table1_navdp_pair.py
  MemNavData/test_hm3d_table2_leg3_runtime.py
  MemNavData/test_hm3d_table2_leg3_analysis.py
  MemNavData/submit_hm3d_table2_leg3_navdp_hpc.sh
)
for file in "${files[@]}"; do
  [[ -f "${file}" && ! -L "${file}" ]] || fail "missing physical ${file}"
done

export PYTHONPATH=${ROOT}:${ROOT}/MemNavData${PYTHONPATH:+:${PYTHONPATH}}
"${LOCAL_MEMNAV_PY}" -m pytest -q \
  MemNavData/test_hm3d_table1_navdp_pair.py \
  MemNavData/test_hm3d_table2_leg3_runtime.py \
  MemNavData/test_hm3d_table2_leg3_analysis.py
"${LOCAL_MEMNAV_PY}" -m py_compile \
  MemNavData/aggregate_hm3d_table1_navdp_pair.py \
  MemNavData/independent_verify_hm3d_table1_navdp_pair.py
"${LOCAL_HAB_PY}" -m py_compile \
  MemNavData/run_hm3d_fullmono_query_history.py \
  MemNavData/eval_shared_online_role_pairs.py \
  MemNavData/eval_2leg_habitat.py
bash -n \
  MemNavData/run_hm3d_fullmono_server_scene.sh \
  MemNavData/slurm_hm3d_table2_leg3_navdp_pair.sbatch \
  MemNavData/slurm_hm3d_table2_leg3_navdp_analysis.sbatch \
  MemNavData/submit_hm3d_table2_leg3_navdp_hpc.sh

readarray -t gate < <(remote "python - '${CONSTRUCTION_VERIFICATION}' '${BENCH_ROOT}/manifest.json' '${PARENT_MANIFEST}' <<'PY'
import hashlib,json,sys
verification=json.load(open(sys.argv[1])); manifest=sys.argv[2]
if verification.get('verified') is not True:
 raise SystemExit('construction verifier did not pass')
if verification.get('construction_only') is not True:
 raise SystemExit('construction verifier is not construction-only')
if verification.get('formal_policy_evaluation_authorized') is not True:
 raise SystemExit('construction power gate did not authorize evaluation')
if verification.get('policy_outcomes_read') is not False:
 raise SystemExit('construction consumed a policy outcome')
digest=hashlib.sha256(open(manifest,'rb').read()).hexdigest()
if digest != verification.get('benchmark_manifest_sha256'):
 raise SystemExit('verified benchmark manifest changed')
parent=json.load(open(sys.argv[3])); scenes=parent.get('scenes')
if not isinstance(scenes,list) or not scenes:
 raise SystemExit('parent scene ledger is invalid')
print(verification['histories']); print(verification['scene_clusters'])
print(digest); print(hashlib.sha256(open(sys.argv[1],'rb').read()).hexdigest())
print(len(scenes)); print(hashlib.sha256(open(sys.argv[3],'rb').read()).hexdigest())
PY" | tr -d '\r')
[[ "${#gate[@]}" -eq 6 ]] || fail "construction gate returned bad receipt"
histories=${gate[0]}; scenes=${gate[1]}; manifest_sha=${gate[2]}
construction_verification_sha=${gate[3]}; parent_scene_count=${gate[4]}
[[ "${gate[5]}" == "${EXPECTED_PARENT_MANIFEST_SHA}" ]] || \
  fail "parent manifest changed"
[[ "${histories}" -ge 16 && "${scenes}" -ge 10 ]] || \
  fail "construction verifier violated frozen power gate"
scene_last=$((parent_scene_count - 1))

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
task_root=${REMOTE_BUNDLES}/hm3d_table2_leg3_navdp_${bundle_key}
task_stage=${task_root}.partial.$$
run_root=${REMOTE_RESULTS}/${RUN_TAG}_${bundle_key:0:8}

remote_identity=$(remote 'id -un' | tr -d '\r')
[[ "${remote_identity}" == yz11502 ]] || fail "wrong remote identity"
remote "set -euo pipefail
test \"\$(sha256sum '${BASE_RECEIPT}' | awk '{print \$1}')\" = '${EXPECTED_BASE_RECEIPT_SHA}'
cd '${BASE_SOURCE_ROOT}' && sha256sum -c --quiet '${BASE_RECEIPT}'
test \"\$(sha256sum '${SERVER_SOURCE_RECEIPT}' | awk '{print \$1}')\" = '${EXPECTED_SERVER_SOURCE_RECEIPT_SHA}'
cd '${SERVER_SOURCE_ROOT}' && sha256sum -c --quiet '${SERVER_SOURCE_RECEIPT}'
test \"\$(sha256sum '${PARENT_MANIFEST}' | awk '{print \$1}')\" = '${EXPECTED_PARENT_MANIFEST_SHA}'"

if remote "test -d '${task_root}'"; then
  remote "test \"\$(sha256sum '${task_root}/SOURCE_BUNDLE.sha256' | awk '{print \$1}')\" = '${task_receipt_sha}' && cd '${task_root}' && sha256sum -c --quiet SOURCE_BUNDLE.sha256"
else
  remote "test ! -e '${task_stage}' && mkdir -p '${task_stage}'"
  timeout 180 rsync -a --timeout=60 --chmod=Du=rwx,Dgo=rx,Fu=rw,Fgo=r \
    -e "ssh -o BatchMode=yes -o ControlMaster=no -S ${SSH_CONTROL_PATH}" \
    "${staging}/" "${SSH_ALIAS}:${task_stage}/"
  remote "cd '${task_stage}' && sha256sum -c --quiet SOURCE_BUNDLE.sha256 && chmod -R a-w '${task_stage}' && mv '${task_stage}' '${task_root}'"
fi

task_receipt=${task_root}/SOURCE_BUNDLE.sha256
protocol=${task_root}/MemNavData/hm3d_table2_leg3_mixed_role_protocol_20260829.json
remote "set -euo pipefail
test ! -e '${run_root}'
mkdir -p '${run_root}/sealed_inputs' '${run_root}/logs' '${run_root}/smoke' '${run_root}/formal' /scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs
cp '${CONSTRUCTION_VERIFICATION}' '${run_root}/sealed_inputs/'
cp '${CONSTRUCTION_RUN}/population/population_receipt.json' '${run_root}/sealed_inputs/'
cp '${protocol}' '${run_root}/sealed_inputs/'
sha256sum '${BENCH_ROOT}/manifest.json' '${CONSTRUCTION_VERIFICATION}' '${CONSTRUCTION_RUN}/population/population_receipt.json' >'${run_root}/sealed_inputs/experiment_inputs.sha256'
chmod -R a-w '${run_root}/sealed_inputs'"

remote "singularity exec --nv -B /scratch/lg154 -B /scratch/yz11502 /share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH='${task_root}:${task_root}/MemNavData:${SERVER_SOURCE_ROOT}:${SERVER_SOURCE_ROOT}/MemNavData:${BASE_SOURCE_ROOT}:${BASE_SOURCE_ROOT}/MemNavData' /scratch/lg154/conda-envs/habitat/bin/python -c 'import MemNavData.run_hm3d_fullmono_query_history as r; assert \"actual_ab\" in r.SCHEMAS'"
remote "singularity exec --nv -B /scratch/lg154 -B /scratch/yz11502 /share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH='${SERVER_SOURCE_ROOT}/NavDP/baselines/memnav:${BASE_SOURCE_ROOT}/NavDP/baselines/memnav' /scratch/lg154/conda-envs/memnav/bin/python -c 'import policy_agent,router_candidates; assert router_candidates.__file__.startswith(\"${SERVER_SOURCE_ROOT}/\"); assert hasattr(router_candidates,\"causal_goal_support_indices\")'"
remote "singularity exec --nv -B /scratch/lg154 -B /scratch/yz11502 /share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH='${SERVER_SOURCE_ROOT}/NavDP/baselines/navdp:${BASE_SOURCE_ROOT}/NavDP/baselines/navdp:${SERVER_SOURCE_ROOT}/NavDP/baselines/memnav:${BASE_SOURCE_ROOT}/NavDP/baselines/memnav' /scratch/lg154/conda-envs/memnav/bin/python -c 'import policy_agent; assert hasattr(policy_agent,\"NavDP_Agent\")'"

common="ALL,TASK_ROOT=${task_root},TASK_RECEIPT=${task_receipt},EXPECTED_TASK_RECEIPT_SHA=${task_receipt_sha},FORMAL_RUN_ROOT=${run_root},BENCH_ROOT=${BENCH_ROOT},CONSTRUCTION_VERIFICATION=${CONSTRUCTION_VERIFICATION},EXPECTED_CONSTRUCTION_VERIFICATION_SHA=${construction_verification_sha}"
pair_common="${common},BASE_SOURCE_ROOT=${BASE_SOURCE_ROOT},BASE_RECEIPT=${BASE_RECEIPT},EXPECTED_BASE_RECEIPT_SHA=${EXPECTED_BASE_RECEIPT_SHA},SERVER_SOURCE_ROOT=${SERVER_SOURCE_ROOT},SERVER_SOURCE_RECEIPT=${SERVER_SOURCE_RECEIPT},EXPECTED_SERVER_SOURCE_RECEIPT_SHA=${EXPECTED_SERVER_SOURCE_RECEIPT_SHA},SOURCE_RUN_ROOT=${SOURCE_RUN_ROOT},PARENT_MANIFEST=${PARENT_MANIFEST},PROTOCOL=${protocol},ROLE_PAIR_SCOPE=paper_heldout"
pair=${task_root}/MemNavData/slurm_hm3d_table2_leg3_navdp_pair.sbatch
analysis=${task_root}/MemNavData/slurm_hm3d_table2_leg3_navdp_analysis.sbatch
remote "sbatch --test-only --array=0 --export='${pair_common},PHASE=smoke' '${pair}' >/dev/null"
remote "sbatch --test-only --array=0-${scene_last}%${CONCURRENCY} --export='${pair_common},PHASE=formal' '${pair}' >/dev/null"
remote "sbatch --test-only --export='${common},MODE=aggregate' '${analysis}' >/dev/null"

smoke_raw=$(remote "sbatch --parsable --array=0 --export='${pair_common},PHASE=smoke' '${pair}'" | tr -d '\r')
smoke_id=${smoke_raw%%;*}; [[ "${smoke_id}" =~ ^[0-9]+$ ]] || fail "bad smoke job"
formal_raw=$(remote "sbatch --parsable --array=0-${scene_last}%${CONCURRENCY} --dependency=afterok:${smoke_id} --kill-on-invalid-dep=yes --export='${pair_common},PHASE=formal' '${pair}'" | tr -d '\r')
formal_id=${formal_raw%%;*}; [[ "${formal_id}" =~ ^[0-9]+$ ]] || fail "bad formal job"
aggregate_raw=$(remote "sbatch --parsable --dependency=afterok:${formal_id} --kill-on-invalid-dep=yes --export='${common},MODE=aggregate' '${analysis}'" | tr -d '\r')
aggregate_id=${aggregate_raw%%;*}; [[ "${aggregate_id}" =~ ^[0-9]+$ ]] || fail "bad aggregate job"
verify_raw=$(remote "sbatch --parsable --dependency=afterok:${aggregate_id} --kill-on-invalid-dep=yes --export='${common},MODE=verify' '${analysis}'" | tr -d '\r')
verify_id=${verify_raw%%;*}; [[ "${verify_id}" =~ ^[0-9]+$ ]] || fail "bad verifier job"

receipt=MemNavData/HM3D_TABLE2_LEG3_NAVDP_SUBMISSION_20260829.json
[[ ! -e "${receipt}" ]] || fail "local submission receipt already exists"
"${LOCAL_MEMNAV_PY}" - "${receipt}" "${run_root}" "${task_root}" \
  "${task_receipt_sha}" "${construction_verification_sha}" \
  "${manifest_sha}" "${histories}" "${scenes}" "${smoke_id}" \
  "${formal_id}" "${aggregate_id}" "${verify_id}" <<'PY'
import json,sys
(path,run,bundle,bundle_sha,construction_sha,manifest_sha,histories,scenes,
 smoke,formal,aggregate,verify)=sys.argv[1:]
payload={
 "schema_version":"hm3d_table2_leg3_navdp_submission_v1_20260829",
 "scope":"conditional Leg-3 after one sealed actual-mono Novel-A/Novel-B prefix",
 "run_root":run,"task_bundle":bundle,"task_receipt_sha256":bundle_sha,
 "construction_verification_sha256":construction_sha,
 "benchmark_manifest_sha256":manifest_sha,"histories":int(histories),
 "scene_clusters":int(scenes),"partial_policy_outcomes_read_at_submission":False,
 "history_contract":"actual_ab","runtime_role_visibility":"none",
 "unconditional_three_leg_joint_sr_reported":False,
 "jobs":{"navdp_smoke":int(smoke),"navdp_formal":int(formal),
         "aggregate":int(aggregate),"independent_verify":int(verify)},
}
open(path,"x").write(json.dumps(payload,indent=2,sort_keys=True)+"\n")
print(json.dumps(payload,indent=2,sort_keys=True))
PY
printf 'RUN_ROOT=%s\nTASK_ROOT=%s\nSMOKE=%s\nFORMAL=%s\nAGGREGATE=%s\nVERIFY=%s\n' \
  "${run_root}" "${task_root}" "${smoke_id}" "${formal_id}" \
  "${aggregate_id}" "${verify_id}"
