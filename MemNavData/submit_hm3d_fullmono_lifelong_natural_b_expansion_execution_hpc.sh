#!/usr/bin/env bash
# Submit the complete result-blind power-expansion DAG.  The deferred stages
# progress through factual B, prefix verification, exact population union,
# Table-II query construction, and paired policy evaluation only after each
# independent gate passes.
set -euo pipefail
umask 0022

ROOT=${ROOT:-/home/asus/Research/Nav-graph-blind}
SSH_ALIAS=${SSH_ALIAS:-alantorch}
LOCAL_MEMNAV_PY=${LOCAL_MEMNAV_PY:-/home/asus/miniconda3/envs/memnav/bin/python}
LOCAL_HAB_PY=${LOCAL_HAB_PY:-/home/asus/miniconda3/envs/habitat/bin/python}
REMOTE_BUNDLES=/scratch/yz11502/Research/Nav-axis-uturn-source-bundles
REMOTE_RESULTS=/scratch/yz11502/Research/Nav-axis-uturn-results/hm3d_fullmono_lifelong_natural_b_expansion_execution_20260830
PARENT_ROOT=/scratch/yz11502/Research/Nav-axis-uturn-results/hm3d_fresh_fullmono_mixed_role_20260820/formal_20260820T143609Z_e6dd44c6
PARENT_MANIFEST=${PARENT_ROOT}/sealed_inputs/parent_manifest.json
EXPECTED_PARENT_MANIFEST_SHA=a96a0b96fab7b7b47709b36cb8eeb9410b42b09f095f87ef01304a68de716dd5
SOURCE_TASK_ROOT=/scratch/yz11502/Research/Nav-axis-uturn-source-bundles/hm3d_fullmono_lifelong_375f0b6879b2ff87
SOURCE_TASK_RECEIPT=${SOURCE_TASK_ROOT}/SOURCE_BUNDLE.sha256
EXPECTED_SOURCE_TASK_RECEIPT_SHA=375f0b6879b2ff87b7019dae4727880d1b03fd3185a1862e6239942a76b5bcc8
SOURCE_PROTOCOL=${SOURCE_TASK_ROOT}/MemNavData/hm3d_fullmono_lifelong_power_expansion_protocol_20260826.json
EXPECTED_SOURCE_PROTOCOL_SHA=127a6796c64eeafd4b48906baad09c48c41edb925fefe0fa964ccb584d4af228
SOURCE_CONSTRUCTION_ROOT=/scratch/yz11502/Research/Nav-axis-uturn-results/hm3d_fullmono_lifelong_power_v3_20260826/formal_20260826T141733Z_375f0b68/construct_ab/scenes
AUDIT_SOURCE_ROOT=/scratch/yz11502/Research/Nav-axis-uturn-source-bundles/hm3d_lifelong_natural_b_expansion_994cfe9585d2467c
AUDIT_SOURCE_RECEIPT=${AUDIT_SOURCE_ROOT}/SOURCE_BUNDLE.sha256
EXPECTED_AUDIT_SOURCE_RECEIPT_SHA=994cfe9585d2467c1c214e923bfd080408b5c8c23f9229d2be57f3c5e15a25a4
AUDIT_RUN_ROOT=/scratch/yz11502/Research/Nav-axis-uturn-results/hm3d_fullmono_lifelong_natural_b_expansion_20260828/natural_b_expansion_audit_20260828T014626Z_994cfe95
BASE_SOURCE_ROOT=/scratch/yz11502/Research/Nav-axis-uturn-source-bundles/final14_mono_factorial_5690569a4373f2d2
BASE_RECEIPT=${BASE_SOURCE_ROOT}/source_inputs.sha256
EXPECTED_BASE_RECEIPT_SHA=5690569a4373f2d2768671418f0c604c4a03aa4b0ffe01baf70b288af03ba216
SERVER_SOURCE_ROOT=${SOURCE_TASK_ROOT}
SERVER_SOURCE_RECEIPT=${SOURCE_TASK_RECEIPT}
EXPECTED_SERVER_SOURCE_RECEIPT_SHA=${EXPECTED_SOURCE_TASK_RECEIPT_SHA}
TABLE2_SERVER_SOURCE_ROOT=/scratch/yz11502/Research/Nav-axis-uturn-source-bundles/hm3d_table1_navdp_authority_transaction_718661db1733d5de
TABLE2_SERVER_SOURCE_RECEIPT=${TABLE2_SERVER_SOURCE_ROOT}/SOURCE_BUNDLE.sha256
EXPECTED_TABLE2_SERVER_SOURCE_RECEIPT_SHA=718661db1733d5de16cd86687eec880a8d02fc5ae5ca982e1ab7d5bde5e96f7d
BASE_SIF=/share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif
REMOTE_MEMNAV_PY=/scratch/lg154/conda-envs/memnav/bin/python
REMOTE_HAB_PY=/scratch/lg154/conda-envs/habitat/bin/python
SSH_CONTROL_PATH=${SSH_CONTROL_PATH:-$(
  ssh -G "${SSH_ALIAS}" 2>/dev/null |
    awk '$1=="controlpath"{value=$2} END{print value}'
)}

cd "${ROOT}"
fail() { echo "ABORT: $*" >&2; exit 2; }
remote() {
  timeout 300 ssh -n -T -o BatchMode=yes -o ControlMaster=no \
    -o ServerAliveInterval=15 -S "${SSH_CONTROL_PATH}" "${SSH_ALIAS}" "$@"
}
job_id() { tr -d '\r' | awk -F';' '/^[0-9]+(;|$)/ {print $1; exit}'; }

[[ -x "${LOCAL_MEMNAV_PY}" && -x "${LOCAL_HAB_PY}" ]] || \
  fail "local interpreters missing"
[[ -n "${SSH_CONTROL_PATH}" && -S "${SSH_CONTROL_PATH}" ]] || \
  fail "authoritative shared SSH socket missing"
timeout 15 ssh -O check -S "${SSH_CONTROL_PATH}" "${SSH_ALIAS}" \
  >/dev/null 2>&1 || fail "shared SSH master is not responsive"

files=(
  MemNavData/hm3d_fullmono_lifelong.py
  MemNavData/hm3d_fullmono_lifelong_natural_b_expansion_execution_protocol_20260830.json
  MemNavData/materialize_hm3d_fullmono_lifelong_natural_b_expansion.py
  MemNavData/independent_verify_hm3d_fullmono_lifelong_natural_b_expansion_materialization.py
  MemNavData/test_materialize_hm3d_fullmono_lifelong_natural_b_expansion.py
  MemNavData/test_independent_verify_hm3d_fullmono_lifelong_natural_b_expansion_materialization.py
  MemNavData/test_hm3d_fullmono_lifelong.py
  MemNavData/materialize_hm3d_fullmono_lifelong_natural_ab.py
  MemNavData/construct_hm3d_fullmono_lifelong_ab.py
  MemNavData/finalize_hm3d_fullmono_lifelong_ab.py
  MemNavData/build_hm3d_fullmono_lifelong_natural_v4_b_shards.py
  MemNavData/test_build_hm3d_fullmono_lifelong_natural_v4_b_shards.py
  MemNavData/collect_hm3d_fullmono_lifelong_b.py
  MemNavData/construct_hm3d_fullmono_lifelong_prefix.py
  MemNavData/finalize_hm3d_fullmono_lifelong_population.py
  MemNavData/independent_verify_hm3d_fullmono_lifelong_natural_v4_population.py
  MemNavData/test_independent_verify_hm3d_fullmono_lifelong_natural_v4_population.py
  MemNavData/merge_hm3d_fullmono_lifelong_populations.py
  MemNavData/independent_verify_hm3d_fullmono_lifelong_population_union.py
  MemNavData/test_hm3d_fullmono_lifelong_population_union.py
  MemNavData/hm3d_table2_leg3_mixed_role_protocol_20260829.json
  MemNavData/hm3d_table2_leg3_mixed_role.py
  MemNavData/construct_hm3d_table2_leg3_mixed_role.py
  MemNavData/finalize_hm3d_table2_leg3_mixed_role.py
  MemNavData/independent_verify_hm3d_table2_leg3_mixed_role.py
  MemNavData/test_hm3d_table2_leg3_mixed_role.py
  MemNavData/test_hm3d_table2_leg3_runtime.py
  MemNavData/test_hm3d_table2_leg3_analysis.py
  MemNavData/build_final14_role_pair_scene.py
  MemNavData/final14_role_pair_contract.py
  MemNavData/shared_online_role_pair_contract.py
  MemNavData/build_shared_online_double_revisit.py
  MemNavData/build_shared_online_role_pairs.py
  MemNavData/audit_shared_online_role_pairs.py
  MemNavData/generate_twoleg.py
  MemNavData/materialize_online_a_traces.py
  MemNavData/deterministic_eval_protocol.py
  MemNavData/final14_mono_factorial.py
  MemNavData/hm3d_fullmono_mixed_role.py
  MemNavData/mdtec_raw_depth_gate_d.py
  MemNavData/run_hm3d_fullmono_query_history.py
  MemNavData/run_hm3d_fullmono_server_scene.sh
  MemNavData/run_final14_mono_factorial_episode.py
  MemNavData/eval_shared_online_role_pairs.py
  MemNavData/eval_2leg_habitat.py
  MemNavData/terminal_uturn.py
  MemNavData/visual_yaw_refinement.py
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
  MemNavData/slurm_hm3d_fullmono_lifelong_natural_b_expansion_materialize.sbatch
  MemNavData/slurm_hm3d_fullmono_lifelong_natural_b_expansion_seal.sbatch
  MemNavData/slurm_hm3d_fullmono_lifelong_natural_b_expansion_materialization_verify.sbatch
  MemNavData/slurm_hm3d_fullmono_lifelong_natural_b_expansion_launch_b.sbatch
  MemNavData/slurm_hm3d_fullmono_lifelong_natural_b_expansion_finish.sbatch
  MemNavData/slurm_hm3d_fullmono_lifelong_natural_b_expansion_finalize_verify.sbatch
  MemNavData/slurm_hm3d_fullmono_lifelong_population_union_launch_table2.sbatch
  MemNavData/slurm_hm3d_fullmono_lifelong_natural_v4_collect_b_shard.sbatch
  MemNavData/slurm_hm3d_fullmono_lifelong_natural_v4_construct_prefix.sbatch
  MemNavData/slurm_hm3d_table2_leg3_construct.sbatch
  MemNavData/slurm_hm3d_table2_leg3_analysis.sbatch
  MemNavData/slurm_hm3d_table2_leg3_power_policy_launch.sbatch
  MemNavData/slurm_hm3d_table2_leg3_navdp_pair.sbatch
  MemNavData/slurm_hm3d_table2_leg3_navdp_analysis.sbatch
  MemNavData/slurm_safe_submit.sh
  MemNavData/submit_hm3d_fullmono_lifelong_natural_b_expansion_execution_hpc.sh
)
for path in "${files[@]}"; do
  [[ -f "${path}" && ! -L "${path}" ]] || fail "missing physical input ${path}"
done

python -m json.tool \
  MemNavData/hm3d_fullmono_lifelong_natural_b_expansion_execution_protocol_20260830.json \
  >/dev/null
shell_files=()
for path in "${files[@]}"; do
  case "${path}" in *.sh|*.sbatch) shell_files+=("${path}");; esac
done
bash -n "${shell_files[@]}"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=${ROOT}:${ROOT}/MemNavData
"${LOCAL_MEMNAV_PY}" -m pytest -q \
  MemNavData/test_hm3d_fullmono_lifelong_population_union.py \
  MemNavData/test_hm3d_table2_leg3_mixed_role.py \
  MemNavData/test_hm3d_table2_leg3_runtime.py \
  MemNavData/test_hm3d_table2_leg3_analysis.py
"${LOCAL_HAB_PY}" -m unittest -q \
  MemNavData.test_hm3d_fullmono_lifelong \
  MemNavData.test_materialize_hm3d_fullmono_lifelong_natural_b_expansion \
  MemNavData.test_independent_verify_hm3d_fullmono_lifelong_natural_b_expansion_materialization \
  MemNavData.test_independent_verify_hm3d_fullmono_lifelong_natural_v4_population
"${LOCAL_MEMNAV_PY}" -m py_compile \
  MemNavData/merge_hm3d_fullmono_lifelong_populations.py \
  MemNavData/independent_verify_hm3d_fullmono_lifelong_population_union.py \
  MemNavData/finalize_hm3d_table2_leg3_mixed_role.py \
  MemNavData/independent_verify_hm3d_table2_leg3_mixed_role.py
"${LOCAL_HAB_PY}" -m py_compile \
  MemNavData/materialize_hm3d_fullmono_lifelong_natural_b_expansion.py \
  MemNavData/construct_hm3d_table2_leg3_mixed_role.py

scratch=$(mktemp -d /tmp/h3life_natbx_exec.XXXXXX)
cleanup() { rm -rf -- "${scratch}"; }
trap cleanup EXIT
mkdir -p "${scratch}/root"
for path in "${files[@]}"; do
  mkdir -p "${scratch}/root/$(dirname "${path}")"
  cp -p -- "${path}" "${scratch}/root/${path}"
done
(
  cd "${scratch}/root"
  find . -type f ! -name SOURCE_BUNDLE.sha256 -print0 | sort -z | \
    xargs -0 sha256sum >SOURCE_BUNDLE.sha256
  sha256sum -c --quiet SOURCE_BUNDLE.sha256
)
task_receipt_sha=$(sha256sum "${scratch}/root/SOURCE_BUNDLE.sha256" | awk '{print $1}')
bundle_key=${task_receipt_sha:0:16}
task_root=${REMOTE_BUNDLES}/hm3d_lifelong_natural_b_expansion_execution_${bundle_key}
task_stage=${task_root}.partial.$$
run_tag=formal_$(date -u +%Y%m%dT%H%M%SZ)_${bundle_key:0:8}
run_root=${REMOTE_RESULTS}/${run_tag}
smoke_root=${REMOTE_RESULTS}/${run_tag}_smoke

protocol_local=MemNavData/hm3d_fullmono_lifelong_natural_b_expansion_execution_protocol_20260830.json
protocol_sha=$(sha256sum "${protocol_local}" | awk '{print $1}')
preflight=$(remote "set -euo pipefail
test \"\$(id -un)\" = yz11502
test \"\$(sha256sum '${PARENT_MANIFEST}' | awk '{print \$1}')\" = '${EXPECTED_PARENT_MANIFEST_SHA}'
test \"\$(sha256sum '${SOURCE_TASK_RECEIPT}' | awk '{print \$1}')\" = '${EXPECTED_SOURCE_TASK_RECEIPT_SHA}'
test \"\$(sha256sum '${SOURCE_PROTOCOL}' | awk '{print \$1}')\" = '${EXPECTED_SOURCE_PROTOCOL_SHA}'
test \"\$(sha256sum '${AUDIT_SOURCE_RECEIPT}' | awk '{print \$1}')\" = '${EXPECTED_AUDIT_SOURCE_RECEIPT_SHA}'
test \"\$(sha256sum '${BASE_RECEIPT}' | awk '{print \$1}')\" = '${EXPECTED_BASE_RECEIPT_SHA}'
test \"\$(sha256sum '${TABLE2_SERVER_SOURCE_RECEIPT}' | awk '{print \$1}')\" = '${EXPECTED_TABLE2_SERVER_SOURCE_RECEIPT_SHA}'
cd '${SOURCE_TASK_ROOT}' && sha256sum -c --quiet '${SOURCE_TASK_RECEIPT}'
cd '${AUDIT_SOURCE_ROOT}' && sha256sum -c --quiet '${AUDIT_SOURCE_RECEIPT}'
cd '${BASE_SOURCE_ROOT}' && sha256sum -c --quiet '${BASE_RECEIPT}'
cd '${TABLE2_SERVER_SOURCE_ROOT}' && sha256sum -c --quiet '${TABLE2_SERVER_SOURCE_RECEIPT}'
test \"\$(sha256sum '${AUDIT_RUN_ROOT}/expansion_audit/summary.json' | awk '{print \$1}')\" = b3ba1ffee79aba0e6aa002fb1bd5b26ae1616350be231aee52a8cc31533da59d
test \"\$(sha256sum '${AUDIT_RUN_ROOT}/independent_expansion_audit_verification.json' | awk '{print \$1}')\" = 2d56a9d6e5c3178567139709c04270a2cb144aaab8a4c0a573ce39e007ce118d
test -x '${REMOTE_MEMNAV_PY}' && test -x '${REMOTE_HAB_PY}' && test -r '${BASE_SIF}'
test ! -e '${run_root}' && test ! -e '${smoke_root}'
echo PREFLIGHT_OK")
[[ "${preflight}" == *PREFLIGHT_OK* ]] || fail "remote preflight incomplete"

if remote "test -d '${task_root}'"; then
  remote "test \"\$(sha256sum '${task_root}/SOURCE_BUNDLE.sha256' | awk '{print \$1}')\" = '${task_receipt_sha}'; cd '${task_root}'; sha256sum -c --quiet SOURCE_BUNDLE.sha256"
else
  remote "test ! -e '${task_stage}'; mkdir -p '${task_stage}'"
  timeout 300 rsync -a --partial \
    --chmod=Du=rwx,Dgo=rx,Fu=rw,Fgo=r \
    -e "ssh -o BatchMode=yes -o ControlMaster=no -S ${SSH_CONTROL_PATH}" \
    "${scratch}/root/" "${SSH_ALIAS}:${task_stage}/"
  remote "cd '${task_stage}'; sha256sum -c --quiet SOURCE_BUNDLE.sha256; chmod -R a-w '${task_stage}'; mv '${task_stage}' '${task_root}'"
fi

task_receipt=${task_root}/SOURCE_BUNDLE.sha256
protocol=${task_root}/${protocol_local}
remote "set -euo pipefail
test \"\$(sha256sum '${protocol}' | awk '{print \$1}')\" = '${protocol_sha}'
mkdir -p '${run_root}/construct_ab/scenes' '${run_root}/logs' '${run_root}/sealed_inputs' '${smoke_root}/construct_ab/scenes' '${smoke_root}/logs' /scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs
cp '${protocol}' '${run_root}/sealed_inputs/'
sha256sum '${PARENT_MANIFEST}' '${AUDIT_RUN_ROOT}/expansion_audit/summary.json' '${AUDIT_RUN_ROOT}/independent_expansion_audit_verification.json' >'${run_root}/sealed_inputs/source_inputs.sha256'
chmod -R a-w '${run_root}/sealed_inputs'"

common="ALL,TASK_ROOT=${task_root},TASK_RECEIPT=${task_receipt},EXPECTED_TASK_RECEIPT_SHA=${task_receipt_sha},BASE_SOURCE_ROOT=${BASE_SOURCE_ROOT},BASE_RECEIPT=${BASE_RECEIPT},EXPECTED_BASE_RECEIPT_SHA=${EXPECTED_BASE_RECEIPT_SHA},SOURCE_TASK_ROOT=${SOURCE_TASK_ROOT},SOURCE_TASK_RECEIPT=${SOURCE_TASK_RECEIPT},EXPECTED_SOURCE_TASK_RECEIPT_SHA=${EXPECTED_SOURCE_TASK_RECEIPT_SHA},AUDIT_SOURCE_ROOT=${AUDIT_SOURCE_ROOT},AUDIT_SOURCE_RECEIPT=${AUDIT_SOURCE_RECEIPT},EXPECTED_AUDIT_SOURCE_RECEIPT_SHA=${EXPECTED_AUDIT_SOURCE_RECEIPT_SHA},SERVER_SOURCE_ROOT=${SERVER_SOURCE_ROOT},SERVER_SOURCE_RECEIPT=${SERVER_SOURCE_RECEIPT},EXPECTED_SERVER_SOURCE_RECEIPT_SHA=${EXPECTED_SERVER_SOURCE_RECEIPT_SHA},TABLE2_SERVER_SOURCE_ROOT=${TABLE2_SERVER_SOURCE_ROOT},TABLE2_SERVER_SOURCE_RECEIPT=${TABLE2_SERVER_SOURCE_RECEIPT},EXPECTED_TABLE2_SERVER_SOURCE_RECEIPT_SHA=${EXPECTED_TABLE2_SERVER_SOURCE_RECEIPT_SHA},RUN_ROOT=${run_root},PARENT_ROOT=${PARENT_ROOT},PROTOCOL=${protocol},EXPECTED_PROTOCOL_SHA=${protocol_sha},AUDIT_RUN_ROOT=${AUDIT_RUN_ROOT},SOURCE_CONSTRUCTION_ROOT=${SOURCE_CONSTRUCTION_ROOT},SOURCE_PROTOCOL=${SOURCE_PROTOCOL}"
materialize=${task_root}/MemNavData/slurm_hm3d_fullmono_lifelong_natural_b_expansion_materialize.sbatch
seal=${task_root}/MemNavData/slurm_hm3d_fullmono_lifelong_natural_b_expansion_seal.sbatch
verify=${task_root}/MemNavData/slurm_hm3d_fullmono_lifelong_natural_b_expansion_materialization_verify.sbatch
launch=${task_root}/MemNavData/slurm_hm3d_fullmono_lifelong_natural_b_expansion_launch_b.sbatch
safe=${task_root}/MemNavData/slurm_safe_submit.sh
remote "source '${safe}'; safe_sbatch --lint-fatal --test-only --qos=gpu48 --time=00:45:00 --array=0 --export='${common},RUN_ROOT=${smoke_root}' '${materialize}' >/dev/null"
remote "source '${safe}'; safe_sbatch --lint-fatal --test-only --qos=gpu48 --time=00:45:00 --array=0-53%4 --export='${common}' '${materialize}' >/dev/null"
remote "source '${safe}'; safe_sbatch --lint-fatal --test-only --partition=cpu_short --export='${common}' '${seal}' >/dev/null"
remote "source '${safe}'; safe_sbatch --lint-fatal --test-only --partition=cpu_short --export='${common}' '${verify}' >/dev/null"
remote "source '${safe}'; safe_sbatch --lint-fatal --test-only --partition=cpu_short --export='${common}' '${launch}' >/dev/null"

smoke_raw=$(remote "source '${safe}'; safe_sbatch --lint-fatal --parsable --qos=gpu48 --time=00:45:00 --array=0 --export='${common},RUN_ROOT=${smoke_root}' '${materialize}'")
smoke_job=$(printf '%s\n' "${smoke_raw}" | job_id)
[[ "${smoke_job}" =~ ^[0-9]+$ ]] || fail "bad smoke job"
formal_raw=$(remote "source '${safe}'; safe_sbatch --lint-fatal --parsable --qos=gpu48 --time=00:45:00 --array=0-53%4 --dependency='afterok:${smoke_job}' --kill-on-invalid-dep=yes --export='${common}' '${materialize}'")
formal_job=$(printf '%s\n' "${formal_raw}" | job_id)
[[ "${formal_job}" =~ ^[0-9]+$ ]] || fail "bad materialization job"
seal_raw=$(remote "source '${safe}'; safe_sbatch --lint-fatal --parsable --partition=cpu_short --dependency='afterok:${formal_job}' --kill-on-invalid-dep=yes --export='${common}' '${seal}'")
seal_job=$(printf '%s\n' "${seal_raw}" | job_id)
[[ "${seal_job}" =~ ^[0-9]+$ ]] || fail "bad seal job"
verify_raw=$(remote "source '${safe}'; safe_sbatch --lint-fatal --parsable --partition=cpu_short --dependency='afterok:${seal_job}' --kill-on-invalid-dep=yes --export='${common}' '${verify}'")
verify_job=$(printf '%s\n' "${verify_raw}" | job_id)
[[ "${verify_job}" =~ ^[0-9]+$ ]] || fail "bad verifier job"
launch_raw=$(remote "source '${safe}'; safe_sbatch --lint-fatal --parsable --partition=cpu_short --dependency='afterok:${verify_job}' --kill-on-invalid-dep=yes --export='${common}' '${launch}'")
launch_job=$(printf '%s\n' "${launch_raw}" | job_id)
[[ "${launch_job}" =~ ^[0-9]+$ ]] || fail "bad deferred launcher job"

receipt=MemNavData/HM3D_NATURAL_B_EXPANSION_EXECUTION_SUBMISSION_20260830.json
[[ ! -e "${receipt}" ]] || fail "local submission receipt exists"
"${LOCAL_MEMNAV_PY}" - "${receipt}" "${run_root}" "${smoke_root}" \
  "${task_root}" "${task_receipt_sha}" "${protocol_sha}" "${smoke_job}" \
  "${formal_job}" "${seal_job}" "${verify_job}" "${launch_job}" <<'PY'
import json,sys
(path,run,smoke,bundle,bundle_sha,protocol_sha,smoke_job,formal,seal,
 verify,launch)=sys.argv[1:]
p={'schema_version':'hm3d_natural_b_expansion_execution_submission_v1_20260830',
   'scope':'complete result-blind expansion through deferred Table-II evaluation',
   'run_root':run,'smoke_root':smoke,'task_bundle':bundle,
   'task_receipt_sha256':bundle_sha,'protocol_sha256':protocol_sha,
   'jobs':{'materialization_smoke':int(smoke_job),
           'materialization_array':int(formal),'materialization_seal':int(seal),
           'materialization_independent_verify':int(verify),
           'factual_B_deferred_launcher':int(launch)},
   'factual_B_or_leg3_outcomes_read_at_submission':False,
   'threshold_relaxation':False,'fallback_completion_allowed':False}
open(path,'x').write(json.dumps(p,indent=2,sort_keys=True)+'\n')
print(json.dumps(p,indent=2,sort_keys=True))
PY
timeout 180 rsync -a --chmod=Fugo=r \
  -e "ssh -o BatchMode=yes -o ControlMaster=no -S ${SSH_CONTROL_PATH}" \
  "${receipt}" "${SSH_ALIAS}:${run_root}/submission_receipt.json"
remote "sha256sum '${run_root}/submission_receipt.json' >'${run_root}/submission_receipt.json.sha256'; chmod a-w '${run_root}/submission_receipt.json' '${run_root}/submission_receipt.json.sha256'"
printf 'RUN_ROOT=%s\nTASK_ROOT=%s\nSMOKE=%s\nMATERIALIZE=%s\nSEAL=%s\nVERIFY=%s\nDEFERRED=%s\n' \
  "${run_root}" "${task_root}" "${smoke_job}" "${formal_job}" \
  "${seal_job}" "${verify_job}" "${launch_job}"
